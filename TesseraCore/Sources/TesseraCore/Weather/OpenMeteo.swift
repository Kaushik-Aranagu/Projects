import Foundation

/// Open-Meteo (free, no key, no account): the forecast and the air quality for a coordinate. Kaushik, Oct 23 10:28 PM:
/// on by default, any free service is fine.
///
/// What leaves the phone: one HTTPS request per half hour at most, carrying the coordinate rounded to 0.1 degree (about
/// 11 km) and nothing else: no key, no id, no cookie, no name. The rounded coordinate is never stored, logged or
/// backed up, no place name is ever shown, and the response is held in memory only. A failed fetch throws; the host
/// falls
/// back to the sample.
public final class OpenMeteoSource: WeatherSource, @unchecked Sendable {
    public typealias Fetch = @Sendable (URL) async throws -> Data

    /// The coordinate rounded to a tenth of a degree.
    public static func rounded(_ value: Double) -> Double {
        (value * 10).rounded() / 10
    }

    public static let cacheSeconds: TimeInterval = 1800

    private let fetch: Fetch
    private let lock = NSLock()
    private var cached: (key: String, at: Date, forecast: WeatherReport)?
    private var inFlight: (key: String, task: Task<WeatherReport, any Error>)?

    public init(fetch: @escaping Fetch = OpenMeteoSource.liveFetch) {
        self.fetch = fetch
    }

    /// One session for every fetch: no cookies, no disk cache.
    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 15
        return URLSession(configuration: configuration)
    }()

    /// A plain GET with no cookies, no redirects away from Open-Meteo, and no caching on disk.
    public static let liveFetch: Fetch = { url in
        let (data, response) = try await session.data(from: url, delegate: SameHostOnly())
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            throw WeatherError("Open-Meteo answered with an error")
        }
        return data
    }

    // MARK: URLs

    static func forecastURL(latitude: Double, longitude: Double) -> URL? {
        var parts = URLComponents(string: "https://api.open-meteo.com/v1/forecast")
        parts?.queryItems = [
            .init(name: "latitude", value: String(format: "%.1f", rounded(latitude))),
            .init(name: "longitude", value: String(format: "%.1f", rounded(longitude))),
            .init(
                name: "current",
                value: "temperature_2m,apparent_temperature,weather_code,is_day,wind_speed_10m,wind_gusts_10m,wind_direction_10m"
            ),
            .init(
                name: "hourly",
                value: "temperature_2m,apparent_temperature,precipitation_probability,weather_code,is_day,wind_speed_10m,wind_gusts_10m,uv_index"
            ),
            .init(
                name: "daily",
                value: "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_probability_max,wind_gusts_10m_max,uv_index_max,sunrise,sunset"
            ),
            .init(name: "forecast_days", value: "7"),
            .init(name: "forecast_hours", value: "48"),
            .init(name: "timezone", value: "auto"),
        ]
        return parts?.url
    }

    static func airURL(latitude: Double, longitude: Double) -> URL? {
        var parts = URLComponents(string: "https://air-quality-api.open-meteo.com/v1/air-quality")
        parts?.queryItems = [
            .init(name: "latitude", value: String(format: "%.1f", rounded(latitude))),
            .init(name: "longitude", value: String(format: "%.1f", rounded(longitude))),
            .init(name: "current", value: "us_aqi,pm2_5,pm10"),
            .init(name: "timezone", value: "auto"),
        ]
        return parts?.url
    }

    // MARK: WeatherSource

    private func held(_ key: String, now: Date) -> WeatherReport? {
        lock.lock()
        defer { lock.unlock() }
        guard let cached, cached.key == key, now >= cached.at,
              now.timeIntervalSince(cached.at) < Self.cacheSeconds
        else {
            return nil
        }
        return cached.forecast
    }

    private func hold(_ forecast: WeatherReport, key: String, now: Date) {
        lock.lock()
        defer { lock.unlock() }
        cached = (key, now, forecast)
    }

    private func joinOrStart(key: String,
                             _ work: @escaping @Sendable () async throws -> WeatherReport)
        -> Task<WeatherReport, any Error>
    {
        lock.lock()
        defer { lock.unlock() }
        if let running = inFlight, running.key == key {
            return running.task
        }
        let task = Task { try await work() }
        inFlight = (key, task)
        return task
    }

    private func finish(key: String) {
        lock.lock()
        defer { lock.unlock() }
        if inFlight?.key == key {
            inFlight = nil
        }
    }

    public func forecast(latitude: Double, longitude: Double, now: Date) async throws -> WeatherReport {
        let key = "\(Self.rounded(latitude)),\(Self.rounded(longitude))"
        if let hit = held(key, now: now) {
            return hit
        }
        guard let url = Self.forecastURL(latitude: latitude, longitude: longitude) else {
            throw WeatherError("no forecast address")
        }
        // A fetch already going for this place is joined, not repeated.
        let airURL = Self.airURL(latitude: latitude, longitude: longitude)
        let task = joinOrStart(key: key) { [fetch] in
            var forecast = try await Self.parse(forecast: fetch(url), fetchedAt: now)
            // Air quality is a second, optional request; a failure leaves it out.
            if let airURL, let data = try? await fetch(airURL) {
                forecast.air = Self.parse(air: data)
            }
            return forecast
        }
        defer { finish(key: key) }
        let forecast = try await task.value
        hold(forecast, key: key, now: now)
        return forecast
    }

    /// Forgets the held forecast (the host calls it when he leaves the Weather room off, or on a memory warning).
    public func forget() {
        lock.lock()
        defer { lock.unlock() }
        cached = nil
    }

    // MARK: Parsing

    private struct ForecastResponse: Decodable {
        struct Current: Decodable {
            let temperature_2m: Double
            let apparent_temperature: Double
            let weather_code: Int
            let is_day: Int
            let wind_speed_10m: Double
            let wind_gusts_10m: Double
            let wind_direction_10m: Double?
        }

        struct Hourly: Decodable {
            let time: [String]
            let temperature_2m: [Double?]
            let apparent_temperature: [Double?]
            let precipitation_probability: [Int?]
            let weather_code: [Int?]
            let is_day: [Int?]
            let wind_speed_10m: [Double?]
            let wind_gusts_10m: [Double?]
            let uv_index: [Double?]
        }

        struct Daily: Decodable {
            let time: [String]
            let weather_code: [Int?]
            let temperature_2m_max: [Double?]
            let temperature_2m_min: [Double?]
            let apparent_temperature_max: [Double?]
            let precipitation_probability_max: [Int?]
            let wind_gusts_10m_max: [Double?]
            let uv_index_max: [Double?]
            let sunrise: [String?]
            let sunset: [String?]
        }

        let utc_offset_seconds: Int
        let current: Current
        let hourly: Hourly
        let daily: Daily
    }

    private struct AirResponse: Decodable {
        struct Current: Decodable {
            let us_aqi: Double?
            let pm2_5: Double?
            let pm10: Double?
        }

        let current: Current
    }

    /// "2026-10-24T14:00" or "2026-10-24" read in the place's own offset (timezone=auto sends local times, no zone).
    static func date(_ text: String, offset: Int) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: offset)
        formatter.dateFormat = text.count > 10 ? "yyyy-MM-dd'T'HH:mm" : "yyyy-MM-dd"
        return formatter.date(from: text)
    }

    static func parse(forecast data: Data, fetchedAt: Date) throws -> WeatherReport {
        let response: ForecastResponse
        do {
            response = try JSONDecoder().decode(ForecastResponse.self, from: data)
        } catch {
            throw WeatherError("the forecast could not be read")
        }
        let offset = response.utc_offset_seconds
        guard let condition = WeatherCondition(wmo: response.current.weather_code) else {
            throw WeatherError("an unknown weather code")
        }
        let now = WeatherReport.Now(
            condition: condition, isDay: response.current.is_day == 1, temperatureC: response.current.temperature_2m,
            apparentC: response.current.apparent_temperature, windKph: response.current.wind_speed_10m,
            gustKph: response.current.wind_gusts_10m, windFromDegrees: response.current.wind_direction_10m
        )
        let h = response.hourly
        var hours: [WeatherReport.Hour] = []
        for index in h.time.indices {
            guard let time = date(h.time[index], offset: offset), let temp = h.temperature_2m[safe: index] ?? nil,
                  let code = h.weather_code[safe: index] ?? nil,
                  let kind = WeatherCondition(wmo: code) else { continue }
            hours.append(WeatherReport.Hour(
                time: time, condition: kind, isDay: (h.is_day[safe: index] ?? 1) == 1, temperatureC: temp,
                apparentC: (h.apparent_temperature[safe: index] ?? nil) ?? temp,
                precipitationChance: (h.precipitation_probability[safe: index] ?? nil) ?? 0,
                windKph: (h.wind_speed_10m[safe: index] ?? nil) ?? 0,
                gustKph: (h.wind_gusts_10m[safe: index] ?? nil) ?? 0,
                uvIndex: (h.uv_index[safe: index] ?? nil) ?? 0
            ))
        }
        let d = response.daily
        var days: [WeatherReport.Day] = []
        for index in d.time.indices {
            guard let day = date(d.time[index], offset: offset), let high = d.temperature_2m_max[safe: index] ?? nil,
                  let low = d.temperature_2m_min[safe: index] ?? nil, let code = d.weather_code[safe: index] ?? nil,
                  let kind = WeatherCondition(wmo: code) else { continue }
            days.append(WeatherReport.Day(
                date: day, condition: kind, highC: high, lowC: low,
                apparentHighC: (d.apparent_temperature_max[safe: index] ?? nil) ?? high,
                precipitationChance: (d.precipitation_probability_max[safe: index] ?? nil) ?? 0,
                gustKph: (d.wind_gusts_10m_max[safe: index] ?? nil) ?? 0,
                uvIndex: (d.uv_index_max[safe: index] ?? nil) ?? 0,
                sunrise: (d.sunrise[safe: index] ?? nil).flatMap { date($0, offset: offset) },
                sunset: (d.sunset[safe: index] ?? nil).flatMap { date($0, offset: offset) }
            ))
        }
        guard !hours.isEmpty, !days.isEmpty else { throw WeatherError("the forecast had no hours or days") }
        return WeatherReport(fetchedAt: fetchedAt, now: now, hourly: hours, daily: days)
    }

    static func parse(air data: Data) -> WeatherReport.AirQuality? {
        guard let response = try? JSONDecoder().decode(AirResponse.self, from: data), let aqi = response.current.us_aqi,
              aqi.isFinite else { return nil }
        return WeatherReport.AirQuality(usAQI: Int(aqi.rounded()), pm25: response.current.pm2_5,
                                        pm10: response.current.pm10)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

/// Follows a redirect only over https and inside Open-Meteo's own host, so the rounded coordinate never travels
/// anywhere else.
private final class SameHostOnly: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(_: URLSession, task: URLSessionTask, willPerformHTTPRedirection _: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest?
    {
        request.url?.scheme == "https" && request.url?.host == task.originalRequest?.url?.host ? request : nil
    }
}
