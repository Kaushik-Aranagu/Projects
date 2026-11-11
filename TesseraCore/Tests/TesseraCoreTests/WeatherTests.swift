import Foundation
@testable import TesseraCore
import Testing

/// The weather model and the Open-Meteo source: every WMO code named, warnings derived from the numbers, the coordinate
/// rounded to 0.1 degree, a 30 minute cache, and nothing that says where.
struct WeatherTests {
    // MARK: Conditions

    @Test func everyWMOCodeOpenMeteoReportsHasACondition() {
        let codes = [
            0,
            1,
            2,
            3,
            45,
            48,
            51,
            53,
            55,
            56,
            57,
            61,
            63,
            65,
            66,
            67,
            71,
            73,
            75,
            77,
            80,
            81,
            82,
            85,
            86,
            95,
            96,
            99,
        ]
        for code in codes {
            #expect(WeatherCondition(wmo: code) != nil, "WMO \(code)")
        }
        // Every case is reached by some code, so Design draws only effects that can happen.
        let reached = Set(codes.compactMap { WeatherCondition(wmo: $0) })
        #expect(reached == Set(WeatherCondition.allCases))
        for code in [4, 10, 44, 100, -1] {
            #expect(WeatherCondition(wmo: code) == nil)
        }
    }

    @Test func rainCloudsAndWhiteCloudsAreTold() {
        #expect(WeatherCondition.partlyCloudy.cloud == .white && WeatherCondition.overcast.cloud == .white)
        #expect(WeatherCondition.clear.cloud == .none && WeatherCondition.mainlyClear.cloud == .none)
        for rainy in [
            WeatherCondition.drizzle,
            .rain,
            .heavyRain,
            .freezingRain,
            .snow,
            .thunderstorm,
            .thunderstormHail,
        ] {
            #expect(rainy.cloud == .rain, "\(rainy)")
        }
        #expect(WeatherCondition.thunderstorm.isPrecipitation && !WeatherCondition.fog.isPrecipitation)
    }

    @Test func wordsReadBySunAndMoon() {
        #expect(WeatherCondition.clear.word(isDay: true) == "Sunny" && WeatherCondition.clear
            .word(isDay: false) == "Clear")
        #expect(WeatherCondition.overcast.word(isDay: true) == "Cloudy")
        #expect(WeatherCondition.snow.scene == .snow && WeatherCondition.rain.scene == .rain && WeatherCondition.fog
            .scene == .clear)
    }

    @Test func windComesFromTheEightPoints() {
        let expected = [(0.0, "north"), (44, "north-east"), (90, "east"), (135, "south-east"), (180, "south"),
                        (225, "south-west"), (270, "west"), (315, "north-west"), (359, "north"), (-45, "north-west"),
                        (720, "north")]
        for (degrees, word) in expected {
            #expect(WeatherUnits.compass(degrees) == word, "\(degrees)")
        }
    }

    // MARK: Warnings

    @Test func warningsComeFromTheNumbers() {
        #expect(WeatherWarning.derive(apparentC: 35, windKph: 5, gustKph: 10) == [.heatWarning])
        #expect(WeatherWarning.derive(apparentC: 34.9, windKph: 5, gustKph: 10) == [])
        #expect(WeatherWarning.derive(apparentC: -15, windKph: 5, gustKph: 10) == [.coldWarning])
        #expect(WeatherWarning.derive(apparentC: 20, windKph: 33, gustKph: 40) == [.windy])
        #expect(WeatherWarning.derive(apparentC: 20, windKph: 33, gustKph: 64) == [.windWarning])
        #expect(WeatherWarning.derive(apparentC: 36, windKph: 70, gustKph: 90) == [.windWarning, .heatWarning])
        // 95 F, 5 F, 20 mph and 40 mph, in the units he reads.
        #expect(abs(WeatherUnits.fahrenheit(WeatherWarning.heatWarningC) - 95) < 0.01)
        #expect(abs(WeatherUnits.fahrenheit(WeatherWarning.coldWarningC) - 5) < 0.01)
        #expect(abs(WeatherUnits.mph(WeatherWarning.windyKph) - 19.9) < 0.1 &&
            abs(WeatherUnits.mph(WeatherWarning.gustWarningKph) - 39.8) < 0.1)
        #expect(WeatherWarning.windy.isWarning == false && WeatherWarning.heatWarning.isWarning)
    }

    // MARK: Source

    static let forecastJSON = """
    {"latitude":37.8,"longitude":-122.4,"utc_offset_seconds":-25200,"timezone":"America/Los_Angeles",
     "current":{"time":"2026-10-24T00:00","temperature_2m":14.2,"apparent_temperature":13.0,"weather_code":61,"is_day":0,
                "wind_speed_10m":10.0,"wind_gusts_10m":70.0,"wind_direction_10m":268.0},
     "hourly":{"time":["2026-10-24T00:00","2026-10-24T01:00","2026-10-24T02:00"],
               "temperature_2m":[14.2,14.0,13.8],"apparent_temperature":[13.0,12.5,36.0],
               "precipitation_probability":[80,60,null],"weather_code":[61,3,0],"is_day":[0,0,0],
               "wind_speed_10m":[10.0,9.0,8.0],"wind_gusts_10m":[20.0,18.0,16.0],"uv_index":[0.0,0.0,null]},
     "daily":{"time":["2026-10-24","2026-10-25"],"weather_code":[61,2],"temperature_2m_max":[18.0,20.5],
              "temperature_2m_min":[11.0,10.0],"apparent_temperature_max":[17.0,19.0],
              "precipitation_probability_max":[85,20],"wind_gusts_10m_max":[40.0,30.0],"uv_index_max":[3.0,5.5],
              "sunrise":["2026-10-24T07:12","2026-10-25T07:13"],"sunset":["2026-10-24T18:10","2026-10-25T18:09"]}}
    """

    static let airJSON = #"{"current":{"us_aqi":57.4,"pm2_5":14.1,"pm10":22.0}}"#

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var urls: [URL] = []
        func add(_ url: URL) {
            lock.lock(); defer { lock.unlock() }
            urls.append(url)
        }

        var all: [URL] {
            lock.lock(); defer { lock.unlock() }
            return urls
        }
    }

    static func source(_ counter: Counter) -> OpenMeteoSource {
        OpenMeteoSource { url in
            counter.add(url)
            return Data((url.host?.hasPrefix("air-quality") == true ? airJSON : forecastJSON).utf8)
        }
    }

    @Test func theResponseBecomesAForecast() async throws {
        let counter = Counter()
        let now = Date(timeIntervalSince1970: 1_792_000_000)
        let forecast = try await Self.source(counter).forecast(latitude: 37.77, longitude: -122.41, now: now)
        #expect(forecast.now.condition == .rain && !forecast.now.isDay && forecast.now.temperatureC == 14.2)
        #expect(forecast.now.warnings == [.windWarning])
        #expect(forecast.now.windFrom == "west")
        #expect(forecast.hourly.count == 3 && forecast.hourly[1].condition == .overcast)
        #expect(forecast.hourly[2].precipitationChance == 0 && forecast.hourly[2].uvIndex == 0)
        #expect(forecast.daily.count == 2 && forecast.daily[0].highC == 18 && forecast.daily[1]
            .condition == .partlyCloudy)
        // Times are read in the place's own offset (UTC-7): 07:12 local is 14:12 UTC.
        let sunrise = try #require(forecast.daily[0].sunrise)
        var utc = Calendar(identifier: .gregorian)
        utc.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        #expect(utc.component(.hour, from: sunrise) == 14 && utc.component(.minute, from: sunrise) == 12)
        #expect(forecast.air?.usAQI == 57 && forecast.air?.category == "Moderate")
        // The heat in hour three shows in the upcoming warnings, after the wind warning now.
        #expect(forecast.warnings == [.windWarning, .heatWarning])
        #expect(forecast.nextPrecipitation(after: forecast.hourly[0].time.addingTimeInterval(-1))?.condition == .rain)
    }

    @Test func onlyTheRoundedCoordinateLeavesThePhoneAndItIsCachedForThirtyMinutes() async throws {
        let counter = Counter()
        let source = Self.source(counter)
        let now = Date(timeIntervalSince1970: 1_792_000_000)
        _ = try await source.forecast(latitude: 37.7749, longitude: -122.4194, now: now)
        let urls = counter.all
        #expect(urls.count == 2)
        for url in urls {
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
            #expect(items.first { $0.name == "latitude" }?.value == "37.8")
            #expect(items.first { $0.name == "longitude" }?.value == "-122.4")
            #expect(url.scheme == "https" && (url.host?.hasSuffix("open-meteo.com") == true))
            #expect(!url.absoluteString.lowercased().contains("key") && !url.absoluteString.contains("37.7749"))
        }
        // Inside 30 minutes, nearby: no new request. Past it, or elsewhere: a new one.
        _ = try await source.forecast(latitude: 37.78, longitude: -122.42, now: now.addingTimeInterval(1700))
        #expect(counter.all.count == 2)
        _ = try await source.forecast(latitude: 37.78, longitude: -122.42, now: now.addingTimeInterval(1801))
        #expect(counter.all.count == 4)
        _ = try await source.forecast(latitude: 40.0, longitude: -74.0, now: now.addingTimeInterval(1802))
        #expect(counter.all.count == 6)
        source.forget()
        _ = try await source.forecast(latitude: 40.0, longitude: -74.0, now: now.addingTimeInterval(1803))
        #expect(counter.all.count == 8)
    }

    @Test func aBrokenResponseThrowsAndAnAirFailureJustLeavesAirOut() async throws {
        let broken = OpenMeteoSource { _ in Data("not json".utf8) }
        await #expect(throws: WeatherError.self) {
            try await broken.forecast(latitude: 1, longitude: 1, now: .now)
        }
        let noAir = OpenMeteoSource { url in
            if url.host?.hasPrefix("air-quality") == true {
                throw WeatherError("down")
            }
            return Data(Self.forecastJSON.utf8)
        }
        let forecast = try await noAir.forecast(latitude: 1, longitude: 1, now: .now)
        #expect(forecast.air == nil && forecast.daily.count == 2)
    }

    @Test func theSampleIsAFullWeek() async throws {
        let forecast = try await SampleWeather().forecast(
            latitude: 0,
            longitude: 0,
            now: Date(timeIntervalSince1970: 1_792_000_000)
        )
        #expect(forecast.hourly.count == 48 && forecast.daily.count == 7 && forecast.air != nil)
    }

    // MARK: Walls

    @Test func noClaudeSenderOrToolReadsTheForecast() throws {
        // Every source file outside Weather/ (Core builds the fragments Claude is sent), not only the Claude senders.
        let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources")
        let files = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "swift" && !$0.path.contains("/TesseraCore/Weather/") } ?? []
        #expect(files.count > 100)
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            #expect(!text.contains("WeatherReport") && !text.contains("OpenMeteo") && !text.contains("WeatherSource")
                && !text.contains("WeatherCondition"), "\(file.lastPathComponent) reads the weather")
        }
    }

    @Test func theForecastRequestAsksForWindDirection() throws {
        let url = try #require(OpenMeteoSource.forecastURL(latitude: 1, longitude: 1))
        let current = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?
            .first { $0.name == "current" }?.value ?? ""
        #expect(current.contains("wind_direction_10m"))
    }

    @Test func twoAsksForTheSamePlaceShareOneFetch() async throws {
        let counter = Counter()
        let source = OpenMeteoSource { url in
            counter.add(url)
            try await Task.sleep(for: .milliseconds(150))
            return url.host?.hasPrefix("air-quality") == true ? Data(Self.airJSON.utf8) : Data(Self.forecastJSON.utf8)
        }
        let now = Date(timeIntervalSince1970: 1_792_000_000)
        async let first = source.forecast(latitude: 37.7, longitude: -122.4, now: now)
        async let second = source.forecast(latitude: 37.7, longitude: -122.4, now: now)
        _ = try await (first, second)
        #expect(counter.all.count == 2)
    }
}
