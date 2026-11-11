import Foundation
import TesseraCore

// CLOUD-WRITTEN, NOT COMPILED. Core's Open-Meteo forecast (`WeatherReport`, Celsius and kph) mapped onto the room's
// `WeatherForecast` (Fahrenheit and mph). A report holds no place, and neither does the result.

enum WeatherCoreAdapter {
    /// One-to-one with Core's cases; `windy` is the overlay for a dry, windy sky (rain, snow and fog keep their own).
    static func kind(_ condition: WeatherCondition, windy: Bool = false) -> WeatherKind {
        switch condition {
        case .clear, .mainlyClear: windy ? .wind : .clear
        case .partlyCloudy: windy ? .wind : .partlyCloudy
        case .overcast: windy ? .wind : .cloudy
        case .fog: .fog
        case .drizzle, .rain, .freezingDrizzle, .freezingRain: .rain
        case .heavyRain: .heavyRain
        case .thunderstorm, .thunderstormHail: .storm
        case .snow, .heavySnow, .snowGrains: .snow
        }
    }

    static func degrees(_ celsius: Double) -> Int {
        Int(WeatherUnits.fahrenheit(celsius).rounded())
    }

    /// The one wind rule, for now, every hour and every day: sustained wind at Core's "windy" line, or gusts at its
    /// wind-warning line (`WeatherWarning.derive`).
    static func isWindy(windKph: Double, gustKph: Double) -> Bool {
        !WeatherWarning.derive(apparentC: nil, windKph: windKph, gustKph: gustKph).isEmpty
    }

    static func hour(_ hour: WeatherReport.Hour) -> WeatherHour {
        WeatherHour(time: hour.time, temperature: degrees(hour.temperatureC),
                    kind: kind(hour.condition, windy: isWindy(windKph: hour.windKph, gustKph: hour.gustKph)))
    }

    /// "good", "moderate" or "poor": the US index in the room's three words.
    static func airWord(_ air: WeatherReport.AirQuality) -> String {
        switch air.usAQI {
        case ..<51: "good"
        case 51 ..< 101: "moderate"
        default: "poor"
        }
    }

    static func forecast(from report: WeatherReport, calendar: Calendar = .current) -> WeatherForecast {
        let days: [WeatherDay] = report.daily.map { day in
            let sameDay = report.hourly.filter { calendar.isDate($0.time, inSameDayAs: day.date) }
            // A day is windy when any of its hours is, or its gusts reach a warning.
            let windy = day.warnings.contains(.windWarning)
                || sameDay.contains { isWindy(windKph: $0.windKph, gustKph: $0.gustKph) }
            return WeatherDay(
                date: day.date, high: degrees(day.highC), low: degrees(day.lowC),
                kind: kind(day.condition, windy: windy), sunrise: day.sunrise, sunset: day.sunset,
                hours: sameDay.map(hour)
            )
        }
        let current = report.now
        let windy = isWindy(windKph: current.windKph, gustKph: current.gustKph)
        let today = report.daily.first
        let uv = today.map { Int($0.uvIndex.rounded()) } ?? Int((report.hourly.first?.uvIndex ?? 0).rounded())
        return WeatherForecast(
            now: report.fetchedAt,
            temperature: degrees(current.temperatureC),
            feelsLike: degrees(current.apparentC),
            kind: kind(current.condition, windy: windy),
            high: today.map { degrees($0.highC) } ?? degrees(current.temperatureC),
            low: today.map { degrees($0.lowC) } ?? degrees(current.temperatureC),
            hours: report.hourly.map(hour),
            days: days,
            air: WeatherAir(quality: report.air.map(airWord), uvIndex: uv,
                            windMph: Int(WeatherUnits.mph(current.windKph).rounded()), windFrom: nil)
        )
    }
}

/// The room's source, backed by any of Core's `WeatherSource`s (`OpenMeteoSource`, `SampleWeather`). The coordinate is
/// the host's: read once from the phone, handed to the source (which rounds it) and never kept here. Nil coordinate or
/// a failed fetch means no forecast, and the room says so in one line.
struct CoreWeatherSource: WeatherForecastSource {
    let source: any WeatherSource
    let coordinate: @Sendable () async -> (latitude: Double, longitude: Double)?
    var now: @Sendable () -> Date = { Date() }

    func forecast() async -> WeatherForecast? {
        guard let place = await coordinate() else { return nil }
        guard let report = try? await source.forecast(latitude: place.latitude, longitude: place.longitude, now: now())
        else { return nil }
        return WeatherCoreAdapter.forecast(from: report)
    }
}
