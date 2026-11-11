import Foundation

// CLOUD-WRITTEN, NOT COMPILED. The sample forecast the room runs on until Core's source is wired: a cloudy morning,
// rain from 4 PM, heavy after 6 PM, clearing by midnight, then a week with rain on Saturday and wind on Tuesday.

struct SampleWeatherSource: WeatherForecastSource {
    var now = Date()
    var calendar = Calendar.current

    func forecast() async -> WeatherForecast? {
        Self.make(now: now, calendar: calendar)
    }

    static func make(now: Date, calendar: Calendar = .current) -> WeatherForecast {
        let start = calendar.date(bySettingHour: 10, minute: 0, second: 0, of: now) ?? now
        let startOfDay = calendar.startOfDay(for: start)

        func hour(_ day: Date, _ hour: Int, _ temperature: Int, _ kind: WeatherKind) -> WeatherHour {
            WeatherHour(time: calendar.date(byAdding: .hour, value: hour, to: day) ?? day,
                        temperature: temperature, kind: kind)
        }
        // Today, midnight to 11 PM.
        let todayTemps = [49, 49, 48, 48, 48, 49, 50, 52, 54, 55, 56, 58, 60, 61, 60, 59, 57, 55, 53, 52, 51, 50, 50, 49]
        let todayKinds: [WeatherKind] = [
            .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy, .cloudy,
            .cloudy, .cloudy, .cloudy, .cloudy, .rain, .rain, .heavyRain, .heavyRain, .heavyRain, .rain, .rain,
            .rain,
        ]
        let today = (0 ..< 24).map { hour(startOfDay, $0, todayTemps[$0], todayKinds[$0]) }

        let outlook: [(high: Int, low: Int, kind: WeatherKind)] = [
            (61, 49, .cloudy), (63, 48, .partlyCloudy), (58, 50, .rain), (60, 47, .clear), (55, 44, .wind),
            (57, 45, .partlyCloudy), (59, 46, .clear),
        ]
        let days: [WeatherDay] = outlook.enumerated().map { index, entry in
            let date = calendar.date(byAdding: .day, value: index, to: startOfDay) ?? startOfDay
            let hours: [WeatherHour] = index == 0 ? today : (0 ..< 24).map { slot in
                // A smooth swing from the low at 5 AM to the high at 3 PM.
                let phase = (Double(slot) - 5) / 10
                let swing = slot >= 5 && slot <= 15 ? phase : (slot < 5 ? 0 : max(0, 1 - Double(slot - 15) / 12))
                let temperature = entry.low + Int((Double(entry.high - entry.low) * swing).rounded())
                return hour(date, slot, temperature, entry.kind)
            }
            return WeatherDay(
                date: date, high: entry.high, low: entry.low, kind: entry.kind,
                sunrise: calendar.date(byAdding: .minute, value: 7 * 60 + 25 - index, to: date) ?? date,
                sunset: calendar.date(byAdding: .minute, value: 18 * 60 + 6 + index, to: date) ?? date,
                hours: hours
            )
        }
        let nowIndex = calendar.component(.hour, from: start)
        let ahead = today[nowIndex...] + (days.count > 1 ? days[1].hours : [])
        return WeatherForecast(
            now: start, temperature: 54, feelsLike: 55, kind: .cloudy, high: 61, low: 49,
            hours: Array(ahead), days: days,
            air: WeatherAir(quality: "good", uvIndex: 2, windMph: 9, windFrom: "west")
        )
    }
}
