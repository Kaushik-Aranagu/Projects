import Foundation

/// A forecast the Weather room and Home read (Design 243): now, the next 48 hours, the next 7 days, air quality. It
/// holds no
/// place: no city, no coordinate, nothing that says where. It is never written to the store, a backup, a log or a
/// widget;
/// a source returns one, the host keeps it in memory.
public struct WeatherReport: Equatable, Sendable {
    public struct Now: Equatable, Sendable {
        public var condition: WeatherCondition
        public var isDay: Bool
        public var temperatureC: Double
        public var apparentC: Double
        public var windKph: Double
        public var gustKph: Double
        /// Where the wind comes from, in degrees (0 is north, 90 east); nil when the source gave none.
        public var windFromDegrees: Double?
        public var warnings: [WeatherWarning]

        /// "west", "north-east": the air line says "from the west". Nil with no direction.
        public var windFrom: String? {
            windFromDegrees.map { WeatherUnits.compass($0) }
        }

        public init(condition: WeatherCondition, isDay: Bool, temperatureC: Double, apparentC: Double, windKph: Double,
                    gustKph: Double, windFromDegrees: Double? = nil)
        {
            self.windFromDegrees = windFromDegrees
            self.condition = condition
            self.isDay = isDay
            self.temperatureC = temperatureC
            self.apparentC = apparentC
            self.windKph = windKph
            self.gustKph = gustKph
            warnings = WeatherWarning.derive(apparentC: apparentC, windKph: windKph, gustKph: gustKph)
        }
    }

    public struct Hour: Equatable, Sendable, Identifiable {
        public var id: Date {
            time
        }

        public var time: Date
        public var condition: WeatherCondition
        public var isDay: Bool
        public var temperatureC: Double
        public var apparentC: Double
        /// 0 to 100.
        public var precipitationChance: Int
        public var windKph: Double
        public var gustKph: Double
        public var uvIndex: Double
    }

    public struct Day: Equatable, Sendable, Identifiable {
        public var id: Date {
            date
        }

        /// Midnight of the day, in the place's own time zone as a moment.
        public var date: Date
        public var condition: WeatherCondition
        public var highC: Double
        public var lowC: Double
        public var apparentHighC: Double
        public var precipitationChance: Int
        public var gustKph: Double
        public var uvIndex: Double
        public var sunrise: Date?
        public var sunset: Date?
        public var warnings: [WeatherWarning] {
            WeatherWarning.derive(apparentC: apparentHighC, windKph: nil, gustKph: gustKph)
        }
    }

    /// The US air quality index and its words.
    public struct AirQuality: Equatable, Sendable {
        public var usAQI: Int
        public var pm25: Double?
        public var pm10: Double?

        public var category: String {
            switch usAQI {
            case ..<51: "Good"
            case 51 ..< 101: "Moderate"
            case 101 ..< 151: "Unhealthy for sensitive groups"
            case 151 ..< 201: "Unhealthy"
            case 201 ..< 301: "Very unhealthy"
            default: "Hazardous"
            }
        }
    }

    public var fetchedAt: Date
    public var now: Now
    public var hourly: [Hour]
    public var daily: [Day]
    public var air: AirQuality?

    public init(fetchedAt: Date, now: Now, hourly: [Hour], daily: [Day], air: AirQuality? = nil) {
        self.fetchedAt = fetchedAt
        self.now = now
        self.hourly = hourly
        self.daily = daily
        self.air = air
    }

    /// Every warning in the next 48 hours or today, without repeats, warnings first. Home says the first.
    public var warnings: [WeatherWarning] {
        var seen = Set<WeatherWarning>()
        var out: [WeatherWarning] = []
        let upcoming = now.warnings + hourly.flatMap {
            WeatherWarning.derive(apparentC: $0.apparentC, windKph: $0.windKph, gustKph: $0.gustKph)
        }
        for warning in upcoming where seen.insert(warning).inserted {
            out.append(warning)
        }
        return out.sorted { ($0.isWarning ? 0 : 1) < ($1.isWarning ? 0 : 1) }
    }

    /// The next hour at which rain, snow or a storm starts, or nil if nothing falls in the next 48 hours.
    public func nextPrecipitation(after date: Date) -> Hour? {
        hourly.first { $0.time > date && $0.condition.isPrecipitation }
    }
}

// MARK: - Units

public enum WeatherUnits {
    /// The eight compass words for a wind direction in degrees: 0 is "north", 45 "north-east", 270 "west".
    public static func compass(_ degrees: Double) -> String {
        let words = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
        let wrapped = degrees.truncatingRemainder(dividingBy: 360)
        let positive = wrapped < 0 ? wrapped + 360 : wrapped
        return words[Int((positive / 45).rounded()) % 8]
    }

    public static func fahrenheit(_ celsius: Double) -> Double {
        celsius * 9 / 5 + 32
    }

    public static func mph(_ kph: Double) -> Double {
        kph / 1.609_344
    }
}

// MARK: - Sources

/// Where a forecast comes from. The coordinate is the caller's, read once from the phone, rounded by the source, and
/// never
/// stored by it.
public protocol WeatherSource: Sendable {
    func forecast(latitude: Double, longitude: Double, now: Date) async throws -> WeatherReport
}

public struct WeatherError: Error, Equatable, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// The on-device sample: a fixed, plausible week. The default when nothing else is on (tests, previews, no network).
public struct SampleWeather: WeatherSource {
    public init() {}

    public func forecast(latitude _: Double, longitude _: Double, now: Date) async throws -> WeatherReport {
        Self.forecast(now: now)
    }

    public static func forecast(now: Date, calendar: Calendar = .current) -> WeatherReport {
        let start = calendar.date(bySetting: .minute, value: 0, of: now) ?? now
        let conditions: [WeatherCondition] = [
            .partlyCloudy,
            .overcast,
            .rain,
            .heavyRain,
            .thunderstorm,
            .mainlyClear,
            .clear,
        ]
        let hours: [WeatherReport.Hour] = (0 ..< 48).map { index in
            let time = start.addingTimeInterval(Double(index) * 3600)
            let hour = calendar.component(.hour, from: time)
            let condition = conditions[(index / 8) % conditions.count]
            let temp = 14 + 6 * sin(Double(hour - 9) / 24 * 2 * .pi)
            return WeatherReport.Hour(
                time: time, condition: condition, isDay: (7 ..< 19).contains(hour), temperatureC: temp,
                apparentC: temp - 1, precipitationChance: condition.isPrecipitation ? 70 : 10,
                windKph: 12 + Double(index % 5), gustKph: 22 + Double(index % 7),
                uvIndex: (7 ..< 19).contains(hour) ? 4 : 0
            )
        }
        let days: [WeatherReport.Day] = (0 ..< 7).map { index in
            let date = calendar.startOfDay(for: calendar.date(byAdding: .day, value: index, to: now) ?? now)
            return WeatherReport.Day(
                date: date, condition: conditions[index % conditions.count], highC: 20 - Double(index % 3),
                lowC: 11 - Double(index % 2), apparentHighC: 19, precipitationChance: index % 3 == 0 ? 60 : 15,
                gustKph: 28, uvIndex: 5, sunrise: date.addingTimeInterval(6.5 * 3600),
                sunset: date.addingTimeInterval(18.5 * 3600)
            )
        }
        let current = WeatherReport.Now(
            condition: .partlyCloudy,
            isDay: (7 ..< 19).contains(calendar.component(.hour, from: now)),
            temperatureC: 17,
            apparentC: 16,
            windKph: 12,
            gustKph: 22
        )
        return WeatherReport(fetchedAt: now, now: current, hourly: hours, daily: days,
                             air: WeatherReport.AirQuality(usAQI: 42, pm25: 9, pm10: 14))
    }
}
