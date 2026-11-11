import Foundation

/// What the sky is doing, named for every WMO weather code Open-Meteo reports (Kaushik, Oct 23 11:51 PM: an effect for
/// every
/// kind of weather Home can show). Day or night is `isDay` on the reading, not a case. The words are what Home and the
/// Weather room say; Design draws one effect per case.
public enum WeatherCondition: String, Codable, CaseIterable, Sendable {
    case clear, mainlyClear, partlyCloudy, overcast
    case fog
    case drizzle, freezingDrizzle
    case rain, heavyRain, freezingRain
    case snow, heavySnow, snowGrains
    case thunderstorm, thunderstormHail

    /// The cloud a case is drawn with: none, a white cloud, or a dark rain cloud. White for partly cloudy and overcast,
    /// dark for everything that falls.
    public enum Cloud: Sendable {
        case none, white, rain
    }

    public var cloud: Cloud {
        switch self {
        case .clear, .mainlyClear: .none
        case .partlyCloudy, .overcast, .fog: .white
        default: .rain
        }
    }

    /// A WMO code (WMO 4677 as Open-Meteo reports it). Nil for a code it does not define.
    public init?(wmo code: Int) {
        switch code {
        case 0: self = .clear
        case 1: self = .mainlyClear
        case 2: self = .partlyCloudy
        case 3: self = .overcast
        case 45, 48: self = .fog
        case 51, 53, 55: self = .drizzle
        case 56, 57: self = .freezingDrizzle
        case 61, 63, 80, 81: self = .rain
        case 65, 82: self = .heavyRain
        case 66, 67: self = .freezingRain
        case 71, 73, 85: self = .snow
        case 75, 86: self = .heavySnow
        case 77: self = .snowGrains
        case 95: self = .thunderstorm
        case 96, 99: self = .thunderstormHail
        default: return nil
        }
    }

    public var isPrecipitation: Bool {
        switch self {
        case .clear, .mainlyClear, .partlyCloudy, .overcast, .fog: false
        default: true
        }
    }

    /// The word Home and the Weather room use (sentence case; "Clear" by night, "Sunny" is the caller's by day).
    public func word(isDay: Bool) -> String {
        switch self {
        case .clear: isDay ? "Sunny" : "Clear"
        case .mainlyClear: isDay ? "Mostly sunny" : "Mostly clear"
        case .partlyCloudy: "Partly cloudy"
        case .overcast: "Cloudy"
        case .fog: "Fog"
        case .drizzle: "Drizzle"
        case .freezingDrizzle: "Freezing drizzle"
        case .rain: "Rain"
        case .heavyRain: "Heavy rain"
        case .freezingRain: "Freezing rain"
        case .snow: "Snow"
        case .heavySnow: "Heavy snow"
        case .snowGrains: "Snow grains"
        case .thunderstorm: "Thunderstorm"
        case .thunderstormHail: "Thunderstorm and hail"
        }
    }

    /// What the mosaic's scene picker reads: clear, rain or snow.
    public var scene: SceneWeather {
        switch self {
        case .drizzle, .rain, .heavyRain, .freezingDrizzle, .freezingRain, .thunderstorm, .thunderstormHail: .rain
        case .snow, .heavySnow, .snowGrains: .snow
        default: .clear
        }
    }
}

/// What is worth saying on top of the condition. Open-Meteo has no alerts feed, so these are derived from the numbers
/// (free data only). `windy` is not a warning; the other three are.
public enum WeatherWarning: String, Codable, CaseIterable, Sendable {
    case windy, windWarning, heatWarning, coldWarning

    /// Sustained wind at or above this is "windy".
    public static let windyKph = 32.0 // 20 mph
    /// Gusts at or above this are a wind warning.
    public static let gustWarningKph = 64.0 // 40 mph
    /// Apparent temperature at or above this is a heat warning (95 F).
    public static let heatWarningC = 35.0
    /// Apparent temperature at or below this is a cold warning (5 F).
    public static let coldWarningC = -15.0

    public var isWarning: Bool {
        self != .windy
    }

    public var word: String {
        switch self {
        case .windy: "Windy"
        case .windWarning: "Wind warning"
        case .heatWarning: "Heat warning"
        case .coldWarning: "Cold warning"
        }
    }

    /// The warnings for a reading. A wind warning replaces plain windy.
    public static func derive(apparentC: Double?, windKph: Double?, gustKph: Double?) -> [WeatherWarning] {
        var out: [WeatherWarning] = []
        if let gustKph, gustKph >= gustWarningKph {
            out.append(.windWarning)
        } else if let windKph, windKph >= windyKph {
            out.append(.windy)
        }
        if let apparentC {
            if apparentC >= heatWarningC {
                out.append(.heatWarning)
            } else if apparentC <= coldWarningC {
                out.append(.coldWarning)
            }
        }
        return out
    }
}
