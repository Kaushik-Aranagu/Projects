import Foundation

// CLOUD-WRITTEN, NOT COMPILED. The Weather room's data seam (sheet 243). The room draws a `WeatherForecast` and asks
// a `WeatherForecastSource` for it; Core's Open-Meteo `WeatherSource` / `WeatherCondition` plug in by conforming (or
// mapping into these types) with no change to the screens. No place is ever named here: a forecast has no city,
// coordinates or location of any kind, so nothing in the room can reach a backup, a log or a widget.

/// What the sky is doing. Core's `WeatherCondition` maps onto this one-to-one.
enum WeatherKind: String, CaseIterable, Sendable, Equatable {
    case clear, partlyCloudy, cloudy, fog, rain, heavyRain, storm, snow, wind

    /// The word as it appears in a sentence, capitalized by the sentence when it starts one.
    var word: String {
        switch self {
        case .clear: "clear"
        case .partlyCloudy: "partly cloudy"
        case .cloudy: "cloudy"
        case .fog: "fog"
        case .rain: "rain"
        case .heavyRain: "heavy rain"
        case .storm: "storms"
        case .snow: "snow"
        case .wind: "wind"
        }
    }

    /// The one word under an hour column.
    var short: String {
        switch self {
        case .clear: "Clear"
        case .partlyCloudy: "Mixed"
        case .cloudy: "Cloudy"
        case .fog: "Fog"
        case .rain: "Rain"
        case .heavyRain: "Heavy"
        case .storm: "Storms"
        case .snow: "Snow"
        case .wind: "Windy"
        }
    }

    var isWet: Bool {
        switch self {
        case .rain, .heavyRain, .storm, .snow: true
        default: false
        }
    }

    /// Whether a day or hour of this kind is worth naming in "The week".
    var isNotable: Bool {
        switch self {
        case .clear, .partlyCloudy, .cloudy: false
        default: true
        }
    }
}

struct WeatherHour: Identifiable, Equatable, Sendable {
    let time: Date
    let temperature: Int
    let kind: WeatherKind

    var id: Date {
        time
    }
}

struct WeatherDay: Identifiable, Equatable, Sendable {
    let date: Date
    let high: Int
    let low: Int
    let kind: WeatherKind
    let sunrise: Date
    let sunset: Date
    /// The day's hours, midnight on; the Day page's columns.
    let hours: [WeatherHour]

    var id: Date {
        date
    }
}

struct WeatherAir: Equatable, Sendable {
    /// "good", "fair", "poor"
    let quality: String
    let uvIndex: Int
    let windMph: Int
    /// "west", "north-east"
    let windFrom: String
}

struct WeatherForecast: Equatable, Sendable {
    let now: Date
    let temperature: Int
    let feelsLike: Int
    let kind: WeatherKind
    /// Today's high and low.
    let high: Int
    let low: Int
    /// From the current hour on, at least 24.
    let hours: [WeatherHour]
    /// Today first.
    let days: [WeatherDay]
    let air: WeatherAir
}

/// Core's Open-Meteo source conforms to this (or is wrapped by a type that does). Returns nil when there is no
/// forecast yet (no permission, offline with an empty cache); the room then says so in one line.
protocol WeatherForecastSource: Sendable {
    func forecast() async -> WeatherForecast?
}

/// The room's whole state of mind about the rain alert: a preference that lives on this phone only.
enum WeatherAlertDefaults {
    static let key = "weather.rainAlert"
}
