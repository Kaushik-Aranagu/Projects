import Foundation

// CLOUD-WRITTEN, NOT COMPILED. The room's sentences, built on the phone from the forecast. A sentence is a list of
// pieces: plain words, a weather word (the landing's two lines, where Builder A's effects play), and a door (a
// weather word in "The week" that opens its day). Sentence case, no middle dots, no city.

enum WeatherPiece: Equatable {
    case plain(String)
    /// A weather word in its condition color. `day` is the day it opens, when it is a door.
    case word(WeatherKind, String, day: Int?)
}

struct WeatherSentence: Equatable {
    var pieces: [WeatherPiece]

    /// The sentence as one string, for VoiceOver and tests.
    var text: String {
        pieces.map { piece in
            switch piece {
            case let .plain(text): text
            case let .word(_, text, _): text
            }
        }.joined()
    }
}

enum WeatherSentences {
    /// "7 PM" / "4 PM": the hour in the house short form, no minutes.
    static func hourName(_ date: Date, calendar: Calendar = .current) -> String {
        let hour = calendar.component(.hour, from: date)
        switch hour {
        case 0: return "midnight"
        case 12: return "noon"
        default: return "\(hour % 12) \(hour < 12 ? "AM" : "PM")"
        }
    }

    static func capitalized(_ word: String) -> String {
        word.prefix(1).uppercased() + word.dropFirst()
    }

    /// Line 1: "Cloudy now, rain from 4 PM." The first change to a wetter or notable kind inside the next 12 hours.
    static func headline(_ forecast: WeatherForecast, calendar: Calendar = .current) -> WeatherSentence {
        var pieces: [WeatherPiece] = [.word(forecast.kind, capitalized(forecast.kind.word), day: nil), .plain(" now")]
        let soon = forecast.hours.prefix(12)
        if let change = soon.first(where: { $0.kind != forecast.kind && $0.kind.isNotable }) {
            pieces += [.plain(", "), .word(change.kind, change.kind.word, day: nil),
                       .plain(" from \(hourName(change.time, calendar: calendar)).")]
        } else {
            pieces.append(.plain(", and no change soon."))
        }
        return WeatherSentence(pieces: pieces)
    }

    /// Line 2: "Heavy rain after 6 PM, clearing by midnight." The worst hour today, and when it ends.
    static func detail(_ forecast: WeatherForecast, calendar: Calendar = .current) -> WeatherSentence? {
        let today = forecast.hours.filter { calendar.isDate($0.time, inSameDayAs: forecast.now) }
        guard let worst = today.first(where: { $0.kind == .heavyRain || $0.kind == .storm }) else { return nil }
        var pieces: [WeatherPiece] = [
            .word(worst.kind, capitalized(worst.kind.word), day: nil),
            .plain(" after \(hourName(worst.time, calendar: calendar))"),
        ]
        if let clear = forecast.hours.first(where: { $0.time > worst.time && !$0.kind.isWet }) {
            pieces.append(.plain(", clearing by \(hourName(clear.time, calendar: calendar))."))
        } else {
            pieces.append(.plain("."))
        }
        return WeatherSentence(pieces: pieces)
    }

    static func dayName(_ date: Date, calendar: Calendar = .current) -> String {
        var calendar = calendar
        calendar.locale = Locale(identifier: "en_US")
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US")
        formatter.dateFormat = "EEEE"
        return formatter.string(from: date)
    }

    /// "The week": one sentence that names only the notable days; each weather word is a door to its day. Today is
    /// left out (the landing is about today); a quiet week says so in plain words.
    static func week(_ forecast: WeatherForecast, calendar: Calendar = .current) -> WeatherSentence {
        let notable = forecast.days.enumerated().dropFirst().filter { $0.element.kind.isNotable }
        guard !notable.isEmpty else {
            return WeatherSentence(pieces: [.plain("A quiet week, nothing to plan around.")])
        }
        var pieces: [WeatherPiece] = [.plain("Mostly easy, with ")]
        for (position, item) in notable.enumerated() {
            let day = item.element
            if position > 0 {
                pieces.append(.plain(position == notable.count - 1 ? " and " : ", "))
            }
            pieces.append(.word(day.kind, day.kind.word, day: item.offset))
            pieces.append(.plain(" on \(dayName(day.date, calendar: calendar))"))
        }
        pieces.append(.plain("."))
        return WeatherSentence(pieces: pieces)
    }

    /// A day's own sentence on its page: "Rain through the afternoon, high of 58." from its hours.
    static func daySentence(_ day: WeatherDay, calendar: Calendar = .current) -> WeatherSentence {
        let wet = day.hours.filter { $0.kind.isWet }
        var pieces: [WeatherPiece] = [.word(day.kind, capitalized(day.kind.word), day: nil)]
        if let first = wet.first, let last = wet.last, day.kind.isWet {
            pieces.append(.plain(" from \(hourName(first.time, calendar: calendar)) to "
                    + "\(hourName(last.time, calendar: calendar)), high of \(day.high)°."))
        } else {
            pieces.append(.plain(" for the day, high of \(day.high)°, low of \(day.low)°."))
        }
        return WeatherSentence(pieces: pieces)
    }

    static func sunLine(_ day: WeatherDay, calendar: Calendar = .current) -> String {
        "Sunrise \(clock(day.sunrise, calendar: calendar)), sunset \(clock(day.sunset, calendar: calendar))."
    }

    /// "7:25 AM"
    static func clock(_ date: Date, calendar: Calendar = .current) -> String {
        let hour = calendar.component(.hour, from: date)
        let minute = calendar.component(.minute, from: date)
        return String(format: "%d:%02d %@", hour % 12 == 0 ? 12 : hour % 12, minute, hour < 12 ? "AM" : "PM")
    }

    /// "7 AM", the time over an hour column.
    static func columnTime(_ date: Date, calendar: Calendar = .current) -> String {
        let hour = calendar.component(.hour, from: date)
        return "\(hour % 12 == 0 ? 12 : hour % 12) \(hour < 12 ? "AM" : "PM")"
    }

    /// "Air is good, UV 2, wind 9 mph from the west." The first piece after "Air is " is the quality word.
    static func airLine(_ air: WeatherAir) -> (before: String, quality: String, after: String) {
        ("Air is ", air.quality, ", UV \(air.uvIndex), wind \(air.windMph) mph from the \(air.windFrom).")
    }
}
