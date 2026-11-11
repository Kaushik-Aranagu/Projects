import Foundation

/// One recurrence model for every dated object (sheet 173, switch repeatsV2): tasks, routines, occasions and meals.
/// A rule is stored in the object's `repeats` field and never copied; the next date is computed from it, never stored
/// as a new object. Days are plain calendar days (year, month, day), so a rule means the same in any time zone.
public struct RepeatRule: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        /// Every N days.
        case days(Int)
        /// Every N weeks, on a weekday (1 is Sunday, as `Calendar` counts).
        case weeks(Int, weekday: Int)
        /// Every N months, on a day of the month (the last day when the month is shorter).
        case months(Int, day: Int)
        /// N days after the last time it was done.
        case afterDone(Int)
    }

    /// The one calendar all of a rule's day arithmetic uses.
    static let utc: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .gmt
        return calendar
    }()

    public var kind: Kind
    /// The first day of the rule.
    public var anchor: DayStamp
    /// The last day it may land on: "Stop repeating" sets it.
    public var end: DayStamp?
    /// "Skip this one": nothing on or before this day counts.
    public var skippedThrough: DayStamp?

    public init(kind: Kind, anchor: DayStamp, end: DayStamp? = nil, skippedThrough: DayStamp? = nil) {
        self.kind = kind
        self.anchor = anchor
        self.end = end
        self.skippedThrough = skippedThrough
    }

    /// A calendar day with no time and no zone.
    public struct DayStamp: Equatable, Comparable, Sendable {
        public var year: Int
        public var month: Int
        public var day: Int

        public init(year: Int, month: Int, day: Int) {
            self.year = year
            self.month = month
            self.day = day
        }

        public init(_ date: Date, calendar: Calendar) {
            let parts = calendar.dateComponents([.year, .month, .day], from: date)
            self.init(year: parts.year ?? 1970, month: parts.month ?? 1, day: parts.day ?? 1)
        }

        public static func < (lhs: DayStamp, rhs: DayStamp) -> Bool {
            (lhs.year, lhs.month, lhs.day) < (rhs.year, rhs.month, rhs.day)
        }

        /// Noon on this day in `calendar`, for showing a weekday name or handing a date to a caller.
        public func date(calendar: Calendar) -> Date {
            calendar.date(from: DateComponents(year: year, month: month, day: day, hour: 12)) ?? .distantPast
        }

        /// Days since 1970-01-01 on one fixed calendar: every step of a rule is whole days on this one count, so no
        /// zone, week numbering or daylight change can move a date.
        var ordinal: Int {
            let date = RepeatRule.utc.date(from: DateComponents(year: year, month: month, day: day, hour: 12))
                ?? .distantPast
            return Int((date.timeIntervalSince1970 / 86400).rounded(.down))
        }

        init(ordinal: Int) {
            let date = Date(timeIntervalSince1970: Double(ordinal) * 86400 + 43200)
            let parts = RepeatRule.utc.dateComponents([.year, .month, .day], from: date)
            self.init(year: parts.year ?? 1970, month: parts.month ?? 1, day: parts.day ?? 1)
        }

        /// 1 Sunday to 7 Saturday.
        var weekday: Int {
            (ordinal + 4) % 7 + 1
        }

        func adding(days: Int) -> DayStamp {
            DayStamp(ordinal: ordinal + days)
        }

        static func daysIn(year: Int, month: Int) -> Int {
            let first = RepeatRule.utc.date(from: DateComponents(year: year, month: month, day: 1)) ?? .distantPast
            return RepeatRule.utc.range(of: .day, in: .month, for: first)?.count ?? 28
        }

        var text: String {
            String(format: "%04d-%02d-%02d", year, month, day)
        }

        init?(text: String) {
            let parts = text.split(separator: "-").compactMap { Int($0) }
            guard parts.count == 3, (1900 ... 2200).contains(parts[0]), (1 ... 12).contains(parts[1]),
                  (1 ... Self.daysIn(year: parts[0], month: parts[1])).contains(parts[2]) else { return nil }
            self.init(year: parts[0], month: parts[1], day: parts[2])
        }
    }

    // MARK: Next

    /// The next day the rule lands on, from `today`, or nil when it has ended. `doneAt` is when it was last done: a
    /// fixed rule skips the occurrence that was done, and "after done" counts from it. A skip moves past one step.
    /// An "after done" chore whose date has passed stays on that date (overdue) until it is done or skipped.
    public func next(today: DayStamp, doneAt: Date? = nil, calendar: Calendar) -> DayStamp? {
        let done = doneAt.map { DayStamp($0, calendar: calendar) }
        let found: DayStamp?
        switch kind {
        case let .afterDone(days):
            let step = max(days, 1)
            // Not done yet: the first one is the anchor itself; after a completion, N days on.
            var candidate = done.map { $0.adding(days: step) } ?? anchor
            if let skippedThrough {
                while candidate <= skippedThrough {
                    candidate = candidate.adding(days: step)
                }
            }
            found = candidate
        default:
            var floor = today
            if let skippedThrough, skippedThrough >= floor {
                floor = skippedThrough.adding(days: 1)
            }
            var candidate = occurrence(onOrAfter: max(floor, anchor))
            // Done on (or after) the occurrence: it is finished, so the following one is next.
            if let done, let first = candidate, done >= first {
                candidate = occurrence(onOrAfter: max(first.adding(days: 1), floor))
            }
            found = candidate
        }
        guard let found else { return nil }
        if let end, found > end {
            return nil
        }
        return found
    }

    /// The first occurrence of a fixed rule on or after `day`, counted in whole days from the anchor.
    private func occurrence(onOrAfter day: DayStamp) -> DayStamp? {
        switch kind {
        case let .days(count):
            let step = max(count, 1)
            let gap = day.ordinal - anchor.ordinal
            let steps = gap <= 0 ? 0 : (gap + step - 1) / step
            return anchor.adding(days: steps * step)
        case let .weeks(count, weekday):
            // The phase is days since the first matching weekday on or after the anchor, over seven: the same
            // answer in any week numbering.
            let first = anchor.adding(days: (weekday - anchor.weekday + 7) % 7)
            let period = 7 * max(count, 1)
            let gap = day.ordinal - first.ordinal
            let steps = gap <= 0 ? 0 : (gap + period - 1) / period
            return first.adding(days: steps * period)
        case let .months(count, dayOfMonth):
            let step = max(count, 1)
            var year = anchor.year, month = anchor.month
            for _ in 0 ..< 2000 {
                let length = DayStamp.daysIn(year: year, month: month)
                let stamp = DayStamp(year: year, month: month, day: min(max(dayOfMonth, 1), length))
                if stamp >= day, stamp >= anchor {
                    return stamp
                }
                month += step
                year += (month - 1) / 12
                month = (month - 1) % 12 + 1
            }
            return nil
        case .afterDone:
            return nil
        }
    }

    // MARK: Text

    /// The stored form: "k=weeks;n=2;w=6;a=2026-10-23;e=2026-12-01;s=2026-10-30".
    public var encoded: String {
        var parts: [String] = switch kind {
        case let .days(count): ["k=days", "n=\(count)"]
        case let .weeks(count, weekday): ["k=weeks", "n=\(count)", "w=\(weekday)"]
        case let .months(count, day): ["k=months", "n=\(count)", "d=\(day)"]
        case let .afterDone(count): ["k=afterDone", "n=\(count)"]
        }
        parts.append("a=\(anchor.text)")
        if let end {
            parts.append("e=\(end.text)")
        }
        if let skippedThrough {
            parts.append("s=\(skippedThrough.text)")
        }
        return parts.joined(separator: ";")
    }

    public init?(encoded text: String?) {
        guard let text else { return nil }
        var map: [String: String] = [:]
        for part in text.split(separator: ";") {
            let pair = part.split(separator: "=", maxSplits: 1).map(String.init)
            if pair.count == 2 {
                map[pair[0]] = pair[1]
            }
        }
        guard let kindText = map["k"], let count = map["n"].flatMap({ Int($0) }), (1 ... 365).contains(count),
              let anchor = map["a"].flatMap({ DayStamp(text: $0) }) else { return nil }
        switch kindText {
        case "days": kind = .days(count)
        case "weeks":
            guard let weekday = map["w"].flatMap({ Int($0) }), (1 ... 7).contains(weekday) else { return nil }
            kind = .weeks(count, weekday: weekday)
        case "months":
            guard let day = map["d"].flatMap({ Int($0) }), (1 ... 31).contains(day) else { return nil }
            kind = .months(count, day: day)
        case "afterDone": kind = .afterDone(count)
        default: return nil
        }
        self.anchor = anchor
        end = map["e"].flatMap { DayStamp(text: $0) }
        skippedThrough = map["s"].flatMap { DayStamp(text: $0) }
    }

    // MARK: Sentence

    /// The rule as a sentence from fixed templates: "Every month on the 1st", "Every other Friday", "Every 3 days
    /// after done". One plain line under the title.
    public func sentence(calendar: Calendar) -> String {
        var line: String
        switch kind {
        case let .days(count):
            line = count == 1 ? "Every day" : "Every \(count) days"
        case let .weeks(count, weekday):
            let name = Self.weekdayName(weekday, calendar: calendar)
            line = count == 1 ? "Every \(name)" : count == 2 ? "Every other \(name)" : "Every \(count) weeks on \(name)"
        case let .months(count, day):
            let which = Self.ordinal(day)
            line = count == 1 ? "Every month on the \(which)" : count == 2 ? "Every other month on the \(which)"
                : "Every \(count) months on the \(which)"
        case let .afterDone(count):
            line = count == 1 ? "Every day after done" : "Every \(count) days after done"
        }
        if let end {
            line += ", until \(Self.monthDay(end, calendar: calendar))"
        }
        return line
    }

    static func ordinal(_ day: Int) -> String {
        let suffix = switch (day % 100, day % 10) {
        case (11 ... 13, _): "th"
        case (_, 1): "st"
        case (_, 2): "nd"
        case (_, 3): "rd"
        default: "th"
        }
        return "\(day)\(suffix)"
    }

    static func weekdayName(_ weekday: Int, calendar _: Calendar) -> String {
        let names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return names[(weekday - 1 + 7) % 7]
    }

    static func monthDay(_ stamp: DayStamp, calendar _: Calendar) -> String {
        let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                      "October", "November", "December"]
        return "\(months[(stamp.month - 1 + 12) % 12]) \(stamp.day)"
    }
}
