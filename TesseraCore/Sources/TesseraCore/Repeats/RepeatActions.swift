import Foundation

/// Sheet 173: setting, skipping and stopping a repeat. Each is one logged change to the object's `repeats` text, so
/// the op log's Undo reverses it. Skipping and stopping never create or delete an object, and a rule is never copied.
public enum Repeats {
    /// The rule on an object, or nil.
    @MainActor public static func rule(of object: ObjectRecord, in store: ObjectStore) -> RepeatRule? {
        RepeatRule(encoded: store.value("repeats", of: object)?.stringValue)
    }

    /// The next day, or nil when it has no rule or has ended.
    @MainActor public static func next(of object: ObjectRecord, in store: ObjectStore, now: Date,
                                       calendar: Calendar) -> RepeatRule.DayStamp?
    {
        guard let rule = rule(of: object, in: store) else { return nil }
        let day = TesseraDay(calendar: calendar).day(of: now)
        return rule.next(today: .init(day, calendar: calendar), doneAt: store.value("doneAt", of: object)?.dateValue,
                         calendar: calendar)
    }

    /// The one plain line under a title: "Every other Friday". A private object's own words never appear in it.
    @MainActor public static func line(for object: ObjectRecord, in store: ObjectStore, calendar: Calendar) -> String? {
        rule(of: object, in: store)?.sentence(calendar: calendar)
    }

    /// "Rent, Friday": the name (its shown label when private) and the day, never an amount and never an action.
    @MainActor public static func noticeLine(for object: ObjectRecord, in store: ObjectStore,
                                             classes: [SystemCatalog.Key: ClassDef], now: Date,
                                             calendar: Calendar) -> String?
    {
        guard let next = next(of: object, in: store, now: now, calendar: calendar) else { return nil }
        let name = ObjectPrivacy.shownLabel(of: object, in: store, classes: classes)
            ?? AppSummaryLines.shown(object.title, in: store, classes: classes, of: object).text
        return "\(name), \(TesseraDay.weekdayName(of: next.date(calendar: calendar), calendar: calendar))"
    }
}

public struct SetRepeat: EngineAction {
    public static let descriptor = ActionDescriptor(
        name: "setRepeat", summary: "Make something repeat",
        parameters: [
            .init("object", .object, "The task, routine or occasion"),
            .init("kind", .string, "days, weeks, months or afterDone"),
            .init("every", .integer, "Every N (days, weeks or months)"),
            .init("weekday", .integer, "1 Sunday to 7 Saturday, for weeks", required: false),
            .init("dayOfMonth", .integer, "1 to 31, for months", required: false),
            .init("anchor", .string, "The first day, as 2026-10-26; set when the action is made", required: false),
        ], isOfferedToModel: false
    )

    public var object: UUID
    public var kind: String
    public var every: Int
    public var weekday: Int?
    public var dayOfMonth: Int?
    /// The first day as "2026-10-26", worked out when the action is made, so a replay gives the same rule.
    public var anchor: String?

    /// The day is read once, here, through his day (1 AM still belongs to yesterday). "Every other Monday" set on a
    /// Wednesday starts on the next Monday, and that Monday is the anchor.
    public init(object: UUID, kind: String, every: Int = 1, weekday: Int? = nil, dayOfMonth: Int? = nil,
                from: Date? = nil, calendar: Calendar = .current)
    {
        self.object = object
        self.kind = kind
        self.every = every
        self.weekday = weekday
        self.dayOfMonth = dayOfMonth
        let base = RepeatRule.DayStamp(TesseraDay(calendar: calendar).day(of: from ?? .now), calendar: calendar)
        if kind == "weeks" {
            let wanted = weekday ?? base.weekday
            self.weekday = wanted
            anchor = base.adding(days: (wanted - base.weekday + 7) % 7).text
        } else {
            anchor = base.text
        }
    }

    /// The rule this builds, or nil when the arguments don't make one.
    func rule() -> RepeatRule? {
        guard (1 ... 365).contains(every), let anchor = anchor.flatMap({ RepeatRule.DayStamp(text: $0) }) else {
            return nil
        }
        switch kind {
        case "days": return RepeatRule(kind: .days(every), anchor: anchor)
        case "weeks":
            let day = weekday ?? anchor.weekday
            guard (1 ... 7).contains(day) else { return nil }
            return RepeatRule(kind: .weeks(every, weekday: day), anchor: anchor)
        case "months":
            let day = dayOfMonth ?? anchor.day
            guard (1 ... 31).contains(day) else { return nil }
            return RepeatRule(kind: .months(every, day: day), anchor: anchor)
        case "afterDone": return RepeatRule(kind: .afterDone(every), anchor: anchor)
        default: return nil
        }
    }

    public func apply(_ transaction: inout Transaction) throws {
        let target = try transaction.object(object)
        guard let rule = rule() else {
            throw ActionError.invalidArguments(action: Self.descriptor.name, reason: "that is not a repeat")
        }
        try transaction.set("repeats", to: .text(rule.encoded), on: target)
    }

    public func summary(in _: ObjectStore) -> String {
        "Set a repeat"
    }

    public func receipt(in _: ObjectStore) -> String? {
        nil
    }
}

public struct SkipRepeat: EngineAction {
    public static let descriptor = ActionDescriptor(
        name: "skipRepeat", summary: "Skip this one",
        parameters: [
            .init("object", .object, "The repeating thing"),
            .init("today", .string, "His day as 2026-10-26; set when the action is made", required: false),
        ], isOfferedToModel: false
    )

    public var object: UUID
    /// His day, worked out when the action is made (1 AM is yesterday), so a replay gives the same result.
    public var today: String?
    /// The zone his day was read in, so a done time is read as the same day on a replay.
    public var zone: String?

    public init(object: UUID, time: Date? = nil, calendar: Calendar = .current) {
        self.object = object
        zone = calendar.timeZone.identifier
        today = RepeatRule.DayStamp(TesseraDay(calendar: calendar).day(of: time ?? .now), calendar: calendar).text
    }

    public func apply(_ transaction: inout Transaction) throws {
        let target = try transaction.object(object)
        guard var rule = Repeats.rule(of: target, in: transaction.store),
              let day = today.flatMap({ RepeatRule.DayStamp(text: $0) })
        else {
            throw ActionError.invalidArguments(action: Self.descriptor.name, reason: "that does not repeat")
        }
        let done = transaction.store.value("doneAt", of: target)?.dateValue
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = zone.flatMap { TimeZone(identifier: $0) } ?? .current
        guard let next = rule.next(today: day, doneAt: done, calendar: calendar) else { return }
        // The next one is skipped: nothing on or before it counts, so the following step is next.
        rule.skippedThrough = next
        try transaction.set("repeats", to: .text(rule.encoded), on: target)
    }

    public func summary(in _: ObjectStore) -> String {
        "Skipped one"
    }

    public func receipt(in _: ObjectStore) -> String? {
        nil
    }
}

public struct StopRepeating: EngineAction {
    public static let descriptor = ActionDescriptor(
        name: "stopRepeating", summary: "Stop repeating",
        parameters: [
            .init("object", .object, "The repeating thing"),
            .init("through", .string, "The last day, 2026-10-26; set when the action is made", required: false),
        ], isOfferedToModel: false
    )

    public var object: UUID
    public var through: String?

    public init(object: UUID, time: Date? = nil, calendar: Calendar = .current) {
        self.object = object
        through = RepeatRule.DayStamp(TesseraDay(calendar: calendar).day(of: time ?? .now), calendar: calendar).text
    }

    public func apply(_ transaction: inout Transaction) throws {
        let target = try transaction.object(object)
        guard var rule = Repeats.rule(of: target, in: transaction.store),
              let day = through.flatMap({ RepeatRule.DayStamp(text: $0) })
        else {
            throw ActionError.invalidArguments(action: Self.descriptor.name, reason: "that does not repeat")
        }
        // Today still counts; nothing after it does.
        rule.end = day
        try transaction.set("repeats", to: .text(rule.encoded), on: target)
    }

    public func summary(in _: ObjectStore) -> String {
        "Stopped repeating"
    }

    public func receipt(in _: ObjectStore) -> String? {
        nil
    }
}
