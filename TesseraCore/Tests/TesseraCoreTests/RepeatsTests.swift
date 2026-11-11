import Foundation
@testable import TesseraCore
import Testing

/// Sheet 173: one recurrence model. A rule in a text field, the next date computed from it, a sentence from fixed
/// templates, and Skip / Stop as logged changes that Undo reverses.
@MainActor
struct RepeatsTests {
    let store = ObjectStore.open(inMemory: true)
    let engine: ActionEngine
    var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }

    init() throws {
        engine = try ActionEngine(store: store, classes: store.prepareCatalog())
    }

    func day(_ month: Int, _ day: Int, year: Int = 2026) -> RepeatRule.DayStamp {
        .init(year: year, month: month, day: day)
    }

    func date(_ month: Int, _ day: Int, hour: Int = 12) -> Date {
        calendar.date(from: DateComponents(year: 2026, month: month, day: day, hour: hour))!
    }

    // MARK: Next dates

    @Test func everyThreeDaysCountsFromTheAnchor() {
        let rule = RepeatRule(kind: .days(3), anchor: day(10, 1))
        #expect(rule.next(today: day(10, 1), calendar: calendar) == day(10, 1))
        #expect(rule.next(today: day(10, 2), calendar: calendar) == day(10, 4))
        #expect(rule.next(today: day(10, 8), calendar: calendar) == day(10, 10))
    }

    @Test func everyOtherFridayLandsOnEveryOtherFriday() {
        // October 23, 2026 is a Friday.
        let rule = RepeatRule(kind: .weeks(2, weekday: 6), anchor: day(10, 23))
        #expect(rule.next(today: day(10, 23), calendar: calendar) == day(10, 23))
        #expect(rule.next(today: day(10, 24), calendar: calendar) == day(11, 6))
        #expect(rule.next(today: day(11, 7), calendar: calendar) == day(11, 20))
    }

    @Test func theFirstOfEveryMonthAndTheEndOfShortMonths() {
        let first = RepeatRule(kind: .months(1, day: 1), anchor: day(10, 5))
        #expect(first.next(today: day(10, 5), calendar: calendar) == day(11, 1))
        let thirty = RepeatRule(kind: .months(1, day: 31), anchor: day(1, 31))
        #expect(thirty.next(today: day(2, 1), calendar: calendar) == day(2, 28))
        #expect(thirty.next(today: day(4, 1), calendar: calendar) == day(4, 30))
    }

    @Test func afterDoneCountsFromTheCompletionTime() {
        let rule = RepeatRule(kind: .afterDone(3), anchor: day(10, 20))
        #expect(rule.next(today: day(10, 20), doneAt: date(10, 20, hour: 9), calendar: calendar) == day(10, 23))
        // Not done yet: the anchor itself, and never in the past.
        #expect(rule.next(today: day(10, 20), calendar: calendar) == day(10, 20))
        // Past its date it reads as overdue: it stays on its day until it is done or skipped.
        #expect(rule.next(today: day(10, 21), calendar: calendar) == day(10, 20))
        var skipped = rule
        skipped.skippedThrough = day(10, 20)
        #expect(skipped.next(today: day(10, 21), calendar: calendar) == day(10, 23))
    }

    @Test func aDoneOccurrenceMovesOnAndAnEndStopsTheRule() {
        var rule = RepeatRule(kind: .weeks(1, weekday: 6), anchor: day(10, 2))
        #expect(rule.next(today: day(10, 23), doneAt: date(10, 23), calendar: calendar) == day(10, 30))
        rule.end = day(10, 23)
        #expect(rule.next(today: day(10, 23), calendar: calendar) == day(10, 23))
        #expect(rule.next(today: day(10, 24), calendar: calendar) == nil)
    }

    @Test func aSkipMovesPastOneStep() {
        var rule = RepeatRule(kind: .weeks(1, weekday: 6), anchor: day(10, 2))
        rule.skippedThrough = day(10, 23)
        #expect(rule.next(today: day(10, 21), calendar: calendar) == day(10, 30))
    }

    // MARK: Text

    @Test func rulesRoundTripAndReadAsSentences() {
        let rules: [(RepeatRule, String)] = [
            (RepeatRule(kind: .days(1), anchor: day(10, 1)), "Every day"),
            (RepeatRule(kind: .days(3), anchor: day(10, 1)), "Every 3 days"),
            (RepeatRule(kind: .weeks(1, weekday: 6), anchor: day(10, 1)), "Every Friday"),
            (RepeatRule(kind: .weeks(2, weekday: 6), anchor: day(10, 1)), "Every other Friday"),
            (RepeatRule(kind: .weeks(3, weekday: 2), anchor: day(10, 1)), "Every 3 weeks on Monday"),
            (RepeatRule(kind: .months(1, day: 1), anchor: day(10, 1)), "Every month on the 1st"),
            (RepeatRule(kind: .months(2, day: 22), anchor: day(10, 1)), "Every other month on the 22nd"),
            (RepeatRule(kind: .months(3, day: 13), anchor: day(10, 1)), "Every 3 months on the 13th"),
            (RepeatRule(kind: .afterDone(3), anchor: day(10, 1)), "Every 3 days after done"),
            (RepeatRule(kind: .days(2), anchor: day(10, 1), end: day(12, 1)), "Every 2 days, until December 1"),
        ]
        for (rule, sentence) in rules {
            #expect(rule.sentence(calendar: calendar) == sentence)
            #expect(RepeatRule(encoded: rule.encoded) == rule)
        }
        #expect(RepeatRule(encoded: "nonsense") == nil && RepeatRule(encoded: nil) == nil)
    }

    // MARK: Actions

    func task(_ title: String = "Water the plants") throws -> ObjectRecord {
        try engine.perform(CreateTask(title: title, startsAt: date(10, 23, hour: 9)), by: .user).output
    }

    @Test func settingSkippingAndStoppingAreLoggedAndUndoable() throws {
        let plants = try task()
        try engine.perform(SetRepeat(object: plants.id, kind: "weeks", every: 1, weekday: 6, from: date(10, 16),
                                     calendar: calendar), by: .user)
        #expect(Repeats.line(for: plants, in: store, calendar: calendar) == "Every Friday")
        #expect(Repeats.next(of: plants, in: store, now: date(10, 21), calendar: calendar) == day(10, 23))

        let skipped = try engine.perform(SkipRepeat(object: plants.id, time: date(10, 21), calendar: calendar),
                                         by: .user)
        #expect(Repeats.next(of: plants, in: store, now: date(10, 21), calendar: calendar) == day(10, 30))
        engine.undo(skipped.record)
        #expect(Repeats.next(of: plants, in: store, now: date(10, 21), calendar: calendar) == day(10, 23))

        let stopped = try engine.perform(StopRepeating(object: plants.id, time: date(10, 24), calendar: calendar),
                                         by: .user)
        #expect(Repeats.next(of: plants, in: store, now: date(10, 25), calendar: calendar) == nil)
        engine.undo(stopped.record)
        #expect(Repeats.next(of: plants, in: store, now: date(10, 25), calendar: calendar) == day(10, 30))
    }

    @Test func aBadRepeatIsRefusedAndNothingIsStored() throws {
        let plants = try task()
        #expect(throws: (any Error).self) {
            try engine.perform(SetRepeat(object: plants.id, kind: "fortnightly", every: 1), by: .user)
        }
        #expect(throws: (any Error).self) {
            try engine.perform(SkipRepeat(object: plants.id), by: .user)
        }
        #expect(Repeats.rule(of: plants, in: store) == nil)
    }

    @Test func theRuleNeverReachesTheDayLogAndEveryActionIsRegistered() {
        for name in ["setRepeat", "skipRepeat", "stopRepeating"] {
            #expect(ActionRegistry.descriptors.contains { $0.name == name })
            #expect(DayStory.dropped.contains(name))
        }
    }

    // MARK: Lines

    @Test func aNoticeIsANameAndADayNeverAnAmount() throws {
        let rent = try task("Pay the rent")
        try engine.perform(SetRepeat(object: rent.id, kind: "weeks", every: 1, weekday: 6, from: date(10, 16),
                                     calendar: calendar), by: .user)
        let classes = engine.classes
        let line = Repeats.noticeLine(for: rent, in: store, classes: classes, now: date(10, 21), calendar: calendar)
        #expect(line == "Pay the rent, Friday")
    }

    @Test func aWalledObjectIsNamedPrivateInItsNotice() throws {
        let lunch = try task("Lunch 650 kcal")
        try engine.perform(SetRepeat(object: lunch.id, kind: "days", every: 1, from: date(10, 20), calendar: calendar),
                           by: .user)
        let line = Repeats.noticeLine(for: lunch, in: store, classes: engine.classes, now: date(10, 21),
                                      calendar: calendar)
        #expect(line?.contains("650") == false && line?.contains("kcal") == false)
    }

    // MARK: Review fixes

    @Test func everyOtherMondayStartsOnTheNextMondayWithThatMondayAsAnchor() throws {
        let chore = try task()
        // Wednesday, October 21, 2026.
        let set = SetRepeat(object: chore.id, kind: "weeks", every: 2, weekday: 2, from: date(10, 21),
                            calendar: calendar)
        #expect(set.anchor == "2026-10-26")
        try engine.perform(set, by: .user)
        let rule = try #require(Repeats.rule(of: chore, in: store))
        #expect(rule.anchor == day(10, 26))
        #expect(rule.next(today: day(10, 21), calendar: calendar) == day(10, 26))
        #expect(rule.next(today: day(10, 27), calendar: calendar) == day(11, 9))
        #expect(Repeats.line(for: chore, in: store, calendar: calendar) == "Every other Monday")
    }

    @Test func theWeekPhaseIsDaysFromTheAnchorWhateverTheCalendarCountsAsAWeek() {
        let rule = RepeatRule(kind: .weeks(2, weekday: 6), anchor: day(10, 23))
        var sunday = calendar, monday = calendar
        sunday.firstWeekday = 1
        monday.firstWeekday = 2
        for today in [day(10, 24), day(11, 7), day(12, 30)] {
            #expect(rule.next(today: today, calendar: sunday) == rule.next(today: today, calendar: monday))
        }
        #expect(rule.next(today: day(11, 7), calendar: sunday) == day(11, 20))
    }

    @Test func oneAMIsYesterdayAndAReplayGivesTheSameRule() throws {
        let chore = try task()
        // 1 AM on Thursday, October 22: still Wednesday's day.
        let set = SetRepeat(object: chore.id, kind: "days", every: 1, from: date(10, 22, hour: 1), calendar: calendar)
        #expect(set.anchor == "2026-10-21")
        let replay = try JSONDecoder().decode(SetRepeat.self, from: JSONEncoder().encode(set))
        #expect(replay.rule() == set.rule())
        let skip = SkipRepeat(object: chore.id, time: date(10, 22, hour: 1), calendar: calendar)
        #expect(skip.today == "2026-10-21")
    }

    @Test func aDayPastTheMonthsLengthIsRefused() {
        #expect(RepeatRule.DayStamp(text: "2026-02-30") == nil)
        #expect(RepeatRule.DayStamp(text: "2026-04-31") == nil)
        #expect(RepeatRule.DayStamp(text: "2028-02-29") != nil)
        #expect(RepeatRule(encoded: "k=days;n=1;a=2026-02-30") == nil)
    }
}
