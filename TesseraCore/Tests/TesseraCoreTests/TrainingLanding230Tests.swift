import Foundation
@testable import TesseraCore
import Testing

/// The Training landing's stat lines and warm-up (sheet 230).
@MainActor
struct TrainingLanding230Tests {
    let store = ObjectStore.open(inMemory: true)
    let classes: [SystemCatalog.Key: ClassDef]
    let engine: ActionEngine
    let stats: TrainingStats
    var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }

    /// Thursday, October 22, 2026, 12:00 UTC.
    var now: Date {
        Date(timeIntervalSince1970: 1_792_670_400)
    }

    init() throws {
        classes = try store.prepareCatalog()
        engine = ActionEngine(store: store, classes: classes)
        stats = TrainingStats(store: store, classes: classes)
    }

    var plan: ObjectRecord {
        get throws { try #require(store.object(id: StarterContent.planID)) }
    }

    @discardableResult
    func session(daysAgo: Double, sets: [(Double, Double)], title: String = "Push A") throws -> ObjectRecord {
        let ended = now.addingTimeInterval(-daysAgo * 86400)
        let workout = try store.create(#require(classes[.workout]), title: title)
        try store.set("endedAt", to: .date(ended), on: workout)
        try store.set("startedAt", to: .date(ended.addingTimeInterval(-3600)), on: workout)
        let entry = try store.create(#require(classes[.exerciseEntry]), parent: workout)
        try store.setClassLink("exercise", on: entry, to: #require(classes[.benchPress]))
        for set in sets {
            let made = try stats.ledger.logSet(weight: set.0, reps: set.1, in: entry)
            try store.set("loggedAt", to: .date(ended), on: made)
        }
        return workout
    }

    // MARK: Week count

    @Test func theWeekCountIsWhatIsDonePlusWhatTheDaysAheadTrain() throws {
        try store.set("sessionsPerWeek", to: .number(4), on: plan)
        try session(daysAgo: 3, sets: [(80, 5)]) // Monday
        try session(daysAgo: 2, sets: [(80, 5)]) // Tuesday
        let count = try stats.weekCount(plan: plan, now: now, calendar: calendar)
        #expect(count.done == 2 && count.planned == 4)
        #expect(TrainingSentences.weekCount(count) == "2 of 4 this week")
        #expect(TrainingSentences.weekCount(.init(done: 3, planned: 0)) == "3 this week")
    }

    // MARK: Weeks in a row

    @Test func weeksInARowCountBackFromThisWeekAndAWeekStillBeingDoneNeverBreaksIt() throws {
        let plan = try plan
        try store.set("sessionsPerWeek", to: .number(2), on: plan)
        // Three full weeks back, two sessions each: Mondays and Wednesdays.
        for week in 1 ... 3 {
            try session(daysAgo: Double(week * 7 + 3), sets: [(80, 5)])
            try session(daysAgo: Double(week * 7 + 1), sets: [(80, 5)])
        }
        try session(daysAgo: 3, sets: [(80, 5)]) // one this week, so far
        #expect(stats.weeksOnPlan(plan: plan, now: now, calendar: calendar) == 3)
        try session(daysAgo: 1, sets: [(80, 5)]) // this week reaches its two
        #expect(stats.weeksOnPlan(plan: plan, now: now, calendar: calendar) == 4)
        #expect(TrainingSentences.weeksInARow(4) == "4 weeks in a row on plan")
        #expect(TrainingSentences.weeksInARow(1) == nil)
    }

    @Test func aShortWeekBreaksTheRunAndNoPlanHasNone() throws {
        let plan = try plan
        try store.set("sessionsPerWeek", to: .number(2), on: plan)
        try session(daysAgo: 8, sets: [(80, 5)]) // last week: one only
        try session(daysAgo: 15, sets: [(80, 5)])
        try session(daysAgo: 16, sets: [(80, 5)])
        #expect(stats.weeksOnPlan(plan: plan, now: now, calendar: calendar) == 0)
        #expect(stats.weeksOnPlan(plan: nil, now: now, calendar: calendar) == 0)
    }

    // MARK: A lift that went up

    @Test func theHeadlineLiftReadsItsGainInWholeStepsAndOnlyAGainHasALine() throws {
        try session(daysAgo: 60, sets: [(80, 5)])
        try session(daysAgo: 55, sets: [(82.5, 5)])
        try session(daysAgo: 5, sets: [(95, 5)])
        try session(daysAgo: 2, sets: [(97.5, 5)])
        let since = now.addingTimeInterval(-70 * 86400)
        let kilos = try #require(stats.headlineGain(since: since, now: now, unit: .kilograms, calendar: calendar))
        #expect(kilos.exercise == "Bench Press" && kilos.steps > 0)
        #expect(TrainingSentences.gain(kilos, calendar: calendar).hasPrefix("+"))
        #expect(TrainingSentences.gain(kilos, calendar: calendar).hasSuffix(" kg on bench press since Aug 13"))
        let pounds = try #require(stats.headlineGain(since: since, now: now, unit: .pounds, calendar: calendar))
        #expect(TrainingSentences.gain(pounds, calendar: calendar).contains(" lb on bench press since "))
        // Down or flat has no line.
        #expect(stats.headlineGain(since: now.addingTimeInterval(-3 * 86400), now: now, calendar: calendar) == nil)
    }

    @Test func aLiftWithAPrivateNameNeverLeadsTheLanding() throws {
        let bench = try #require(classes[.benchPress])
        let old = bench.name
        bench.name = "Lexapro press"
        defer { bench.name = old }
        try session(daysAgo: 60, sets: [(80, 5)])
        try session(daysAgo: 55, sets: [(82.5, 5)])
        try session(daysAgo: 5, sets: [(95, 5)])
        try session(daysAgo: 2, sets: [(97.5, 5)])
        #expect(stats.headlineGain(since: now.addingTimeInterval(-70 * 86400), now: now, calendar: calendar) == nil)
    }

    @Test func aLiftThatIsNotUpHasNoGain() throws {
        try session(daysAgo: 60, sets: [(100, 5)])
        try session(daysAgo: 55, sets: [(100, 5)])
        try session(daysAgo: 5, sets: [(80, 5)])
        try session(daysAgo: 2, sets: [(80, 5)])
        #expect(stats.headlineGain(since: now.addingTimeInterval(-70 * 86400), now: now, calendar: calendar) == nil)
    }

    // MARK: Warm-up

    @Test func aWarmUpIsSetReadAndTakenOff() throws {
        let plan = try plan
        let routine = try #require(store.linkedObjects("routines", of: plan).first)
        #expect(stats.warmUp(of: routine) == nil)
        let set = try engine.perform(SetWarmUp(routine: routine.id, name: "Rower", minutes: 8,
                                               note: "Keep it easy, about 22 strokes a minute."), by: .user)
        #expect(set.record.receipt == "Warm-up set, 8 min")
        let warm = try #require(stats.warmUp(of: routine))
        #expect(warm.minutes == 8 && warm.note == "Keep it easy, about 22 strokes a minute.")
        #expect(TrainingSentences.warmUpLine(warm) == "Starts with 8 min on the rower")
        // A new name with no minutes keeps the minutes.
        try engine.perform(SetWarmUp(routine: routine.id, name: "Bike"), by: .user)
        #expect(stats.warmUp(of: routine)?.minutes == 8 && stats.warmUp(of: routine)?.name == "Bike")
        // The note stays when none is given, and goes when it is empty.
        #expect(stats.warmUp(of: routine)?.note == "Keep it easy, about 22 strokes a minute.")
        try engine.perform(SetWarmUp(routine: routine.id, name: "Bike", note: ""), by: .user)
        #expect(stats.warmUp(of: routine)?.note == nil)
        let off = try engine.perform(SetWarmUp(routine: routine.id, name: nil), by: .user)
        #expect(off.record.receipt == "Warm-up taken off" && stats.warmUp(of: routine) == nil)
        #expect(TrainingSentences.warmUpLine(nil) == nil)
        // Undo brings it back.
        engine.undo(off.record)
        #expect(stats.warmUp(of: routine) != nil)
    }

    @Test func aWarmUpRefusesBadMinutesAndWordsAndAPrivateRoutineShowsNone() throws {
        let plan = try plan
        let routine = try #require(store.linkedObjects("routines", of: plan).first)
        #expect(throws: ActionError.self) {
            try engine.perform(SetWarmUp(routine: routine.id, name: "Rower", minutes: 0), by: .user)
        }
        // No minutes and none set before: refused.
        #expect(throws: ActionError.self) {
            try engine.perform(SetWarmUp(routine: routine.id, name: "Rower"), by: .user)
        }
        #expect(throws: ActionError.self) {
            try engine.perform(SetWarmUp(routine: routine.id, name: "Lexapro dose", minutes: 5), by: .user)
        }
        #expect(throws: ActionError.self) {
            try engine.perform(SetWarmUp(routine: routine.id, name: "Rower", minutes: 5,
                                         note: "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"), by: .user)
        }
        try engine.perform(SetWarmUp(routine: routine.id, name: "Rower", minutes: 8), by: .user)
        routine.title = "Shoulder rehab and Lexapro"
        #expect(stats.warmUp(of: routine) == nil)
    }
}
