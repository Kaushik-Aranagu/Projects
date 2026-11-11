import Foundation

/// The Training landing's stat lines (sheet 230): "3 of 6 this week", "5 weeks in a row on plan" and "+10 lb on bench
/// since Oct 1". Pure reads from the ledger; no number leads and nothing reads "lost" or "down".
@MainActor
public extension TrainingStats {
    struct WeekCount: Equatable, Sendable {
        public let done: Int
        public let planned: Int
    }

    /// Sessions done this week against the sessions the plan has for it: what is done, plus the days still ahead that
    /// the plan trains (today included when nothing is done yet). A rest day ahead is not a session.
    func weekCount(plan: ObjectRecord?, now: Date, calendar: Calendar = .current) -> WeekCount {
        let week = ledger.week(containing: now, plan: plan, now: now, calendar: calendar)
        let done = week.reduce(0) { $0 + $1.done.count }
        let ahead = week.filter { day in
            day.done.isEmpty && !day.isRest && day.planned != nil && (day.state == .today || day.state == .upcoming)
        }.count
        return WeekCount(done: done, planned: done + ahead)
    }

    /// How many sessions a week the plan asks for: its own number, else one per routine up to six; on fixed days, the
    /// days that have a routine.
    func sessionsPerWeek(of plan: ObjectRecord, now: Date, calendar: Calendar = .current) -> Int {
        if PlanSchedule.mode(of: plan, store: store) == .fixed, PlanSchedule.hasFixedDays(plan, store: store) {
            let days = TesseraDay(calendar: calendar).weekDays(containing: now)
            return days.filter { PlanSchedule.routine(on: $0, in: plan, store: store, calendar: calendar) != nil }.count
        }
        let routines = store.linkedObjects("routines", of: plan).count
        return Int(store.number("sessionsPerWeek", of: plan) ?? Double(min(max(routines, 1), 6)))
    }

    /// Weeks in a row on plan: the weeks, newest first from this one, in which the sessions done reached what the plan
    /// asks for. This week counts once it has reached it; a week still being done never breaks the run. Weeks before
    /// the
    /// plan started don't count. A plan with no sessions asked for has no run.
    func weeksOnPlan(plan: ObjectRecord?, now: Date, calendar: Calendar = .current) -> Int {
        guard let plan else { return 0 }
        let expected = sessionsPerWeek(of: plan, now: now, calendar: calendar)
        guard expected > 0 else { return 0 }
        let tessera = TesseraDay(calendar: calendar)
        let startsOn = store.value("startsOn", of: plan)?.dateValue.map { tessera.week(containing: $0).start }
        var count = 0
        var cursor = tessera.week(containing: now).start
        let thisWeek = ledger.week(containing: now, plan: plan, now: now, calendar: calendar)
        if thisWeek.reduce(0, { $0 + $1.done.count }) >= expected {
            count += 1
        }
        for _ in 0 ..< 104 {
            guard let before = calendar.date(byAdding: .day, value: -7, to: cursor) else { break }
            cursor = tessera.week(containing: before.addingTimeInterval(3600 * 12)).start
            if let startsOn, cursor < startsOn {
                break
            }
            let week = ledger.week(containing: cursor.addingTimeInterval(3600 * 12), plan: plan, now: now,
                                   calendar: calendar)
            guard week.reduce(0, { $0 + $1.done.count }) >= expected else { break }
            count += 1
        }
        return count
    }

    /// A lift that went up since a date, in display steps (2.5 kg or 5 lb).
    struct LiftGain: Equatable, Sendable {
        public let exercise: String
        public let exerciseID: UUID
        public let since: Date
        public let steps: Int
        public let unit: MassUnit

        /// The gain in kilograms, whole steps of the unit.
        public var kilograms: Double {
            let size = StatsMass.step(unit)
            return Double(steps) * size / (unit == .pounds ? MassUnit.poundsPerKilogram : 1)
        }
    }

    /// The lift to lead the landing with and how far it is up since `since`: of the lifts with counted sessions since
    /// then, the one done most often (ties by name), its best estimated max over its first 28 days there against its
    /// best
    /// in the last 28 days, in whole steps. Nil when there isn't enough to say, or it is not up.
    func headlineGain(since: Date, now: Date, unit: MassUnit = .kilograms, calendar _: Calendar = .current)
        -> LiftGain?
    {
        let recentStart = now.addingTimeInterval(-28 * 86400)
        var best: (gain: LiftGain, sessions: Int)?
        let last = BodyWeight.lastKg(in: store)
        let grouped = Dictionary(grouping: counted(), by: { $0.exercise.id })
        for group in grouped.values {
            let exercise = group[0].exercise
            // A lift whose name is a private word never leads the landing.
            guard !ObjectPrivacy.isPrivateTitle(exercise.name, lastWeightKg: last) else { continue }
            let all = group.map(\.session).filter { $0.date >= since && $0.date <= now }.sorted { $0.date < $1.date }
            func top(_ session: TrainingLedger.Session) -> Double? {
                session.sets.filter(\.isE1RMEligible).map(\.e1RM).max()
            }
            let eligible = all.filter { top($0) != nil }
            guard eligible.count >= 2, let first = eligible.first else { continue }
            let thenEnd = first.date.addingTimeInterval(28 * 86400)
            guard thenEnd <= recentStart,
                  let thenKg = eligible.filter({ $0.date < thenEnd }).compactMap(top).max(),
                  let nowKg = eligible.filter({ $0.date >= recentStart }).compactMap(top).max() else { continue }
            let size = StatsMass.step(unit)
            let steps = Int(((StatsMass.inDisplay(nowKg, unit) - StatsMass.inDisplay(thenKg, unit)) / size)
                .rounded(.toNearestOrAwayFromZero))
            guard steps > 0 else { continue }
            let gain = LiftGain(
                exercise: exercise.name,
                exerciseID: exercise.id,
                since: since,
                steps: steps,
                unit: unit
            )
            if best == nil || eligible.count > best!.sessions
                || (eligible.count == best!.sessions && exercise.name < best!.gain.exercise)
            {
                best = (gain, eligible.count)
            }
        }
        return best?.gain
    }
}

public extension TrainingSentences {
    /// "3 of 6 this week"; with nothing planned, "3 this week".
    static func weekCount(_ count: TrainingStats.WeekCount) -> String {
        count.planned > 0 ? "\(count.done) of \(count.planned) this week" : "\(count.done) this week"
    }

    /// "5 weeks in a row on plan", only from two weeks up.
    static func weeksInARow(_ weeks: Int) -> String? {
        weeks >= 2 ? "\(weeks) weeks in a row on plan" : nil
    }

    /// "+10 lb on bench press since Oct 1". Only a gain has a line.
    @MainActor
    static func gain(_ gain: TrainingStats.LiftGain, calendar: Calendar = .current) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = Locale(identifier: "en_US")
        formatter.setLocalizedDateFormatFromTemplate("MMMd")
        let amount = StatsMass.text(kg: gain.kilograms, unit: gain.unit)
        return "+\(amount) on \(gain.exercise.lowercased()) since \(formatter.string(from: gain.since))"
    }

    /// "Starts with 8 min on the rower" for the landing brief (no final stop, like every shown line), or nil with no
    /// warm-up.
    static func warmUpLine(_ warmUp: WarmUp?) -> String? {
        warmUp.map { "Starts with \($0.phrase)" }
    }
}
