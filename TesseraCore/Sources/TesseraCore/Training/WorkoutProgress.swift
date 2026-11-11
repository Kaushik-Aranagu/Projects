import Foundation

/// Where a workout stands: the next set to do, as a resume point and a one-line next step.
@MainActor
public struct WorkoutProgress {
    public let entry: ObjectRecord?
    /// Zero-based index of the next set within `entry`.
    public let setIndex: Int
    public let weight: Double?
    public let reps: Double?
    public let exerciseName: String?
    public let entriesStarted: Int
    public let entryCount: Int

    public var isComplete: Bool {
        entry == nil
    }

    /// "Bench press, set 3: 80 kg × 8". Nil once every planned set is logged.
    public var nextStep: String? {
        nextStep(unit: .kilograms)
    }

    /// The same line with the weight in his unit (stored kilograms read through `MassUnit`): "Bench press, set 3: 176.5
    /// lb × 8". Nothing stored changes.
    public func nextStep(unit: MassUnit) -> String? {
        guard let exerciseName else { return nil }
        var line = "\(exerciseName), set \(setIndex + 1)"
        if let weight, let reps {
            line += ": \(unit.text(kilograms: weight)) \(unit.symbol) × \(Self.format(reps))"
        }
        return line
    }

    public var resumePoint: ResumePoint {
        ResumePoint(focus: entry?.id, index: setIndex, weight: weight, reps: reps)
    }

    public init(of workout: ObjectRecord, store: ObjectStore) {
        let entries = workout.orderedChildren
        // A warm-up never moves the next working set along (it is a kind, as the ledger reads it).
        let isWorking = { (set: ObjectRecord) in store.setKind(of: set) != .warmup }
        entryCount = entries.count
        entriesStarted = entries.count { $0.orderedChildren.contains(where: isWorking) }
        let next = entries.lazy.compactMap { entry -> (ObjectRecord, [ObjectRecord])? in
            let logged = entry.orderedChildren.filter(isWorking)
            let planned = Int(store.number("targetSets", of: entry) ?? 3)
            return logged.count < planned ? (entry, logged) : nil
        }.first
        entry = next?.0
        setIndex = next?.1.count ?? 0
        let last = next?.1.last
        weight = next.flatMap { store.number("targetWeight", of: $0.0) } ?? last
            .flatMap { store.number("weight", of: $0) }
        reps = next.flatMap { store.number("targetReps", of: $0.0) } ?? last.flatMap { store.number("reps", of: $0) }
        exerciseName = next.flatMap { store.linkedClass("exercise", of: $0.0)?.name }
    }

    nonisolated static func format(_ value: Double) -> String {
        // Up to two decimals, trailing zeros dropped: 80, 82.5, 1.25.
        value.rounded() == value ? String(Int(value)) : String(format: "%.2f", value)
            .replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
    }
}

extension Transaction {
    /// Refreshes the workout's process with its next set, if it has an open one.
    mutating func touchWorkoutProcess(_ workout: ObjectRecord, at now: Date) throws {
        guard let process = openProcess(for: workout) else { return }
        // Logging into a set-aside workout picks it back up.
        if store.value("state", of: process)?.stringValue == ProcessState.suspended.rawValue {
            try makeRunning(process, at: now)
        }
        let progress = WorkoutProgress(of: workout, store: store)
        try saveContext(process, resumePoint: progress.resumePoint, nextStep: progress.nextStep, at: now)
    }
}

extension MassUnit {
    /// Stored kilograms as a number in a sentence, through Measure so the same weight reads the same on the Units page
    /// and on every line (82.5 kg is 181.9 lb). Kilograms stay as stored.
    func text(kilograms: Double) -> String {
        guard self != .kilograms, let moved = Measure(kilograms, "kg").converted(to: symbol) else {
            return WorkoutProgress.format(kilograms)
        }
        return MeasureFormat.number(moved.value, unit: symbol)
    }

    /// The smallest change worth saying in this unit: half a kilo or half a pound (numbers show one decimal).
    var step: Double {
        0.5
    }

    /// A raise in stored kilograms, in this unit and rounded to its step ("Up 5.5 lb", not "Up 5.51 lb").
    func text(kilogramsDelta delta: Double) -> String {
        let shown = self == .kilograms ? delta : delta * Self.poundsPerKilogram
        // A raise is never "Up 0": at least one step.
        let stepped = max((shown / step).rounded() * step, shown > 0 ? step : 0)
        return MeasureFormat.number(stepped, unit: symbol)
    }
}
