import Foundation

/// A progression rule, stored as text on an exercise class or a routine slot so it inherits and can be edited.
/// v1 understands double progression: `dp(+2.5, 8, 12)`.
public enum ProgressionRule: Hashable, Sendable {
    /// When every working set hits `repMax`, add `increment` and drop back to `repMin`.
    case doubleProgression(increment: Double, repMin: Int, repMax: Int)

    public init?(parsing text: String) {
        let compact = text.replacingOccurrences(of: " ", with: "").lowercased()
        guard compact.hasPrefix("dp("), compact.hasSuffix(")") else { return nil }
        let parts = compact.dropFirst(3).dropLast().split(separator: ",").map(String.init)
        guard parts.count == 3, let increment = Double(parts[0].replacingOccurrences(of: "+", with: "")),
              let low = Int(parts[1]), let high = Int(parts[2]), low >= 1, high >= low, increment > 0
        else { return nil }
        self = .doubleProgression(increment: increment, repMin: low, repMax: high)
    }

    public var text: String {
        switch self {
        case let .doubleProgression(increment, low, high):
            "dp(+\(WorkoutProgress.format(increment)), \(low), \(high))"
        }
    }
}

/// The prefilled target for the next session, with a one-line reason ("Why this weight").
public struct SetTarget: Hashable, Sendable {
    public var weight: Double?
    public var reps: Int
    public var reason: String
}

@MainActor
public extension TrainingLedger {
    /// Next session's target for an exercise: from the rule when one applies, else last time's weight and reps,
    /// else the bottom of the rep range with no weight. The increment snaps to what the gym can load.
    func nextTarget(for exercise: ClassDef, slot: ObjectRecord? = nil,
                    excludingWorkout excludedID: UUID? = nil, unit: MassUnit = .kilograms) -> SetTarget
    {
        // With no slot given (the exercise page), use the plan's slot for this exercise, so its rep range counts.
        let slot = slot ?? store.planSlot(for: exercise)
        let classLow = store.classValue("repMin", of: exercise)?.numericValue.map(Int.init) ?? 8
        let classHigh = store.classValue("repMax", of: exercise)?.numericValue.map(Int.init) ?? 12
        let ruleText = slot.flatMap { store.value("progression", of: $0)?.stringValue }
            ?? store.classValue("progression", of: exercise)?.stringValue
        let rule = ruleText.flatMap(ProgressionRule.init(parsing:))
            ?? store.equipmentKind(of: exercise).flatMap {
                ProgressionRule.default(for: $0, low: classLow, high: max(classHigh, classLow))
            }
        let low = slot.flatMap { store.number("repMin", of: $0) }.map(Int.init) ?? rule.map(\.repMin) ?? classLow
        let high = slot.flatMap { store.number("repMax", of: $0) }.map(Int.init) ?? rule.map(\.repMax) ?? classHigh

        guard let last = sessions(of: exercise, excludingWorkout: excludedID).first,
              let top = last.sets.max(by: { $0.weight < $1.weight })
        else { return SetTarget(weight: nil, reps: low, reason: "First time: start light, \(low) reps") }

        let working = last.sets.filter { $0.weight >= top.weight - 0.001 }
        let reps = working.map { Int($0.reps.rounded()) }
        guard case let .doubleProgression(increment, _, _)? = rule else {
            return SetTarget(weight: top.weight, reps: reps.min() ?? low, reason: "Same as last time")
        }
        if reps.allSatisfy({ $0 >= high }) {
            let profile = store.loadProfile(for: exercise)
            let raised = profile.nearest(to: top.weight + increment)
            let next = raised > top.weight + 0.001 ? raised : profile.next(above: top.weight) ?? top.weight + increment
            return SetTarget(weight: next, reps: low,
                             reason: "Up \(unit.text(kilogramsDelta: next - top.weight)) \(unit.symbol) from last time")
        }
        let target = min((reps.min() ?? low) + 1, high)
        return SetTarget(weight: top.weight, reps: max(target, low),
                         reason: "Same weight; aim for \(max(target, low)) reps on every set")
    }
}

extension ProgressionRule {
    /// The equipment's default rule, when neither a slot nor the class sets one: barbells add 2.5 kg,
    /// dumbbells 2 kg, machines and cables a 5 kg plate. Bodyweight work progresses by reps only.
    static func `default`(for equipment: EquipmentKind, low: Int, high: Int) -> ProgressionRule? {
        switch equipment {
        case .barbell, .ezBar, .smith: .doubleProgression(increment: 2.5, repMin: low, repMax: high)
        case .dumbbell, .kettlebell: .doubleProgression(increment: 2, repMin: low, repMax: high)
        case .machine, .cable: .doubleProgression(increment: 5, repMin: low, repMax: high)
        case .bodyweight, .band: nil
        }
    }

    var repMin: Int {
        if case let .doubleProgression(_, low, _) = self {
            low
        } else {
            1
        }
    }

    var repMax: Int {
        if case let .doubleProgression(_, _, high) = self {
            high
        } else {
            1
        }
    }
}

public extension ObjectStore {
    /// A light first-session weight for an exercise with no history: the empty bar, light dumbbells or a kettlebell,
    /// the first plate on a stack, no added load on bodyweight work. Nil for bands (nothing to load).
    func startingWeight(for exercise: ClassDef) -> Double? {
        let profile = loadProfile(for: exercise)
        switch equipmentKind(of: exercise) {
        case .barbell, .ezBar, .smith: return (profile as? PlateLoad)?.bar ?? profile.nearest(to: 20)
        case .dumbbell: return profile.nearest(to: 8)
        case .kettlebell: return profile.nearest(to: 12)
        case .machine, .cable: return profile.nearest(to: 20)
        case .bodyweight: return 0
        case .band: return nil
        case nil: return profile.nearest(to: 20)
        }
    }

    /// The routine slot that programs this exercise (or a parent class of it), preferring the starter plan's
    /// routines: where a rep range for it lives when there's no workout in hand.
    func planSlot(for exercise: ClassDef) -> ObjectRecord? {
        guard let slotClass = allClasses().first(where: { $0.systemKey == SystemCatalog.Key.slot.systemKey })
        else { return nil }
        let lineage = Set(lineage(of: exercise).map(\.id))
        let slots = objects(of: slotClass).filter { slot in
            linkedClass("exercise", of: slot).map { lineage.contains($0.id) } ?? false
        }
        let starter = Set(object(id: StarterContent.planID).map { linkedObjects("routines", of: $0).map(\.id) } ?? [])
        let ranked = slots.sorted { lhs, rhs in
            let lhsPlan = lhs.parent.map { starter.contains($0.id) } ?? false
            let rhsPlan = rhs.parent.map { starter.contains($0.id) } ?? false
            if lhsPlan != rhsPlan {
                return lhsPlan
            }
            return lhs.createdAt < rhs.createdAt
        }
        // An exact match beats a parent class's slot.
        return ranked.first { linkedClass("exercise", of: $0)?.id == exercise.id } ?? ranked.first
    }
}
