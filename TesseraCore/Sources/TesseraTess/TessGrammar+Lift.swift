import Foundation
import TesseraCore

/// Named lifts: "bench 80x8", "squat 100 5", "bench 3 sets of 8 at 80", "pull ups 3x8".
extension TessGrammar {
    func parseLift(_ text: String, entry: ObjectRecord?) -> Parse? {
        var lift = LiftWords()
        // "8 reps only" is 8 (never groceries). "bench 3 sets of 8 at 80" is 3 sets of 8 at 80; with no weight said
        // it never becomes 3 kg (QA F1 on #282).
        var rest = Substring(text.replacing(/(\d) reps?(?: only)?\b/) { "\($0.1)" }
            .replacing(/(\d) only\b/) { "\($0.1)" })
        if let said = rest.firstMatch(of: /(\d+) sets? of (\d+)/), let sets = Int(said.1), let reps = Double(said.2) {
            (lift.sets, lift.reps, lift.setsSaid) = (sets, reps, true)
            rest.removeSubrange(said.range)
            // "3 sets of 8 at 7": the number after "at" is the weight, and goes through the small-"at N" check (F5).
            // "@ 7" with a space counts too (QA on #289).
            lift.weightAfterAt = rest.firstMatch(of: /(?:\bat\b|@)\s*\d/) != nil
            rest = Substring(rest.replacing(/(?:\bat\b|@)/, with: " "))
        }
        if let scheme = rest.firstMatch(of: /(\d+(?:\.\d+)?)(?: ?x ?| by )(\d+)(?: (?:at|@) (\d+(?:\.\d+)?))?/) {
            lift.readScheme(Double(scheme.1) ?? 0, Double(scheme.2) ?? 0, at: scheme.3.flatMap { Double($0) },
                            weightElsewhere: Self.hasOtherNumber(rest, excluding: scheme.range))
            rest.removeSubrange(scheme.range)
        }
        rest.split(separator: " ").forEach { lift.read($0) }
        guard !lift.words.isEmpty, let exercise = resolveExercise(lift.words.joined(separator: " ")) else { return nil }
        // "pull ups 3x8": on a bodyweight lift a bare N x M is N sets of M, never N kg (A181).
        let bodyweight = store.equipmentKind(of: exercise) == .bodyweight
        // In a live workout with that lift in focus, it asks like a bare "10x8" does (Product 9:40 PM PT).
        if bodyweight, lift.weightFromScheme, let entry, store.linkedClass("exercise", of: entry)?.id == exercise.id,
           let first = lift.weight, first.rounded() == first, first >= 1, let reps = lift.reps
        {
            return .askSetCount(TessSetCountAsk(entry: entry.id, count: Int(first), reps: reps, bodyweight: true))
        }
        if bodyweight, lift.weightFromScheme, let first = lift.weight, first <= 10, first.rounded() == first {
            (lift.sets, lift.weight) = (Int(first), 0)
        }
        // "pull-ups 8" with Pull-Up in focus: one number on a bodyweight lift is reps (Research on #282, as K898).
        if bodyweight, !lift.weightFromScheme, !lift.setsSaid, lift.reps == nil, let only = lift.weight, let entry,
           store.linkedClass("exercise", of: entry)?.id == exercise.id
        {
            (lift.reps, lift.weight) = (only, 0)
        }
        if bodyweight, lift.weight == nil, lift.reps != nil {
            // "dips 3 sets of 10", or "pull-ups 8" in focus: at bodyweight (Research on #282).
            lift.weight = 0
        }
        // "leg press 5x5", "curls 3x8", "bench 3x8": on any loaded kit a bare N x M (no "at", no unit) with N up to 10
        // never writes N kg. With a last weight in focus, the set-count ask decides as SQ8 does; with none, ask
        // (QA F1, F2, F3; Research on #282).
        let dial = store.weightDial(for: exercise)
        let inFocus = entry.flatMap { store.linkedClass("exercise", of: $0)?.id == exercise.id ? $0 : nil }
        let last = inFocus.flatMap { lastSet(in: $0) }
        if !bodyweight, lift.weightFromScheme, !lift.unitSaid, let first = lift.weight, first <= 10,
           first.rounded() == first, let reps = lift.reps
        {
            let name = TessCaptureEngine.shortName(exercise.name)
            guard let inFocus, let last else {
                return .ask("\(name) \(Int(first)) × \(TessVoice.number(reps)) at what weight?")
            }
            // In focus it always asks and never reaches the weight write (QA F4): under half the last weight it may
            // be sets ("bench 8x8" at 80); otherwise, at what weight.
            if first < last.weight / 2 {
                return .askSetCount(TessSetCountAsk(entry: inFocus.id, count: Int(first), reps: reps,
                                                    bodyweight: false))
            }
            return .ask("\(name) \(Int(first)) × \(TessVoice.number(reps)) at what weight?")
        }
        if lift.setsSaid, lift.weight == nil, let reps = lift.reps {
            let name = TessCaptureEngine.shortName(exercise.name)
            return .ask("\(name) \(lift.sets) × \(TessVoice.number(reps)) at what weight?")
        }
        // "bench 3x5 at 7", "curls 3x10 at 5": a small "at N" the kit can't load may be a time: ask, never log
        // either (QA on #274, F3). Pounds convert first.
        if lift.weightAfterAt, let at = lift.weight, at <= 12, (1 ... 20).contains(lift.sets),
           !dial.detents.contains(where: { abs($0 - lift.unit.toKilograms(at)) < 0.01 })
        {
            let name = TessCaptureEngine.shortName(exercise.name)
            return .ask("\(name) \(lift.sets) × \(TessVoice.number(lift.reps ?? 0)) at what weight?")
        }
        let (sets, unit, reps, weight) = (lift.sets, lift.unit, lift.reps, lift.weight)
        guard let reps, let weight, (1 ... 20).contains(sets), reps >= 1, reps <= 100, reps.rounded() == reps,
              weight > 0 || bodyweight && weight == 0, unit.toKilograms(weight) <= 1000 else { return nil }
        if let entry, sets == 1, store.linkedClass("exercise", of: entry)?.id == exercise.id {
            return .set(entry: entry, weight: unit.toKilograms(weight), reps: reps, kind: .working, phrase: .full,
                        unit: unit)
        }
        return .lift(exercise: exercise, sets: sets, reps: reps, weight: weight, unit: unit)
    }

    /// The numbers and words of a lift phrase.
    struct LiftWords {
        var sets = 1
        var unit = TessUnits.weight
        var reps: Double?
        var weight: Double?
        var words: [Substring] = []
        /// "kg" or "lb" was said.
        var unitSaid = false
        /// "3 sets of 8" said the sets in words.
        var setsSaid = false
        /// The weight is the first number of a bare "N x M" (no "at", no other number).
        var weightFromScheme = false
        /// The weight came after "at" ("3x5 at 80"), where a small number may be a clock time.
        var weightAfterAt = false

        mutating func readScheme(_ first: Double, _ second: Double, at atWeight: Double?, weightElsewhere: Bool) {
            if let atWeight {
                // "3x8 at 80": sets x reps at weight.
                (sets, reps, weight) = (Int(first), second, atWeight)
                weightAfterAt = true
            } else if first <= 10, second <= 30, weightElsewhere {
                // "3x10 bench 80": sets x reps, weight elsewhere.
                (sets, reps) = (Int(first), second)
            } else {
                // "80x8": weight x reps.
                (weight, reps) = (first, second)
                weightFromScheme = true
            }
        }

        mutating func read(_ token: Substring) {
            if let parsed = MassUnit(parsing: String(token)) {
                unit = parsed
                unitSaid = true
            } else if let value = Double(token) {
                if weight == nil {
                    weight = value
                } else if reps == nil {
                    reps = value
                }
            } else if !["for", "at", "@", "of", "x", "sets", "reps", "log", "logged", "add", "did"].contains(token) {
                words.append(token)
            }
        }
    }

    static func hasOtherNumber(_ text: Substring, excluding range: Range<Substring.Index>) -> Bool {
        var copy = text
        copy.removeSubrange(range)
        return copy.split(separator: " ").contains { Double($0) != nil }
    }
}
