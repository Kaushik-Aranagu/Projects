import Foundation
import TesseraCore

/// SQ8 (Product, 7:53 PM PT Oct 15): a bare "2x10" in a live workout that may be 2 sets of 10 or 2 kg x 10. Tess asks
/// "2 × 10: 2 sets, or 2 kg?" and writes nothing until he answers: a tap on either answer, or saying it.
public struct TessSetCountAsk: Sendable, Equatable {
    public enum Choice: Sendable, Equatable {
        /// N sets of M at the last set's weight.
        case sets
        /// One set of N kg (added load on a bodyweight lift) x M, snapped to the kit.
        case kilograms
        /// One set of N lb x M: the pounds he said, stored as kilograms.
        case pounds
    }

    /// The weight answer that the tapped ghost row stands for: in the unit she asked in.
    public var weightChoice: Choice {
        unit == .pounds ? .pounds : .kilograms
    }

    /// The exercise entry in focus when he said it.
    public let entry: UUID
    public let count: Int
    public let reps: Double
    /// A bodyweight lift: the weight answer is added load ("+5 kg").
    public let bodyweight: Bool
    /// Said with it ("warmup 2x10", "2x10 @8"): kept on whichever answer logs (QA Q9).
    public let kind: SetKind
    public let rpe: Double?
    /// The unit she asks in, taken when the ask is made.
    public let unit: MassUnit

    public init(entry: UUID, count: Int, reps: Double, bodyweight: Bool, kind: SetKind = .working,
                rpe: Double? = nil, unit: MassUnit = TessUnits.weight)
    {
        self.unit = unit
        self.entry = entry
        self.count = count
        self.reps = reps
        self.bodyweight = bodyweight
        self.kind = kind
        self.rpe = rpe
    }

    /// "2 × 10: 2 sets, or 2 kg?" (exact, Research's Q01 to Q19).
    public var line: String {
        "\(count) × \(TessVoice.number(reps)): \(setsAnswer), or \(kilogramsAnswer)?"
    }

    /// The two answers' words, for the ghost rows: "2 sets" and "2 kg" (or "+5 kg").
    public var setsAnswer: String {
        "\(count) sets"
    }

    public var kilogramsAnswer: String {
        "\(bodyweight ? "+" : "")\(count) \(unit.symbol)"
    }

    /// A said answer: "sets", "2 sets", "kg", "2 kg", "+2 kg", "kilos". Nil for anything else (the ask then drops).
    @MainActor public func choice(for phrase: String) -> Choice? {
        let text = TessGrammar.normalize(phrase).replacing(/\bkilos?\b|\bkgs\b/, with: "kg").replacing(
            /\bpounds?\b|\blbs\b/,
            with: "lb"
        )
        let said = text.replacing(/^\+/, with: "").replacing(/(\d)(kg|lb)$/) { "\($0.1) \($0.2)" }
        let count = String(count)
        if said == "sets" || said == "\(count) sets" {
            return .sets
        }
        // The unit he says is the unit he gets, whichever unit she asked in.
        if said == "kg" || said == "\(count) kg" {
            return .kilograms
        }
        if said == "lb" || said == "\(count) lb" {
            return .pounds
        }
        return nil
    }
}

extension TessGrammar.Parse {
    /// A question, not a write: a guarded phrase or the SQ8 ask.
    var isAsk: Bool {
        switch self {
        case .ask, .askSetCount, .askDatedLift: true
        default: false
        }
    }
}

/// A lift said with a past day ("bench 80x8 mon", Product 8:36 PM PT Oct 15): "Bench 80 × 8 for Monday?" with LOG
/// and NOT NOW. LOG puts it in that day's workout; NOT NOW writes nothing. Never a task.
public struct TessDatedLiftAsk: Sendable, Equatable {
    public enum Choice: Sendable, Equatable {
        case log, notNow
    }

    public let exercise: UUID
    /// "Bench", as Tess says it.
    public let name: String
    public let sets: Int
    public let reps: Double
    public let weight: Double
    public let unit: MassUnit
    /// Noon on that day, in the zone it was said in.
    public let day: Date
    /// "Monday".
    public let dayName: String

    /// "Bench 80 × 8", or "Bench 3 × 10 at 80".
    var lift: String {
        let load = TessVoice.number(weight) + (unit == .kilograms ? "" : " \(unit.symbol)")
        return sets > 1 ? "\(name) \(sets) × \(TessVoice.number(reps)) at \(load)"
            : "\(name) \(load) × \(TessVoice.number(reps))"
    }

    /// "Bench 80 × 8 for Monday?"
    public var line: String {
        "\(lift) for \(dayName)?"
    }

    /// "Bench 80 × 8 logged for Monday."
    public var loggedLine: String {
        "\(lift) logged for \(dayName)."
    }

    /// A said answer: "log", "log it", "yes" or "not now", "no". Nil for anything else (the ask then drops).
    @MainActor public func choice(for phrase: String) -> Choice? {
        switch TessGrammar.normalize(phrase) {
        case "log", "log it", "yes", "yep", "yeah": .log
        case "not now", "no", "nope": .notNow
        default: nil
        }
    }
}
