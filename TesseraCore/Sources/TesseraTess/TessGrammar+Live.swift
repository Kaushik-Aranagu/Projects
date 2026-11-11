import Foundation
import TesseraCore

/// The live-workout set rules (S701 to S706, docs/research/fixtures/set-logger.md): while a workout is in front, a
/// number that could be a weight, a signed change or "at N" after a set never becomes a wrong set or a task.
extension TessGrammar {
    /// What the live guard decided: write this, or nothing (the model asks).
    struct Live {
        let parse: Parse?
    }

    /// S701 to S706 (docs/research/fixtures/set-logger.md): the live-workout set rules. Nil when the phrase isn't a
    /// set phrase (the other rules then read it); `Live(parse: nil)` when it is one the grammar mustn't guess.
    func parseLive(_ phrase: String, text _: String, entry: ObjectRecord) -> Live? {
        // Dictation ends sentences: "Same at 9." (QA Q1).
        let raw = phrase.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            .replacing(/[.!]+$/, with: "").replacing(/,/, with: " ")
            // "2 kilos x 10" says its unit (K888).
            .replacing(/(\d) ?(?:kilos?|kgs)\b/) { "\($0.1) kg" }
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
        let model = Live(parse: nil)
        let exercise = store.linkedClass("exercise", of: entry)
        let kindOfKit = exercise.flatMap { store.equipmentKind(of: $0) }
        let dial = exercise.map { store.weightDial(for: $0) }
        if let words = parseLiveWords(raw, entry: entry, kindOfKit: kindOfKit, dial: dial) {
            return words
        }
        // S705: kind words, read before the RPE so "warmup 60x5 at 8" keeps both (QA Q5).
        var core = Self.normalize(raw)
        var kind: SetKind?
        if let match = core.firstMatch(of: /^(?:warm ?-?up) (.+)$/) {
            core = String(match.1)
            kind = .warmup
        }
        // S704: "at N" / "@N" after a set is RPE (read before "to failure", so "to failure at 9" works).
        var rpe: Double?
        if let match = core.firstMatch(of: /^(.+?) ?(?: at |@ ?)(\d+(?:\.\d+)?)$/),
           Self.isSetCore(String(match.1))
        {
            // A set with an "at" that isn't an RPE ("same at 11", "8 at 4"): never a task here (QA Q3).
            guard let value = Double(match.2), (5 ... 10).contains(value), (value * 2).rounded() == value * 2 else {
                return model
            }
            rpe = value
            core = String(match.1)
        }
        if let match = core.firstMatch(of: /^(.+?) to failure$/) {
            core = String(match.1)
            kind = .failure
        }
        if kind == nil, let match = core.firstMatch(of: /^(.+) (?:warm ?-?up)$/) {
            core = String(match.1)
            kind = .warmup
        }
        // S701: a bare number is reps only if it can't be a weight. "N reps" says so: only the range applies.
        if let number = Self.number(core) {
            let saidReps = core.hasSuffix(" reps")
            let shown = TessVoice.number(number)
            guard let last = lastSet(in: entry) else {
                return Live(parse: .ask(saidReps ? "\(shown) reps: at what weight?" :
                        "\(shown): \(TessUnits.symbol), or reps?"))
            }
            guard number.rounded() == number, (1 ... 50).contains(number),
                  saidReps ||
                  // On a bodyweight move the last "weight" is 0 added: nothing to confuse reps with (QA Q4).
                  kindOfKit == .bodyweight && last.weight == 0
                  || !nearLastWeight(number, last: last.weight, dial: dial)
            else {
                let reps = number.rounded() == number && (1 ... 50).contains(number)
                return Live(parse: .ask(reps ? "\(shown): \(shown) \(TessUnits.symbol), or \(shown) reps?" :
                        "\(shown) for how many reps?"))
            }
        }
        // SQ8 (Product, 7:53 PM PT): a bare "N x M" may be N sets of M. "3 sets of 10" says so.
        // Whatever RPE or kind came with it: "2x10 @8" and "warmup 2x10" ask too, and keep both (QA Q9).
        if let sets = setCount(core, entry: entry, bodyweight: kindOfKit == .bodyweight, kind: kind ?? .working,
                               rpe: rpe)
        {
            return Live(parse: sets)
        }
        guard let parsed = parseRelativeSet(core, entry: entry) else {
            return Self.looksLikeSet(core) || rpe != nil || kind != nil ? model : nil
        }
        guard case let .set(entry, weight, reps, setKind, how, unit, _, _) = parsed else { return Live(parse: parsed) }
        let withExtras = Parse.set(entry: entry, weight: weight, reps: reps, kind: kind ?? setKind, phrase: how,
                                   unit: unit, rpe: rpe)
        return Live(parse: snap(withExtras, entry: entry))
    }

    /// S702, S705, S706: load words ("bw 12", "+10 for 6", "bar for 10"), signed changes, "skip", a bare yes or
    /// no, a weight with no reps, "fail at 6". Nil when the phrase is none of these.
    private func parseLiveWords(_ raw: String, entry: ObjectRecord, kindOfKit: EquipmentKind?,
                                dial: WeightDial?) -> Live?
    {
        let model = Live(parse: nil)
        // S706: load words.
        if let match = raw.firstMatch(of: /^(?:bodyweight|bw) (\d+)$/), let reps = Double(match.1) {
            guard kindOfKit == .bodyweight else { return model }
            return Live(parse: .set(entry: entry, weight: 0, reps: reps, kind: .working, phrase: .full))
        }
        if let match = raw.firstMatch(of: /^\+(\d+(?:\.\d+)?) ?(?:for|x) ?(\d+)$/), let added = Double(match.1),
           let reps = Double(match.2)
        {
            guard kindOfKit == .bodyweight else { return model }
            return Live(parse: snap(.set(entry: entry, weight: added, reps: reps, kind: .working, phrase: .full),
                                    entry: entry))
        }
        if let match = raw.firstMatch(of: /^bar ?(?:for|x) ?(\d+)$/), let reps = Double(match.1) {
            guard let bar = dial?.bar else { return model }
            return Live(parse: .set(entry: entry, weight: bar, reps: reps, kind: .working, phrase: .full))
        }
        // S702: a signed change alone, and "skip" (the screen's), write nothing.
        // No yes/no questions: nothing remembers them, so "yes" would be read fresh (QA Q10). Say what to say.
        if let match = raw.firstMatch(of: /^([+-]|up |down )(\d+(?:\.\d+)?)(?: ?kg)?$/), let change = Double(match.2) {
            let sign: Double = match.1 == "-" || match.1 == "down " ? -1 : 1
            guard let last = lastSet(in: entry) else { return Live(parse: .ask("Say the set, like 80 × 8.")) }
            let next = TessVoice.number(last.weight + sign * change)
            return Live(parse: .ask("Say the set, like \(next) × \(TessVoice.number(last.reps))."))
        }
        if raw == "skip" {
            return Live(parse: .ask("Skip is on the workout screen."))
        }
        // A bare "yes" or "ok" answers nothing here; it never becomes a task.
        if ["yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure"].contains(raw) {
            return Live(parse: .ask(lastSet(in: entry).map {
                "Say the set, like \(TessVoice.number($0.weight)) × \(TessVoice.number($0.reps))."
            } ?? "Say the set, like 80 × 8."))
        }
        // "81 kg": a weight with no reps.
        if let match = raw.firstMatch(of: /^(\d+(?:\.\d+)?) ?(kg|lb)$/), let weight = Double(match.1) {
            return Live(parse: .ask("\(TessVoice.number(weight)) \(match.2) for how many reps?"))
        }
        // S705: "fail at 6" is unclear (rep 6, or 6 reps).
        if let match = raw.firstMatch(of: /^fail(?:ed)?(?: at)? (\d+)$/) {
            return Live(parse: .ask("Fail at \(match.1): rep \(match.1), or \(match.1) reps?"))
        }
        return nil
    }

    /// SQ8: "2x10" (written with x, no unit said) asks when N is 6 or less and under half the last set's weight, and
    /// on a bodyweight lift whatever N is, first set included. "3 sets of 10" logs every set at the last weight.
    func setCount(_ core: String, entry: ObjectRecord, bodyweight: Bool, kind: SetKind = .working,
                  rpe: Double? = nil) -> Parse?
    {
        let last = lastSet(in: entry)
        if let match = core.firstMatch(of: /^(\d+) sets? of (\d+)(?: reps)?$/), let count = Int(match.1),
           let reps = Double(match.2), (1 ... 10).contains(count), (1 ... 100).contains(reps)
        {
            guard let weight = last?.weight ?? (bodyweight ? 0 : nil)
            else { return .ask("\(count) × \(match.2) at what weight?") }
            return .sets(Array(repeating: .set(entry: entry, weight: weight, reps: reps, kind: kind, phrase: .full,
                                               rpe: rpe), count: count))
        }
        guard let match = core.firstMatch(of: /^(\d+) ?(?:x|by) ?(\d+)$/), let count = Int(match.1),
              let reps = Double(match.2), count >= 1, (1 ... 100).contains(reps) else { return nil }
        let asks = bodyweight || count <= 6 && last.map { Double(count) < $0.weight / 2 } == true
        guard asks, count <= 20 else { return nil }
        return .askSetCount(TessSetCountAsk(entry: entry.id, count: count, reps: reps, bodyweight: bodyweight,
                                            kind: kind, rpe: rpe))
    }

    /// A169, A170: a set and an errand in one breath ("8 and call mom at 7", "same, call mom thu"). The set part goes
    /// through the live guard; the rest is read as if no workout were in front. Nil unless both parts write.
    func parseSetAndErrand(_ phrase: String, context: TessContext, entry: ObjectRecord) -> Parse? {
        parseLiftAndErrand(phrase, context: context, entry: entry)
    }

    /// A lift (or a set in focus) and an errand in one phrase, either order: "bench 80x8, call mom thu" (A162),
    /// "buy milk and bench 80x8" (A059), "bench 3x5 at 80 then call mom at 7" (A167), and with no joining word when
    /// the errand has a day: "bench 80x8 dentist fri" (A171). Nil unless both parts write.
    func parseLiftAndErrand(_ phrase: String, context: TessContext, entry: ObjectRecord?) -> Parse? {
        // A lift has a number; most phrases don't, and each try below reads its parts again.
        guard entry != nil || phrase.contains(where: \.isNumber), phrase.split(separator: " ").count >= 2 else {
            return nil
        }
        for separator in [/,?\sand\sthen\s/, /,?\sthen\s/, /\sand\s/, /,\s?/] {
            guard let split = phrase.firstMatch(of: separator.ignoresCase()) else { continue }
            let left = String(phrase[..<split.range.lowerBound])
            let right = String(phrase[split.range.upperBound...])
            if let lift = liftPart(left, entry: entry), let errand = errandPart(right, context: context) {
                return .sets([lift, errand])
            }
            if let lift = liftPart(right, entry: entry), let errand = errandPart(left, context: context) {
                return .sets([errand, lift])
            }
        }
        // No joining word: a lift, then words that make a dated task ("dentist fri").
        let words = phrase.split(separator: " ").map(String.init)
        for cut in stride(from: words.count - 1, through: 2, by: -1)
            where words[cut - 1].contains(where: \.isNumber) && !words[cut].contains(where: \.isNumber)
        {
            let rest = words[cut...].joined(separator: " ")
            // A day word alone is the dated-lift ask's ("bench 80x8 mon"), not an errand; set words never start one
            // ("80 x 6 to failure at 9").
            let setWords: Set = [
                "to",
                "at",
                "for",
                "x",
                "by",
                "reps",
                "rep",
                "kg",
                "kgs",
                "lb",
                "lbs",
                "warmup",
                "warm",
                "failure",
                "fail",
                "sets",
                "set",
                "drop",
                "same",
                "rpe",
                "@",
            ]
            guard !setWords.contains(words[cut].lowercased()),
                  TessDates.weekdays[rest.lowercased()] == nil, !["today", "yesterday"].contains(rest.lowercased()),
                  let lift = liftPart(words[..<cut].joined(separator: " "), entry: entry),
                  let errand = errandPart(rest, context: context), case .task(_, .some) = errand
            else { continue }
            return .sets([lift, errand])
        }
        return nil
    }

    /// A set in focus or a named lift; never an ask.
    private func liftPart(_ text: String, entry: ObjectRecord?) -> Parse? {
        let normalized = Self.normalize(text)
        if let entry, let set = parseLive(text, text: normalized, entry: entry)?.parse {
            return set.isAsk ? nil : set
        }
        guard let lift = parseLift(normalized, entry: entry) else { return nil }
        switch lift {
        case .lift, .set: return lift
        default: return nil
        }
    }

    /// Anything Tess writes that isn't a set, read as if no workout were in front; a task needs a day or time.
    private func errandPart(_ text: String, context: TessContext) -> Parse? {
        guard Self.normalize(text).firstMatch(of: /^[\d.,x ]+$|^(?:same|again|repeat|one more)\b/) == nil else {
            return nil
        }
        var errandContext = context
        errandContext.entry = nil
        errandContext.process = nil
        guard let errand = parse(text, context: errandContext), !errand.isAsk else { return nil }
        switch errand {
        case .lift, .set, .sets, .ignoredInWorkout, .task(_, nil): return nil
        default: return errand
        }
    }

    /// What can carry an RPE: a set shape, "same" and its kin, or bare reps ("call mom at 7" can't).
    static func isSetCore(_ text: String) -> Bool {
        looksLikeSet(text) || ["same", "same again", "again", "repeat", "one more"].contains(text)
            || number(text) != nil || text.hasSuffix(" to failure") && looksLikeSet(text)
    }

    /// "80 x 8", "100 for 5", "80x8": a set shape.
    static func looksLikeSet(_ text: String) -> Bool {
        text.firstMatch(of: /^\d+(?:\.\d+)?(?: ?kg| ?lb)?(?: ?x ?| for | by )\d+/) != nil
    }

    /// Within two loads of the last weight: "81" after 80 is a weight, not 81 reps.
    private func nearLastWeight(_ number: Double, last: Double, dial: WeightDial?) -> Bool {
        guard let dial, !dial.detents.isEmpty else { return abs(number - last) <= 5 }
        guard let lightest = dial.detents.first, let heaviest = dial.detents.last,
              number >= lightest - 0.001, number <= heaviest + 0.001 else { return false }
        let index = { (weight: Double) in dial.detents.firstIndex { $0 >= dial.snap(weight) - 0.001 } ?? 0 }
        return abs(index(number) - index(last)) <= 2
    }

    /// S703: a kilogram weight snaps to the nearest load the exercise's kit can make (a tie to the lighter); outside
    /// the kit's range, nil (the model asks). Pounds stay as said; "same" and bare reps reuse a logged weight.
    func snap(_ parse: Parse, entry: ObjectRecord) -> Parse? {
        guard case let .set(entry, weight, reps, kind, how, unit, rpe, _) = parse, unit == .kilograms,
              how == .full || how == .drop,
              let exercise = store.linkedClass("exercise", of: entry) else { return parse }
        let dial = store.weightDial(for: exercise)
        // Below the lightest load ("0 x 8" on a barbell, QA Q6) or above the heaviest: the model asks.
        guard let lightest = dial.detents.first, let heaviest = dial.detents.last else { return parse }
        guard weight >= lightest - 0.001, weight <= heaviest + 0.001 else { return nil }
        let snapped = dial.snap(weight)
        return .set(entry: entry, weight: snapped, reps: reps, kind: kind, phrase: how, unit: unit, rpe: rpe,
                    said: abs(snapped - weight) > 0.001 ? weight : nil)
    }
}
