import Foundation
import TesseraCore

/// The deterministic first step: gym shorthand and dated tasks, parsed instantly and offline.
/// Returns typed actions; it never writes. Anything it doesn't recognise goes to the model.
@MainActor
public struct TessGrammar {
    public enum Parse {
        /// Log sets of a named exercise (into the entry in focus if it's the same exercise, else today's workout).
        case lift(exercise: ClassDef, sets: Int, reps: Double, weight: Double, unit: MassUnit)
        /// A set in the entry in focus: "same", "8", "80 x 8", "drop to 70 for 10".
        /// `weight` is in kilograms; `unit` is how it was said.
        /// `rpe` from "at 8" / "@8"; `said` is the weight as said when it was snapped to a loadable one ("81" → 80).
        case set(entry: ObjectRecord, weight: Double, reps: Double, kind: SetKind, phrase: SetPhrase,
                 unit: MassUnit = .kilograms, rpe: Double? = nil, said: Double? = nil)
        case task(title: String, startsAt: Date?)
        /// "note: bar felt heavy" (or "note bar felt heavy").
        case note(body: String, on: Date? = nil)
        /// A line with a secret word: a private note on the phone, never the model, Claude, memory or the miss log.
        /// `remembered` when he said "remember …" (Tess adds "I don't keep passwords.").
        case privateNote(body: String, remembered: Bool, on: Date? = nil)
        /// Kitchen: "buy milk", "milk, eggs, 2 lemons".
        case groceries(lines: [String])
        /// "cook carbonara" (now, `on` nil) or "cook carbonara thu" (a dated task linking the recipe).
        case cook(recipe: ObjectRecord, on: Date?)
        /// "timer pasta 10", "10 min timer".
        case timer(label: String, minutes: Double)
        /// "list: packing".
        case list(name: String)
        /// "copy packing for lisbon": a new list named Lisbon with Packing's items, all open.
        case copyList(source: String, name: String)
        /// "shop": start shopping the groceries list.
        case shop
        /// Pasted text with an Ingredients heading.
        case recipe(text: String)
        /// Several sets said at once, comma separated: "bench 80x8, 82.5x6" or "80x8, 80x8, 80x6".
        case sets([Parse])
        /// "move dentist to friday": an open task, to a new time.
        case reschedule(task: ObjectRecord, to: Date)
        /// "rest day Sunday": the rest weekday, 1 (Sunday) to 7. Never a task (RG3).
        case setRestWeekday(weekday: Int)
        /// "next" / "done" while cooking.
        case cookStep(cook: ObjectRecord, next: Bool)
        /// Bare input the running workout owns but Tess doesn't act on (a rest timer is the workout screen's).
        case ignoredInWorkout
        /// SQ8: a bare "2x10" that may be 2 sets: nothing is written until he answers (`TessSetCountAsk`).
        case askSetCount(TessSetCountAsk)
        /// "bench 80x8 mon": a lift on a past day, asked before it's logged (`TessDatedLiftAsk`).
        case askDatedLift(TessDatedLiftAsk)
        /// A guarded set phrase the grammar mustn't guess ("81", "fail at 6"): Tess's question, if no model reads it.
        case ask(String)
        /// "spent 12.40 on coffee", "paid rent 1200": a spend, dated `on` (a past day) or now.
        case spend(MoneyAmount, note: String, on: Date?)
        /// Money pools: "got 3200 paycheck", "got 500 rupees". Read only with the switch on, on the phone.
        case inflow(MoneyAmount, source: String, pool: UUID?, note: String)
        /// Money pools: "moved 1000 to India, got 82650, fee 5".
        case poolMove(from: UUID, to: UUID, sent: MoneyAmount, arrived: MoneyAmount, fee: MoneyAmount?)
        /// The groceries total after a shop.
        case shopTotal(MoneyAmount)
        /// "skip" while a groceries total is awaited.
        case skipShopTotal
    }

    public enum SetPhrase: Equatable, Sendable {
        case same, repsOnly, full, drop
    }

    public let store: ObjectStore
    public let classes: [SystemCatalog.Key: ClassDef]

    public init(store: ObjectStore, classes: [SystemCatalog.Key: ClassDef]) {
        self.store = store
        self.classes = classes
    }

    /// The rules ahead of every process and task rule: a note, a pasted recipe, a question (no write at all:
    /// `.some(nil)`) and a rest weekday. Nil when none of them applies.
    func parseBeforeTasks(_ phrase: String, text: String, context: TessContext) -> Parse?? {
        if let note = parseNote(phrase) {
            return .some(note)
        }
        if let recipe = parsePastedRecipe(phrase) {
            return .some(recipe)
        }
        // "how much this month" is answered here, from the log, on the phone: it never reaches a model.
        if let month = parseMonthQuestion(text, context: context) {
            return .some(month)
        }
        // Rule 2: a question never writes; it goes to the model's read-only answer. The raw phrase, so a typed "?"
        // still counts (normalize trims it).
        if Self.isQuestion(text) || phrase.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("?") {
            return .some(nil)
        }
        // A rest weekday is a setting, not a task called "Rest day" (RG3): ahead of every task rule.
        if let rest = parseSetRestWeekday(text: text) {
            return .some(rest)
        }
        return nil
    }

    public func parse(_ phrase: String, context: TessContext) -> Parse? {
        // A wake word said with it ("Okay Tess, 80 for 8.") is not part of the phrase.
        // Only "Tess," / "Tess:" or "okay/hey Tess": "Tess's ideas list" and "Tess dinner fri" keep their Tess (QA W1).
        // "Tess 80 for 8" (dictation dropped the comma): a bare "Tess" before a number goes too (QA L3 on #287).
        let phrase = phrase.replacing(/^(?i)\s*(?:(?:ok(?:ay)?|hey)\s+tess(?:[,.:]\s*|\s+)|tess[,:]\s*|tess\s+(?=\d))/,
                                      with: "")
        let text = Self.normalize(phrase)
        guard !text.isEmpty else { return nil }
        // Secret words first, before notes, recipes and questions (Research 134 R1): nothing with one leaves the phone.
        if let secret = parseSecret(phrase, text: text, context: context) ?? parseRememberedMoney(phrase) {
            return secret
        }
        switch parseBeforeTasks(phrase, text: text, context: context) {
        case let .some(.some(early)): return early
        case .some(.none): return nil
        case .none: break
        }
        // Rule 3: the process in front decides bare input.
        if let bare = parseForProcess(text, context: context) {
            return bare
        }
        let entry = context.entry.flatMap { store.object(id: $0) }
        if let entry, let live = parseInFocus(phrase, text: text, context: context, entry: entry) {
            return live.parse
        }
        // Money pools (switch On): after the secret words and the process in front, before generic money.
        if let pooled = parsePools(phrase, context: context) {
            return pooled
        }
        // Money (slice 4): after the relative sets above, before comma sets and lifts, so "80 x5" and
        // "bench 82.50 x 8" stay training and "6.50 cab" is a spend.
        if let money = parseMoney(phrase, text: text, context: context, entry: entry) {
            return money
        }
        if phrase.contains(","), let sets = parseCommaSets(phrase, entry: entry) {
            guard let entry else { return sets }
            // Every part snaps to a loadable weight; one out of the kit's range sends the lot to the model.
            guard case let .sets(parts) = sets else { return sets }
            let snapped = parts.map { snap($0, entry: entry) }
            return snapped.contains { $0 == nil } ? nil : .sets(snapped.compactMap(\.self))
        }
        if let both = parseLiftAndErrand(phrase, context: context, entry: entry) {
            return both
        }
        if let dated = parseDatedLift(text, context: context) {
            return dated
        }
        if let lift = parseLift(text, entry: entry) {
            return lift
        }
        // "bench 80x8 at 7": a set now, or a plan for 7 pm? Never a task called "Bench 80x8" (A184): the model reads
        // it.
        if let timed = text.firstMatch(of: /^(.+) at \d{1,2}(?::\d{2})? ?(?:am|pm)?$/),
           parseLift(String(timed.1), entry: nil) != nil
        {
            return nil
        }
        // Money said ("spent $23.47 at trader joe's sat") is never a task or a list until Money exists: the miss line.
        if Self.isMoney(phrase), phrase.lowercased().firstMatch(of: /^(?:spent|spend|paid|pay|cost)\b/) != nil
            || phrase.firstMatch(of: /^[$£€₹]\s?\d/) != nil
        {
            // Something to do with a day wins over its amount (12:13 PM PT): "pay rent 1800 friday" is a task, said
            // back without its words. Money said in the past with a day is a private note on that day ("paid rent
            // 2100 fri", Product 1:15 PM PT); with no day it stays the miss line.
            return Self.isSomethingToDo(phrase) ? parseTask(phrase, context: context) : parsePrivateLog(
                phrase,
                context: context
            )
        }
        // A bare set with nothing in focus ("80 for 8" while shopping) is never an item or a task: the model reads it.
        if text.firstMatch(of: /^\d+(?:\.\d+)?(?: ?kg| ?lb)?(?: ?x ?| for | by )\d+(?: reps)?$/) != nil {
            return nil
        }
        // A body fact is private before it's a dated log (Product P14/P15a, 12:30 PM PT): "slept 6 hours last night",
        // "took advil 400mg yesterday", "take advil 400mg at 9 pm". A past-tense log with a day ("watered plants
        // sat") is a note dated that day, never a future task.
        // A dated past log of money is private too ("bought a tv for 800 sat", QA F2 on #534).
        let walledLog = Self.isSomethingToDo(phrase) ? nil
            : parseBodyFact(phrase, context: context) ??
            (PrivateFacts.isMoneyShaped(phrase) ? parsePrivateLog(phrase, context: context) : nil)
        if let log = walledLog ?? parsePastLog(phrase, context: context) {
            return log
        }
        // After lifts, so "bench 80x8, 82.5x6" stays a lift.
        if let kitchen = parseKitchen(phrase, context: context) {
            return kitchen
        }
        if let reschedule = parseReschedule(phrase, context: context) {
            return reschedule
        }
        // A reschedule whose task doesn't resolve goes to the model rather than becoming "Move the piano to".
        if phrase.firstMatch(of: Self.rescheduleShape) != nil {
            return nil
        }
        return parseTaskOrBodyFact(phrase, context: context)
    }

    /// A body fact with a day ("headache since tuesday") is a private note, not a task (QA R1 on #471), unless it
    /// opens with something to do: "call the doctor about my knee thu" and "refill lexapro 10mg fri" stay tasks.
    func parseTaskOrBodyFact(_ phrase: String, context: TessContext) -> Parse? {
        Self.isSomethingToDo(phrase) ? parseTask(phrase, context: context) ?? parseBodyFact(phrase, context: context)
            : parseBodyFact(phrase, context: context) ?? parseTask(phrase, context: context)
    }

    /// The live workout's rules for a phrase with an exercise in focus; nil when they don't apply (the other rules
    /// then read it), `Live(parse: nil)` when the phrase is a set the grammar mustn't guess.
    private func parseInFocus(_ phrase: String, text: String, context: TessContext, entry: ObjectRecord,
                              spokenRead: Bool = false) -> Live?
    {
        // Number words typed or said ("a hundred, five", K885) are read as digits by the live rules first; one pass.
        let words = phrase.lowercased().split { !$0.isLetter }.map(String.init)
        if !spokenRead, words.contains(where: { TessSpoken.isNumberWord($0) || $0 == "plus" || $0 == "minus" }) {
            let spoken = TessSpoken.numbers(phrase)
            if spoken != phrase, let live = parseInFocus(spoken, text: Self.normalize(spoken), context: context,
                                                         entry: entry, spokenRead: true)
            {
                return live
            }
        }
        // In a live workout, set phrases are guarded first (S701 to S706): a number that could be a weight, a signed
        // change, "at N" after a set: never a wrong set and never a task.
        if !phrase.contains(","), let both = parseSetAndErrand(phrase, context: context, entry: entry) {
            return Live(parse: both)
        }
        if !phrase.contains(","), let live = parseLive(phrase, text: text, entry: entry) {
            return live
        }
        // A comma in a live workout (QA Q2): every part goes through the guard ("8, 81" asks: 81 may be a weight); a
        // comma that only split one set ("100 for 5, at 8") is read whole.
        if phrase.contains(",") {
            let parts = phrase.split(separator: ",").map(String.init)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            let lives = parts.map { parseLive($0, text: Self.normalize($0), entry: entry) }
            if parts.count >= 2, lives.allSatisfy({ $0 != nil }) {
                var sets = lives.compactMap { $0?.parse }
                guard sets.count == parts.count else { return Live(parse: nil) }
                // After a set, a bare number is its next weight: Tess asks for the reps (K874).
                if sets.count > 1, sets.last?.isAsk == true,
                   let number = Self.number(Self.normalize(parts[parts.count - 1])),
                   !parts[parts.count - 1].contains("reps")
                {
                    sets[sets.count - 1] = .ask("\(TessVoice.number(number)) for how many reps?")
                }
                // A guarded part at the end asks after the rest log ("80 for 8, 81": K874); anywhere else, a guess.
                return Live(parse: sets.dropLast().contains(where: \.isAsk) ? nil : .sets(sets))
            }
            if let both = parseSetAndErrand(phrase, context: context, entry: entry) {
                return Live(parse: both)
            }
            if let live = parseLive(parts.joined(separator: " "), text: text, entry: entry) {
                return live
            }
        }
        return nil
    }

    /// Lead words that ask rather than tell ("what did i weigh monday", "anything due tmrw").
    static func isQuestion(_ text: String) -> Bool {
        let first = text.split(separator: " ").first.map(String.init) ?? ""
        // "is" and "are" phrases ask (Product, 2026-10-15). "do laundry thu" stays a capture: "do" only asks with a
        // subject after it. "show" only asks as "show me / my / all".
        let leads: Set = ["what", "whats", "what's", "when", "how", "did", "anything", "everything", "is", "are",
                          "who", "who's", "whos", "where", "where's", "when's", "whens", "which"]
        return leads.contains(first) || text.hasPrefix("am i ") || text.hasSuffix("?")
            || text.firstMatch(of: /^(do|does|can|could|should|have) (i|we|you)\b/) != nil
            || text.firstMatch(of: /^show (me|my|all)\b/) != nil
    }

    /// Rule 3: while a workout, cook or shop is running (Now), bare input belongs to it.
    private func parseForProcess(_ text: String, context: TessContext) -> Parse? {
        guard let subject = context.process.flatMap({ store.object(id: $0) }), let classDef = subject.classDef else {
            return nil
        }
        func isA(_ key: SystemCatalog.Key) -> Bool {
            classes[key].map { store.isSubclass(classDef, ofClassID: $0.id) } ?? false
        }
        if isA(.cook) {
            if ["next", "next step", "done", "finished", "done with that"].contains(text) {
                return .cookStep(cook: subject, next: text.hasPrefix("next"))
            }
        } else if isA(.workout) {
            // "timer 90" / "rest 90" in a workout is the rest between sets, not a kitchen timer: leave it to the
            // workout screen (no kitchen reading).
            // Only "rest 90", "timer 90", "rest timer 90" (not "restock protein thu" or "restaurant fri").
            if text.firstMatch(of: /^(rest|timer)(\s+timer)?\s+\d/) != nil {
                return .ignoredInWorkout
            }
        } else if subject.id == Groceries.listID {
            // At the shop, a bare thing is for the list, but not an errand ("call mom"), a word for the shop itself
            // ("done", "thanks"), or a verb the kitchen rules read ("buy milk" goes there, without the verb).
            let first = text.split(separator: " ").first.map(String.init) ?? ""
            if text
                .firstMatch(
                    of: /^(done|finished|finished shopping|done shopping|shop|shopping|thanks|thank you|ok|cancel|stop)$/
                )
                != nil || Self.startsWithErrand(text) || ["buy", "get", "add", "pick"].contains(first)
                // "milk for groceries" is the kitchen rule's; a set shape ("80 for 8") is never an item (Research
                // on #282).
                || text
                .firstMatch(
                    of: /\s(?:for|on|to|onto)\s+(?:the\s+|my\s+)?(?:groceries|grocery list|shopping list|shopping)$/
                ) !=
                nil
                || text.firstMatch(of: /\d\s*(?:x|×|for|by)\s*\d/) != nil || Self.isMoney(text)
            {
                return nil
            }
            if !Self.hasDate(text, context: context),
               !text.contains(where: \.isNumber) || IngredientLine.parse(text)?.amount != nil,
               text.split(separator: " ").count <= 4
            {
                return .groceries(lines: [text])
            }
        }
        return nil
    }

    static let rescheduleShape = /^(?i)(?:move|push|shift|reschedule|put)\s+(.+?)\s+(?:to|till|until)\s+(.+)$/

    /// "move dentist to friday", "push the dentist to thu 3pm": exactly one open task whose title has every word
    /// said, to the new day. Nil when no single task matches; `parse` then sends the phrase to the model.
    private func parseReschedule(_ phrase: String, context: TessContext) -> Parse? {
        guard let match = phrase.firstMatch(of: Self.rescheduleShape) else { return nil }
        let found = TessDates.find(in: String(match.2).split(separator: " ").map(String.init), now: context.now,
                                   calendar: context.calendar)
        guard var date = found.date, found.remaining.isEmpty || found.remaining == ["next"] else { return nil }
        let name = String(match.1).lowercased().replacing(/^(the|my)\s+/, with: "")
        let open = TaskList(store: store, classes: classes).all().filter { !$0.isDone }
        let words = name.split(separator: " ")
        let matches = open.filter { item in
            let title = Set(item.title.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init))
            return words.allSatisfy { title.contains(String($0)) }
        }
        guard matches.count == 1, let task = matches.first?.task else { return nil }
        // Keep what wasn't said: a new day keeps the task's time; a new time keeps its day.
        let calendar = context.calendar
        if let old = store.value("startsAt", of: task)?.dateValue {
            if found.saidDay, !found.saidTime {
                let parts = calendar.dateComponents([.hour, .minute], from: old)
                date = calendar.date(bySettingHour: parts.hour ?? 9, minute: parts.minute ?? 0, second: 0, of: date)
                    ?? date
            } else if found.saidTime, !found.saidDay {
                let parts = calendar.dateComponents([.hour, .minute], from: date)
                date = calendar.date(bySettingHour: parts.hour ?? 9, minute: parts.minute ?? 0, second: 0, of: old)
                    ?? date
            }
        }
        // Keeping the old day or hour must not land in the past ("move dentist to 3pm" after 3 PM; QA G10).
        if date <= context.now, let said = found.date {
            date = said
        }
        return .reschedule(task: task, to: date)
    }

    /// Past verbs that start a log ("watered", "saw", "went"). Words that only look past ("need", "feed") don't.
    static let pastVerbs: Set = ["did", "went", "saw", "met", "ate", "had", "ran", "took", "got", "bought", "made",
                                 "paid", "sent", "rang", "spoke", "slept", "woke", "fed", "drank", "swam", "rode",
                                 "left", "came", "gave", "wrote", "called", "was", "were"]
    static let notPast: Set = ["need", "feed", "seed", "weed", "bleed", "breed", "speed", "shed", "embed", "exceed",
                               "proceed", "succeed", "red", "bed", "shred", "sled", "wed"]

    static func isPastVerb(_ word: String) -> Bool {
        let word = word.lowercased()
        return !notPast.contains(word) && (pastVerbs.contains(word) || word.count >= 5 && word.hasSuffix("ed"))
    }

    /// "watered plants sat", "saw mum yesterday": a note dated that day (Product, 2026-10-15).
    /// A past-tense log of money, dated, kept private: "paid rent 2100 fri".
    func parsePrivateLog(_ phrase: String, context: TessContext) -> Parse? {
        guard case let .note(_, on)? = parsePastLog(phrase, context: context) else { return nil }
        let body = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        return .privateNote(body: body.prefix(1).uppercased() + body.dropFirst(), remembered: false, on: on)
    }

    private func parsePastLog(_ phrase: String, context: TessContext) -> Parse? {
        let words = phrase.split(separator: " ").map(String.init)
        guard let first = words.first, Self.isPastVerb(first),
              let (day, rest) = TessDates.pastDay(in: words, now: context.now, calendar: context.calendar),
              !rest.isEmpty else { return nil }
        let body = rest.joined(separator: " ")
        // Midday on that day, so it sits inside the day wherever its edges are drawn.
        let noon = context.calendar.date(bySettingHour: 12, minute: 0, second: 0, of: day) ?? day
        return .note(body: body.prefix(1).uppercased() + body.dropFirst(), on: noon)
    }

    /// "note: …" or "note …" at the start makes a note of the rest, keeping its text as typed.
    private func parseNote(_ phrase: String) -> Parse? {
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let match = trimmed.firstMatch(of: /^(?i)note[:\s]\s*(.+)$/) else { return nil }
        let body = String(match.1).trimmingCharacters(in: .whitespacesAndNewlines)
        return body.isEmpty ? nil : .note(body: body)
    }

    // MARK: Sets in focus

    func parseRelativeSet(_ text: String, entry: ObjectRecord) -> Parse? {
        let last = lastSet(in: entry)
        if ["same", "same again", "again", "repeat", "one more"].contains(text), let last {
            return .set(entry: entry, weight: last.weight, reps: last.reps, kind: .working, phrase: .same)
        }
        if let reps = Self.number(text), reps.rounded() == reps, reps >= 1, reps <= 100, let last {
            return .set(entry: entry, weight: last.weight, reps: reps, kind: .working, phrase: .repsOnly)
        }
        if let match = text.firstMatch(of: /^drop(?: to)? (\d+(?:\.\d+)?)(?: (kg|lb))? (?:for|x) (\d+)$/),
           let weight = Double(match.1), let reps = Double(match.3)
        {
            let unit = match.2.flatMap { MassUnit(parsing: String($0)) } ?? TessUnits.weight
            return .set(entry: entry, weight: unit.toKilograms(weight), reps: reps, kind: .drop, phrase: .drop,
                        unit: unit)
        }
        if let match = text.firstMatch(of: /^(\d+(?:\.\d+)?)(?: ?(kg|lb))?(?: ?x ?| for | by )(\d+)$/),
           let weight = Double(match.1), let reps = Double(match.3)
        {
            let unit = match.2.flatMap { MassUnit(parsing: String($0)) } ?? TessUnits.weight
            return .set(entry: entry, weight: unit.toKilograms(weight), reps: reps, kind: .working, phrase: .full,
                        unit: unit)
        }
        return nil
    }

    func lastSet(in entry: ObjectRecord) -> (weight: Double, reps: Double)? {
        for set in entry.orderedChildren.reversed() {
            if let weight = store.number("weight", of: set), let reps = store.number("reps", of: set) {
                return (weight, reps)
            }
        }
        return nil
    }

    // MARK: Named lifts

    /// "3x10 bench 80", "bench 3x10 80", "bench 80x8", "bench 80 for 8", "bench 3x8 at 80", "squat 100 5".
    /// "bench 80x8, 82.5x6": each comma part is a set, the exercise carried forward from the first. With an entry in
    /// focus, "80x8, 80x8, 80x6" logs into it. "Squat 100, 5" is one set (weight, then reps).
    private func parseCommaSets(_ phrase: String, entry: ObjectRecord?) -> Parse? {
        let parts = phrase.split(separator: ",").map { Self.normalize(String($0)) }.filter { !$0.isEmpty }
        guard parts.count >= 2 else { return nil }
        // "Squat 100, 5" (dictated): weight and reps split by a comma. Not when the first part is a whole lift
        // already ("bench 80x8, 6" is two sets; QA Q7a).
        if parts.count == 2, Self.number(parts[1]) != nil, parseLift(parts[0], entry: entry) == nil,
           let lift = parseLift(
               parts.joined(separator: " "),
               entry: entry
           )
        {
            return lift
        }
        var parsed: [Parse] = []
        // The exercise's name carries forward, never the first part's numbers: "bench 80 for 8, 82.5 for 6".
        var exerciseName: String?
        for part in parts {
            if let lift = parseLift(part, entry: entry) {
                parsed.append(lift)
                switch lift {
                case let .lift(exercise, _, _, _, _): exerciseName = exercise.name.lowercased()
                case let .set(entry, _, _, _, _, _, _, _):
                    exerciseName = store.linkedClass("exercise", of: entry)?.name.lowercased()
                default: break
                }
            } else if let name = exerciseName, let lift = parseLift("\(name) \(part)", entry: entry) {
                parsed.append(lift)
            } else if let entry, let live = parseLive(part, text: part, entry: entry) {
                // Through the live guard, so "bench 80 for 8, 81" never logs 81 reps (QA Q7b, K874). After a set, a
                // bare number is its next weight: Tess asks for the reps.
                guard let set = live.parse, !parsed.contains(where: \.isAsk) else { return nil }
                if set.isAsk, let number = Self.number(part), !part.hasSuffix(" reps") {
                    parsed.append(.ask("\(TessVoice.number(number)) for how many reps?"))
                } else {
                    parsed.append(set)
                }
            } else if let entry, let set = parseRelativeSet(part, entry: entry) {
                parsed.append(set)
            } else {
                return nil
            }
        }
        return .sets(parsed)
    }

    static let equipmentWords = ["db": "dumbbell", "bb": "barbell", "kb": "kettlebell", "ez": "ez"]

    /// Common gym names for built-in lifts.
    static let aliases: [String: String] = [
        "bench": "bench press", "bp": "bench press", "ohp": "overhead press", "press": "overhead press",
        "squat": "back squat", "squats": "back squat", "dl": "deadlift", "deads": "deadlift",
        "rdl": "romanian deadlift", "row": "barbell row", "rows": "barbell row", "pullups": "pull-up",
        "pull-ups": "pull-up",
        "pull ups": "pull-up", "chinups": "chin-up", "curls": "dumbbell curl", "curl": "dumbbell curl", "dips": "dip",
        "lat pulldown": "lat pulldown",
    ]

    /// The exercise class whose name best matches: exact, then alias, then every word contained, shortest first.
    public func resolveExercise(_ query: String) -> ClassDef? {
        guard let root = classes[.exercise] else { return nil }
        let candidates = ClassResolver.descendants(of: root).filter { store.isConcreteExercise($0, root: root) }
        // Gym shorthand for equipment: "db press" is a dumbbell press.
        let normalized = Self.normalize(query).split(separator: " ")
            .map { Self.equipmentWords[String($0)] ?? String($0) }.joined(separator: " ")
        let target = Self.aliases[normalized] ?? normalized
        let names = candidates.map { ($0, Self.normalize($0.name)) }
        if let exact = names.first(where: { $0.1 == target || $0.1 == normalized }) {
            return exact.0
        }
        // Library exercises carry their own aliases ("Flat Bench", "RDL") on their class objects.
        for (classDef, _) in names {
            guard let aliases = store.classValue("aliases", of: classDef)?.stringValue else { continue }
            let list = aliases.split(separator: ",").map { Self.normalize(String($0)) }
            if list.contains(normalized) {
                return classDef
            }
        }
        // Whole words only, three letters or more ("p" isn't Dip, "rolls" isn't Rollout).
        let words = target.split(separator: " ").map { $0.hasSuffix("s") && $0.count > 3 ? $0.dropLast() : $0 }
        // Gym shorthand ("ez", "db") is exempt from the floor.
        let shorthand = Set(Self.equipmentWords.keys).union(Self.equipmentWords.values)
        guard !words.isEmpty, words.allSatisfy({ $0.count >= 3 || shorthand.contains(String($0)) }) else { return nil }
        // Whole words of the name or its library aliases ("ez curl" finds Preacher Curl through "EZ-Bar Preacher
        // Curl"), the shortest name winning.
        func wordSet(_ text: String) -> Set<Substring> {
            Set(text.split { !$0.isLetter && !$0.isNumber }
                .map { $0.hasSuffix("s") && $0.count > 3 ? $0.dropLast() : $0 })
        }
        return names
            .filter { classDef, name in
                var known = wordSet(name)
                for alias in (store.classValue("aliases", of: classDef)?.stringValue ?? "").split(separator: ",") {
                    known.formUnion(wordSet(Self.normalize(String(alias))))
                }
                return words.allSatisfy { known.contains($0) }
            }
            .min { $0.1.count < $1.1.count }?.0
    }

    // MARK: Tasks

    /// A task with a day or time cue ("dentist thu 3pm"), or one that says it's a task ("todo call mom").
    /// Research 134 R1 to R7. A value after the secret word makes a private note; with no value a task shape stays a
    /// task ("password reset email friday", "code review friday 3pm"); anything else with a secret word is a private
    /// note too, because nothing with one goes to the model. A question about it is answered on the phone.
    func parseSecret(_ phrase: String, text: String, context: TessContext) -> Parse? {
        guard SecretWords.mentions(phrase) else { return nil }
        var body = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        var remembered = false
        if let match = body.firstMatch(of: /^(?i)note[:\s]\s*(.+)$/) {
            body = String(match.1)
        } else if let match = body.firstMatch(of: /^(?i)remember(?:\s+that)?\s+(.+)$/) {
            body = String(match.1)
            remembered = true
        }
        if Self.isQuestion(text) || body.hasSuffix("?") {
            return .ask("It's in your private notes.")
        }
        if !SecretWords.hasValue(body), !remembered, let task = parseTask(phrase, context: context) {
            return task
        }
        // A list or kitchen shape with no value wins (Research 139): "buy pins", "get a combination lock" are
        // groceries, read by the rules below. They still never reach a model: the router keeps them on the grammar.
        if !SecretWords.hasValue(body), !remembered,
           phrase.firstMatch(of: /^(?i)\s*(?:buy|get|grab|pick up|add)\s+\S/) != nil
        {
            return nil
        }
        return .privateNote(body: body.prefix(1).uppercased() + body.dropFirst(), remembered: remembered)
    }

    func parseTask(_ phrase: String, context: TessContext) -> Parse? {
        var words = phrase.split(separator: " ").map(String.init)
        var explicit = false
        if let first = words.first?.lowercased().trimmingCharacters(in: .punctuationCharacters),
           ["todo", "task", "remind"].contains(first)
        {
            explicit = true
            words.removeFirst()
            if words.first?.lowercased() == "me" {
                words.removeFirst()
            }
            if words.first?.lowercased() == "to" {
                words.removeFirst()
            }
        }
        let found = TessDates.find(in: words, now: context.now, calendar: context.calendar)
        var remaining = found.remaining
        // "remind me tomorrow to call mom": the "to" sat after the date.
        if remaining.first?.lowercased() == "to", explicit {
            remaining.removeFirst()
        }
        let title = remaining.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ",;:"))
            .trimmingCharacters(in: .whitespaces)
        guard !title.isEmpty, explicit || found.date != nil else { return nil }
        // "plank 2:30" is a hold, not a time: leave it to the model.
        if found.bareColon, !found.saidDay, resolveExercise(title.lowercased()) != nil {
            return nil
        }
        return .task(title: title.prefix(1).uppercased() + title.dropFirst(), startsAt: found.date)
    }

    // MARK: Helpers

    static func normalize(_ text: String) -> String {
        text.lowercased()
            .replacingOccurrences(of: "×", with: "x")
            .replacingOccurrences(of: ",", with: " ")
            // "180lbs" and "80kg" become "180 lb" and "80 kg".
            .replacing(/(\d)(kgs?|lbs?)\b/) { "\($0.1) \($0.2.hasPrefix("k") ? "kg" : "lb")" }
            .replacing(/\blbs\b/, with: "lb")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .trimmingCharacters(in: .punctuationCharacters)
    }

    static func number(_ text: String) -> Double? {
        Double(text.replacingOccurrences(of: " only", with: "").replacingOccurrences(of: " reps", with: ""))
    }
}

/// Finds a day and time in a phrase: "thu 3pm", "tomorrow", "today 18:30", "friday at 9".
enum TessDates {
    static let weekdays: [String: Int] = [
        "sun": 1, "sunday": 1, "mon": 2, "monday": 2, "tue": 3, "tues": 3, "tuesday": 3, "wed": 4, "wednesday": 4,
        "thu": 5, "thur": 5, "thurs": 5, "thursday": 5, "fri": 6, "friday": 6, "sat": 7, "saturday": 7,
    ]

    /// What a phrase said about when: the date, the words left over, and which parts were said.
    struct Found {
        var date: Date?
        var remaining: [String]
        var saidDay = false
        var saidTime = false
        /// The time came from a bare "2:30" with no "at" (could be a duration: "plank 2:30").
        var bareColon = false
    }

    /// Cues that settle a bare hour: "wake me at 7" is morning, "drinks at 9" is evening.
    static let amCues: Set = ["wake", "alarm", "morning", "breakfast"]
    static let pmCues: Set = ["tonight", "evening", "dinner", "drinks", "pub", "party"]
    /// Words that only lead into a day or time ("on fri", "by thu", "this sat", "for tomorrow").
    static let lead: Set = ["on", "by", "this", "for"]

    static func extract(from words: [String], now: Date, calendar: Calendar) -> (Date?, [String]) {
        let found = find(in: words, now: now, calendar: calendar)
        return (found.date, found.remaining)
    }

    /// Bare hours (Product, 2026-10-15): 1 to 6 mean pm, 7 to 11 the next one today (another day: am), 12 is noon;
    /// an am or pm cue settles it. A time today that has passed lands tomorrow.
    static func find(in words: [String], now: Date, calendar: Calendar) -> Found {
        var day: Date?
        var time: (hour: Int, minute: Int, explicit: Bool)?
        var remaining: [String] = []
        var fromWeekday = false, saidToday = false, bareColon = false, tonightOnly = false
        let lower = Set(words.map { $0.lowercased().trimmingCharacters(in: .punctuationCharacters) })
        func dropLead() {
            if let last = remaining.last?.lowercased(), lead.contains(last) {
                remaining.removeLast()
            }
        }
        var index = 0
        while index < words.count {
            let word = words[index].lowercased().trimmingCharacters(in: .punctuationCharacters)
            if word == "today" || word == "tonight" {
                day = calendar.startOfDay(for: now)
                saidToday = true
                if word == "tonight", time == nil {
                    time = (20, 0, true)
                    tonightOnly = true
                }
                dropLead()
            } else if word == "tomorrow" || word == "tmrw" {
                // His tomorrow: at 01:00 Saturday (still his Friday) "tomorrow" is Saturday, the coming daytime.
                day = TesseraDay(calendar: calendar).day(1, after: now)
                dropLead()
            } else if let weekday = weekdays[word] {
                day = nextDate(weekday: weekday, after: now, calendar: calendar)
                fromWeekday = true
                dropLead()
            } else if word == "at", index + 1 < words.count, let parsed = clock(words[index + 1].lowercased()) {
                time = parsed
                tonightOnly = false
                index += 1
            } else if let parsed = clock(word), word.contains(where: \.isLetter) || word.contains(":") {
                time = parsed
                tonightOnly = false
                bareColon = !word.contains(where: \.isLetter)
                dropLead()
            } else {
                remaining.append(words[index])
            }
            index += 1
        }
        guard day != nil || time != nil else { return Found(date: nil, remaining: words) }
        var base = day ?? calendar.startOfDay(for: now)
        let today = day == nil || saidToday
        // A weekday that names today ("thu" on a Thursday) also counts as today for the hour rule only.
        let hourToday = today || fromWeekday && day.map { calendar.isDate($0, inSameDayAs: now) } == true
        var hour = time?.hour ?? 9, minute = time?.minute ?? 0
        if let time, !time.explicit, (1 ... 12).contains(time.hour) {
            if !lower.isDisjoint(with: pmCues), time.hour < 12 {
                hour = time.hour + 12
            } else if !lower.isDisjoint(with: amCues) || time.hour == 12 {
                hour = time.hour
            } else if time.hour <= 6 {
                hour = time.hour + 12
            } else if hourToday, let morning = calendar.date(
                bySettingHour: time.hour,
                minute: minute,
                second: 0,
                of: base
            ),
                morning <= now,
                let evening = calendar.date(bySettingHour: time.hour + 12, minute: minute, second: 0, of: base),
                evening > now
            {
                hour = time.hour + 12 // 7 to 11: the next one today (both passed: tomorrow morning)
            }
        }
        // "tonight" said after 20:00 stays tonight until 23:30: the next whole hour (Product, 4:41 PM PT; QA G11).
        var secondPass: Date?
        let lateCutoff = calendar.date(bySettingHour: 23, minute: 30, second: 0, of: now) ?? now
        if tonightOnly, let eight = calendar.date(bySettingHour: 20, minute: 0, second: 0, of: base), eight <= now,
           now < lateCutoff
        {
            hour = min(calendar.component(.hour, from: now) + 1, 23)
            minute = hour == 23 && calendar.component(.hour, from: now) == 23 ? 59 : 0
        } else if today, time != nil,
                  let candidate = calendar.date(bySettingHour: hour, minute: minute, second: 0, of: base),
                  candidate <= now
        {
            // The fall-back night's repeated hour: "1:30am" said at 1:45 am PDT is the 1:30 am PST still ahead, not
            // tomorrow's (Research DS8, board 0ag).
            secondPass = Self.laterPass(hour: hour, minute: minute, of: base, after: now, calendar: calendar)
            if secondPass == nil {
                base = calendar.date(byAdding: .day, value: 1, to: base) ?? base
            }
        }
        var date = secondPass ?? calendar.date(bySettingHour: hour, minute: minute, second: 0, of: base)
        // "thu" said on a Thursday means today (Product, 2026-10-15). With no time, never in the past: the next
        // whole hour. With a time that's passed, next week.
        if let resolved = date, resolved <= now {
            date = Self.ahead(resolved, now: now, fromWeekday: fromWeekday, saidDay: day != nil, saidTime: time != nil,
                              hour: hour, minute: minute, calendar: calendar)
        }
        // A time Tess picked (none was said) never lands in his quiet hours, 23:00 to 07:00: it moves to 7 am.
        // A time he says himself is kept (Product 0ab, 1:22 PM PT Oct 16).
        if time == nil, let resolved = date {
            date = Self.outOfQuietHours(resolved, calendar: calendar)
        }
        return Found(date: date, remaining: remaining, saidDay: day != nil, saidTime: time != nil,
                     bareColon: bareColon)
    }

    /// A past day for a log: "sat" is the last Saturday (today if it's Saturday), "yesterday", "today", "last night".
    /// Nil when no day was said. The day word goes; the rest stays.
    static func pastDay(in words: [String], now: Date, calendar: Calendar) -> (Date, [String])? {
        // His day: at 00:40 Saturday, "today" is still Friday and "yesterday" Thursday.
        let today = TesseraDay(calendar: calendar).today(now)
        var day: Date?
        var remaining: [String] = []
        var index = 0
        while index < words.count {
            let word = words[index].lowercased().trimmingCharacters(in: .punctuationCharacters)
            let nextWord = index + 1 < words.count ? words[index + 1].lowercased() : ""
            if word == "yesterday" || word == "last" && nextWord.hasPrefix("night") {
                day = calendar.date(byAdding: .day, value: -1, to: today)
                index += word == "last" ? 1 : 0
            } else if word == "today" || word == "tonight" || word == "this" && nextWord == "morning" {
                day = today
                index += word == "this" ? 1 : 0
            } else if let weekday = weekdays[word] {
                let back = (calendar.component(.weekday, from: today) - weekday + 7) % 7
                day = calendar.date(byAdding: .day, value: -back, to: today)
                if let last = remaining.last?.lowercased(), ["on", "last"].contains(last) {
                    remaining.removeLast()
                }
            } else {
                remaining.append(words[index])
            }
            index += 1
        }
        return day.map { ($0, remaining) }
    }

    /// A clock time and whether it said am or pm (or was 13 or more, or 0).
    static func clock(_ text: String) -> (hour: Int, minute: Int, explicit: Bool)? {
        let word = text.trimmingCharacters(in: .punctuationCharacters)
        guard let parsed = parseTime(word) else { return nil }
        let explicit = word == "noon" || word.hasSuffix("am") || word.hasSuffix("pm") || parsed.hour >= 13
            || parsed.hour == 0
        return (parsed.hour, parsed.minute, explicit)
    }

    /// The next given weekday; today counts only if it hasn't passed (it's the same day, so we keep it).
    /// 23:00 to 07:00 becomes the next 7 am.
    static func outOfQuietHours(_ date: Date, calendar: Calendar, quiet: QuietHours = .current) -> Date {
        quiet.leaving(date, calendar: calendar)
    }

    /// The next date with that weekday, counted from his day: "thu" said at 1 am Friday (still his Thursday) is
    /// today, not next week (TW29).
    static func nextDate(weekday: Int, after now: Date, calendar: Calendar) -> Date? {
        let his = TesseraDay(calendar: calendar).day(of: now)
        let offset = (weekday - calendar.component(.weekday, from: his) + 7) % 7
        return calendar.date(byAdding: .day, value: offset, to: his)
    }

    /// "3pm", "3:30pm", "15:00", "9" (with "at"), "noon".
    static func parseTime(_ text: String) -> (hour: Int, minute: Int)? {
        let word = text.trimmingCharacters(in: .punctuationCharacters)
        if word == "noon" {
            return (12, 0)
        }
        guard let match = word.firstMatch(of: /^(\d{1,2})(?::(\d{2}))?(am|pm)?$/), let hour = Int(match.1) else {
            return nil
        }
        let minute = match.2.flatMap { Int($0) } ?? 0
        var result = hour
        if match.3 == "pm", hour < 12 {
            result += 12
        }
        if match.3 == "am", hour == 12 {
            result = 0
        }
        guard (0 ... 23).contains(result), (0 ... 59).contains(minute) else { return nil }
        return (result, minute)
    }
}
