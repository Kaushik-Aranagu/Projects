import Foundation
import TesseraCore

/// The tools Tess's model routes may call (Tess model layer T2): one list for the on-device model and Claude.
/// Models can't know ids, so every tool takes names ("Bench Press", "Groceries", "dentist") and resolves them here.
/// Each call performs real typed actions as Tess, pending by default, so it lands as a ghost row with undo.
/// Nothing destructive is here, and nothing a system clock should drive.
@MainActor
public enum TessTools {
    /// One tool: its name, what it's for, its input schema, and what it does.
    public struct Tool {
        public let name: String
        public let description: String
        public let input: TessSchema
        /// Money: the phone's own rungs only. Claude's tool list never carries it (money actions are never offered to
        /// Claude).
        public let deviceOnly: Bool
        let run: @MainActor (_ arguments: [String: Any], _ call: Call) throws -> [OpRecord]

        init(name: String, description: String, input: TessSchema, deviceOnly: Bool = false,
             run: @escaping @MainActor (_ arguments: [String: Any], _ call: Call) throws -> [OpRecord])
        {
            self.name = name
            self.description = description
            self.input = input
            self.deviceOnly = deviceOnly
            self.run = run
        }
    }

    /// What a tool call runs against.
    @MainActor
    struct Call {
        let engine: ActionEngine
        let grammar: TessGrammar
        let pending: Bool
        let now: Date
        /// His calendar: a time with no zone ("2026-10-21T15:00:00") is read in it, never as UTC (QA T1).
        let calendar: Calendar

        var store: ObjectStore {
            engine.store
        }

        func perform(_ action: some EngineAction) throws -> OpRecord {
            try engine.perform(action, by: .tess, pending: pending).record
        }
    }

    public static var all: [Tool] {
        [logSet, createTask, completeTask, rescheduleTask, addGroceries, addToList, createList, createNote,
         createRecipe, query, startShop, cookTogether, setServeTime, checkListItem, startWorkout, startCooking,
         logSpend,
         pinNote, rainAlert, restDay, tendRoutine,
         plantRoutine, pauseRoutine, rehearse]
    }

    /// The tools that write (every tool but the reads and rehearse), by name: a nonisolated list the request builder
    /// can read off the main actor. A test keeps it equal to `all`.
    public nonisolated static let writeNames: Set<String> = [
        "logSet", "createTask", "completeTask", "rescheduleTask", "addGroceries", "addToList", "createList",
        "createNote", "createRecipe", "startShop", "cookTogether", "setServeTime", "checkListItem", "startWorkout",
        "startCooking", "logSpend", "pinNote", "rainAlert", "restDay", "tendRoutine", "plantRoutine", "pauseRoutine",
    ]

    /// The tools Claude may be given: money too while `ClaudeSees.moneyWrites` is on. Never the rehearsal, which is
    /// offered through its own spec and runs on the phone.
    public static var claudeTools: [Tool] {
        all.filter { $0.name != RehearseFlags.tool && !ClaudeSees.keepsOnPhone($0.deviceOnly) }
    }

    public static func tool(named name: String) -> Tool? {
        all.first { $0.name == name }
    }

    /// Runs a tool call from its JSON arguments. Returns what was logged; throws `TessToolError` with a line the
    /// model can read and correct.
    /// Most ghost rows one phrase may leave (injection audit gap 2): a page or a pasted line that coaxes "add 50
    /// tasks" gets this many and then a plain refusal. Claude's capture and the on-device model both stop here.
    public static let maxRecordsPerPhrase = 10

    /// What a tool call answers once a phrase is at its cap.
    static let capLine = "Error: that is enough for one phrase."

    /// `budget`: how many more ghosts this phrase may leave (QA on #666). A list of lines or items is cut to it before
    /// the tool runs, and anything past it is taken back after, on both the on-device and the Claude path.
    public static func run(_ name: String, arguments: Data, engine: ActionEngine, pending: Bool = true,
                           now: Date = .now, calendar: Calendar = .current,
                           budget: Int = maxRecordsPerPhrase) throws -> [OpRecord]
    {
        guard let tool = tool(named: name) else { throw TessToolError("no tool called \(name)") }
        guard budget > 0 else { throw TessToolError(capLine) }
        guard var object = try? JSONSerialization.jsonObject(with: arguments) as? [String: Any] else {
            throw TessToolError("unreadable arguments")
        }
        for key in ["lines", "items"] {
            if let list = object[key] as? [Any], list.count > budget {
                object[key] = Array(list.prefix(budget))
            }
        }
        var records = try runChecked(tool, object, engine: engine, pending: pending, now: now, calendar: calendar)
        if records.count > budget {
            for record in records[budget...].reversed() {
                if record.state == .pending {
                    _ = try? engine.discard(record)
                } else {
                    engine.undo(record)
                }
            }
            records.removeLast(records.count - budget)
        }
        return records
    }

    private static func runChecked(_ tool: Tool, _ object: [String: Any], engine: ActionEngine, pending: Bool,
                                   now: Date, calendar: Calendar) throws -> [OpRecord]
    {
        let call = Call(engine: engine, grammar: TessGrammar(store: engine.store, classes: engine.classes),
                        pending: pending, now: now, calendar: calendar)
        // A tool that fails part-way takes back what it already did, so no half-done ghost is left behind.
        let mark = engine.history(limit: 1).first?.id
        do {
            return try tool.run(object, call)
        } catch let answer as TessToolAnswer {
            throw answer
        } catch {
            for record in engine.history(limit: 50) {
                if record.id == mark {
                    break
                }
                if record.state == .pending {
                    _ = try? engine.discard(record)
                } else if record.state != .undone {
                    engine.undo(record)
                }
            }
            // A refusal the actions word themselves is what he hears, not the error's debug text.
            if let line = TakeRest.refusalLine(error) {
                throw TessToolError(line)
            }
            throw (error as? TessToolError) ?? TessToolError("That didn't work.")
        }
    }

    /// Claude's tool definitions, strict: every object closed, every field required (optional ones nullable).
    public static func definitionsJSON() throws -> Data {
        let tools: [[String: Any]] = claudeTools.map { tool in
            ["name": tool.name, "description": tool.description, "strict": true, "input_schema": tool.input.json]
        }
        return try JSONSerialization.data(withJSONObject: tools, options: [.sortedKeys])
    }

    // MARK: The tools

    static let unit = TessSchema.choice(["kg", "lb"], "Weight unit")

    static var logSet: Tool {
        Tool(name: "logSet", description: "Log sets of an exercise into today's workout",
             input: .object([
                 ("exercise", .string("The exercise, e.g. Bench Press")),
                 ("weight", .number("Weight per rep")),
                 ("reps", .integer("Reps per set")),
                 ("sets", .integer("How many sets, usually 1")),
                 ("unit", unit),
             ])) { args, call in
            let name = try args.string("exercise")
            guard let exercise = call.grammar.resolveExercise(name) else {
                throw TessToolError("no exercise called \(name)")
            }
            let unit = MassUnit(parsing: args["unit"] as? String ?? TessUnits.symbol) ?? TessUnits.weight
            return try [call.perform(LogExercise(exercise: exercise.id, weight: args.number("weight"),
                                                 reps: args.number("reps"), sets: Int(args.number("sets")),
                                                 unit: unit))]
        }
    }

    static var createTask: Tool {
        Tool(name: "createTask", description: "Add a task, optionally at a time and about things (e.g. a recipe)",
             input: .object([
                 ("title", .string("What to do, verb first")),
                 ("startsAt", .nullable(.string("When, ISO 8601 date-time"))),
                 ("about", .array(.string("A thing it's about, by name"))),
             ])) { args, call in
            let about = (args["about"] as? [String] ?? []).compactMap { call.object(named: $0) }
            return try [call.perform(CreateTask(
                title: args.string("title"),
                startsAt: args.date("startsAt", in: call.calendar),
                about: about.map(\.id)
            ))]
        }
    }

    static var completeTask: Tool {
        Tool(name: "completeTask", description: "Mark a task done",
             input: .object([("task", .string("The task, by name"))]))
        { args, call in
            let task = try call.task(named: args.string("task"))
            return try [call.perform(SetTaskDone(task: task.id))]
        }
    }

    static var rescheduleTask: Tool {
        Tool(name: "rescheduleTask", description: "Move a task to another time",
             input: .object([
                 ("task", .string("The task, by name")),
                 ("startsAt", .string("When, ISO 8601 date-time")),
             ])) { args, call in
            let task = try call.task(named: args.string("task"))
            guard let date = args.date("startsAt", in: call.calendar)
            else { throw TessToolError("startsAt isn't a date") }
            return try [call.perform(MoveTask(task: task.id, to: date))]
        }
    }

    static var addGroceries: Tool {
        Tool(name: "addGroceries", description: "Put things on the groceries list",
             input: .object([("lines", .array(.string("One thing, with its amount if said, e.g. 2 lemons")))]))
        {
            args, call in
            let lines = args["lines"] as? [String] ?? []
            guard !lines.isEmpty else { throw TessToolError("nothing to add") }
            return try [call.perform(AddGroceries(lines: lines))]
        }
    }

    static let item = TessSchema.object([
        ("text", .string("The item")),
        ("section", .nullable(.string("A section, if said"))),
        ("done", .boolean("Already done")),
    ])

    static var addToList: Tool {
        Tool(name: "addToList", description: "Add items to a list by its name",
             input: .object([("list", .string("The list, e.g. Packing")), ("items", .array(item))]))
        { args, call in
            let list = try call.list(named: args.string("list"))
            return try call.addItems(args["items"] as? [[String: Any]] ?? [], to: list)
        }
    }

    static var createList: Tool {
        Tool(name: "createList", description: "Make a new list, with items if given",
             input: .object([("name", .string("What the list is")), ("items", .array(item))]))
        { args, call in
            let name = try args.string("name")
            // A name that is already a list opens it: the items go onto that one (G40).
            let record = try call.perform(CreateList(name: name))
            guard let list = CreateList.existing(named: name, in: call.store, classes: call.engine.classes) else {
                return [record]
            }
            let items = args["items"] as? [[String: Any]] ?? []
            // "new list packing" with nothing in it yet is fine.
            return items.isEmpty ? [record] : try [record] + call.addItems(items, to: list)
        }
    }

    static var createNote: Tool {
        Tool(name: "createNote", description: "Write a note; @Name links it to a thing",
             input: .object([("body", .string("The note"))]))
        { args, call in
            try [call.perform(CreateNote(body: args.string("body")))]
        }
    }

    static var createRecipe: Tool {
        Tool(name: "createRecipe", description: "Save a recipe",
             input: .object([
                 ("title", .string("The recipe's name")),
                 ("serves", .nullable(.integer("How many it serves"))),
                 ("totalMinutes", .nullable(.integer("Total time in minutes"))),
                 ("ingredients", .array(.object([
                     ("amount", .nullable(.string("e.g. 200, 1/2, 2-3"))),
                     ("unit", .nullable(.string("e.g. g, tbsp, cloves"))),
                     ("name", .string("The ingredient, with prep after a comma")),
                 ]))),
                 ("steps", .array(.string("One step"))),
             ])) { args, call in
            try [call.perform(CreateRecipe(text: TessTools.recipeText(args)))]
        }
    }

    /// The structured recipe written out as text, so the same parser (and its kept source text) reads it.
    static func recipeText(_ args: [String: Any]) throws -> String {
        var lines = try [args.string("title")]
        if let serves = args["serves"] as? Int {
            lines.append("Serves \(serves)")
        }
        if let total = args["totalMinutes"] as? Int {
            lines.append("Total time: \(total) min")
        }
        lines.append("Ingredients")
        for ingredient in args["ingredients"] as? [[String: Any]] ?? [] {
            let parts = [ingredient["amount"] as? String, ingredient["unit"] as? String, ingredient["name"] as? String]
            lines.append(parts.compactMap(\.self).joined(separator: " "))
        }
        lines.append("Method")
        lines += args["steps"] as? [String] ?? []
        return lines.joined(separator: "\n")
    }

    /// Plans several saved recipes as one cook. The model names them and may give a time to eat; it never sees an id, a
    /// private recipe (it cannot be named) or anything the plan holds, and what it is told back is "Several dishes
    /// planned".
    static var cookTogether: Tool {
        Tool(
            name: "cookTogether",
            description: "Plan cooking several saved recipes together, optionally for a time to eat",
            input: .object([
                ("recipes", .array(.string("A saved recipe, by name"))),
                ("serveAt", .nullable(.string("When to eat, ISO 8601 date-time"))),
            ])
        ) { args, call in
            let names = args["recipes"] as? [String] ?? []
            guard names.count >= 2 else { throw TessToolError("name at least two recipes") }
            let recipes = try names.map { name -> ObjectRecord in
                guard let recipe = call.grammar.recipe(named: name),
                      !isPrivateForClaude(recipe, store: call.store, classes: call.engine.classes)
                else { throw TessToolError("no recipe called \(name)") }
                return recipe
            }
            return try [call.perform(StartCookSession(recipes: recipes.map(\.id),
                                                      serveAt: args.date("serveAt", in: call.calendar),
                                                      time: call.now))]
        }
    }

    /// Moves the time to eat of the cook that is planned or under way.
    static var setServeTime: Tool {
        Tool(name: "setServeTime", description: "Change when to eat for the cook that is planned",
             input: .object([("serveAt", .string("When to eat, ISO 8601 date-time"))]))
        { args, call in
            let book = CookSessionBook(store: call.store, classes: call.engine.classes)
            guard let session = book.current() else { throw TessToolError("no cook is planned") }
            guard let when = args.date("serveAt", in: call.calendar) else { throw TessToolError("unreadable time") }
            return try [call.perform(SetServeTime(session: session.id, serveAt: when, time: call.now))]
        }
    }

    static var startShop: Tool {
        Tool(name: "startShop", description: "Start shopping the groceries list", input: .object([])) { _, call in
            try [call.perform(StartShop(time: call.now))]
        }
    }

    /// What answerQuery may count: training, tasks, notes, lists and the kitchen. Never a money or body class.
    static let countable: [SystemCatalog.Key] = [.workout, .exerciseEntry, .set, .routine, .exercise, .task, .note,
                                                 .list, .listItem, .recipe, .cook, .ingredient]

    /// Whether an object is private for what goes to Claude: the one title check for every countable class (a list,
    /// an item, an ingredient, a workout, a routine, a set), the cook rule for a cook, and the note, task, memory and
    /// recipe rules. An object inside a private one (a set of a private workout) is private with it. Nothing is,
    /// while Claude sees everything (`ClaudeSees`, his Oct 27 call).
    static func isPrivateForClaude(_ object: ObjectRecord, store: ObjectStore,
                                   classes: [SystemCatalog.Key: ClassDef]) -> Bool
    {
        guard !ClaudeSees.everything else { return false }
        let last = BodyWeight.lastKg(in: store)
        var current: ObjectRecord? = object
        var depth = 0
        while let item = current, depth < 8 {
            if ObjectPrivacy.label(of: item, in: store, classes: classes) != nil
                || store.value("isPrivate", of: item) == .bool(true)
                || ObjectPrivacy.isPrivateTitle(item.title, lastWeightKg: last)
                || item.title == ObjectPrivacy.routineLabel
                || item.classDef?.id == classes[.cook]?.id
                && ObjectPrivacy.cookTitle(item, in: store) == ObjectPrivacy.recipeLabel
            {
                return true
            }
            current = item.parent
            depth += 1
        }
        return false
    }

    /// Read-only: the query AST (docs/research/questions-eval.md), answered from the store on the phone. Counts, bests,
    /// averages and lists come from the data, never from a model's arithmetic. It sees only `countable` classes minus
    /// anything private for Claude, exactly as `answerQuery` did.
    static var query: Tool {
        Tool(name: "query", description: """
             Look things up in the log; read only. from: classes (workouts, sets, cooks, recipes, tasks, list items, notes, \
             lists, routines, exercises). where: field op value (ops: = != < <= > >= ~ isa has none; a field may be a path \
             like exercise.pattern or workout.routine; value nil means empty). during: a range on a date field (default \
             field per class). agg: list, count, sum, avg, min, max (a numeric field like weight, reps, e1rm, volume) or \
             latest. group splits a count or sum by a field. Use this for every number: never work one out yourself.
             """,
             input: .object([
                 ("from", .array(.string("Class names, e.g. sets, workouts, tasks"))),
                 ("where", .array(.object([
                     ("field", .string("A field or path, e.g. reps, exercise, isDone, due")),
                     ("op", .choice(Array(Query.ops.sorted()), "Operator")),
                     ("value", .nullable(.string("The value; nil for empty"))),
                 ]))),
                 ("during", .nullable(.object([
                     (
                         "range",
                         .string(
                             "today, tomorrow, thisWeek, lastWeek, thisMonth, lastMonth, thisYear, last7d, last30d, last90d, date:YYYY-MM-DD or month:YYYY-MM"
                         )
                     ),
                     ("field", .nullable(.string("Date field; null for the class's own"))),
                 ]))),
                 ("group", .nullable(.string("Field to split by"))),
                 ("agg", .object([
                     ("kind", .choice(Array(Query.aggs.sorted()), "What to work out")),
                     ("field", .nullable(.string("Numeric field for sum, avg, min, max; date field for latest"))),
                 ])),
                 ("sort", .nullable(.object([
                     ("field", .string("Field to sort by")),
                     ("descending", .boolean("Largest or latest first")),
                 ]))),
                 ("limit", .nullable(.integer("Most rows, 1 to 50"))),
             ])) { args, call in
            guard let data = try? JSONSerialization.data(withJSONObject: args), let query = Query(json: data) else {
                throw TessToolError("unreadable query")
            }
            do {
                let result = try QueryEngine.run(query, engine: call.engine, now: call.now, calendar: call.calendar)
                throw TessToolAnswer(line: result.line)
            } catch let error as QueryError {
                throw TessToolError(error.description)
            }
        }
    }
}

// MARK: - Resolving names

@MainActor
extension TessTools.Call {
    /// An open task whose title matches: exactly, else the one containing every word said.
    func task(named name: String) throws -> ObjectRecord {
        // A bound call (a place Tess was shown) names the task by id: a private one is a label, never a name.
        if let byID = object(id: name, of: .task), store.value("isDone", of: byID) != .bool(true) {
            return byID
        }
        let tasks = TaskList(store: store, classes: engine.classes).all().filter { !$0.isDone }
        let query = name.lowercased()
        if let exact = tasks.first(where: { $0.title.lowercased() == query }) {
            return exact.task
        }
        let words = query.split(separator: " ")
        let matches = tasks.filter { item in words.allSatisfy { item.title.lowercased().contains($0) } }
        guard matches.count == 1, let match = matches.first else {
            throw TessToolError(matches.isEmpty ? "no open task called \(name)" : "more than one task like \(name)")
        }
        return match.task
    }

    /// A list by name ("groceries" is the groceries list). An exact name wins only when exactly one list has it; else
    /// a prefix of three letters or more, when exactly one list starts with it. Two matches write nothing and ask
    /// which (G40). A one- or two-letter prefix never matches.
    func list(named name: String) throws -> ObjectRecord {
        try ListLookup.find(name, store: store, classes: engine.classes)
    }

    /// The fewest letters that count as a prefix of a list's name.
    /// Any thing by name, recipes first ("about" links).
    func object(named name: String) -> ObjectRecord? {
        if let recipe = grammar.recipe(named: name) {
            return recipe
        }
        let query = name.lowercased()
        return store.context.objects.values
            .first { !$0.isDeleted && !$0.isClassObject && $0.title.lowercased() == query }
    }

    func addItems(_ items: [[String: Any]], to list: ObjectRecord) throws -> [OpRecord] {
        var records: [OpRecord] = []
        for item in items {
            guard let text = item["text"] as? String, !text.isEmpty else { continue }
            // Groceries goes through the Add line's path: one row per ingredient, linked, in its aisle (G16). A line
            // the wall turns into a private note takes the plain path, which makes that note.
            if list.id == Groceries.listID, let kept = AddListItem.kept(text, store: store) {
                let result = try engine.perform(AddGroceries(lines: [kept]), by: .tess, pending: pending)
                records.append(result.record)
                if item["done"] as? Bool == true, let row = result.output.first {
                    try records.append(perform(CheckListItem(item: row.id)))
                }
                continue
            }
            let record = try perform(AddListItem(list: list.id, text: text, section: item["section"] as? String))
            records.append(record)
            if item["done"] as? Bool == true, let id = record.objectIDs.first {
                try records.append(perform(CheckListItem(item: id)))
            }
        }
        guard !records.isEmpty else { throw TessToolError("no items to add") }
        return records
    }
}

/// A list by name for Tess's phrases and tools: an exact name wins only when exactly one list has it; else a prefix of
/// three letters or more, when exactly one list starts with it. Two matches write nothing and ask which (G40).
@MainActor
public enum ListLookup {
    /// The fewest letters that count as a prefix of a list's name.
    static let prefixMinimum = 3

    static func find(_ name: String, store: ObjectStore,
                     classes: [SystemCatalog.Key: ClassDef]) throws -> ObjectRecord
    {
        let query = name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let lists = ListContents(store: store, classes: classes).lists()
        func title(_ list: ObjectRecord) -> String {
            list.title.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // A private word is never echoed back, even as his own query.
        let shown = ObjectPrivacy.isPrivateWord(name, in: store) ? "that" : name
        let exact = lists.filter { title($0) == query }
        if exact.count == 1, let list = exact.first {
            return list
        }
        if exact.count > 1 {
            throw TessToolError(ListLookupLine.moreThanOne(shown))
        }
        let prefixed = query.count >= prefixMinimum ? lists.filter { title($0).hasPrefix(query) } : []
        if prefixed.count > 1 {
            throw TessToolError(ListLookupLine.moreThanOne(shown))
        }
        guard let list = prefixed.first else { throw TessToolError("no list called \(shown)") }
        return list
    }
}

/// The line for a name that matches two lists (G40), for the app to say the same words.
public enum ListLookupLine {
    public static func moreThanOne(_ name: String) -> String {
        "More than one list like \(name)."
    }
}

/// A line a model can read and act on ("no task called dentist").
public struct TessToolError: Error, CustomStringConvertible {
    public let description: String

    init(_ description: String) {
        self.description = description
    }
}

/// A read-only tool's answer (no ops): the line to show.
public struct TessToolAnswer: Error {
    public let line: String
}

extension [String: Any] {
    func string(_ key: String) throws -> String {
        guard let value = self[key] as? String, !value.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw TessToolError("\(key) is missing")
        }
        return value
    }

    func number(_ key: String) throws -> Double {
        if let value = self[key] as? Double {
            return value
        }
        if let value = self[key] as? Int {
            return Double(value)
        }
        throw TessToolError("\(key) isn't a number")
    }

    /// An ISO 8601 date: with a zone or offset, as written; without one, in his calendar's zone (QA T1).
    func date(_ key: String, in calendar: Calendar) -> Date? {
        guard let text = self[key] as? String else { return nil }
        let local = Date.ISO8601FormatStyle(timeZone: calendar.timeZone).year().month().day()
            .time(includingFractionalSeconds: false).timeSeparator(.colon)
        return (try? Date(text, strategy: .iso8601)) ?? (try? Date(text, strategy: local))
    }
}

// MARK: - Schema

/// A small JSON-schema shape that emits Claude's strict schema and converts to the on-device model's schema, so
/// one definition feeds both routes.
public indirect enum TessSchema: Sendable {
    case string(String)
    case number(String)
    case integer(String)
    case boolean(String)
    case choice([String], String)
    case array(TessSchema)
    case object([(String, TessSchema)])
    case nullable(TessSchema)

    var summary: String {
        switch self {
        case let .string(text), let .number(text), let .integer(text), let .boolean(text), let .choice(_, text): text
        case let .array(item): item.summary
        case .object: ""
        case let .nullable(inner): inner.summary
        }
    }

    /// Strict JSON schema: objects closed, every property required, optional ones as a type union with null.
    public var json: [String: Any] {
        switch self {
        case let .string(text): ["type": "string", "description": text]
        case let .number(text): ["type": "number", "description": text]
        case let .integer(text): ["type": "integer", "description": text]
        case let .boolean(text): ["type": "boolean", "description": text]
        case let .choice(options, text): ["type": "string", "enum": options, "description": text]
        case let .array(item): ["type": "array", "items": item.json]
        case let .object(properties):
            [
                "type": "object", "additionalProperties": false,
                "properties": Dictionary(uniqueKeysWithValues: properties.map { ($0.0, $0.1.json) }),
                "required": properties.map(\.0),
            ]
        case let .nullable(inner): Self.nullable(inner.json)
        }
    }

    /// A nullable schema: the type takes null. A choice can't be a type union with an enum (the API refuses "Enum
    /// value does not match declared type"), so it is one of its choices or null.
    private static func nullable(_ inner: [String: Any]) -> [String: Any] {
        if inner["enum"] != nil {
            var choice = inner
            let description = choice.removeValue(forKey: "description")
            var json: [String: Any] = ["anyOf": [choice, ["type": "null"]]]
            json["description"] = description
            return json
        }
        var json = inner
        json["type"] = [inner["type"] as? String ?? "string", "null"]
        return json
    }
}
