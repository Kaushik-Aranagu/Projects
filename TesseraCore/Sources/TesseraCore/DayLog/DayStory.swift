import Foundation

/// One line of the Day log (docs/research/day-log-first.md, day-log-acceptance.md, Design's day-log-v1.md).
public struct DayLine: Equatable, Sendable {
    /// Which tier the page draws it in, so it never reads that out of the text (K8).
    public enum Kind: String, Sendable, Equatable {
        /// A thing that happened and is done: a finished workout, a cook, a capture, a note.
        case finished
        /// Today's workout, under way: numbers swap as sets land.
        case running
        /// A past workout never finished: the faint row, no words about it.
        case unfinished
        /// "Rest day", first, no time.
        case rest
        /// "Also: …", last, no time.
        case tail
    }

    /// When it started; nil for "Rest day" and the "Also:" tail.
    public let time: Date?
    /// Plain spaces around ×; the page swaps in no-break spaces when it draws.
    public let text: String
    /// What a tap opens (the first).
    public let objectIDs: [UUID]
    public let kind: Kind
    /// "about 330 kcal" for a finished workout, only when he turned calories on (`CaloriesOptions`). Kept apart from
    /// `text` on purpose: the words of the line feed memory, Home and Claude, and a calorie number never goes there.
    public var calories: String?

    public init(time: Date?, text: String, objectIDs: [UUID], kind: Kind, calories: String? = nil) {
        self.time = time
        self.text = text
        self.objectIDs = objectIDs
        self.kind = kind
        self.calories = calories
    }

    /// "07:10": the column, 24 h, zero padded, in the calendar's zone (the zone he's in now).
    public func clock(_ calendar: Calendar) -> String? {
        time.map { DayStory.clock($0, calendar: calendar) }
    }

    /// "7:10 am, Push Day A, 52 minutes. Bench 82.5 by 8, best." for VoiceOver.
    public func spoken(_ calendar: Calendar) -> String {
        let words = text.replacingOccurrences(of: " × ", with: " by ")
            .replacing(/(\d+) min\b/) { "\($0.1) minutes" }
        guard let time else { return words }
        return "\(DayStory.sentenceTime(time, calendar: calendar)), \(words)"
    }
}

/// What the fold needs to know about objects; the store in the app, a fake in a test if wanted.
@MainActor
public protocol DayLookup {
    func title(of id: UUID) -> String?
    /// The class's system key ("task", "note", "workout", "listItem"…).
    func kind(of id: UUID) -> String?
    func parent(of id: UUID) -> UUID?
    /// The workout above this object (entry or set), or the object itself when it is one.
    func groupRoot(of id: UUID) -> UUID?
    /// The root's own day time: a workout's `startedAt`, else the object's `createdAt` (a past-dated note's `on`).
    func dayTime(of id: UUID) -> Date?
    /// The zone of the record that created or started the root; nil means the calendar's zone (pre-v3 rows).
    func dayZone(of id: UUID) -> TimeZone?
    /// A workout's facts.
    func workout(_ id: UUID) -> (title: String, startedAt: Date?, endedAt: Date?, sets: Int)?
    /// "Bench 82.5 × 8", and whether it beat a record.
    func topSet(inWorkout id: UUID) -> (text: String, isPR: Bool)?
    /// A field's key ("isDone"), for reading a record's values.
    func fieldKey(_ id: UUID) -> String?
    /// A spend's amount as typed, its currency and its one word (nil word when the word is walled).
    func spend(_ id: UUID) -> (amount: Decimal, currency: String?, word: String?)?
    /// "about 330 kcal" for a finished workout, or nil (off, private, unfinished, no weight).
    func caloriesLine(forWorkout id: UUID) -> String?
}

public extension DayLookup {
    func caloriesLine(forWorkout _: UUID) -> String? {
        nil
    }

    func spend(_: UUID) -> (amount: Decimal, currency: String?, word: String?)? {
        nil
    }
}

/// Folds the op log into a day's story: records by action, by his 4 AM day, groups by their root's start, loose things
/// counted, at most 6 lines. Pure: it never writes, and the same input always reads the same.
public enum DayStory {
    public static let maxLines = 6
    /// An unfinished workout reads as under way this long after its start, whatever the day (DL70, QA S2 on #318).
    public static let liveFor: TimeInterval = 6 * 3600
    /// The records a day can need: from 14 h before its 04:00 start (zones east of here) to 20 h after its end
    /// (zones west, a cook or shop finished late, a note said the next day).
    static let windowBefore: TimeInterval = 14 * 3600, windowAfter: TimeInterval = 20 * 3600
    static let notesAfter: TimeInterval = 8 * 86400

    /// The 04:00 day start for a moment.
    public static func dayStart(for date: Date, calendar: Calendar) -> Date {
        TesseraDay(calendar: calendar).start(of: date)
    }

    /// Workout actions that fold into the workout's one line.
    static let workoutActions: Set = ["startWorkout", "startRoutine", "logSet", "logNextSet", "logExercise",
                                      "addExercise", "swapExercise", "finishWorkout", "finishWorkoutEarly"]
    static let ownLines: [String: String] = [
        "createRecipe": "Saved",
        "createList": "New list:",
        "copyList": "New list:",
        "createRoutine": "New routine:",
        "createPlan": "New plan:",
    ]
    /// Edits, reorders, plumbing: never a line (day-log-first.md, the 26 drops).
    static let dropped: Set = [
        "editSet", "editRecipe", "editNote", "renameObject", "renameListSection", "moveListItem", "clearDoneItems",
        "setListReusable", "resetList", "startListRun", "finishListRun", "putAwayListRun", "setListPickOne",
        "acceptPick",
        "addSlot", "moveSlot", "editSlot", "removeSlot", "swapSlotExercise", "deleteObject",
        "removeRecipeFromGroceries", "moveTask", "keepPlannedTask", "rollForwardTasks", "waitOnTask", "startTask",
        "goToCookStep",
        "startKitchenTimer", "stopKitchenTimer", "startProcess", "saveContext", "suspendProcess", "resumeProcess",
        "endProcess", "writeSettings", "createClass", "setRoutineKind", "applyDayName", "syncCalendar", "syncHealth",
        "closeDay", "createPlace", "renamePlace", "setPlaceSections", "movePlace", "removePlace", "setItemPlace", "setMeProfile", "startFocusBlock", "focusCheckIn", "acceptFirstStep", "holdNotice",
        "releaseHeldNotices",
        "editSpend", "logShopTotal", "skipShopTotal", "queueBook", "editBook", "startReading", "setBookAside",
        // Multi-dish Cook: the session has no Day log line yet (the dishes' cooks are stamped as ever).
        "startCookSession", "setServeTime", "addDish", "dropDish", "advanceDishStep", "pauseCook", "resumeCook",
        "finishCookSession",
        // Money pools: amounts and pool names never reach the Day log.
        "createPool", "logInflow", "reassignSpend", "logMove", "recordRate", "archivePool",
        "makeKind", "keepNoteAsObject", "keepAllNotes",
        // Occasions: links and dates only; the page reads them, the Day log does not.
        "createOccasion", "linkToOccasion", "unlinkFromOccasion", "setAway",
        "repeatWorkout", "markArrived", "joinTwins", "keepTwinsApart",
        // Repeats: a rule's words never reach the Day log.
        "setRepeat", "skipRepeat", "stopRepeating",
    ]

    /// Actions with their own fold rule (the rest are workout, own-line or dropped).
    static let folded: Set = ["startCooking", "finishCooking", "startShop", "finishShop", "createTask", "addGroceries",
                              "addRecipeToGroceries", "addListItem", "setTaskDone", "checkListItem", "createNote",
                              "logSpend",
                              "setBookPage", "finishBook"]

    /// Whether an action has a rule here (a test fails when a new action has none).
    public static func hasRule(for action: String) -> Bool {
        workoutActions.contains(action) || ownLines[action] != nil || dropped.contains(action) || folded
            .contains(action)
    }

    /// The lines for one of his days (`day`: any moment in it, read in `calendar`). `now` decides running vs past;
    /// `isRestDay` puts "Rest day" first.
    /// One fold, rule by rule in day-log-first.md's order, so it reads against the spec top to bottom.
    @MainActor
    // swiftlint:disable:next cyclomatic_complexity function_body_length
    public static func lines(for records: [OpRecord], on day: Date, calendar: Calendar, lookup: any DayLookup,
                             now: Date = .now, isRestDay: Bool = false) -> [DayLine]
    {
        let target = key(TesseraDay(calendar: calendar).day(of: day), in: calendar)
        func onDay(_ instant: Date, zone: TimeZone?) -> Bool {
            var zoned = calendar
            zoned.timeZone = zone ?? calendar.timeZone
            return key(TesseraDay(calendar: zoned).day(of: instant), in: zoned) == target
        }
        func zone(of record: OpRecord) -> TimeZone? {
            record.zone.flatMap(TimeZone.init(identifier:))
        }
        func valueSet(_ record: OpRecord, _ field: String) -> FieldValue? {
            for op in record.ops {
                if case let .setValue(_, id, _, new) = op, lookup.fieldKey(id) == field {
                    return new
                }
            }
            return nil
        }
        func created(_ record: OpRecord, kind: String) -> [UUID] {
            record.ops.compactMap { op in
                if case let .create(object, _, _, _) = op, lookup.kind(of: object) == kind {
                    return object
                }
                return nil
            }
        }

        let standing = records.filter { $0.state == .committed && $0.actor != .system }
            .sorted { $0.createdAt < $1.createdAt }
        // Same-day pairs cancel: a thing made and deleted that day; a done counts by its last state that day.
        var deletedToday = Set<UUID>()
        var doneToday: [UUID: Bool] = [:]
        // A shop is its start and the finish that closes it; a later start replaces an open one (QA L3 on #318).
        var shopFinish: [UUID: OpRecord] = [:]
        var openShop: OpRecord?
        var cookFinish: [UUID: OpRecord] = [:]
        for record in standing {
            switch record.action {
            case "startShop": openShop = record
            case "finishShop":
                if let start = openShop {
                    shopFinish[start.id] = record
                }
                openShop = nil
            case "finishCooking":
                for id in record.objectIDs where cookFinish[id] == nil {
                    cookFinish[id] = record
                }
            default: break
            }
            guard record.action == "deleteObject" || record.action == "setTaskDone",
                  onDay(record.createdAt, zone: zone(of: record)) else { continue }
            if record.action == "deleteObject" {
                for op in record.ops {
                    if case let .softDelete(object) = op {
                        deletedToday.insert(object)
                    }
                }
            } else if let done = valueSet(record, "isDone") {
                for id in record.objectIDs {
                    doneToday[id] = done == .bool(true)
                }
            }
        }

        // Workouts and own lines, and whether each is a workout (they keep their place first under the cap).
        var lines: [(line: DayLine, group: Bool)] = []
        // Every root checked once, on its day or not, and each object's root looked up once (QA S1 on #318).
        var seenRoots = Set<UUID>()
        var roots: [UUID: UUID] = [:]
        var rootless = Set<UUID>()
        func root(of id: UUID) -> UUID? {
            if let known = roots[id] {
                return known
            }
            guard !rootless.contains(id) else { return nil }
            guard let found = lookup.groupRoot(of: id) else {
                rootless.insert(id)
                return nil
            }
            roots[id] = found
            return found
        }
        // Loose counts, each at its first time.
        struct Count {
            var time: Date
            var objects: [UUID] = []
            var titles: [String] = []
        }
        var tasks: Count?, groceries: Count?, dones: Count?, notes: Count?, spends: Count?
        // Reading: the page he started the day on and ended it on, per book; a finish takes the book's one line.
        var pages: [UUID: (from: Int, to: Int, time: Date)] = [:]
        var finishedBooks: [UUID: Date] = [:]
        var ticks: [UUID: Count] = [:]
        func add(_ count: inout Count?, at time: Date, objects: [UUID], titles: [String] = []) {
            if count == nil {
                count = Count(time: time)
            }
            count?.objects += objects
            count?.titles += titles
        }

        for record in standing where !dropped.contains(record.action) {
            let action = record.action
            // Workouts: one line per workout, on the day it started.
            if workoutActions.contains(action) {
                guard let root = record.objectIDs.lazy.compactMap(root(of:)).first,
                      seenRoots.insert(root).inserted, let workout = lookup.workout(root),
                      let started = workout.startedAt, onDay(started, zone: lookup.dayZone(of: root)),
                      workout.sets > 0 // a 0-set workout is no session
                else { continue }
                let top = lookup.topSet(inWorkout: root)
                let topText = top.map { ". \($0.text)\($0.isPR ? ", best" : "")." } ?? "."
                if let ended = workout.endedAt {
                    let minutes = Int((ended.timeIntervalSince(started) / 60).rounded())
                    lines.append((DayLine(time: started, text: "\(workout.title), \(minutes) min\(topText)",
                                          objectIDs: [root], kind: .finished), true))
                } else if now >= started, now.timeIntervalSince(started) < liveFor {
                    // Still going: in ink on the day it started, even past 04:00 (DL70).
                    lines.append((DayLine(time: started, text: "\(workout.title), \(sets(workout.sets))",
                                          objectIDs: [root], kind: .running), true))
                } else {
                    lines.append((DayLine(time: started, text: "\(workout.title), \(sets(workout.sets))\(topText)",
                                          objectIDs: [root], kind: .unfinished), true))
                }
                continue
            }
            switch action {
            case "startCooking":
                // A cook files on the day it was finished, read in the finish's zone, like a done task (Product G2,
                // trip-week TW4). The row shows the finish time and its length, an hour or more in h and min
                // ("9:05 am Cooked overnight oats, 6 h 55 min"; Product 2:48 PM PT).
                guard let cook = record.objectIDs.first(where: { lookup.kind(of: $0) == "cook" }),
                      let finish = cookFinish[cook], onDay(finish.createdAt, zone: zone(of: finish)),
                      seenRoots.insert(cook).inserted
                else { continue }
                let minutes = Int((finish.createdAt.timeIntervalSince(record.createdAt) / 60).rounded())
                let length = minutes < 60 ? "\(minutes) min"
                    : minutes % 60 == 0 ? "\(minutes / 60) h" : "\(minutes / 60) h \(minutes % 60) min"
                lines.append((DayLine(time: finish.createdAt,
                                      text: "Cooked \(lookup.title(of: cook) ?? "it"), \(length)",
                                      objectIDs: [cook], kind: .finished), false))
            case "startShop":
                guard onDay(record.createdAt, zone: zone(of: record)), let finish = shopFinish[record.id] else {
                    continue
                }
                let items = finish.ops.count {
                    if case .softDelete = $0 {
                        true
                    } else {
                        false
                    }
                }
                guard items > 0 else { continue }
                lines.append((DayLine(time: record.createdAt,
                                      text: "Shopped, \(items) \(items == 1 ? "item" : "items")",
                                      objectIDs: [Groceries.listID], kind: .finished), false))
            case "createTask":
                guard onDay(record.createdAt, zone: zone(of: record)) else { continue }
                let made = created(record, kind: "task").filter { !deletedToday.contains($0) }
                guard !made.isEmpty else { continue }
                add(&tasks, at: record.createdAt, objects: made)
            case "addGroceries", "addRecipeToGroceries", "addListItem":
                guard onDay(record.createdAt, zone: zone(of: record)) else { continue }
                let made = created(record, kind: "listItem")
                    .filter { lookup.parent(of: $0) == Groceries.listID && !deletedToday.contains($0) }
                guard !made.isEmpty else { continue }
                add(&groceries, at: record.createdAt, objects: made)
            case "setTaskDone":
                guard onDay(record.createdAt, zone: zone(of: record)), valueSet(record, "isDone") == .bool(true)
                else { continue }
                // Done, reopened, done again: one done, at its first time (QA L2 on #318).
                let counted = Set(dones?.objects ?? [])
                let done = record.objectIDs.filter {
                    doneToday[$0] == true && !counted.contains($0) && lookup.kind(of: $0) == "task"
                }
                add(&dones, at: record.createdAt, objects: done, titles: done.compactMap(lookup.title))
            case "checkListItem":
                guard onDay(record.createdAt, zone: zone(of: record)), valueSet(record, "isDone") == .bool(true)
                else { continue }
                for item in record.objectIDs where lookup.kind(of: item) == "listItem" {
                    guard let list = lookup.parent(of: item), list != Groceries.listID else { continue }
                    var count: Count? = ticks[list]
                    add(&count, at: record.createdAt, objects: [item])
                    ticks[list] = count
                }
            case "logSpend":
                guard onDay(record.createdAt, zone: zone(of: record)) else { continue }
                let made = created(record, kind: "spend").filter { !deletedToday.contains($0) }
                guard !made.isEmpty else { continue }
                add(&spends, at: record.createdAt, objects: made)
            case "setBookPage":
                guard onDay(record.createdAt, zone: zone(of: record)) else { continue }
                for op in record.ops {
                    guard case let .setValue(object, id, old, new) = op, lookup.fieldKey(id) == "page",
                          case let .number(to) = new else { continue }
                    var from = 0
                    if let old, case let .number(value) = old {
                        from = Int(value)
                    }
                    pages[object] = (pages[object]?.from ?? from, Int(to), pages[object]?.time ?? record.createdAt)
                }
            case "finishBook":
                guard onDay(record.createdAt, zone: zone(of: record)) else { continue }
                for book in record.objectIDs where lookup.kind(of: book) == "book" {
                    finishedBooks[book] = finishedBooks[book] ?? record.createdAt
                }
            case "createNote":
                // A note said afterwards lands on the day it's about (its own createdAt), in its record's zone.
                for note in created(record, kind: "note") where !deletedToday.contains(note) {
                    // A past-dated note ("watered the plants sat") carries its day as its own time, well before
                    // the record's; otherwise the record's time.
                    let own = lookup.dayTime(of: note) ?? record.createdAt
                    let time = own < record.createdAt.addingTimeInterval(-60) ? own : record.createdAt
                    guard onDay(time, zone: zone(of: record)) else { continue }
                    add(&notes, at: time, objects: [note], titles: [lookup.title(of: note) ?? "note"])
                }
            default:
                if let lead = ownLines[action], onDay(record.createdAt, zone: zone(of: record)) {
                    // A copy is a new list (copyList does not start with "create").
                    let key = action == "copyList" ? "list" : String(action.dropFirst("create".count)).lowercased()
                    for object in created(record, kind: key) where !deletedToday.contains(object) {
                        lines.append((DayLine(time: record.createdAt,
                                              text: "\(lead) \(lookup.title(of: object) ?? "")",
                                              objectIDs: [object], kind: .finished), false))
                    }
                }
            }
        }

        for (book, time) in finishedBooks.sorted(by: { $0.value < $1.value }) {
            lines.append((DayLine(time: time, text: "Finished \(lookup.title(of: book) ?? "a book")", objectIDs: [book],
                                  kind: .finished), false))
        }
        for (book, moved) in pages.sorted(by: { $0.value.time < $1.value.time })
            where finishedBooks[book] == nil && moved.to > moved.from
        {
            let title = lookup.title(of: book) ?? "a book"
            // No earlier page: "Read Dune to p 142"; a real start page: "Read Dune, p 120 to 142".
            let span = moved.from == 0 ? "to p \(moved.to)" : "p \(moved.from) to \(moved.to)"
            lines.append((DayLine(time: moved.time, text: "Read \(title)\(moved.from == 0 ? " " : ", ")\(span)",
                                  objectIDs: [book], kind: .finished), false))
        }

        // The loose counts, each one line at its first time.
        struct Loose {
            let line: DayLine
            let size: Int
            let tail: String
        }
        var loose: [Loose] = []
        if tasks != nil || groceries != nil {
            let first = min(tasks?.time ?? .distantFuture, groceries?.time ?? .distantFuture)
            var parts: [String] = []
            if let tasks {
                parts.append("\(tasks.objects.count) \(tasks.objects.count == 1 ? "task" : "tasks")")
            }
            if let groceries {
                parts.append("\(groceries.objects.count) \(groceries.objects.count == 1 ? "grocery" : "groceries")")
            }
            let size = (tasks?.objects.count ?? 0) + (groceries?.objects.count ?? 0)
            loose.append(Loose(line: DayLine(time: first, text: "Added \(parts.joined(separator: ", "))",
                                             objectIDs: (tasks?.objects ?? []) + (groceries?.objects ?? []),
                                             kind: .finished),
                               size: size, tail: parts.joined(separator: ", ")))
        }
        if let spends, let line = spendLine(spends.objects, at: spends.time, lookup: lookup) {
            loose.append(Loose(line: line, size: spends.objects.count,
                               tail: "\(spends.objects.count) \(spends.objects.count == 1 ? "spend" : "spends")"))
        }
        if let dones, !dones.objects.isEmpty {
            let text = dones.objects.count <= 2 ? "Done: \(dones.titles.joined(separator: ", "))"
                : "\(dones.objects.count) tasks done"
            loose.append(Loose(line: DayLine(time: dones.time, text: text, objectIDs: dones.objects, kind: .finished),
                               size: dones.objects.count, tail: "\(dones.objects.count) done"))
        }
        if let notes {
            // A private note reads "Private note" and nothing more (Research 134 SW2); its title already is that.
            let text = notes.objects.count == 1
                ? (notes.titles[0] == SecretWords.title ? SecretWords.title : "Note: \(notes.titles[0])")
                : "\(notes.objects.count) notes"
            loose.append(Loose(line: DayLine(time: notes.time, text: text, objectIDs: notes.objects, kind: .finished),
                               size: notes.objects.count,
                               tail: "\(notes.objects.count) \(notes.objects.count == 1 ? "note" : "notes")"))
        }
        for (list, count) in ticks.sorted(by: { $0.value.time < $1.value.time }) {
            let name = lookup.title(of: list) ?? "a list"
            loose.append(Loose(line: DayLine(time: count.time, text: "Ticked \(count.objects.count) on \(name)",
                                             objectIDs: [list] + count.objects, kind: .finished),
                               size: count.objects.count, tail: "ticked \(count.objects.count) on \(name)"))
        }

        // At most 6 lines: workouts first, then own lines, then the largest counts; the rest is one "Also:" tail,
        // which takes a slot of its own even when own lines alone would fill them (QA L1 on #318).
        let cap = maxLines - (isRestDay ? 1 : 0)
        var drawn: [DayLine]
        if lines.count + loose.count > cap {
            let fixed = lines.sorted { ($0.group ? 0 : 1, $0.line.time ?? .distantFuture)
                < ($1.group ? 0 : 1, $1.line.time ?? .distantFuture)
            }
            let keptFixed = min(fixed.count, cap - 1)
            let ranked = loose.sorted { ($0.size, $1.line.time ?? .distantFuture) > (
                $1.size,
                $0.line.time ?? .distantFuture
            ) }
            let room = cap - 1 - keptFixed
            drawn = fixed.prefix(keptFixed).map(\.line) + ranked.prefix(room).map(\.line)
            let rest = (fixed.dropFirst(keptFixed).map { Loose(line: $0.line, size: 1, tail: $0.line.text) }
                + ranked.dropFirst(room))
                .sorted { ($0.line.time ?? .distantFuture) < ($1.line.time ?? .distantFuture) }
            drawn.append(DayLine(time: nil, text: "Also: \(rest.map(\.tail).joined(separator: ", ")).",
                                 objectIDs: rest.flatMap(\.line.objectIDs), kind: .tail))
        } else {
            drawn = lines.map(\.line) + loose.map(\.line)
        }
        // By instant, never by the "HH:mm" text; the tail last.
        drawn.sort { lhs, rhs in
            switch (lhs.time, rhs.time) {
            case let (left?, right?): left < right
            case (_?, nil): true
            default: false
            }
        }
        // Calories ride beside a finished workout's line, never in its words.
        drawn = drawn.map { line in
            guard line.kind == .finished, let id = line.objectIDs.first, lookup.kind(of: id) == "workout",
                  let text = lookup.caloriesLine(forWorkout: id) else { return line }
            var withCalories = line
            withCalories.calories = text
            return withCalories
        }
        if isRestDay {
            drawn.insert(DayLine(time: nil, text: "Rest day", objectIDs: [], kind: .rest), at: 0)
        }
        return drawn
    }
}

extension DayStory {
    /// "Spent 12.40, lunch" for one; "Spent 3 times, 26.90 (lunch, coffee, taxi)" for several. The total is shown only
    /// when every spend has the same currency (never converted); words are lowercased, deduplicated, at most 3.
    @MainActor
    static func spendLine(_ ids: [UUID], at time: Date, lookup: any DayLookup) -> DayLine? {
        let facts = ids.compactMap { id in lookup.spend(id).map { (id, $0) } }
        guard !facts.isEmpty else { return nil }
        let currencies = Set(facts.map { $0.1.currency ?? "" })
        let total: String? = currencies.count == 1
            ? (facts[0].1.currency.flatMap { SpendLedger.symbols[$0] } ?? "") + MoneyAmount
            .format(facts.reduce(Decimal(0)) { $0 + $1.1.amount })
            : nil
        var words: [String] = []
        for fact in facts {
            if let word = fact.1.word, !word.isEmpty, !words.contains(word) {
                words.append(word)
            }
        }
        let shown = words.prefix(3).joined(separator: ", ")
        let text: String
        if facts.count == 1 {
            text = "Spent " + [total, shown.isEmpty ? nil : shown].compactMap(\.self).joined(separator: ", ")
        } else {
            let lead = "Spent \(facts.count) times"
            text = [lead, total].compactMap(\.self).joined(separator: ", ") + (shown.isEmpty ? "" : " (\(shown))")
        }
        return DayLine(time: time, text: text, objectIDs: facts.map(\.0), kind: .finished)
    }
}

/// Words and keys the fold and its lines share.
extension DayStory {
    static func sets(_ count: Int) -> String {
        "\(count) \(count == 1 ? "set" : "sets")"
    }

    /// (year, month, day) of a day key, in the calendar it was made with.
    static func key(_ date: Date, in calendar: Calendar) -> DateComponents {
        calendar.dateComponents([.year, .month, .day], from: date)
    }

    static func clock(_ date: Date, calendar: Calendar) -> String {
        let parts = calendar.dateComponents([.hour, .minute], from: date)
        return String(format: "%02d:%02d", parts.hour ?? 0, parts.minute ?? 0)
    }

    /// "6:12 pm", "11:20 pm", "12 pm".
    static func sentenceTime(_ date: Date, calendar: Calendar) -> String {
        HouseTime.clock(date, calendar: calendar)
    }
}

/// The Day log's view of the store: titles as stored, a workout's facts and top set, each root's day zone.
@MainActor
public struct StoreDayLookup: DayLookup {
    public let store: ObjectStore
    public let classes: [SystemCatalog.Key: ClassDef]
    /// Calories for finished workouts, when he turned them on; nil (the default) shows none.
    public var calories: CaloriesOptions?
    let fields: [UUID: String]
    /// Each root's zone, read once: the record that started it, else the one that created it (QA S1 on #318).
    let zones: [UUID: TimeZone]

    public init(store: ObjectStore, classes: [SystemCatalog.Key: ClassDef], records: [OpRecord]) {
        self.store = store
        self.classes = classes
        var started: [UUID: TimeZone?] = [:], made: [UUID: TimeZone?] = [:]
        for record in records {
            let zone = record.zone.flatMap(TimeZone.init(identifier:))
            if ["startWorkout", "startRoutine", "logExercise"].contains(record.action) {
                for id in record.objectIDs where started[id] == nil {
                    started[id] = .some(zone)
                }
            }
            for op in record.ops {
                if case let .create(object, _, _, _) = op, made[object] == nil {
                    made[object] = .some(zone)
                }
            }
        }
        var zones: [UUID: TimeZone] = [:]
        // A start with no zone (a pre-v3 row) falls back to the create's.
        for id in Set(started.keys).union(made.keys) {
            if let zone = started[id].flatMap(\.self) ?? made[id].flatMap(\.self) {
                zones[id] = zone
            }
        }
        self.zones = zones
        fields = Dictionary(
            store.context.all(FieldDef.self).map { ($0.id, $0.key) },
            uniquingKeysWith: { first, _ in first }
        )
    }

    public func title(of id: UUID) -> String? {
        guard let object = store.object(id: id) else { return nil }
        // A private recipe's cook reads "Private recipe" in the story, old cooks included (G21b).
        if let cook = classes[.cook], let recipe = classes[.recipe], let classDef = object.classDef,
           ClassResolver.isSubclass(classDef, of: cook) || ClassResolver.isSubclass(classDef, of: recipe)
        {
            return ObjectPrivacy.cookTitle(object, in: store)
        }
        if let book = classes[.book], let classDef = object.classDef, ClassResolver.isSubclass(classDef, of: book) {
            // Bare words wall a book here, as everywhere a title can reach a memory line: when unsure, wall.
            if let label = ObjectPrivacy.label(of: object, in: store, classes: classes) {
                return label
            }
            return PrivateWords.isPrivateTitle(object.title, lastWeightKg: BodyWeight.lastKg(in: store))
                ? ObjectPrivacy.bookLabel : object.title
        }
        // A list with a walled name reads "Private list" in the story, as it does everywhere else (G75).
        if let list = classes[.list], let classDef = object.classDef, ClassResolver.isSubclass(classDef, of: list) {
            return ListContents.shownName(of: object, in: store)
        }
        return object.title
    }

    public func spend(_ id: UUID) -> (amount: Decimal, currency: String?, word: String?)? {
        guard kind(of: id) == "spend", let object = store.object(id: id) else { return nil }
        let typed = store.value("typed", of: object)?.stringValue ?? ""
        let amount = MoneyAmount.parse(typed)?.amount
            ?? store.number("amount", of: object).flatMap { Decimal(string: String($0)) } ?? 0
        let currency = store.value("currency", of: object)?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
        // A walled word never reaches the line (the amount may; the word may not).
        let word = store.value("category", of: object)?.stringValue.flatMap {
            PrivateWords.isPrivateTitle($0, lastWeightKg: nil) ? nil : $0
        }
        return (amount, currency, word)
    }

    public func kind(of id: UUID) -> String? {
        guard let classDef = store.object(id: id)?.classDef else { return nil }
        for (key, system) in classes where key != .thing && store.isSubclass(classDef, ofClassID: system.id) {
            if classDef.id == system.id {
                return key.rawValue
            }
        }
        return classes.first { store.isSubclass(classDef, ofClassID: $0.value.id) && $0.key != .thing }?.key.rawValue
    }

    public func parent(of id: UUID) -> UUID? {
        store.object(id: id)?.parent?.id
    }

    public func groupRoot(of id: UUID) -> UUID? {
        var object = store.object(id: id)
        while let current = object {
            if kind(of: current.id) == "workout" {
                return current.id
            }
            object = current.parent
        }
        return nil
    }

    public func dayTime(of id: UUID) -> Date? {
        guard let object = store.object(id: id) else { return nil }
        return store.value("startedAt", of: object)?.dateValue ?? object.createdAt
    }

    public func dayZone(of id: UUID) -> TimeZone? {
        zones[id]
    }

    public func workout(_ id: UUID) -> (title: String, startedAt: Date?, endedAt: Date?, sets: Int)? {
        guard let workout = store.object(id: id) else { return nil }
        let sets = workout.orderedChildren.reduce(0) { total, entry in
            total + entry.orderedChildren.count { !$0.isDeleted && store.number("reps", of: $0) != nil }
        }
        return (store.displayTitle(of: workout), store.value("startedAt", of: workout)?.dateValue,
                store.value("endedAt", of: workout)?.dateValue, sets)
    }

    public func topSet(inWorkout id: UUID) -> (text: String, isPR: Bool)? {
        guard let workout = store.object(id: id) else { return nil }
        struct Best {
            var e1rm: Double, text: String, exercise: ClassDef?, weight: Double, reps: Double, flag: Bool?
        }
        var best: Best?
        for entry in workout.orderedChildren {
            let exercise = store.linkedClass("exercise", of: entry)
            let name = Self.shortName(exercise?.name ?? entry.title)
            for set in entry.orderedChildren where !set.isDeleted {
                guard let weight = store.number("weight", of: set), let reps = store.number("reps", of: set) else {
                    continue
                }
                if store.value("kind", of: set) == .choice("warmup") {
                    continue
                }
                let e1rm = weight * (1 + reps / 30)
                if best == nil || e1rm > best!.e1rm { // swiftlint:disable:this force_unwrapping
                    best = Best(e1rm: e1rm, text: "\(name) \(shownWeight(weight)) × \(Self.number(reps))",
                                exercise: exercise, weight: weight, reps: reps,
                                flag: store.value("wasRecord", of: set).map { $0 == .bool(true) })
                }
            }
        }
        guard let best else { return nil }
        // The flag written when it was logged (Product 72954ac) keeps the PR after a later session beats it; sets
        // from before the flag fall back to one records read for the top set's lift (QA S1 on #318).
        if let flag = best.flag {
            return (best.text, flag)
        }
        let isPR = best.exercise.map {
            !TrainingLedger(store: store, classes: classes)
                .recordsBeaten(weight: best.weight, reps: best.reps, for: $0, excludingWorkout: id).isEmpty
        } ?? false
        return (best.text, isPR)
    }

    public func fieldKey(_ id: UUID) -> String? {
        fields[id]
    }

    public func caloriesLine(forWorkout id: UUID) -> String? {
        guard let calories, let workout = store.object(id: id) else { return nil }
        return CaloriesBurned.line(forWorkout: workout, store: store, classes: classes, options: calories)
    }

    /// A stored load (kilograms) as he chose to see weights: bare kilograms by default, "181.9 lb" when he picked
    /// pounds. Stored values never change.
    func shownWeight(_ kilograms: Double) -> String {
        let units = displayUnits
        guard let measure = units.weight(kilograms: kilograms),
              measure.unit != "kg" else { return Self.number(kilograms) }
        return measure.formatted
    }

    var displayUnits: DisplayUnits {
        guard let object = store.object(id: SettingsStore.objectID), !object.isDeleted else { return DisplayUnits() }
        return DisplayUnits(encoded: store.value("displayUnits", of: object)?.stringValue)
    }

    static func shortName(_ name: String) -> String {
        ExerciseWords.short(name)
    }

    static func number(_ value: Double) -> String {
        value.rounded() == value ? String(Int(value)) : String(format: "%g", value)
    }
}

@MainActor
public extension DayStory {
    /// The app's call: one of his days from a window of the engine's history around it (never the whole log).
    static func lines(engine: ActionEngine, on day: Date, calendar: Calendar = .current, now: Date = .now,
                      isRestDay: Bool = false, calories: CaloriesOptions? = nil) -> [DayLine]
    {
        let start = dayStart(for: day, calendar: calendar)
        let end = dayStart(for: start.addingTimeInterval(30 * 3600), calendar: calendar)
        let after = end.addingTimeInterval(windowAfter)
        // A note said up to a week later can be about this day ("watered the plants sat").
        let records = engine.history(
            from: start.addingTimeInterval(-windowBefore),
            to: after.addingTimeInterval(notesAfter)
        )
        .filter { $0.createdAt < after || $0.action == "createNote" }
        var lookup = StoreDayLookup(store: engine.store, classes: engine.classes, records: records)
        lookup.calories = calories
        return lines(for: records, on: day, calendar: calendar, lookup: lookup, now: now, isRestDay: isRestDay)
    }

    /// The store's committed history for one of his days, from records the caller already has.
    static func lines(store: ObjectStore, classes: [SystemCatalog.Key: ClassDef], records: [OpRecord], on day: Date,
                      calendar: Calendar = .current, now: Date = .now, isRestDay: Bool = false) -> [DayLine]
    {
        lines(for: records, on: day, calendar: calendar,
              lookup: StoreDayLookup(store: store, classes: classes, records: records), now: now,
              isRestDay: isRestDay)
    }
}
