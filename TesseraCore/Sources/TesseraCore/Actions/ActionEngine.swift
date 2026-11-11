import Foundation

/// Runs typed actions and keeps the operation log: commit, pending previews, undo and redo.
///
/// - Every action is one `OpRecord` holding the ops it applied, its actor and a one-line summary.
/// - A pending action (Tess's ghost row) is already applied, so it shows; `commit` makes it final,
///   `discard` reverts it.
/// - Undo and redo replay ops only where the data still looks the way the op left it. Anything changed since
///   (by a later edit or a sync) is left alone and reported as a conflict, never overwritten.
@MainActor
public final class ActionEngine {
    public struct Result<Output> {
        public let record: OpRecord
        public let output: Output
    }

    public enum ReplayOutcome: Equatable {
        case applied
        /// Some ops were skipped because the data changed since; the rest were applied.
        case partial(skipped: [Op])
        case nothingToDo
    }

    public let store: ObjectStore
    public let classes: [SystemCatalog.Key: ClassDef]
    /// How long a pending preview waits before `commitExpired` makes it final.
    public var pendingGracePeriod: TimeInterval = 2

    public init(store: ObjectStore, classes: [SystemCatalog.Key: ClassDef]) {
        self.store = store
        self.classes = classes
    }

    /// Set by Tess's router while one route runs, so the ops it performs record which route and model did them.
    public var attribution: (route: OpRoute, model: String?)?
    /// The zone stamped on each record: the device's zone now (autoupdating, so a flight is seen without a relaunch;
    /// QA Z1), unless a test sets it.
    public var zone: () -> TimeZone = { TimeZone.autoupdatingCurrent }
    /// The clock actions read when the caller gave no time (the app sets its own, so `-demoTime` and the test world
    /// hold; Kitchen actions use it for "tomorrow's 4 am" and "this weekday").
    public var now: () -> Date = { Date.now }
    /// Gregorian in the engine's zone: the calendar days are read in.
    public func calendar() -> Calendar {
        var calendar = Calendar.current
        calendar.timeZone = zone()
        return calendar
    }

    // MARK: Performing

    /// Runs an action and logs it. If it throws, everything it wrote so far is rolled back.
    @discardableResult
    public func perform<A: EngineAction>(_ action: A, by actor: Actor,
                                         pending: Bool = false) throws -> Result<A.Output>
    {
        var transaction = Transaction(store: store, classes: classes, now: now(), calendar: calendar())
        let output: A.Output
        do {
            output = try action.apply(&transaction)
        } catch {
            _ = replay(transaction.ops.reversed().map(\.inverse), guarded: false)
            throw error
        }
        let record = OpRecord(
            action: A.descriptor.name,
            actor: actor,
            summary: action.summary(in: store, output: output),
            ops: transaction.ops,
            state: pending ? .pending : .committed
        )
        record.receipt = action.receipt(in: store, output: output)
        if let attribution, actor == .tess {
            record.routeRaw = attribution.route.rawValue
            record.model = attribution.model
        }
        // Which zone he was in, so a day can be read where it happened (a New York session stays Wednesday's).
        record.zone = zone().identifier
        // A mount that changed nothing writes nothing (the record is returned but never stored).
        if (transaction.ops.isEmpty && !action.logsWhenEmpty) || !action.recordsOps {
            return Result(record: record, output: output)
        }
        store.context.insert(record)
        if !pending {
            notify(record)
        }
        return Result(record: record, output: output)
    }

    /// Decodes a tool call (from Tess, Claude or an intent) and runs it. Dates are ISO 8601.
    @discardableResult
    public func perform(named name: String, arguments: Data, by actor: Actor,
                        pending: Bool = false) throws -> OpRecord
    {
        let action = try ActionRegistry.decode(name: name, arguments: arguments)
        return try performErased(action, by: actor, pending: pending)
    }

    private func performErased(_ action: some EngineAction, by actor: Actor, pending: Bool) throws -> OpRecord {
        try perform(action, by: actor, pending: pending).record
    }

    // MARK: Observers

    /// Called on the main actor for every action once it's committed (straight away, or when a preview commits).
    /// Integrations hang off this: Apple Health saves a workout when `finishWorkout` lands.
    public func observe(_ observer: @escaping @MainActor (OpRecord) -> Void) {
        observers.append(observer)
    }

    private var observers: [@MainActor (OpRecord) -> Void] = []

    private func notify(_ record: OpRecord) {
        for observer in observers {
            observer(record)
        }
    }

    // MARK: Pending previews

    public func commit(_ record: OpRecord) throws {
        guard record.state == .pending else { throw ActionError.notPending(record.id) }
        record.state = .committed
        notify(record)
    }

    /// Reverts a preview the person dismissed.
    @discardableResult
    public func discard(_ record: OpRecord) throws -> ReplayOutcome {
        guard record.state == .pending else { throw ActionError.notPending(record.id) }
        return undo(record)
    }

    /// Commits every preview older than the grace period, except a room patch that widens reach (it waits for Allow).
    /// Call it from a timer or on the next interaction.
    public func commitExpired(now: Date = .now) {
        for record in pendingRecords()
            where now.timeIntervalSince(record.createdAt) >= pendingGracePeriod && !RoomReach.holds(record, in: store)
        {
            record.state = .committed
            notify(record)
        }
    }

    public func pendingRecords() -> [OpRecord] {
        store.context.all(OpRecord.self)
            .filter { $0.state == .pending }
            .sorted { $0.createdAt < $1.createdAt }
    }

    /// Whether an object only exists or changed because of a preview (draw it as a ghost).
    public func isPending(_ objectID: UUID) -> Bool {
        pendingRecords().contains { $0.objectIDs.contains(objectID) }
    }

    // MARK: Undo & redo

    /// Reverts an action. Works on any committed or pending record, not only the latest.
    @discardableResult
    public func undo(_ record: OpRecord) -> ReplayOutcome {
        guard record.state != .undone else { return .nothingToDo }
        let outcome = replay(record.ops.reversed().map(\.inverse), guarded: true)
        record.state = .undone
        return outcome
    }

    @discardableResult
    public func redo(_ record: OpRecord) -> ReplayOutcome {
        guard record.state == .undone else { return .nothingToDo }
        let outcome = replay(record.ops, guarded: true)
        record.state = .committed
        return outcome
    }

    /// Undoes the latest action, optionally only one by a given actor ("undo what Tess did") or of a given
    /// kind ("undo the last set" from the Lock Screen: `undoLatest(action: LogSet.descriptor.name)`).
    @discardableResult
    public func undoLatest(by actor: Actor? = nil, action: String? = nil) -> OpRecord? {
        // With neither an actor nor an action named, the app's own housekeeping (the 04:00 roll, a wrap-up) is skipped:
        // Undo takes back
        // what he did, never the roll that ran at 04:00 (QA D1).
        let match = { (record: OpRecord) in
            record.state != .undone
                && (actor.map { record.actor == $0 } ?? (action != nil || record.actor != .system))
                && (action == nil || record.action == action)
        }
        guard let record = history(limit: 200).first(where: match) else { return nil }
        undo(record)
        return record
    }

    /// A set taken back by `takeBackLatestSet()`: the record that undid it and its number in its exercise.
    public struct TakenBackSet {
        public let record: OpRecord
        /// 1-based, counted among the exercise's sets before it went.
        public let number: Int
    }

    /// Takes back exactly one set: the newest one logged (typed, the prefill hold, the Lock Screen's Done, Tess) in
    /// the workout that's still going. Never reaches into a finished workout, and never takes more than that set:
    /// when the action that logged it also made other things (Quick Capture's "3x10 bench 80" logs three sets and
    /// may start the workout), only the newest of its sets is deleted, through the op log, and the rest stays.
    @discardableResult
    public func takeBackLatestSet(by actor: Actor = .intent) -> TakenBackSet? {
        guard let setClass = classes[.set], let workoutClass = classes[.workout] else { return nil }
        for record in history(limit: 300) where record.state != .undone {
            var created = 0
            var newest: ObjectRecord?
            for op in record.ops {
                guard case let .create(objectID, classID, _, _) = op else { continue }
                created += 1
                guard let classDef = store.classDef(id: classID), store.isSubclass(classDef, ofClassID: setClass.id),
                      let set = store.object(id: objectID), !set.isDeleted else { continue }
                newest = set
            }
            guard let set = newest, let workout = store.ancestor(of: set, ofClass: workoutClass) else { continue }
            guard store.value("endedAt", of: workout) == nil else { return nil }
            let siblings = set.parent?.orderedChildren.filter { !$0.isDeleted } ?? [set]
            let number = (siblings.firstIndex { $0.id == set.id } ?? siblings.count - 1) + 1
            if created == 1 {
                undo(record)
                return TakenBackSet(record: record, number: number)
            }
            guard let deletion = try? perform(DeleteObject(object: set.id), by: actor) else { return nil }
            return TakenBackSet(record: deletion.record, number: number)
        }
        return nil
    }

    /// `takeBackLatestSet()`, returning only the record.
    @discardableResult
    public func undoLatestSet() -> OpRecord? {
        takeBackLatestSet()?.record
    }

    // MARK: History

    /// Newest first.
    public func history(limit: Int = 100) -> [OpRecord] {
        Array(newestFirst().prefix(limit))
    }

    /// Records made in `start ..< end`, newest first. Filters before it sorts, so one day's read never sorts years of
    /// history (the Day log's window, QA S1 on #318).
    public func history(from start: Date, to end: Date) -> [OpRecord] {
        store.context.operations.values.filter { $0.createdAt >= start && $0.createdAt < end }
            .sorted { ($0.createdAt, $0.id.uuidString) > ($1.createdAt, $1.id.uuidString) }
    }

    /// Actions that touched an object, newest first (for the object page's history scrubber).
    public func history(of objectID: UUID, limit: Int = 100) -> [OpRecord] {
        Array(newestFirst().lazy.filter { $0.objectIDs.contains(objectID) }.prefix(limit))
    }

    private func newestFirst() -> [OpRecord] {
        store.context.all(OpRecord.self).sorted { ($0.createdAt, $0.id.uuidString) > ($1.createdAt, $1.id.uuidString) }
    }

    // MARK: Replay

    private func replay(_ ops: [Op], guarded: Bool) -> ReplayOutcome {
        var skipped: [Op] = []
        for operation in ops where !apply(operation, guarded: guarded) {
            skipped.append(operation)
        }
        return skipped.isEmpty ? .applied : .partial(skipped: skipped)
    }

    /// Applies one op. When guarded, only if the current state still matches the op's "before".
    private func apply(_ operation: Op, guarded: Bool) -> Bool {
        // Classes are not objects: the two class ops are applied here.
        switch operation {
        case let .createClass(classID, parent, name, symbol, color):
            guard store.classDef(id: classID) == nil else { return !guarded }
            store.createClass(name: name, parent: parent.flatMap { store.classDef(id: $0) }, symbolName: symbol,
                              colorRole: color, id: classID)
            return true
        case let .deleteClass(classID, _, _, _, _):
            guard let classDef = store.classDef(id: classID) else { return !guarded }
            return store.deleteClass(classDef)
        default:
            break
        }
        guard let object = store.object(id: operation.objectID) else { return false }
        switch operation {
        case .create, .restore:
            // Redoing a creation brings the same object back.
            return setDeleted(false, object, guarded: guarded)
        case .softDelete:
            return setDeleted(true, object, guarded: guarded)
        case let .setSortIndex(_, old, new):
            if guarded, object.sortIndex != old {
                return false
            }
            object.sortIndex = new
            store.noteChange(object)
        case let .setTitle(_, old, new):
            if guarded, object.title != old {
                return false
            }
            object.title = new
            object.updatedAt = .now
            store.noteChange(object)
        case let .setValue(_, field, old, new):
            if guarded, object.values[field] != old {
                return false
            }
            store.setRaw(fieldID: field, to: new, on: object)
        case let .setLinks(_, field, old, new):
            if guarded, store.ownLinkTargets(fieldID: field, of: object) != old {
                return false
            }
            store.setLinksRaw(fieldID: field, on: object, to: new)
        case .createClass, .deleteClass:
            return false
        }
        return true
    }

    private func setDeleted(_ deleted: Bool, _ object: ObjectRecord, guarded: Bool) -> Bool {
        guard object.isDeleted != deleted else { return !guarded }
        if deleted {
            store.softDelete(object)
        } else {
            store.restore(object)
        }
        return true
    }
}
