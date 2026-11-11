import Foundation

// patchRoom (R1 spec sections 1 and 2): one sentence's change to a room, written as create, setValue, softDelete and
// setSortIndex in one op group, pending as Tess (the ghost). The validator runs inside `apply`, so nothing invalid is
// ever written. A patch that widens the room's reach waits for his Allow: `RoomReach.holds` keeps the engine from
// committing it on the drying line.

public struct PatchRoom: EngineAction {
    public struct Output: Equatable, Sendable {
        public let room: UUID
        /// The receipt line, no full stop.
        public let say: String
        /// The grants this patch adds; non-empty means the patch waits for Allow.
        public let widening: [String]
    }

    public static let descriptor = ActionDescriptor(
        name: "patchRoom", summary: "Change a room Tess built",
        parameters: [.init("patch", .json, "room, say and ops (add, set, remove, move, look, grant)")],
        isOfferedToModel: false
    )

    public var patch: RoomPatch

    public init(patch: RoomPatch) {
        self.patch = patch
    }

    public func apply(_ transaction: inout Transaction) throws -> Output {
        let store = transaction.store
        let book = RoomBook(store: store, classes: transaction.classes)
        let validator = RoomPatchValidator(store: store, classes: transaction.classes)
        let roomObject = book.room(named: patch.room)
        let current = roomObject.map(book.document(of:)) ?? RoomDocument(name: patch.room)
        switch validator.validate(patch, against: current) {
        case let .refused(reason):
            throw ActionError.invalidArguments(action: Self.descriptor.name, reason: reason)
        case let .ok(document, widening):
            let room = try roomObject ?? transaction.create(transaction.classDef(.room), title: patch.room)
            try write(document, over: current, on: room, in: &transaction)
            return Output(room: room.id, say: RoomPatchValidator.stripped(patch.say), widening: widening)
        }
    }

    /// Writes only what differs, so undo puts back exactly the old room.
    @MainActor private func write(_ document: RoomDocument, over current: RoomDocument, on room: ObjectRecord,
                                  in transaction: inout Transaction) throws
    {
        let store = transaction.store
        if document.name != current.name || store.value("name", of: room) == nil {
            try transaction.set("name", to: .text(document.name), on: room)
        }
        if document.color != current.color || store.value("color", of: room) == nil {
            try transaction.set("color", to: .choice(document.color), on: room)
        }
        if document.look != current.look || store.value("look", of: room) == nil {
            try transaction.set("look", to: .text(RoomDocument.text(document.look)), on: room)
        }
        if document.reach != current.reach || store.value("reach", of: room) == nil {
            try transaction.set("reach", to: .text(RoomDocument.text(document.reach)), on: room)
        }
        let partClass = try transaction.classDef(.roomPart)
        var existing: [String: ObjectRecord] = [:]
        for child in room.orderedChildren {
            existing[child.title] = child
        }
        let kept = Set(document.parts.map(\.id))
        for (id, object) in existing where !kept.contains(id) {
            transaction.delete(object)
        }
        for (index, part) in document.parts.enumerated() {
            let object: ObjectRecord = if let found = existing[part.id] {
                found
            } else {
                transaction.create(partClass, title: part.id, parent: room)
            }
            if current.part(part.id) != part || existing[part.id] == nil {
                try transaction.set("kind", to: .choice(part.kind), on: object)
                try transaction.set("place", to: .choice(part.place), on: object)
                try transaction.set("props", to: .text(RoomDocument.text(part.props)), on: object)
                try transaction.set("bind", to: part.bind.map { .text($0) }, on: object)
            }
            transaction.setSortIndex(index, on: object)
        }
    }

    public func summary(in _: ObjectStore) -> String {
        "Changed \(patch.room)"
    }

    public func receipt(in _: ObjectStore, output: Output) -> String? {
        output.say
    }

    public func receipt(in _: ObjectStore) -> String? {
        nil
    }
}

/// The reach a room has been allowed, and the hold a widening patch is under.
@MainActor
public enum RoomReach {
    /// True for a pending patch that changes a room's reach: it waits for Allow and never commits on the drying line.
    public static func holds(_ record: OpRecord, in store: ObjectStore) -> Bool {
        guard record.state == .pending, record.action == PatchRoom.descriptor.name else { return false }
        return record.ops.contains { op in
            guard case let .setValue(objectID, fieldID, old, new) = op,
                  let object = store.object(id: objectID),
                  object.classDef?.systemKey == SystemCatalog.Key.room.systemKey,
                  let field = object.classDef.flatMap({ store.field("reach", of: $0) }),
                  field.id == fieldID
            else { return false }
            func grants(_ value: FieldValue?) -> Set<String> {
                let text = value?.stringValue ?? "[]"
                return Set((try? JSONDecoder().decode([String].self, from: Data(text.utf8))) ?? [])
            }
            // Only a grant the room did not have before widens it; setting an empty reach does not.
            return !grants(new).subtracting(grants(old)).isEmpty
        }
    }

    /// Allow: the held patch commits with its grants. Undo on its receipt reverts the patch and the grants together.
    @discardableResult
    public static func allow(_ record: OpRecord, in engine: ActionEngine) -> Bool {
        guard holds(record, in: engine.store) else { return false }
        return (try? engine.commit(record)) != nil
    }
}

public extension RoomDocument {
    /// A short sentence for what a grant lets the room do ("write Plant.lastWatered" is "change Plant lastWatered").
    static func words(for grant: String) -> String {
        let parts = grant.split(separator: " ", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return grant }
        let target = parts[1].replacingOccurrences(of: ".", with: " ")
        return parts[0] == "write" ? "change \(target)" : "read \(target)"
    }
}
