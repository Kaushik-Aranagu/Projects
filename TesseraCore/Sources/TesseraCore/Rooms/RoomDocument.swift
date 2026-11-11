import Foundation

// A room as plain data (R1 spec section 1). The objects are the truth (a `room` with `roomPart` children); this is
// the shape a screen and the validator read, and the shape of the fixture Builder A builds against.

public struct RoomPart: Codable, Equatable, Sendable {
    /// The part's stable id in its room (its object title), so later sentences can name it.
    public var id: String
    public var kind: String
    public var place: String
    public var props: [String: RawJSON]
    /// A class name (optionally a query) whose objects the part shows; resolved at render time, never copied.
    public var bind: String?

    public init(id: String, kind: String, place: String, props: [String: RawJSON] = [:], bind: String? = nil) {
        self.id = id
        self.kind = kind
        self.place = place
        self.props = props
        self.bind = bind
    }

    public func text(_ key: String) -> String? {
        if case let .string(value)? = props[key] {
            value
        } else {
            nil
        }
    }
}

public struct RoomDocument: Codable, Equatable, Sendable {
    public var name: String
    /// A color role from `RoomKit.colorRoles`.
    public var color: String
    /// LookCatalog trait values, e.g. density: airy.
    public var look: [String: String]
    /// Grants beyond reading the classes the parts bind, e.g. "write Plant.lastWatered".
    public var reach: [String]
    /// In draw order; within a place, the order is the order here.
    public var parts: [RoomPart]

    public init(name: String, color: String = "slate", look: [String: String] = [:], reach: [String] = [],
                parts: [RoomPart] = [])
    {
        self.name = name
        self.color = color
        self.look = look
        self.reach = reach
        self.parts = parts
    }

    public func part(_ id: String) -> RoomPart? {
        parts.first { $0.id == id }
    }

    /// The classes this room reads: every bound class (and the binds of its doors).
    public var boundClasses: [String] {
        var names: [String] = []
        for part in parts {
            if let bind = part.bind {
                names.append(Self.className(of: bind))
            }
            if case let .array(doors)? = part.props["doors"] {
                for case let .object(door) in doors {
                    if case let .string(bind)? = door["bind"] {
                        names.append(Self.className(of: bind))
                    }
                }
            }
        }
        return names
    }

    /// "Plant" or "Plant where ..." is the class "Plant".
    static func className(of bind: String) -> String {
        bind.split(separator: " ", maxSplits: 1).first.map(String.init) ?? bind
    }

    /// The color roles in use: the room's own and every part's.
    public var colorsInUse: Set<String> {
        var roles: Set<String> = [color]
        for part in parts {
            if let role = part.text("color") {
                roles.insert(role)
            }
        }
        return roles
    }
}

public extension RoomDocument {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    func json() throws -> Data {
        try Self.encoder.encode(self)
    }
}

/// Reads and writes rooms as objects.
@MainActor
public struct RoomBook {
    public let store: ObjectStore
    public let classes: [SystemCatalog.Key: ClassDef]

    public init(store: ObjectStore, classes: [SystemCatalog.Key: ClassDef]) {
        self.store = store
        self.classes = classes
    }

    public func rooms() -> [ObjectRecord] {
        guard let roomClass = classes[.room] else { return [] }
        return store.objects(of: roomClass)
    }

    public func room(named name: String) -> ObjectRecord? {
        rooms().first { $0.title.caseInsensitiveCompare(name) == .orderedSame }
    }

    public func document(of room: ObjectRecord) -> RoomDocument {
        let look = decode([String: String].self, store.value("look", of: room)?.stringValue) ?? [:]
        let reach = decode([String].self, store.value("reach", of: room)?.stringValue) ?? []
        let parts = room.orderedChildren.compactMap { child -> RoomPart? in
            guard let kind = store.value("kind", of: child)?.stringValue,
                  let place = store.value("place", of: child)?.stringValue else { return nil }
            let props = decode([String: RawJSON].self, store.value("props", of: child)?.stringValue) ?? [:]
            let bind = store.value("bind", of: child)?.stringValue
            return RoomPart(id: child.title, kind: kind, place: place, props: props,
                            bind: bind?.isEmpty == false ? bind : nil)
        }
        return RoomDocument(name: store.value("name", of: room)?.stringValue ?? room.title,
                            color: store.value("color", of: room)?.stringValue ?? "slate",
                            look: look, reach: reach, parts: parts)
    }

    private func decode<T: Decodable>(_ type: T.Type, _ text: String?) -> T? {
        text.flatMap { try? JSONDecoder().decode(type, from: Data($0.utf8)) }
    }
}

public extension RoomDocument {
    /// The compact JSON text a field holds (sorted keys, so the same value is the same text).
    static func text(_ value: some Encodable) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return (try? String(data: encoder.encode(value), encoding: .utf8)) ?? ""
    }
}
