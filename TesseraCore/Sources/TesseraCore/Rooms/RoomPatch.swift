import Foundation

/// What `patchRoom` carries (R1 spec section 2): one sentence's change to a room, as ops on named parts.
public struct RoomPatch: Codable, Equatable, Sendable {
    public struct Step: Codable, Equatable, Sendable {
        /// add, set, remove, move, look or grant.
        public var op: String
        public var id: String?
        public var kind: String?
        public var place: String?
        public var props: [String: RawJSON]?
        public var bind: String?
        public var trait: String?
        public var value: String?
        /// "write Plant.lastWatered"
        public var reach: String?
        public var index: Int?

        public init(op: String, id: String? = nil, kind: String? = nil, place: String? = nil,
                    props: [String: RawJSON]? = nil, bind: String? = nil, trait: String? = nil, value: String? = nil,
                    reach: String? = nil, index: Int? = nil)
        {
            self.op = op
            self.id = id
            self.kind = kind
            self.place = place
            self.props = props
            self.bind = bind
            self.trait = trait
            self.value = value
            self.reach = reach
            self.index = index
        }
    }

    /// The room's name, e.g. "Chart room".
    public var room: String
    /// The receipt line, with no full stop at the end.
    public var say: String
    public var ops: [Step]

    public init(room: String, say: String, ops: [Step]) {
        self.room = room
        self.say = say
        self.ops = ops
    }

    public static func parse(_ data: Data) throws -> RoomPatch {
        try JSONDecoder().decode(RoomPatch.self, from: data)
    }
}
