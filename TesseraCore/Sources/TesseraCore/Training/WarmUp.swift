import Foundation

/// The warm-up before a routine's first lift (sheet 230): "Rower", 8 minutes, "Keep it easy, about 22 strokes a
/// minute."
/// It is three fields on the routine; empty is no warm-up. A private routine has none to show.
public struct WarmUp: Equatable, Sendable {
    public static let minutesRange = 1 ... 30
    public static let nameLimit = 30
    public static let noteLimit = 120

    public let name: String
    public let minutes: Int
    public let note: String?

    /// "8 min on the rower": the phrase the landing brief puts the minutes in.
    public var phrase: String {
        "\(minutes) min on the \(name.lowercased())"
    }
}

@MainActor
public extension TrainingStats {
    /// A routine's warm-up, or nil: none set, or the routine is private (it shows as "Private workout" and gives
    /// nothing).
    func warmUp(of routine: ObjectRecord) -> WarmUp? {
        guard ObjectPrivacy.shownLabel(of: routine, in: store, classes: classes) == nil,
              let name = store.value("warmupName", of: routine)?.stringValue?.trimmingCharacters(in: .whitespaces),
              !name.isEmpty, let minutes = store.number("warmupMinutes", of: routine).map({ Int($0) }),
              WarmUp.minutesRange.contains(minutes) else { return nil }
        let note = store.value("warmupNote", of: routine)?.stringValue?.trimmingCharacters(in: .whitespaces)
        return WarmUp(name: name, minutes: minutes, note: note?.isEmpty == false ? note : nil)
    }
}

/// Sets, changes or clears a routine's warm-up. A name that holds a secret word, a body fact or a key is refused.
public struct SetWarmUp: EngineAction {
    public static let descriptor = ActionDescriptor(
        name: "setWarmUp", summary: "Set a routine's warm-up",
        parameters: [
            .init("routine", .object, "The routine"),
            .init("name", .string, "What it is, e.g. Rower; empty clears the warm-up", required: false),
            .init("minutes", .integer, "1 to 30", required: false),
            .init(
                "note",
                .string,
                "A line for the warm-up screen; left out keeps the one set, empty takes it off",
                required: false
            ),
        ], isOfferedToModel: false
    )

    public var routine: UUID
    public var name: String?
    public var minutes: Int?
    public var note: String?

    public init(routine: UUID, name: String?, minutes: Int? = nil, note: String? = nil) {
        self.routine = routine
        self.name = name
        self.minutes = minutes
        self.note = note
    }

    public func apply(_ transaction: inout Transaction) throws {
        let object = try transaction.object(routine, is: .routine)
        let action = Self.descriptor.name
        let trimmed = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            for key in ["warmupName", "warmupMinutes", "warmupNote"] {
                try transaction.set(key, to: nil, on: object)
            }
            return
        }
        let line = (note ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let last = BodyWeight.lastKg(in: transaction.store)
        guard trimmed.count <= WarmUp.nameLimit, line.count <= WarmUp.noteLimit,
              !looksLikeClaudeKey(trimmed), !looksLikeClaudeKey(line),
              !ObjectPrivacy.isPrivateTitle(trimmed, lastWeightKg: last),
              !ObjectPrivacy.isPrivateTitle(line, lastWeightKg: last)
        else { throw ActionError.invalidArguments(action: action, reason: "refused: words") }
        // No minutes given: the ones already set stay.
        let kept = transaction.store.number("warmupMinutes", of: object).map { Int($0) }
        guard let minutes = minutes ?? kept, WarmUp.minutesRange.contains(minutes) else {
            throw ActionError.invalidArguments(action: action, reason: "refused: minutes")
        }
        try transaction.set("warmupName", to: .text(trimmed), on: object)
        try transaction.set("warmupMinutes", to: .number(Double(minutes)), on: object)
        // No note given: the one already set stays; an empty note takes it off.
        if note != nil {
            try transaction.set("warmupNote", to: line.isEmpty ? nil : .text(line), on: object)
        }
    }

    public func summary(in store: ObjectStore) -> String {
        "\(store.object(id: routine).map { ObjectPrivacy.shownLabel(of: $0, in: store, classes: PrivateWords.systemClasses(in: store.context)) ?? $0.title } ?? "A routine"), warm-up changed"
    }

    public func receipt(in _: ObjectStore) -> String? {
        let empty = (name ?? "").trimmingCharacters(in: .whitespaces).isEmpty
        return empty ? "Warm-up taken off" : minutes.map { "Warm-up set, \($0) min" } ?? "Warm-up changed"
    }
}
