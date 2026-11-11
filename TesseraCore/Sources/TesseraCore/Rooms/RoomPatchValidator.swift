import Foundation

/// Checks a patch before anything is written (R1 spec section 4). Pure engine code: it applies the patch to a copy of
/// the room, checks the room that would result, and either returns it or one plain reason. The first rejection goes
/// back to the model for one retry; a second is shown to him.
@MainActor
public struct RoomPatchValidator {
    public enum Outcome: Equatable {
        /// The room as it would be, and the grants this patch adds (a widening patch waits for his Allow).
        case ok(RoomDocument, widening: [String])
        case refused(String)
    }

    public static let moneyLine = "Rooms can't change money"

    let store: ObjectStore
    let classes: [SystemCatalog.Key: ClassDef]

    public init(store: ObjectStore, classes: [SystemCatalog.Key: ClassDef]) {
        self.store = store
        self.classes = classes
    }

    public func validate(_ patch: RoomPatch, against current: RoomDocument) -> Outcome {
        // A key never rides in a room, in any of its words.
        if let text = Self.allText(of: patch).first(where: looksLikeClaudeKey) {
            _ = text
            return .refused("That looks like a key, and a key never goes in a room")
        }
        var document = current
        var added: [String] = []
        for step in patch.ops {
            if let reason = apply(step, to: &document, added: &added) {
                return .refused(reason)
            }
        }
        // A write the parts make needs its grant, even when the patch did not name one.
        for does in document.parts.compactMap({ $0.kind == "actionWord" ? $0.text("does") : nil })
            where !document.reach.contains(does)
        {
            if let reason = checkGrant(does) {
                return .refused(reason)
            }
            document.reach.append(does)
            added.append(does)
        }
        document = Self.cleaned(document)
        if let reason = check(document) {
            return .refused(reason)
        }
        return .ok(document, widening: added.filter { !current.reach.contains($0) })
    }

    // MARK: Steps

    private func apply(_ step: RoomPatch.Step, to document: inout RoomDocument, added: inout [String]) -> String? {
        switch step.op {
        case "add":
            guard let id = step.id?.trimmingCharacters(in: .whitespaces), !id.isEmpty else {
                return "A new part needs an id"
            }
            guard document.part(id) == nil else { return "The part \(id) is already there, so set it instead" }
            guard let kind = step.kind, RoomKit.kind(kind) != nil else {
                return "\(step.kind ?? "That") isn't a kind of part this room has"
            }
            guard let place = step.place else { return "The part \(id) needs a place" }
            guard document.parts.count < RoomKit.maxParts else {
                return "A room holds at most \(RoomKit.maxParts) parts"
            }
            document.parts.append(RoomPart(id: id, kind: kind, place: place, props: step.props ?? [:],
                                           bind: step.bind?.isEmpty == false ? step.bind : nil))
            return nil
        case "set":
            guard let id = step.id, let index = document.parts.firstIndex(where: { $0.id == id }) else {
                return "There is no part called \(step.id ?? "that")"
            }
            for (key, value) in step.props ?? [:] {
                if value == .null {
                    document.parts[index].props[key] = nil
                } else {
                    document.parts[index].props[key] = value
                }
            }
            if let kind = step.kind {
                document.parts[index].kind = kind
            }
            if let place = step.place {
                document.parts[index].place = place
            }
            if let bind = step.bind {
                document.parts[index].bind = bind.isEmpty ? nil : bind
            }
            return nil
        case "remove":
            guard let id = step.id, document.part(id) != nil else {
                return "There is no part called \(step.id ?? "that") to remove"
            }
            document.parts.removeAll { $0.id == id }
            return nil
        case "move":
            guard let id = step.id, let index = document.parts.firstIndex(where: { $0.id == id }) else {
                return "There is no part called \(step.id ?? "that") to move"
            }
            var part = document.parts.remove(at: index)
            if let place = step.place {
                part.place = place
            }
            let sameplace = document.parts.indices.filter { document.parts[$0].place == part.place }
            let target = max(0, min(step.index ?? sameplace.count, sameplace.count))
            let insertAt = target < sameplace
                .count ? sameplace[target] : (sameplace.last.map { $0 + 1 } ?? document.parts.count)
            document.parts.insert(part, at: insertAt)
            return nil
        case "look":
            return applyLook(step, to: &document)
        case "grant":
            guard let grant = step.reach?.trimmingCharacters(in: .whitespaces), !grant.isEmpty else {
                return "A grant names what it lets the room do"
            }
            if let reason = checkGrant(grant) {
                return reason
            }
            if !document.reach.contains(grant) {
                document.reach.append(grant)
                added.append(grant)
            }
            return nil
        default:
            return "\(step.op) isn't something a patch can do"
        }
    }

    private func applyLook(_ step: RoomPatch.Step, to document: inout RoomDocument) -> String? {
        guard let trait = step.trait, let value = step.value else { return "A look change needs a trait and a value" }
        if trait == "color" {
            guard RoomKit.colorRoles.contains(value) else {
                return "\(value) isn't a color role; rooms use roles, never a hex"
            }
            document.color = value
            return nil
        }
        let values = Self.traitValues[trait]
        guard let values else { return "\(trait) isn't something a room's look can change" }
        guard values.contains(value) else {
            return "\(trait) takes \(values.sorted().joined(separator: ", ")), not \(value)"
        }
        document.look[trait] = value
        return nil
    }

    /// Every trait the studio drew, by name, with the values it takes in any place.
    static let traitValues: [String: Set<String>] = {
        var all: [String: Set<String>] = [:]
        for forms in LookCatalog.places.values {
            for trait in forms.traits {
                all[trait.name, default: []].formUnion(trait.values)
            }
        }
        return all
    }()

    // MARK: Grants

    /// "write Plant.lastWatered" or "read Plant"; money is never writable, and the class and field must exist.
    private func checkGrant(_ grant: String) -> String? {
        let words = grant.split(separator: " ", maxSplits: 1).map(String.init)
        guard words.count == 2, ["read", "write"].contains(words[0]) else {
            return "A grant reads \"read Class\" or \"write Class.field\""
        }
        let target = words[1].split(separator: ".", maxSplits: 1).map(String.init)
        guard let classDef = resolveClass(target[0]) else { return "There is no class called \(target[0])" }
        if words[0] == "write" {
            if isMoney(classDef) {
                return Self.moneyLine
            }
            guard target.count == 2 else { return "A write names the field, like write \(target[0]).field" }
            guard ClassResolver.field(target[1], of: classDef) != nil else {
                return "\(target[0]) has no field called \(target[1])"
            }
        }
        return nil
    }

    func resolveClass(_ name: String) -> ClassDef? {
        let clean = name.trimmingCharacters(in: .whitespaces)
        return store.classDef(named: clean)
            ?? store.allClasses().first { $0.name.caseInsensitiveCompare(clean) == .orderedSame }
    }

    func isMoney(_ classDef: ClassDef) -> Bool {
        RoomKit.moneyClassKeys.contains { key in
            classes[key].map { ClassResolver.isSubclass(classDef, of: $0) } ?? false
        }
    }

    // MARK: The room that would result

    private func check(_ document: RoomDocument) -> String? {
        // Color roles only, and a few.
        for role in document.colorsInUse where !RoomKit.colorRoles.contains(role) {
            return "\(role) isn't a color role; rooms use roles, never a hex"
        }
        if document.colorsInUse.count > RoomKit.maxColors {
            return "A room uses \(RoomKit.maxColors) colors at most"
        }
        for part in document.parts {
            if let reason = check(part, in: document) {
                return reason
            }
        }
        for place in RoomKit.singleOccupantPlaces
            where document.parts.filter({ $0.place == place }).count > 1
        {
            return "The \(place) place holds one part"
        }
        // Reach: reading a class is free, so only writes are listed; money is never among them.
        for grant in document.reach {
            if let reason = checkGrant(grant) {
                return reason
            }
        }
        return nil
    }

    private func check(_ part: RoomPart, in document: RoomDocument) -> String? {
        guard let kind = RoomKit.kind(part.kind) else { return "\(part.kind) isn't a kind of part this room has" }
        guard kind.places.contains(part.place) else {
            return "A \(part.kind) can't sit in the \(part.place) place; it takes \(kind.places.joined(separator: " or "))"
        }
        for key in part.props.keys where !kind.props.contains(key) {
            return "A \(part.kind) has no \(key) prop; it takes \(kind.props.joined(separator: ", "))"
        }
        let bound = part.bind.flatMap { resolveClass(RoomDocument.className(of: $0)) }
        if let bind = part.bind, bound == nil {
            return "There is no class called \(RoomDocument.className(of: bind))"
        }
        for (key, value) in part.props {
            if let reason = checkText(value, key: key, of: part) {
                return reason
            }
        }
        if let role = part.text("color"), !RoomKit.colorRoles.contains(role) {
            return "\(role) isn't a color role; rooms use roles, never a hex"
        }
        return checkProps(of: part, kind: kind, bound: bound, in: document)
    }

    /// Text: at most 80 characters a line; tokens must be known.
    private func checkText(_ value: RawJSON, key: String, of part: RoomPart) -> String? {
        switch value {
        case let .string(text):
            if let reason = lineLength(text) {
                return reason
            }
            if key == "text" || key == "label" || key == "verb" || key == "object" {
                return checkTokens(in: text, of: part)
            }
        case let .array(items):
            for item in items {
                if let reason = checkText(item, key: key, of: part) {
                    return reason
                }
            }
        case let .object(object):
            for (inner, item) in object {
                if let reason = checkText(item, key: inner, of: part) {
                    return reason
                }
            }
        default: break
        }
        return nil
    }

    private func lineLength(_ text: String) -> String? {
        text.count > RoomKit.maxText ? "A line holds \(RoomKit.maxText) characters at most" : nil
    }

    private func checkTokens(in text: String, of part: RoomPart) -> String? {
        let bound = part.bind.flatMap { resolveClass(RoomDocument.className(of: $0)) }
        var rest = Substring(text)
        while let open = rest.firstIndex(of: "{"), let close = rest[open...].firstIndex(of: "}") {
            let token = String(rest[rest.index(after: open) ..< close])
            if !["count", "top"].contains(token) {
                guard let bound else { return "{\(token)} needs the part to bind a class" }
                if ClassResolver.field(token, of: bound) == nil {
                    return "\(bound.name) has no field called \(token)"
                }
            }
            rest = rest[rest.index(after: close)...]
        }
        return nil
    }

    private func checkProps(of part: RoomPart, kind _: RoomKit.Kind, bound: ClassDef?, in document: RoomDocument)
        -> String?
    {
        func field(_ key: String) -> String? {
            guard let name = part.text(key) else { return nil }
            guard let bound else { return "A \(part.kind) needs a class to bind before it can show \(name)" }
            if name == "title" {
                return nil
            }
            return ClassResolver.field(name, of: bound) == nil ? "\(bound.name) has no field called \(name)" : nil
        }
        switch part.kind {
        case "number":
            if let reason = field("field") {
                return reason
            }
            if let agg = part.text("agg"), !RoomKit.aggregates.contains(agg) {
                return "agg is one of \(RoomKit.aggregates.joined(separator: ", "))"
            }
            if part.text("field") == nil, part.text("agg") != nil, part.text("agg") != "count" {
                return "That aggregate needs a field"
            }
            if part.bind == nil {
                return "A number needs a class to bind"
            }
        case "row", "list":
            if part.bind == nil {
                return "A \(part.kind) needs a class to bind"
            }
            if let reason = field("big") ?? field("small") {
                return reason
            }
            if let sort = part.text("sort"), let reason = checkSort(sort, bound: bound) {
                return reason
            }
            if case let .number(rows)? = part.props["max"] {
                let limit = part.kind == "row" ? 4 : RoomKit.maxRows
                if rows < 1 || rows > Double(limit) {
                    return "A \(part.kind) shows 1 to \(limit) rows"
                }
            }
        case "line":
            break
        case "lines":
            if case let .array(items)? = part.props["lines"], items.count > 2 {
                return "Lines holds two lines at most"
            }
        case "stack", "grid":
            if let reason = checkChildren(of: part, in: document) {
                return reason
            }
            if part.kind == "grid", case let .number(columns)? = part.props["columns"], !(2 ... 4).contains(columns) {
                return "A grid has 2 to 4 columns"
            }
            if part.kind == "stack", let axis = part.text("axis"), !["h", "v"].contains(axis) {
                return "A stack's axis is h or v"
            }
        case "glyph":
            if let symbol = part.text("symbol"), !RoomKit.glyphs.contains(symbol) {
                return "\(symbol) isn't in the vetted symbols"
            }
        case "doors":
            if case let .array(doors)? = part.props["doors"] {
                for case let .object(door) in doors {
                    if case let .string(bind)? = door["bind"],
                       resolveClass(RoomDocument.className(of: bind)) == nil
                    {
                        return "There is no class called \(RoomDocument.className(of: bind))"
                    }
                }
            }
        case "actionWord":
            if let does = part.text("does") {
                if let reason = checkGrant(does) {
                    return reason
                }
                if !does.hasPrefix("write ") {
                    return "An action word's does is a write, like write Plant.lastWatered"
                }
            }
        case "top":
            guard let target = part.text("on"), let on = document.part(target), ["row", "list"].contains(on.kind) else {
                return "A top highlight names the row or list it marks"
            }
            if let name = part.text("field") {
                guard let bound = on.bind.flatMap({ resolveClass(RoomDocument.className(of: $0)) }),
                      ClassResolver.field(name, of: bound) != nil
                else { return "\(on.id) has no field called \(name) to rank by" }
            }
        default: break
        }
        return nil
    }

    private func checkSort(_ sort: String, bound: ClassDef?) -> String? {
        var text = sort.hasPrefix("-") ? String(sort.dropFirst()) : sort
        if let open = text.firstIndex(of: "("), text.hasSuffix(")") {
            let function = String(text[..<open])
            guard RoomKit.sortFunctions.contains(function) else { return "\(function) isn't a sort function" }
            text = String(text[text.index(after: open) ..< text.index(before: text.endIndex)])
        }
        guard let bound else { return "A sort needs a class to bind" }
        if text == "title" {
            return nil
        }
        return ClassResolver.field(text, of: bound) == nil ? "\(bound.name) has no field called \(text)" : nil
    }

    private func checkChildren(of part: RoomPart, in document: RoomDocument) -> String? {
        guard case let .array(children)? = part.props["children"], !children.isEmpty else {
            return "A \(part.kind) names the parts it lays out"
        }
        for case let .string(id) in children {
            guard let child = document.part(id) else { return "There is no part called \(id) to lay out" }
            guard child.place == part.place else { return "\(id) sits in the \(child.place) place, not \(part.place)" }
            if child.id == part.id {
                return "A \(part.kind) can't hold itself"
            }
            if ["stack", "grid"].contains(child.kind) {
                return "A \(part.kind) can't hold another \(child.kind)"
            }
        }
        return nil
    }

    // MARK: Text of a patch

    static func allText(of patch: RoomPatch) -> [String] {
        var texts = [patch.room, patch.say]
        func collect(_ value: RawJSON) {
            switch value {
            case let .string(text): texts.append(text)
            case let .array(items): items.forEach(collect)
            case let .object(object): object.values.forEach(collect)
            default: break
            }
        }
        for step in patch.ops {
            texts.append(contentsOf: [step.id, step.kind, step.place, step.bind, step.trait, step.value, step.reach]
                .compactMap(\.self))
            step.props?.values.forEach(collect)
        }
        return texts
    }

    /// A shown line never ends in a full stop; the patch is cleaned before it is written.
    static func stripped(_ text: String) -> String {
        var line = text.trimmingCharacters(in: .whitespaces)
        while line.hasSuffix(".") || line.hasSuffix("。") {
            line.removeLast()
        }
        return line
    }

    /// Shown text with its trailing full stop taken off.
    static func cleaned(_ document: RoomDocument) -> RoomDocument {
        var document = document
        for index in document.parts.indices {
            for key in ["text", "label", "verb", "object"] {
                if case let .string(text)? = document.parts[index].props[key] {
                    document.parts[index].props[key] = .string(stripped(text))
                }
            }
            if case let .array(items)? = document.parts[index].props["lines"] {
                document.parts[index].props["lines"] = .array(items.map { item in
                    if case let .string(text) = item {
                        .string(stripped(text))
                    } else {
                        item
                    }
                })
            }
        }
        return document
    }
}
