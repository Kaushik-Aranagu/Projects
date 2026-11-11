import Foundation
@testable import TesseraCore
import Testing

/// The Chart room's classes, ops and validator (R1 spec sections 1, 4 and 7): one case per rule, op-group round trips,
/// the drying line, and the hold a widening patch is under.
@MainActor
struct RoomPatchTests {
    let store = ObjectStore.open(inMemory: true)
    let classes: [SystemCatalog.Key: ClassDef]
    let engine: ActionEngine

    init() throws {
        classes = try store.prepareCatalog()
        engine = ActionEngine(store: store, classes: classes)
        let thing = try #require(classes[.thing])
        let plant = store.createClass(name: "Plant", parent: thing)
        for (index, (key, kind)) in [("lastWatered", FieldKind.date), ("waterEvery", .number)].enumerated() {
            let def = FieldDef(name: key, key: key, kind: kind, sortIndex: index)
            def.owner = plant
            store.context.insert(def)
        }
        store.invalidateCaches()
    }

    var validator: RoomPatchValidator {
        RoomPatchValidator(store: store, classes: classes)
    }

    var book: RoomBook {
        RoomBook(store: store, classes: classes)
    }

    func patch(_ steps: RoomPatch.Step...) -> RoomPatch {
        RoomPatch(room: "Chart room", say: "Done", ops: steps)
    }

    func add(_ id: String, _ kind: String, _ place: String, _ props: [String: RawJSON] = [:], bind: String? = nil)
        -> RoomPatch.Step
    {
        RoomPatch.Step(op: "add", id: id, kind: kind, place: place, props: props, bind: bind)
    }

    func refusal(_ patch: RoomPatch, from document: RoomDocument = RoomDocument(name: "Chart room")) -> String? {
        if case let .refused(reason) = validator.validate(patch, against: document) {
            reason
        } else {
            nil
        }
    }

    // MARK: The fixture

    @Test func theFixtureIsAValidRoomAndBuildsFromOneBigPatch() {
        let steps = RoomFixture.chart.parts.map {
            RoomPatch.Step(op: "add", id: $0.id, kind: $0.kind, place: $0.place, props: $0.props, bind: $0.bind)
        } + [RoomPatch.Step(op: "look", trait: "density", value: "airy"),
             RoomPatch.Step(op: "look", trait: "motion", value: "calm")]
        let result = validator.validate(RoomPatch(room: "Chart room", say: "Plants, set", ops: steps),
                                        against: RoomDocument(name: "Chart room"))
        guard case let .ok(document, widening) = result else {
            Issue.record("\(result)")
            return
        }
        #expect(document.parts == RoomFixture.chart.parts)
        #expect(document.look == RoomFixture.chart.look)
        #expect(widening == ["write Plant.lastWatered"])
        #expect(document.colorsInUse.count <= RoomKit.maxColors)
    }

    @Test func theFixtureFileIsTheFixture() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let file = root.appendingPathComponent("docs/design/tess-built-rooms/fixture-chart-room.json")
        let wanted = try RoomDocument.encoder.encode(RoomFixture.chart)
        if ProcessInfo.processInfo.environment["TESSERA_WRITE_FIXTURE"] == "1" {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try wanted.write(to: file)
        }
        let onDisk = try Data(contentsOf: file)
        #expect(try JSONDecoder().decode(RoomDocument.self, from: onDisk) == RoomFixture.chart)
    }

    // MARK: One case per rule

    @Test func kindsAndPlaces() {
        #expect(refusal(patch(add("a", "carousel", "widget")))?.contains("isn't a kind") == true)
        #expect(refusal(patch(add("a", "title", "widget")))?.contains("can't sit") == true)
        #expect(refusal(patch(add("a", "title", "title"), add("b", "title", "title")))?.contains("holds one") == true)
        let many = (0 ... RoomKit.maxParts).map { add("p\($0)", "line", "caption", ["text": .string("x")]) }
        #expect(refusal(RoomPatch(room: "Chart room", say: "x", ops: many))?.contains("at most 24") == true)
    }

    @Test func props() {
        #expect(refusal(patch(add("a", "title", "title", ["size": .number(9)])))?.contains("no size prop") == true)
        let long = String(repeating: "x", count: 81)
        #expect(refusal(patch(add("a", "title", "title", ["text": .string(long)])))?.contains("80 characters") == true)
        #expect(refusal(patch(add("a", "line", "caption", ["text": .string("{nope}")])))?
            .contains("needs the part") == true)
        #expect(refusal(patch(add("a", "line", "caption", ["text": .string("{nope}")], bind: "Plant")))?
            .contains("no field called nope") == true)
        #expect(refusal(patch(add(
            "a",
            "lines",
            "needs",
            ["lines": .array([.string("a"), .string("b"), .string("c")])]
        )))?
            .contains("two lines") == true)
        #expect(refusal(patch(add("a", "glyph", "glyph", ["symbol": .string("skull")])))?.contains("vetted") == true)
    }

    @Test func aTrailingFullStopIsStripped() {
        let result = validator.validate(patch(add("a", "title", "title", ["text": .string("Plants.")])),
                                        against: RoomDocument(name: "Chart room"))
        guard case let .ok(document, _) = result else {
            Issue.record("\(result)")
            return
        }
        #expect(document.part("a")?.text("text") == "Plants")
    }

    @Test func aKeyIsRefusedInAnyWord() {
        let key = "sk-ant-api03-" + String(repeating: "a1", count: 12)
        #expect(refusal(patch(add("a", "title", "title", ["text": .string(key)])))?.contains("key") == true)
        #expect(refusal(RoomPatch(room: "Chart room", say: key, ops: []))?.contains("key") == true)
    }

    @Test func lookAndColor() {
        #expect(refusal(patch(.init(op: "look", trait: "color", value: "#FF0000")))?.contains("role") == true)
        #expect(refusal(patch(.init(op: "look", trait: "sparkle", value: "lots")))?.contains("can change") == true)
        #expect(refusal(patch(.init(op: "look", trait: "motion", value: "wild")))?.contains("takes") == true)
        #expect(refusal(patch(add("a", "glyph", "glyph", ["symbol": .string("leaf"), "color": .string("#fff")])))?
            .contains("role") == true)
        let colors = ["ember", "wine", "cobalt", "sage"].enumerated().map { index, role in
            add("g\(index)", "title", index == 0 ? "title" : "caption", ["color": .string(role)])
        }
        #expect(refusal(patch(
            .init(op: "look", trait: "color", value: "violet"),
            colors[0],
            colors[1],
            colors[2],
            colors[3]
        ))
            != nil)
        guard case .ok = validator.validate(patch(.init(op: "look", trait: "motion", value: "calm")),
                                            against: RoomDocument(name: "Chart room"))
        else {
            Issue.record("a calm look is allowed")
            return
        }
    }

    @Test func binds() {
        #expect(refusal(patch(add("a", "number", "keystone", ["agg": .string("count")], bind: "Pet")))?
            .contains("no class called Pet") == true)
        #expect(refusal(patch(add("a", "row", "widget", ["big": .string("height")], bind: "Plant")))?
            .contains("no field called height") == true)
        #expect(refusal(patch(add("a", "row", "widget", ["sort": .string("-cube(lastWatered)")], bind: "Plant")))?
            .contains("sort function") == true)
        #expect(refusal(patch(add("a", "number", "keystone", ["agg": .string("count")])))?.contains("bind") == true)
        #expect(refusal(patch(add("a", "row", "widget", ["max": .number(9)], bind: "Plant")))?
            .contains("1 to 4") == true)
    }

    @Test func layoutAndModifiers() {
        let one = add("n1", "number", "keystone", ["agg": .string("count")], bind: "Plant")
        #expect(refusal(patch(one, add("s", "stack", "widget", ["children": .array([.string("n1")])])))?
            .contains("not widget") == true)
        #expect(refusal(patch(one, add("g", "grid", "keystone", ["children": .array([.string("n1")]),
                                                                 "columns": .number(5)])))?.contains("2 to 4") == true)
        #expect(refusal(patch(add("t", "top", "widget", ["on": .string("nothing")])))?.contains("row or list") == true)
    }

    @Test func reachAndMoney() {
        #expect(refusal(patch(.init(op: "grant", reach: "write Pool.name")))?
            .contains(RoomPatchValidator.moneyLine) == true)
        #expect(refusal(patch(.init(op: "grant", reach: "write Spend.amount")))?
            .contains(RoomPatchValidator.moneyLine) == true)
        #expect(refusal(patch(add("w", "actionWord", "actionWord", ["does": .string("write Pool.name")])))?
            .contains(RoomPatchValidator.moneyLine) == true)
        #expect(refusal(patch(.init(op: "grant", reach: "write Plant.height")))?.contains("no field") == true)
        #expect(refusal(patch(.init(op: "grant", reach: "delete Plant")))?.contains("grant reads") == true)
        // An action word's write is a grant the patch is not allowed to leave out.
        guard case let .ok(document, widening) = validator.validate(
            patch(add("w", "actionWord", "actionWord", ["does": .string("write Plant.lastWatered")])),
            against: RoomDocument(name: "Chart room")
        ) else {
            Issue.record("a plant write is allowed")
            return
        }
        #expect(document.reach == ["write Plant.lastWatered"] && widening == ["write Plant.lastWatered"])
    }

    @Test func stepsNameTheirParts() {
        #expect(refusal(patch(.init(op: "set", id: "nope", props: [:])))?.contains("no part called nope") == true)
        #expect(refusal(patch(.init(op: "remove", id: "nope")))?.contains("to remove") == true)
        #expect(refusal(patch(.init(op: "move", id: "nope", index: 0)))?.contains("to move") == true)
        #expect(refusal(patch(.init(op: "explode")))?.contains("can do") == true)
        #expect(refusal(patch(add("a", "title", "title", ["text": .string("x")]),
                              add("a", "title", "title", ["text": .string("y")])))?.contains("already there") == true)
    }

    // MARK: Writing

    func fingerprint(_ room: ObjectRecord?) -> RoomDocument? {
        room.map { book.document(of: $0) }
    }

    @Test func aPatchIsOnePendingGroupThatUndoesAndRedoesToTheSameRoom() throws {
        let sentence = patch(add("title", "title", "title", ["text": .string("Plants")]),
                             add("line", "line", "switcher", ["text": .string("{count} plants")], bind: "Plant"),
                             .init(op: "look", trait: "motion", value: "calm"))
        let result = try engine.perform(PatchRoom(patch: sentence), by: .tess, pending: true)
        #expect(result.record.state == .pending && result.output.widening.isEmpty)
        let room = try #require(book.room(named: "Chart room"))
        let built = book.document(of: room)
        #expect(built.parts.map(\.id) == ["title", "line"] && built.look == ["motion": "calm"])
        // Every op is an existing one, so undo and redo come free.
        for op in result.record.ops {
            switch op {
            case .create, .setValue, .softDelete, .setSortIndex: break
            default: Issue.record("an op outside the four: \(op)")
            }
        }
        engine.undo(result.record)
        #expect(fingerprint(book.room(named: "Chart room")) == nil)
        engine.redo(result.record)
        #expect(fingerprint(book.room(named: "Chart room")) == built)
    }

    @Test func aSecondPatchEditsMovesAndRemovesAndUndoesToTheFirst() throws {
        try engine.perform(PatchRoom(patch: patch(add("a", "line", "caption", ["text": .string("one")]),
                                                  add("b", "line", "caption", ["text": .string("two")]))), by: .tess)
        let room = try #require(book.room(named: "Chart room"))
        let first = book.document(of: room)
        let second = try engine.perform(PatchRoom(patch: patch(
            .init(op: "set", id: "a", props: ["text": .string("uno.")]),
            .init(op: "move", id: "b", index: 0),
            add("c", "title", "title", ["text": .string("T")]),
            .init(op: "remove", id: "a")
        )), by: .tess, pending: true)
        let after = book.document(of: room)
        #expect(after.parts.map(\.id) == ["b", "c"])
        engine.undo(second.record)
        #expect(book.document(of: room) == first)
        engine.redo(second.record)
        #expect(book.document(of: room) == after)
    }

    @Test func aPendingPatchCommitsOnTheDryingLineAndDismissingUndoesIt() throws {
        let one = try engine.perform(PatchRoom(patch: patch(add("t", "title", "title", ["text": .string("A")]))),
                                     by: .tess, pending: true)
        engine.commitExpired(now: .now.addingTimeInterval(3))
        #expect(one.record.state == .committed)
        let two = try engine.perform(PatchRoom(patch: patch(add("u", "line", "caption", ["text": .string("B")]))),
                                     by: .tess, pending: true)
        try engine.discard(two.record)
        #expect(book.room(named: "Chart room").map { book.document(of: $0).parts.map(\.id) } == ["t"])
        #expect(two.record.state == .undone)
    }

    @Test func aWideningPatchWaitsForAllowAndUndoTakesTheGrantBack() throws {
        let widening = patch(add("w", "actionWord", "actionWord",
                                 ["verb": .string("Water"), "does": .string("write Plant.lastWatered")]))
        let result = try engine.perform(PatchRoom(patch: widening), by: .tess, pending: true)
        #expect(result.output.widening == ["write Plant.lastWatered"])
        engine.commitExpired(now: .now.addingTimeInterval(60))
        #expect(result.record.state == .pending, "the drying line never commits a widening patch")
        let room = try #require(book.room(named: "Chart room"))
        #expect(book.document(of: room).reach == ["write Plant.lastWatered"])
        #expect(RoomReach.allow(result.record, in: engine) && result.record.state == .committed)
        engine.undo(result.record)
        #expect(book.room(named: "Chart room") == nil, "the room and its grant go back together")
    }

    @Test func undoingAWideningPatchRemovesItsGrantInTheSameGroup() throws {
        try engine.perform(PatchRoom(patch: patch(add("t", "title", "title", ["text": .string("A")]))), by: .tess)
        let result = try engine.perform(PatchRoom(patch: patch(.init(op: "grant", reach: "write Plant.lastWatered"))),
                                        by: .tess, pending: true)
        let room = try #require(book.room(named: "Chart room"))
        #expect(book.document(of: room).reach == ["write Plant.lastWatered"])
        try engine.discard(result.record)
        #expect(book.document(of: room).reach.isEmpty)
    }

    @Test func aRefusedPatchWritesNothing() throws {
        let before = store.context.all(OpRecord.self).count
        #expect(throws: ActionError.self) {
            try engine.perform(PatchRoom(patch: patch(.init(op: "grant", reach: "write Pool.name"))), by: .tess)
        }
        #expect(store.context.all(OpRecord.self).count == before)
        #expect(book.room(named: "Chart room") == nil)
    }

    @Test func thePatchDecodesFromTheModelsJSON() throws {
        let json = """
        {"room":"Chart room","say":"Plants, set","ops":[
          {"op":"add","id":"title","kind":"title","place":"title","props":{"text":"Plants"}},
          {"op":"add","id":"line","kind":"line","place":"switcher",
           "props":{"text":"{count} plants, the {top} is thirsty"},"bind":"Plant"},
          {"op":"set","id":"line","props":{"text":"{count} plants"}},
          {"op":"look","trait":"motion","value":"calm"},
          {"op":"grant","reach":"write Plant.lastWatered"},
          {"op":"move","id":"line","place":"caption","index":0}]}
        """
        let decoded = try RoomPatch.parse(Data(json.utf8))
        #expect(decoded.ops.count == 6 && decoded.ops[0].props?["text"] == .string("Plants"))
        let action = try ActionRegistry.decode(name: "patchRoom",
                                               arguments: JSONEncoder().encode(["patch": decoded]))
        #expect(action is PatchRoom)
    }
}
