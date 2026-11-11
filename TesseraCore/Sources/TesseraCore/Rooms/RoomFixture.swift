import Foundation

/// The Chart room as Builder A builds against it (R1 spec section 8, step 1): every kind of part at least once, over a
/// "Plant" class with `lastWatered` (date) and `waterEvery` (number), and one write grant. It is also the document the
/// validator's own tests start from; the same JSON is `docs/design/tess-built-rooms/fixture-chart-room.json`.
public enum RoomFixture {
    public nonisolated static let chart = RoomDocument(
        name: "Chart room", color: "slate", look: ["density": "airy", "motion": "calm"],
        reach: ["write Plant.lastWatered"],
        parts: [
            RoomPart(id: "title", kind: "title", place: "title", props: ["text": .string("Plants")]),
            RoomPart(id: "line", kind: "line", place: "switcher",
                     props: ["text": .string("{count} plants, the {top} is thirsty")], bind: "Plant"),
            RoomPart(id: "need", kind: "lines", place: "needs",
                     props: ["lines": .array([.string("Water the thirsty ones first"), .string("Mist the ferns")])]),
            RoomPart(id: "days", kind: "number", place: "keystone",
                     props: ["label": .string("plants"), "agg": .string("count")], bind: "Plant"),
            RoomPart(id: "soon", kind: "number", place: "keystone",
                     props: ["label": .string("days apart"), "unit": .string("d"), "field": .string("waterEvery"),
                             "agg": .string("max")], bind: "Plant"),
            // A stack or grid draws its children; a child is not drawn again on its own.
            RoomPart(id: "pair", kind: "stack", place: "keystone",
                     props: ["children": .array([.string("days"), .string("soon")]), "axis": .string("h")]),
            RoomPart(id: "row", kind: "row", place: "widget",
                     props: ["big": .string("waterEvery"), "small": .string("title"),
                             "sort": .string("-daysSince(lastWatered)"), "max": .number(4)], bind: "Plant"),
            RoomPart(id: "next", kind: "row", place: "widget",
                     props: ["big": .string("waterEvery"), "small": .string("title"), "max": .number(2)],
                     bind: "Plant"),
            RoomPart(id: "both", kind: "grid", place: "widget",
                     props: ["children": .array([.string("row"), .string("next")]), "columns": .number(2)]),
            RoomPart(id: "top", kind: "top", place: "widget",
                     props: ["on": .string("row"), "field": .string("lastWatered")]),
            RoomPart(id: "list", kind: "list", place: "page",
                     props: ["big": .string("waterEvery"), "small": .string("title"),
                             "sort": .string("-daysSince(lastWatered)"), "max": .number(8)], bind: "Plant"),
            RoomPart(id: "leaf", kind: "glyph", place: "glyph",
                     props: ["symbol": .string("leaf"), "color": .string("sage")]),
            RoomPart(id: "doors", kind: "doors", place: "dial",
                     props: ["doors": .array([.object(["name": .string("Plants"), "bind": .string("Plant")])])]),
            RoomPart(id: "water", kind: "actionWord", place: "actionWord",
                     props: ["verb": .string("Water"), "object": .string("the thirstiest plant"),
                             "does": .string("write Plant.lastWatered")]),
        ]
    )
}
