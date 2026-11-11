@testable import TesseraCore
import Testing

/// Every measured number shows its unit, apart from the numeral so a view can draw it at about half the size (Kaushik,
/// Oct 23 11:24 PM). The one-line text reads the same as before.
struct MeasurePartsTests {
    typealias Part = MeasureFormat.Part

    @Test func aWeightIsItsNumeralAndItsUnit() {
        #expect(MeasureFormat.parts(of: Measure(82.5, "kg")) == [Part(number: "82.5", unit: "kg")])
        #expect(MeasureFormat.parts(of: Measure(185, "lb")) == [Part(number: "185", unit: "lb")])
        #expect(MeasureFormat.parts(of: Measure(5.0, "km")).first?.unit == "km")
    }

    @Test func degreesStickToTheNumeral() {
        let parts = MeasureFormat.parts(of: Measure(72, "°F"))
        #expect(parts == [Part(number: "72", unit: "°F", spaced: false)] && parts[0].text == "72°F")
    }

    @Test func timePastAnHourIsTwoPairs() {
        #expect(MeasureFormat.parts(of: Measure(95, "min")) == [Part(number: "1", unit: "hr"),
                                                                Part(number: "35", unit: "min")])
        #expect(MeasureFormat.parts(of: Measure(2, "hr")) == [Part(number: "2", unit: "hr")])
        #expect(MeasureFormat.parts(of: Measure(35, "min")) == [Part(number: "35", unit: "min")])
    }

    @Test func kitchenAmountsKeepTheirFractionsAndUnits() {
        #expect(MeasureFormat.parts(of: Measure(1.5, "cup", kind: .kitchenVolume)) == [Part(number: "1½",
                                                                                            unit: "cups")])
        #expect(MeasureFormat.parts(of: Measure(0.25, "tsp", kind: .kitchenVolume)) == [Part(number: "¼", unit: "tsp")])
        #expect(MeasureFormat.parts(of: Measure(250, "g", kind: .kitchenMass)) == [Part(number: "250", unit: "g")])
        #expect(MeasureFormat.parts(of: Measure(0.01, "g", kind: .kitchenMass)) == [Part(number: "less than 1",
                                                                                         unit: "g")])
    }

    @Test func moneyKeepsItsSignOrItsCurrency() {
        #expect(MeasureFormat.parts(of: Measure(12.5, "USD", kind: .money)) == [Part(number: "$12.50", unit: "",
                                                                                     spaced: false)])
        #expect(MeasureFormat.parts(of: Measure(12.5, "CHF", kind: .money)) == [Part(number: "12.50", unit: "CHF")])
    }

    @Test func theOneLineTextIsUnchanged() {
        for (measure, text) in [
            (Measure(82.5, "kg"), "82.5 kg"), (Measure(72, "°F"), "72°F"), (Measure(95, "min"), "95 min"),
            (Measure(1.5, "hr"), "1 hr 30 min"), (Measure(1.5, "cup", kind: .kitchenVolume), "1½ cups"),
            (Measure(12.5, "USD", kind: .money), "$12.50"),
        ] {
            #expect(MeasureFormat.string(measure) == text, "\(text)")
        }
    }
}
