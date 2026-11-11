import Foundation

/// The kinds of number that carry a small unit beside them (Design sheet 115: the data plane is the number and the
/// unit, each its own; the control plane lets the number be edited alone and the unit be swiped alone). A measure is
/// a value with its kind and unit. Converting it never changes what is stored: it only expresses the same quantity
/// in another unit.
public enum MeasureKind: String, Codable, Sendable, CaseIterable {
    case weight, distance, time, length, temperature, money
    /// Kitchen amounts (recipes, groceries): small mass and volume, US customary for volume. Never converted to each
    /// other: a cup of flour is not a number of grams without a density.
    case kitchenMass, kitchenVolume

    /// The units that fit this kind, in the order the unit wheel swipes through them. Money has none of its own: its
    /// wheel is the currencies the rates table knows.
    public var units: [String] {
        switch self {
        case .weight: ["kg", "lb"]
        case .distance: ["km", "mi"]
        case .time: ["min", "hr"]
        case .length: ["cm", "in"]
        case .temperature: ["°C", "°F"]
        case .kitchenMass: ["g", "kg", "oz", "lb"]
        case .kitchenVolume: ["ml", "l", "tsp", "tbsp", "cup", "fl oz", "pt", "qt"]
        case .money: []
        }
    }

    /// The unit a value of this kind is stored in (money: as typed, so there is none).
    public var base: String? {
        units.first
    }

    /// Whether a unit of this kind belongs on a wheel at all without a rates table.
    public var convertsOffline: Bool {
        self != .money
    }

    /// The kind a unit symbol belongs to, for the non-money kinds.
    public static func kind(ofUnit unit: String) -> MeasureKind? {
        allCases.first { $0.units.contains(unit) }
    }
}

/// A value in a unit. Immutable: `converted(to:)` returns a new measure of the same quantity.
public struct Measure: Equatable, Sendable {
    public let kind: MeasureKind
    public let value: Double
    /// "kg", "lb", ... or, for money, the currency code the amount was typed in ("" when none was said).
    public let unit: String
    /// True for a converted amount of money: it reads as an estimate ("about 24.80 EUR").
    public let isEstimate: Bool
    /// The date of the rate used to estimate it.
    public let rateDate: Date?
    /// True when this came from `converted(to:)` into another unit: the view puts "about" before it (`aboutFormatted`).
    public let isConverted: Bool

    public init(_ value: Double, _ unit: String, kind: MeasureKind? = nil, isEstimate: Bool = false,
                rateDate: Date? = nil, isConverted: Bool = false)
    {
        self.isConverted = isConverted
        self.kind = kind ?? MeasureKind.kind(ofUnit: unit) ?? .money
        self.value = value
        self.unit = unit
        self.isEstimate = isEstimate
        self.rateDate = rateDate
    }

    public static func money(_ amount: Double, currency: String?) -> Measure {
        Measure(amount, currency ?? "", kind: .money)
    }

    /// The same quantity in another unit of its kind; nil when it cannot be done (a money wheel with no rates, or
    /// a unit of another kind).
    public func converted(to target: String, rates: CurrencyRates? = nil) -> Measure? {
        if kind == .money {
            guard !unit.isEmpty, let rates, let moved = rates.convert(value, from: unit, to: target) else {
                return unit == target ? self : nil
            }
            return unit == target ? self : Measure(moved, target, kind: .money, isEstimate: true, rateDate: rates.date)
        }
        guard kind.units.contains(target), let moved = UnitConverter.convert(value, from: unit, to: target, kind: kind)
        else { return nil }
        return Measure(moved, target, kind: kind, isConverted: target != unit)
    }

    /// "about 1 cup" for a converted amount, the plain form otherwise. Money keeps its own "about" form.
    public var aboutFormatted: String {
        isConverted && kind != .money && !formatted.hasPrefix("less than") ? "about \(formatted)" : formatted
    }

    /// "181.9 lb", "5.6 km"; money as typed ("€12.40", "12.40") or, estimated, "about 24.80 EUR".
    public var formatted: String {
        MeasureFormat.string(self)
    }
}

/// Pure conversion between units of one kind. Values are `Double`; display rounding lives in `MeasureFormat`.
public enum UnitConverter {
    static let poundsPerKilogram = 2.204_622_621_8
    static let milesPerKilometre = 0.621_371_192_237_334
    static let inchesPerCentimetre = 1 / 2.54

    /// The value in the target unit, or nil when the two units are not of the same convertible kind.
    public static func convert(_ value: Double, from: String, to: String) -> Double? {
        guard let kind = MeasureKind.kind(ofUnit: from), kind == MeasureKind.kind(ofUnit: to) else { return nil }
        return convert(value, from: from, to: to, kind: kind)
    }

    /// The value in the target unit of a known kind ("kg" is in both weight and kitchen mass, so the kind says which).
    public static func convert(_ value: Double, from: String, to: String, kind: MeasureKind) -> Double? {
        guard kind != .money, kind.units.contains(from), kind.units.contains(to) else { return nil }
        if from == to {
            return value
        }
        if kind == .kitchenMass || kind == .kitchenVolume {
            guard let source = kitchenFactors[from], let target = kitchenFactors[to] else { return nil }
            return value * source / target
        }
        return fromBase(toBase(value, unit: from), unit: to)
    }

    /// Grams per unit, and millilitres per unit (US customary), for the kitchen kinds.
    static let kitchenFactors: [String: Double] = [
        "g": 1, "kg": 1000, "oz": 28.349_523_125, "lb": 453.592_37,
        "ml": 1, "l": 1000, "tsp": 4.928_921_593_75, "tbsp": 14.786_764_781_25, "cup": 236.588_236_5,
        "fl oz": 29.573_529_562_5, "pt": 473.176_473, "qt": 946.352_946,
    ]

    private static func toBase(_ value: Double, unit: String) -> Double {
        switch unit {
        case "lb": value / poundsPerKilogram
        case "mi": value / milesPerKilometre
        case "hr": value * 60
        case "in": value / inchesPerCentimetre
        case "°F": (value - 32) * 5 / 9
        default: value
        }
    }

    private static func fromBase(_ value: Double, unit: String) -> Double {
        switch unit {
        case "lb": value * poundsPerKilogram
        case "mi": value * milesPerKilometre
        case "hr": value / 60
        case "in": value * inchesPerCentimetre
        case "°F": value * 9 / 5 + 32
        default: value
        }
    }
}

/// How a measure reads. Sensible digits per unit: 82.5 kg is 181.9 lb; 5 km is 3.11 mi; 90 min is 1.5 hr.
public enum MeasureFormat {
    /// Decimal places shown at most; trailing zeros are dropped.
    static let places: [String: Int] = [
        "kg": 1, "lb": 1, "km": 2, "mi": 2, "min": 0, "hr": 2, "cm": 0, "in": 1, "°C": 1, "°F": 1,
        "g": 0, "ml": 0, "l": 2, "oz": 1, "tsp": 2, "tbsp": 2, "cup": 2, "fl oz": 1,
    ]

    public static func number(_ value: Double, unit: String) -> String {
        let digits = places[unit] ?? 2
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = digits
        formatter.usesGroupingSeparator = false
        let rounded = (value * pow(10, Double(digits))).rounded() / pow(10, Double(digits))
        return formatter.string(from: NSNumber(value: rounded == 0 ? 0 : rounded)) ?? String(value)
    }

    /// A kitchen amount as a plain decimal: two places for cups, spoons and pounds (1 oz is 0.06 lb), one for ounces;
    /// a positive amount below that reads "less than 0.01".
    static func kitchenDecimal(_ value: Double, unit: String) -> String {
        let digits = unit == "oz" ? 1 : 2
        let step = pow(10, -Double(digits))
        if value > 0, value < step / 2 {
            return "less than \(places3(step))"
        }
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = digits
        formatter.usesGroupingSeparator = false
        return formatter.string(from: NSNumber(value: (value / step).rounded() * step)) ?? String(value)
    }

    static func places3(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 3
        formatter.usesGroupingSeparator = false
        return formatter.string(from: NSNumber(value: (value * 1000).rounded() / 1000)) ?? String(value)
    }

    /// "1 hr 35 min", "2 hr", "35 min"; nil for a negative or absurd value.
    static func hoursAndMinutes(_ hours: Double) -> String? {
        guard hours >= 0, hours < 10000 else { return nil }
        let minutes = Int((hours * 60).rounded())
        let whole = minutes / 60, rest = minutes % 60
        switch (whole, rest) {
        case (0, _): return "\(rest) min"
        case (_, 0): return "\(whole) hr"
        default: return "\(whole) hr \(rest) min"
        }
    }

    static let signs = ["EUR": "€", "GBP": "£", "USD": "$", "INR": "₹"]

    /// Units a cook reads in fractions: spoons, cups, fluid ounces, pints and quarts, ounces and pounds.
    static let spoonAndCupUnits: Set = ["tsp", "tbsp", "cup", "fl oz", "pt", "qt", "oz", "lb"]
    static let fractionGlyphs: [(Double, String)] = [
        (0, ""), (1.0 / 8, "⅛"), (1.0 / 4, "¼"), (1.0 / 3, "⅓"), (3.0 / 8, "⅜"), (1.0 / 2, "½"), (5.0 / 8, "⅝"),
        (2.0 / 3, "⅔"), (3.0 / 4, "¾"), (7.0 / 8, "⅞"), (1, ""),
    ]

    /// One drawn pair: the number and its unit, apart, so a screen sizes the unit at about half the numeral (every
    /// measured number shows its unit, Kaushik's rule 11:24 PM Oct 23). `spaced` is false for degrees ("72°F").
    public struct Part: Equatable, Sendable {
        public let number: String
        public let unit: String
        public let spaced: Bool

        public init(number: String, unit: String, spaced: Bool = true) {
            self.number = number
            self.unit = unit
            self.spaced = spaced
        }

        /// The pair as one line of text: "82.5 kg", "72°F".
        public var text: String {
            unit.isEmpty ? number : number + (spaced ? " " : "") + unit
        }
    }

    /// A kitchen amount as its number and its unit: "1½" and "cups", "250" and "g", "less than 1" and "g".
    static func kitchenPart(_ value: Double, unit: String) -> Part {
        guard value.isFinite, value >= 0 else { return Part(number: "\(value)", unit: unit) }
        guard spoonAndCupUnits.contains(unit) else {
            if unit == "g" || unit == "ml" {
                // Under 0.05 it is "less than 1 g", not "0 g"; under one, one decimal.
                if value > 0, value < 0.05 {
                    return Part(number: "less than 1", unit: unit)
                }
                if value < 1, value > 0 {
                    return Part(number: number(value, unit: "kg"), unit: unit)
                }
            }
            // Kitchen kilograms and litres need three places (250 g is 0.25 kg, 5 ml is 0.005 l).
            if unit == "kg" || unit == "l" {
                return Part(number: places3(value), unit: unit)
            }
            return Part(number: number(value, unit: unit), unit: unit)
        }
        let whole = value.rounded(.down)
        let rest = value - whole
        let snapped = fractionGlyphs.min { abs($0.0 - rest) < abs($1.0 - rest) } ?? (0, "")
        // Cups snap a little wider (250 ml reads "1 cup"); everything else stays within a hair.
        let tolerance = unit == "cup" ? 0.06 : 0.02
        // A positive amount that snaps to nothing is never "0 cup": it reads as a decimal instead.
        guard abs(snapped.0 - rest) <= tolerance, !(whole == 0 && snapped.0 == 0 && value > 0) else {
            return Part(number: kitchenDecimal(value, unit: unit), unit: unit == "cup" && value > 1 ? "cups" : unit)
        }
        let carried = snapped.0 == 1 ? whole + 1 : whole
        let fraction = snapped.0 == 1 ? "" : snapped.1
        let text = (carried > 0 ? "\(Int(carried))" : "") + fraction
        let shown = carried + (snapped.0 == 1 ? 0 : snapped.0)
        return Part(number: text.isEmpty ? "0" : text, unit: unit == "cup" && shown > 1.0001 ? "cups" : unit)
    }

    /// "1½ cups", "¼ tsp", "2 tbsp", "250 g", "1.5 kg", "5 ml": whole grams and millilitres; spoons and cups snap to
    /// ⅛ ¼ ⅓ ⅜ ½ ⅝ ⅔ ¾ ⅞ when within a hair of one, else two decimals.
    static func kitchen(_ value: Double, unit: String) -> String {
        guard value.isFinite, value >= 0 else { return "\(value) \(unit)" }
        return kitchenPart(value, unit: unit).text
    }

    /// What a measure draws, as separate parts, each in its own role: a weight or any measured number is its numeral
    /// and
    /// its unit, the unit drawn at about half the numeral's size. Time past an hour is two pairs ("1" "hr", "35"
    /// "min");
    /// money is its amount (with the currency sign when it has one) and the currency when estimated; everything else is
    /// one pair. Every part of a measured number has a unit; a view never formats a number or a unit by hand.
    public static func parts(of measure: Measure) -> [Part] {
        if measure.kind == .money {
            let amount = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), measure.value)
            if measure.isEstimate {
                return [Part(number: amount, unit: measure.unit)]
            }
            // As typed: the sign against the number when the currency has one, the bare number otherwise.
            if let sign = signs[measure.unit] {
                return [Part(number: sign + amount, unit: "", spaced: false)]
            }
            return [Part(number: amount, unit: measure.unit)]
        }
        if measure.kind == .kitchenMass || measure.kind == .kitchenVolume {
            return [kitchenPart(measure.value, unit: measure.unit)]
        }
        if measure.unit == "hr" || measure.unit == "min" {
            let totalMinutes = measure.unit == "hr" ? measure.value * 60 : measure.value
            if totalMinutes >= 60, totalMinutes < 600_000 {
                let minutes = Int(totalMinutes.rounded())
                var parts = [Part(number: "\(minutes / 60)", unit: "hr")]
                if minutes % 60 != 0 {
                    parts.append(Part(number: "\(minutes % 60)", unit: "min"))
                }
                return parts
            }
        }
        let attached = measure.unit == "°C" || measure.unit == "°F"
        return [Part(number: number(measure.value, unit: measure.unit), unit: measure.unit, spaced: !attached)]
    }

    public static func string(_ measure: Measure) -> String {
        if measure.kind == .money {
            let amount = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), measure.value)
            if measure.isEstimate {
                return "about \(amount) \(measure.unit)"
            }
            // As typed: the sign against the number when the currency has one, the bare number otherwise.
            return signs[measure.unit].map { $0 + amount } ?? amount
        }
        // Kitchen amounts: spoons and cups in the fractions a cook reads, grams and millilitres whole.
        if measure.kind == .kitchenMass || measure.kind == .kitchenVolume {
            return kitchen(measure.value, unit: measure.unit)
        }
        // Hours read as the cook clock does, "1 hr 35 min", so the app has one way to say time.
        if measure.unit == "hr", let text = hoursAndMinutes(measure.value) {
            return text
        }
        return Part(number: number(measure.value, unit: measure.unit), unit: measure.unit,
                    spaced: !(measure.unit == "°C" || measure.unit == "°F")).text
    }
}
