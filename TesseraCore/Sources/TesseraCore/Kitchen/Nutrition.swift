import Foundation

/// What a food holds per 100 g, as USDA FoodData Central publishes it (SR Legacy, public domain). A missing nutrient is
/// nil, never 0.
public struct FoodFacts: Equatable, Sendable {
    public let name: String
    public let fdcId: Int
    public let energyKcal: Double?
    public let protein: Double?
    public let fat: Double?
    public let carbohydrate: Double?
    public let fiber: Double?
    public let sugars: Double?
    /// Milligrams.
    public let sodium: Double?
    public let saturatedFat: Double?
    /// How the entry is prepared: a raw or dry food, a cooked one, or a canned one.
    public let form: Form

    public enum Form: String, Sendable {
        case raw, cooked, canned
    }
}

/// One serving of a recipe: the sum of its lines' foods, divided by the servings. Rounded only for display.
public struct NutritionServing: Equatable, Sendable {
    public let servings: Double
    public let energyKcal: Double
    public let protein: Double?
    public let fat: Double?
    public let carbohydrate: Double?
}

/// The bundled food table and the sums built on it. It lives on the phone: nothing here is written to the op log, a
/// backup, a log, a widget or a Claude payload, and a private ingredient or recipe has no facts at all. A number is
/// shown only when every counted line has a food and a gram weight for the unit it is written in; otherwise there is no
/// number (never a guess).
public enum Nutrition {
    // MARK: Per 100 g

    private static func number(_ text: Substring) -> Double? {
        text.isEmpty ? nil : Double(text)
    }

    /// Every food, keyed by `IngredientLine.key(for:)` of its name.
    static let table: [String: FoodFacts] = {
        var map: [String: FoodFacts] = [:]
        for line in FoodTableData.foods.split(separator: "\n") {
            let parts = line.split(separator: "|", omittingEmptySubsequences: false)
            guard parts.count == 11, let fdc = Int(parts[1]) else { continue }
            let facts = FoodFacts(
                name: String(parts[0]), fdcId: fdc, energyKcal: number(parts[2]), protein: number(parts[3]),
                fat: number(parts[4]), carbohydrate: number(parts[5]), fiber: number(parts[6]),
                sugars: number(parts[7]), sodium: number(parts[8]), saturatedFat: number(parts[9]),
                form: FoodFacts.Form(rawValue: String(parts[10])) ?? .raw
            )
            map[IngredientLine.key(for: facts.name)] = facts
        }
        return map
    }()

    /// Gram weight of one portion, by food: label ("cup, chopped", "medium", "clove") to grams.
    static let portions: [Int: [(label: String, grams: Double)]] = {
        var map: [Int: [(label: String, grams: Double)]] = [:]
        for line in FoodTableData.portions.split(separator: "\n") {
            let parts = line.split(separator: "|", omittingEmptySubsequences: false)
            guard parts.count == 3, let fdc = Int(parts[0]), let grams = Double(parts[2]) else { continue }
            map[fdc, default: []].append((String(parts[1]), grams))
        }
        return map
    }()

    /// The food for an ingredient name ("Tomatoes" finds tomato), or nil: never a near match.
    public static func per100g(_ ingredient: String) -> FoodFacts? {
        table[IngredientLine.key(for: ingredient)]
    }

    /// The food for an ingredient object; nil for a private one (it keeps no facts).
    @MainActor
    public static func per100g(_ ingredient: ObjectRecord, store: ObjectStore,
                               classes: [SystemCatalog.Key: ClassDef]) -> FoodFacts?
    {
        guard ObjectPrivacy.label(of: ingredient, in: store, classes: classes) == nil else { return nil }
        return per100g(ingredient.title)
    }

    // MARK: Grams of a line

    /// Grams for `amount` of `unit` of this food, or nil when the unit has no weight (a pinch, a can) or the food has
    /// no portion that matches it.
    static func grams(amount: Double, unit: KitchenUnit?, original: String, prep: String?, food: FoodFacts) -> Double? {
        guard amount > 0, amount.isFinite else { return nil }
        guard let unit else {
            return portion(food, labels: sizeLabels(in: original), exact: false).map { $0 * amount }
        }
        switch unit.dimension {
        case .mass:
            return unit.metricFactor.map { $0 * amount }
        case .volume:
            guard let ml = unit.metricFactor else { return nil }
            // Cups, spoons: this food's own portion, matched on the unit and the prep ("cup, chopped").
            let cups = ml * amount / 240
            if let cup = portion(food, labels: ["cup"], prep: prep) {
                return cup * cups
            }
            let spoons: [(String, Double)] = [("tbsp", 15), ("tsp", 5)]
            for (label, size) in spoons {
                if let one = portion(food, labels: [label], prep: prep) {
                    return one * ml * amount / size
                }
            }
            return nil
        case .count:
            switch unit {
            case .clove: return portion(food, labels: ["clove"]).map { $0 * amount }
            case .slice: return portion(food, labels: ["slice"], prep: prep).map { $0 * amount }
            case .piece: return portion(food, labels: sizeLabels(in: original), exact: false).map { $0 * amount }
            default: return nil
            }
        }
    }

    /// The size word the line says, then the medium portion when it says none ("2 eggs" is two medium eggs, as FDC's
    /// first size), then large, then whole.
    private static func sizeLabels(in original: String) -> [String] {
        let lower = original.lowercased()
        var labels: [String] = []
        for size in ["extra large", "jumbo", "large", "medium", "small"] where lower.contains(size) {
            labels.append(size)
            break
        }
        return labels + ["medium", "large", "whole", "each"]
    }

    /// One portion's grams: the first label that matches a portion exactly, or (with a prep word) one starting with
    /// the label and naming the prep; with `exact` false the label may also lead a longer one ("medium (2-1/2 dia)").
    private static func portion(_ food: FoodFacts, labels: [String], prep: String? = nil,
                                exact: Bool = true) -> Double?
    {
        guard let list = portions[food.fdcId] else { return nil }
        for label in labels {
            if let prep, let word = prep.lowercased().split(whereSeparator: { !$0.isLetter }).first.map(String.init),
               let hit = list.first(where: { $0.label.hasPrefix(label) && $0.label.contains(word) })
            {
                return hit.grams
            }
            if let hit = list.first(where: { $0.label == label }) {
                return hit.grams
            }
            if !exact,
               let hit = list.first(where: { $0.label.hasPrefix(label + " ") || $0.label.hasPrefix(label + ",") })
            {
                return hit.grams
            }
        }
        return nil
    }

    // MARK: Which food a line means

    private static let cookedWords: Set<String> = [
        "cooked", "boiled", "roasted", "steamed", "fried", "baked", "grilled", "braised", "poached", "sauteed",
        "sautéed", "stewed",
    ]
    private static let cannedWords: Set<String> = ["canned", "tinned", "drained"]

    /// The food a recipe line means, read against how the line says it is prepared. "2 cups cooked rice" is the cooked
    /// row, never the dry one (a cup of cooked rice is a third of a cup of dry); a line that says cooked or canned with
    /// no cooked or canned row for that food has no food at all.
    static func food(named name: String, original: String, prep: String?) -> FoodFacts? {
        let words = Set("\(original) \(prep ?? "")".lowercased().split { !$0.isLetter && $0 != "é" }.map(String.init))
        let cooked = !words.isDisjoint(with: cookedWords)
        let canned = !words.isDisjoint(with: cannedWords)
        if cooked, let row = per100g("cooked \(name)") {
            return row
        }
        guard let food = per100g(name) else { return nil }
        if cooked || canned {
            // Only a row that is itself prepared stands in for a prepared line.
            return food.form == .raw ? nil : food
        }
        return food
    }

    // MARK: Per serving

    /// One serving of a recipe, or nil when any counted line has no food or no gram weight, the recipe or an ingredient
    /// is private, or there is no serving count. Optional and to-taste lines are not counted.
    @MainActor
    public static func perServing(recipe: ObjectRecord, servings: Double? = nil, store: ObjectStore,
                                  classes: [SystemCatalog.Key: ClassDef]) -> NutritionServing?
    {
        guard ObjectPrivacy.label(of: recipe, in: store, classes: classes) == nil,
              let count = servings ?? store.number("serves", of: recipe), count > 0, count.isFinite else { return nil }
        let lines = RecipeBook(store: store, classes: classes).lines(of: recipe)
        guard !lines.isEmpty else { return nil }
        var energy = 0.0
        var protein: Double? = 0, fat: Double? = 0, carbs: Double? = 0
        for line in lines where !line.line.optional && !line.line.toTaste {
            let target = line.ingredient ?? line.object
            guard ObjectPrivacy.label(of: target, in: store, classes: classes) == nil,
                  line.ingredient != nil || !ObjectPrivacy.isPrivateTitle(
                      line.line.name, lastWeightKg: BodyWeight.lastKg(in: store)
                  ),
                  let food = food(named: line.ingredient?.title ?? line.line.name, original: line.line.original,
                                  prep: line.line.prep) ?? food(named: line.line.name, original: line.line.original,
                                                                prep: line.line.prep),
                  let amount = line.line.amount.map({ ($0 + (line.line.amountMax ?? $0)) / 2 }),
                  let grams = grams(amount: amount, unit: line.line.unit, original: line.line.original,
                                    prep: line.line.prep, food: food),
                  let kcal = food.energyKcal else { return nil }
            let share = grams / 100
            energy += kcal * share
            protein = food.protein.flatMap { value in protein.map { $0 + value * share } }
            fat = food.fat.flatMap { value in fat.map { $0 + value * share } }
            carbs = food.carbohydrate.flatMap { value in carbs.map { $0 + value * share } }
        }
        return NutritionServing(servings: count, energyKcal: energy / count, protein: protein.map { $0 / count },
                                fat: fat.map { $0 / count }, carbohydrate: carbs.map { $0 / count })
    }
}
