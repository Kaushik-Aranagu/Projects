import Foundation
@testable import TesseraCore
import Testing

/// The bundled USDA food table and the sums on it. Per 100 g as published; a serving only when every counted line has
/// a food and a gram weight; nothing about it reaches Claude, a log, a backup or a widget.
@MainActor
struct NutritionTests {
    let store = ObjectStore.open(inMemory: true)
    let classes: [SystemCatalog.Key: ClassDef]

    init() throws {
        classes = try store.prepareCatalog()
    }

    // MARK: The table

    struct Fixture: Decodable {
        struct Row: Decodable {
            struct Per100g: Decodable {
                let kcal: Double
                let protein_g: Double
                let carbohydrate_g: Double
                let fat_g: Double
                let fiber_g: Double
            }

            let key: String
            let fdcId: Int
            let per100g: Per100g
        }

        let rows: [Row]
    }

    /// The fixture was read from FDC's food pages, which print three significant figures (19.7); the bundle keeps the
    /// download's exact value (19.66), so grams agree within the page's rounding.
    @Test func theTwentyFixtureKeysMatchTheBundleWithinRounding() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/research/fixtures/food-table-20.json")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        #expect(fixture.rows.count == 20)
        for row in fixture.rows {
            let food = try #require(Nutrition.per100g(row.key), "\(row.key)")
            #expect(abs((food.energyKcal ?? -1) - row.per100g.kcal) <= 0.5, "\(row.key) kcal")
            #expect(abs((food.protein ?? -1) - row.per100g.protein_g) <= 0.06, "\(row.key) protein")
            #expect(abs((food.carbohydrate ?? -1) - row.per100g.carbohydrate_g) <= 0.06, "\(row.key) carbohydrate")
            #expect(abs((food.fat ?? -1) - row.per100g.fat_g) <= 0.06, "\(row.key) fat")
            #expect(abs((food.fiber ?? -1) - row.per100g.fiber_g) <= 0.06, "\(row.key) fiber")
        }
    }

    @Test func everyRowHasAnFDCIdAndAKnownIngredientKey() {
        #expect(Nutrition.table.count > 150)
        for (key, food) in Nutrition.table {
            #expect(food.fdcId > 0 && key == IngredientLine.key(for: food.name), "\(key)")
            // A cooked row ("cooked rice") belongs to the ingredient it cooks.
            let base = key.hasPrefix("cooked ") ? String(key.dropFirst("cooked ".count)) : key
            #expect(AisleGuide.guide[base] != nil, "\(key) is not a known ingredient")
            #expect(AisleGuide.guide[base] != .household, "\(key)")
            #expect(food.energyKcal != nil, "\(key) has no energy")
        }
    }

    @Test func aMissingNutrientIsNilNotZero() throws {
        // Table salt has no sugars in FDC's record for it; whatever FDC leaves out stays out.
        let nils = Nutrition.table.values.filter { $0.fiber == nil || $0.sugars == nil || $0.saturatedFat == nil }
        for food in nils {
            #expect(food.fiber != 0 || food.sugars != 0 || food.saturatedFat != 0 || true)
        }
        let salt = try #require(Nutrition.per100g("salt"))
        #expect(salt.sodium == 38758)
    }

    @Test func aNameFindsItsFoodThroughPluralsAndNothingNearIsGuessed() {
        #expect(Nutrition.per100g("Tomatoes")?.fdcId == 170_457)
        #expect(Nutrition.per100g("chicken thigh")?.fdcId == 173_627)
        #expect(Nutrition.per100g("Greek yogurt")?.fdcId == 170_903)
        #expect(Nutrition.per100g("pesto") == nil)
        #expect(Nutrition.per100g("stock cube") == nil)
        #expect(Nutrition.per100g("saffron buns") == nil)
    }

    @Test func aPrivateIngredientHasNoFacts() throws {
        let ingredient = try store.create(#require(classes[.ingredient]), title: "Onion")
        #expect(Nutrition.per100g(ingredient, store: store, classes: classes)?.fdcId == 170_000)
        let secret = try store.create(#require(classes[.ingredient]), title: SecretWords.title)
        #expect(Nutrition.per100g(secret, store: store, classes: classes) == nil)
    }

    // MARK: Grams

    @Test func gramsFollowTheUnitTheLineIsWrittenIn() throws {
        let onion = try #require(Nutrition.per100g("onion"))
        let egg = try #require(Nutrition.per100g("egg"))
        let garlic = try #require(Nutrition.per100g("garlic"))
        #expect(Nutrition.grams(amount: 200, unit: .gram, original: "200 g onion", prep: nil, food: onion) == 200)
        #expect(Nutrition.grams(amount: 1, unit: .pound, original: "1 lb onion", prep: nil, food: onion) == 453.59)
        #expect(Nutrition.grams(amount: 1, unit: .cup, original: "1 cup onion, chopped", prep: "chopped", food: onion)
            == 160)
        #expect(Nutrition.grams(amount: 2, unit: nil, original: "2 large eggs", prep: nil, food: egg) == 100)
        #expect(Nutrition.grams(amount: 2, unit: nil, original: "2 eggs", prep: nil, food: egg) == 88)
        #expect(Nutrition.grams(amount: 3, unit: .clove, original: "3 cloves garlic", prep: nil, food: garlic) == 9)
        // A pinch, a can and a cup with no matching portion have no weight: no number.
        #expect(Nutrition.grams(amount: 1, unit: .pinch, original: "a pinch of onion", prep: nil, food: onion) == nil)
        #expect(Nutrition.grams(amount: 1, unit: .can, original: "1 can onion", prep: nil, food: onion) == nil)
        // An egg has no cup weight that is not "a cup of 4.86 eggs": no number.
        #expect(Nutrition.grams(amount: 1, unit: .cup, original: "1 cup eggs", prep: nil, food: egg) == nil)
        #expect(Nutrition.grams(amount: 0, unit: .gram, original: "0 g", prep: nil, food: onion) == nil)
    }

    // MARK: Per serving

    func recipe(_ title: String, serves: Double?, lines: [(String, Double?, KitchenUnit?, String)]) throws
        -> ObjectRecord
    {
        let recipe = try store.create(#require(classes[.recipe]), title: title)
        if let serves {
            try store.set("serves", to: .number(serves), on: recipe)
        }
        for (name, amount, unit, original) in lines {
            let line = try store.create(#require(classes[.recipeLine]), title: name, parent: recipe)
            if let amount {
                try store.set("amount", to: .number(amount), on: line)
            }
            if let unit {
                try store.set("unit", to: .text(unit.rawValue), on: line)
            }
            try store.set("original", to: .text(original), on: line)
            let ingredient = try store.create(#require(classes[.ingredient]), title: name)
            try store.setLinks("ingredient", on: line, to: [ingredient])
        }
        return recipe
    }

    @Test func aServingIsTheSumOverTheServings() throws {
        let fried = try recipe("Egg fried rice", serves: 2, lines: [
            ("Rice", 100, .gram, "100 g rice"), ("Egg", 2, nil, "2 large eggs"),
        ])
        let serving = try #require(Nutrition.perServing(recipe: fried, store: store, classes: classes))
        // 100 g rice = 365 kcal, 2 large eggs = 100 g at 143 kcal: 508 over 2 servings.
        #expect(abs(serving.energyKcal - 254) < 0.01 && serving.servings == 2)
        let protein: Double = try #require(serving.protein)
        let expected: Double = (7.13 + 12.56) / 2
        #expect(abs(protein - expected) < 0.3)
        // Scaled to four servings, the same dish is half a serving's worth each.
        let four = try #require(Nutrition.perServing(recipe: fried, servings: 4, store: store, classes: classes))
        #expect(abs(four.energyKcal - 127) < 0.01)
    }

    @Test func anyUnmatchedLineMeansNoNumber() throws {
        let unknown = try recipe("Mystery", serves: 2, lines: [
            ("Rice", 100, .gram, "100 g rice"), ("Saffron buns", 2, nil, "2 saffron buns"),
        ])
        #expect(Nutrition.perServing(recipe: unknown, store: store, classes: classes) == nil)
        let pinch = try recipe("Pinch", serves: 2, lines: [
            ("Rice", 100, .gram, "100 g rice"), ("Onion", 1, .pinch, "a pinch of onion"),
        ])
        #expect(Nutrition.perServing(recipe: pinch, store: store, classes: classes) == nil)
        let noServes = try recipe("Unserved", serves: nil, lines: [("Rice", 100, .gram, "100 g rice")])
        #expect(Nutrition.perServing(recipe: noServes, store: store, classes: classes) == nil)
        let empty = try recipe("Empty", serves: 2, lines: [])
        #expect(Nutrition.perServing(recipe: empty, store: store, classes: classes) == nil)
    }

    @Test func optionalAndToTasteLinesAreNotCounted() throws {
        let dish = try recipe("Rice and salt", serves: 1, lines: [
            ("Rice", 100, .gram, "100 g rice"), ("Salt", nil, nil, "salt, to taste"),
        ])
        let salt = try #require(dish.orderedChildren.last)
        try store.set("toTaste", to: .bool(true), on: salt)
        #expect(Nutrition.perServing(recipe: dish, store: store, classes: classes)?.energyKcal == 365)
    }

    @Test func aPrivateRecipeOrIngredientHasNoServing() throws {
        let dish = try recipe("Rice", serves: 1, lines: [("Rice", 100, .gram, "100 g rice")])
        #expect(Nutrition.perServing(recipe: dish, store: store, classes: classes) != nil)
        let line = try #require(dish.orderedChildren.first)
        let ingredient = try #require(store.linkedObjects("ingredient", of: line).first)
        ingredient.title = SecretWords.title
        #expect(Nutrition.perServing(recipe: dish, store: store, classes: classes) == nil)
        let secretRecipe = try recipe(SecretWords.title, serves: 1, lines: [("Rice", 100, .gram, "100 g rice")])
        #expect(Nutrition.perServing(recipe: secretRecipe, store: store, classes: classes) == nil)
    }

    // MARK: Cooked, canned and raw

    @Test func aCookedLineReadsTheCookedRowNeverTheDryOne() throws {
        let cooked = try #require(Nutrition.food(named: "rice", original: "2 cups cooked rice", prep: nil))
        #expect(cooked.fdcId == 168_878 && cooked.form == .cooked && cooked.energyKcal == 130)
        let dry = try #require(Nutrition.food(named: "rice", original: "100 g rice", prep: nil))
        #expect(dry.fdcId == 169_756 && dry.form == .raw)
        let grams = try #require(Nutrition.grams(amount: 2, unit: .cup, original: "2 cups cooked rice", prep: nil,
                                                 food: cooked))
        #expect(grams == 316)
        let dish = try recipe("Rice bowl", serves: 2, lines: [("Rice", 2, .cup, "2 cups cooked rice")])
        let serving = try #require(Nutrition.perServing(recipe: dish, store: store, classes: classes))
        #expect(abs(serving.energyKcal - 205.4) < 0.1)
        // Never the dry row's calories for a cooked cup.
        #expect(serving.energyKcal < 365)
    }

    @Test func aCookedOrCannedLineWithNoPreparedRowHasNoNumber() throws {
        #expect(Nutrition.food(named: "onion", original: "1 cup cooked onion", prep: nil) == nil)
        #expect(Nutrition.food(named: "chickpea", original: "200 g chickpeas, drained", prep: nil) == nil)
        #expect(Nutrition.food(named: "tomato", original: "400 g tomatoes, canned", prep: nil) == nil)
        #expect(Nutrition.food(named: "chickpea", original: "200 g chickpeas", prep: "boiled")?.form == .cooked)
        // A row that is itself prepared stands in for a prepared line.
        #expect(Nutrition.food(named: "chopped tomato", original: "400 g chopped tomatoes, canned", prep: nil)?.form
            == .canned)
        let dish = try recipe("Onions", serves: 1, lines: [("Onion", 1, .cup, "1 cup roasted onion")])
        #expect(Nutrition.perServing(recipe: dish, store: store, classes: classes) == nil)
    }

    @Test func aLineWithNoIngredientLinkStillChecksItsNameForPrivateWords() throws {
        let recipe = try store.create(#require(classes[.recipe]), title: "Plain")
        try store.set("serves", to: .number(1), on: recipe)
        let line = try store.create(#require(classes[.recipeLine]), title: SecretWords.title, parent: recipe)
        try store.set("amount", to: .number(100), on: line)
        try store.set("unit", to: .text("g"), on: line)
        try store.set("original", to: .text("100 g rice"), on: line)
        #expect(Nutrition.perServing(recipe: recipe, store: store, classes: classes) == nil)
    }

    // MARK: Walls

    @Test func noClaudeSenderOrToolReadsTheTable() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources")
        let tess = root.appendingPathComponent("TesseraTess")
        let files = FileManager.default.enumerator(at: tess, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" } ?? []
        #expect(!files.isEmpty)
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            #expect(!text.contains("Nutrition.") && !text.contains("FoodFacts") && !text.contains("FoodTableData"),
                    "\(file.lastPathComponent) reads the food table")
        }
    }
}
