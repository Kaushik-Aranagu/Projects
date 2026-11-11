# Food table

`TesseraCore/Sources/TesseraCore/Kitchen/FoodTableData.swift` is generated from USDA FoodData Central, SR Legacy
(public domain, CC0 1.0): per 100 g as published, plus the gram weight of one portion.

Rebuild: download `FoodData_Central_sr_legacy_food_csv_2018-04.zip` from fdc.nal.usda.gov, unzip it next to these
scripts, then

```bash
python3 build.py > food-table.json && python3 gen.py
```

`build.py` holds the name to FDC id table (a name with no good match is left out, never guessed); `gen.py` writes
`FoodTableData.swift`. Source line for the app: "Food facts: USDA FoodData Central (public domain, CC0 1.0)."

Notes on how a line reads the table (`Nutrition.swift`):
- A line that says cooked, boiled, roasted or canned reads the cooked row (`cooked rice`) when there is one; with none, a raw
  or dry row is never used, so there is no number.
- A count with no size word ("2 eggs") uses the medium portion, then large, then whole.
