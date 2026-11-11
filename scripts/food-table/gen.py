import csv, json, re
D='FoodData_Central_sr_legacy_food_csv_2018-04/'
t=json.load(open('food-table.json'))
ids={r['fdcId'] for r in t['rows']}
def num(v): return '' if v is None else (('%g' % v))
def form(d):
    d=d.lower()
    if 'canned' in d: return 'canned'
    if any(w in d for w in ('cooked','boiled','roasted','baked','fried','grilled')): return 'cooked'
    return 'raw'
foods=[]
seen=set()
for r in t['rows']:
    foods.append('|'.join([r['name'],str(r['fdcId']),num(r['energyKcal']),num(r['protein']),num(r['fat']),num(r['carbohydrate']),num(r['fiber']),num(r['sugars']),num(r['sodium']),num(r['saturatedFat']),form(r['description'])]))
por=[]
for r in csv.DictReader(open(D+'food_portion.csv')):
    fid=int(r['fdc_id'])
    if fid not in ids: continue
    mod=r['modifier'].strip().lower().replace('"','').replace('|',' ')
    amt=float(r['amount'] or 1)
    if not mod or r['gram_weight']=='' : continue
    g=float(r['gram_weight'])/amt
    if len(mod)>40 or amt<=0: continue
    por.append(f"{fid}|{mod}|{g:.2f}".replace('.00|','|') if False else f"{fid}|{mod}|{g:g}")
src='''// Generated from USDA FoodData Central, SR Legacy (public domain, CC0 1.0): per 100 g, as published, and the
// gram weight of one portion. Do not edit; rebuild from the FDC download (docs/research/food-table-usda.md).
enum FoodTableData {
    /// name|fdcId|energyKcal|protein|fat|carbohydrate|fiber|sugars|sodium(mg)|saturatedFat|form (raw, cooked or canned),\n    /// empty when FDC has none.
    static let foods = """
'''+'\n'.join(foods)+'''
"""

    /// fdcId|portion label|grams for one.
    static let portions = """
'''+'\n'.join(por)+'''
"""
}
'''
open('FoodTableData.swift','w').write(src)
print(len(foods),len(por),len(src))
