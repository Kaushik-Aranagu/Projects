import csv, json, sys
D='FoodData_Central_sr_legacy_food_csv_2018-04/'
M = {
# produce
"onion":170000,"garlic":169230,"ginger":169231,"shallot":170499,"spring onion,scallion,green onion":170005,
"leek":169246,"carrot":170393,"celery":169988,"potato":170026,"sweet potato":168482,"tomato,cherry tomato":170457,
"cucumber":168409,"courgette,zucchini":169291,"aubergine,eggplant":169228,"bell pepper":170427,
"chilli,chili":170497,"lettuce":169249,"spinach":168462,"rocket":169387,"kale":168421,"cabbage":169975,
"broccoli":170379,"cauliflower":169986,"mushroom":169251,"pea":170419,"green bean":169961,"sweetcorn":169998,
"avocado":171705,"lemon":167746,"lime":168155,"orange":169097,"apple":171688,"banana":173944,"pear":169118,
"grape":174683,"strawberry":167762,"blueberry":171711,"raspberry":167755,"mango":169910,"pineapple":169124,
"basil":172232,"parsley":170416,"coriander,cilantro":169997,"mint":173475,"dill":172233,"thyme":173470,
"rosemary":173473,"chive":169994,"bay leaf":170917,"beetroot":169145,"radish":169276,"fennel":169385,
"asparagus":168389,"pak choi":170390,"butternut squash":169295,"pumpkin":168448,"squash":170489,
# bakery
"bread":174924,"baguette,sourdough":172675,"pitta,pita":174915,"tortilla":175037,"naan":171845,"bun,roll":172796,
"bagel":175049,"croissant":174987,
# meat and seafood
"chicken":171052,"chicken breast":171077,"chicken thigh":173627,"beef mince,mince":174036,"pork":167818,
"chorizo":173859,"turkey":171505,"duck":172410,"salmon":175167,"cod":171955,"prawn,shrimp":174210,
"mussel":174216,"haddock":171964,"sea bass":175142,"mackerel":175119,"sausage":171631,
# dairy
"milk":171265,"butter":173410,"double cream":170859,"single cream":170857,"soured cream,sour cream":171257,
"yogurt,yoghurt":171284,"greek yogurt":170903,"cheddar":173414,"parmesan":170848,"mozzarella":170845,"feta":173420,
"ricotta":170851,"cream cheese":173418,"egg":171287,
# pantry
"flour":168894,"plain flour":168894,"self-raising flour":168895,"bread flour":168896,"sugar,caster sugar":169655,
"brown sugar":168833,"icing sugar":169656,"rice":169756,"basmati rice,jasmine rice":169756,"pasta,spaghetti,penne,linguine,fusilli,lasagne,macaroni,orecchiette,rigatoni,farfalle,orzo,ziti":169736,
"egg noodle,noodle":169731,"couscous":169699,"quinoa":168874,"bulgur":170688,"lentil":172420,"red lentil":174284,
"oat":173904,"yeast":175043,"baking powder":172804,"bicarbonate of soda,baking soda":175040,"cornflour":169698,
"cocoa":169593,"dark chocolate":170273,"honey":169640,"maple syrup":169661,"almond":170158,"walnut":170187,
"peanut":172430,"cashew":170162,"pine nut":170591,"raisin":168165,
# cans and jars
"chopped tomato,tinned tomato,diced tomato,crushed tomato":170051,"tomato puree,passata":170460,"tomato paste":170459,
"tomato sauce":170054,"coconut milk":170173,"chickpea":173756,"black bean":173734,"baked bean":175182,
"kidney bean":175195,"pinto bean":175199,"cannellini bean":175202,"refried bean":172438,"tuna":173709,
"olive":169094,"caper":172238,"anchovy":174183,"jam":169641,"peanut butter":172470,"mustard,dijon mustard":172234,
"mayonnaise":171009,"ketchup":168556,"soy sauce":174277,"worcestershire sauce":171610,"vinegar":172237,
"balsamic vinegar":172241,"olive oil":171413,"vegetable oil,oil":172370,"sesame oil":171016,"coconut oil":171412,
"chicken stock,chicken broth":172884,"tahini":170189,"enchilada sauce":174074,
# spices
"salt,sea salt,kosher salt":173468,"black pepper":170931,"paprika,smoked paprika":171329,"cumin,ground cumin":170923,
"coriander seed":170922,"turmeric":172231,"cayenne":170932,"chilli powder,chili powder":171319,
"curry powder":170924,"cinnamon":171320,"nutmeg":171326,"oregano,dried oregano":171328,"cardamom":170919,
"star anise":171316,"vanilla,vanilla extract":173471,"fennel seed":171323,"mustard seed":170929,"saffron":170934,
# frozen
"frozen pea":170016,"frozen spinach":169287,"ice cream":167575,"frozen berry":168173,
}
# Cooked forms: a line that says cooked, boiled or roasted is read against these ("cooked rice"), never the raw row.
M.update({
"cooked rice":168878,"cooked basmati rice,cooked jasmine rice":168878,"cooked pasta,cooked spaghetti,cooked penne":169737,
"cooked lentil":172421,"cooked chickpea":173757,"cooked black bean":173735,"cooked kidney bean":173740,
"cooked pinto bean":175200,"cooked couscous":169700,"cooked quinoa":168917,"cooked bulgur":170287,
"cooked egg":173424,"cooked chicken breast":171477,"cooked salmon":175168,"cooked oat":173905,"cooked chicken thigh":172388,"cooked potato":170438,
"cooked sweet potato":168483,"cooked broccoli":169967,"cooked spinach":168463,"cooked carrot":170394,
})
IDS={'energyKcal':'1008','protein':'1003','fat':'1004','carbohydrate':'1005','fiber':'1079','sugars':'2000','sodium':'1093','saturatedFat':'1258'}
want=set(M.values())
food={r['fdc_id']:r for r in csv.DictReader(open(D+'food.csv'))}
nut={}
for r in csv.DictReader(open(D+'food_nutrient.csv')):
    if int(r['fdc_id']) in want: nut.setdefault(int(r['fdc_id']),{})[r['nutrient_id']]=r['amount']
rows=[]; bad=[]
for names,fid in M.items():
    f=food.get(str(fid))
    if not f: bad.append(fid); continue
    n=nut.get(fid,{})
    vals={k:(float(n[i]) if i in n else None) for k,i in IDS.items()}
    for name in names.split(','):
        rows.append({'name':name,'fdcId':fid,'dataType':'SR Legacy','description':f['description'],**vals})
rows.sort(key=lambda r:r['name'])
print(len(rows),'rows; missing ids:',bad,file=sys.stderr)
missing_energy=[r['name'] for r in rows if r['energyKcal'] is None]
print('no energy:',missing_energy,file=sys.stderr)
json.dump({'source':'USDA FoodData Central, SR Legacy (public domain, CC0 1.0)','basis':'per 100 g, as published','rows':rows},sys.stdout,indent=1,sort_keys=True)
