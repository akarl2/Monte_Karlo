import collections
import random
import sys
import tkinter
import pandas
from Database import *
from Reactions import *
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 100)

#Converts string name to class name
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

#specify reaction chemicals and reaction type
a = DETA()
b = Adipic_Acid()
rt = PolyEsterification()
Samples = 1000
#Specify extend of reaction (EOR)
EOR = 1

#Starting material mass and moles
a.mass = 85.33
b.mass = 250
a.mol = round(a.mass / a.mw, 3) * Samples
b.mol = round(b.mass / b.mw, 3) * Samples
unreacted = b.mol - (EOR * b.mol)
b.mol = EOR * b.mol
bms = b.mol

#Creates final product name(s) from starting material name(s)
final_product_names = [a.sn, b.sn]
final_product_names.extend([f"{a.sn}({1})_{b.sn}({str(i)})" for i in range(1, 1001)])

#Creates final product molar masses from final product name(s)
final_product_masses = ({a.sn: round(a.mw, 1), b.sn: round(b.mw, 1)})
if rt != PolyEsterification():
    final_product_masses.update({f"{a.sn}({i})_{b.sn}({str(i)})": round(a.mw + i * b.mw - i * rt.wl, 1) for i in range(1, 1001)})
elif rt == PolyEsterification():
    final_product_masses.update({f"{a.sn}({i})_{b.sn}({str(i)})": round(i * a.mw + i * b.mw - i * rt.wl, 1) for i in range(1, 1001)})
    final_product_masses.update({f"{a.sn}({i-1})_{b.sn}({str(i)})": round(i-1 * a.mw + i * b.mw - i * rt.wl, 1) for i in range(2, 1001)})
print(final_product_masses)

#Specifty rate constants
prgK = 1
srgK = 0
cgK = 1

#Creats starting composition list
composition = []
try:
    for i in range(0, int(a.mol)):
        composition.extend(group for group in a.comp)
except TypeError:
    for i in range(0, int(a.mol)):
        composition.append(a.mw)
print(composition)

#Create weights from starting composition list
weights = []
for group in composition:
    if group == a.prgmw:
        weights.append(prgK)
    else:
        weights.append(srgK)
print(weights)
#Reacts away b.mol until gone.  Still need to add different rate constants(weights)
if rt != PolyEsterification():
    while b.mol >= 0:
        MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
        if MC[1] == a.prgmw or MC[1] == a.srgmw:
            composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
            b.mol -= 1
            weights[MC[0]] = cgK
        else:
            composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
            b.mol -= 1
        print(round(100-(b.mol/bms*100), 2))
elif rt == PolyEsterification():
    while b.mol >= 0:
        MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
        if MC[1] == a.prgmw or MC[1] == a.srgmw:
            composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
            b.mol -= 1
            weights[MC[0]] = cgK
        else:
            composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
            b.mol -= 1
        print(round(100-(b.mol/bms*100), 2))

#Seperates composition into compounds
try:
    composition = [composition[x:x+len(a.comp)] for x in range(0, len(composition), len(a.comp))]
    composition_tuple = [tuple(l) for l in composition]
except TypeError:
    composition = [composition[x:x+1] for x in range(0, len(composition), 1)]
    composition_tuple = [tuple(l) for l in composition]

#Tabulates final composition and converts to dataframe
rxn_summary = collections.Counter(composition_tuple)
RS = []
for key in rxn_summary:
    MS = round(sum(key), 1)
    for item in final_product_masses:
        if MS == final_product_masses[item]:
            RS.append((item, rxn_summary[key]))
rxn_summary_df = pandas.DataFrame(RS, columns=['Product', 'Count'])
rxn_summary_df.set_index('Product', inplace=True)
rxn_summary_df.loc[f"{b.sn}"] = [unreacted]


#Add colums to dataframe
rxn_summary_df['Molar Mass'] = rxn_summary_df.index.map(final_product_masses.get)
rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
rxn_summary_df['Mass'] = rxn_summary_df['Molar Mass'] * rxn_summary_df['Count']
rxn_summary_df['(%)'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)


print(rxn_summary_df)

#---------------------------------------------------User-Interface----------------------------------------------#
# window = tkinter.Tk()
# window.title("Monte Karlo")
# window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
# a.mass = tkinter.Entry(window)
# a.mass.grid(row=1, column=1)
# b.mass = tkinter.Entry(window)
# b.mass.grid(row=2, column=1)
# a = tkinter.Entry(window)
# a.grid(row=3, column=1)
# b = tkinter.Entry(window)
# b.grid(row=4, column=1)
#
# # Add a label for the interations entry
# tkinter.Label(window, text="Grams of A: ").grid(row=1, column=0)
# tkinter.Label(window, text="Grams of B: ").grid(row=2, column=0)
# tkinter.Label(window, text="Reactant A: ").grid(row=3, column=0)
# tkinter.Label(window, text="Reactant B: ").grid(row=4, column=0)
#
#
# window.mainloop()