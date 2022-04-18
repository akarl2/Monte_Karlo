import collections
import random
from Database import *
from Reactions import *
import sys
import tkinter as tk
import pandas

#Converts string name to class name
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

#specify reaction chemicals and reaction type
a = Glycerol()
b = C181()
rt = Esterification()
EOR = 1

#Starting material mass and moles
a.mass = 92.09
b.mass = 282.47
a.mol = round(a.mass / a.mw, 3) * 10000
b.mol = round(b.mass / b.mw, 3) * 10000
unreacted = b.mol - (EOR * b.mol)
b.mol = EOR * b.mol
bms = b.mol


#Define limiting reagent
try:
    if len(a.comp) >= len(b.comp):
        species = len(a.comp)
except TypeError:
    species = len(a.comp)

#Creates final product name(s) from starting material name(s)
final_product_names = [a.sn, b.sn]
final_product_names.extend([f"{a.sn}({1})_{b.sn}({str(i)})" for i in range(1, species + 1)])

#Creates final product molar masses from final product name(s)
final_product_masses = ({a.sn: round(a.mw, 1), b.sn: round(b.mw, 1)})
final_product_masses.update({f"{a.sn}({1})_{b.sn}({str(i)})": round(a.mw + i * b.mw - i * rt.wl, 1) for i in range(1, species + 1)})

#Creates starting molar amounts from final product names
starting_molar_amounts = ({a.sn: [a.mol], b.sn: [b.mol]})
starting_molar_amounts.update({f"{a.sn}({1})_{b.sn}({str(i)})": [0] for i in range(1, species + 1)})

#Creates finish molar amounts from final product names
final_molar_amounts = ({a.sn: [0], b.sn: [0]})
final_molar_amounts.update({f"{a.sn}({1})_{b.sn}({str(i)})": [0] for i in range(1, species + 1)})

#Specifty rate constants
k1 = 1
k2 = 0

#Creats starting composition list
composition = []
for i in range(0, int(a.mol)):
    composition.extend(group.__name__ for group in a.comp)


#Creates weights from starting commposition list
weights = []
for group in composition:
    if group == a.prg.__name__:
        weights.append(k1)
    else:
        weights.append(k2)

#Reacts away b.mol until gone.  Still need to add different rate constants(weights)
while b.mol != 0:
    MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
    if MC[1] != rt.rp.__name__:
        composition[MC[0]] = rt.rp.__name__
        b.mol -= 1
        weights[MC[0]] = 0
    else:
        pass
    print(round(100-(b.mol/bms*100), 2))

#Seperates composition into compounds
composition = [composition[x:x+len(a.comp)] for x in range(0, len(composition), len(a.comp))]
composition_tuple = [tuple(l) for l in composition]

#tabulates results from reacion
rxn_summary = collections.Counter(composition_tuple)

#Convert rxn_summary to dataframe
rxn_summary_df = pandas.DataFrame.from_dict(rxn_summary, orient='index')
rxn_summary_df.loc[f"{b.sn}"] = [unreacted]


print(rxn_summary_df)
