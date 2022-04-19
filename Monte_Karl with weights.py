import collections
import random
from Database import *
from Reactions import *
import sys
import tkinter
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
a.mass = 92.09382
b.mass = 564.94
a.mol = round(a.mass / a.mw, 3) * 1000
b.mol = round(b.mass / b.mw, 3) * 1000
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
print(final_product_masses)

#Creates starting molar amounts from final product names
starting_molar_amounts = ({a.sn: [a.mol], b.sn: [b.mol]})
starting_molar_amounts.update({f"{a.sn}({1})_{b.sn}({str(i)})": [0] for i in range(1, species + 1)})

#Creates finish molar amounts from final product names
final_molar_amounts = ({a.sn: [0], b.sn: [0]})
final_molar_amounts.update({f"{a.sn}({1})_{b.sn}({str(i)})": [0] for i in range(1, species + 1)})

#Specifty rate constants
k1 = 1
k2 = .5

#Creats starting composition list
composition = []
for i in range(0, int(a.mol)):
    composition.extend(group for group in a.compmw)

#Creates weights from starting commposition list
weights = []
for group in composition:
    if group == a.prgmw:
        weights.append(k1)
    else:
        weights.append(k2)

#Reacts away b.mol until gone.  Still need to add different rate constants(weights)
while b.mol >= 0:
    MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
    if MC[1] != a.prgmw or MC[1] != a.srgmw:
        composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
        b.mol -= 1
        weights[MC[0]] = 0
    else:
        pass
    print(round(100-(b.mol/bms*100), 2))

#Seperates composition into compounds
composition = [composition[x:x+len(a.comp)] for x in range(0, len(composition), len(a.comp))]
composition_tuple = [tuple(l) for l in composition]

#Tabulates final composition
rxn_summary = collections.Counter(composition_tuple)
print(rxn_summary)

RS = []
for key in rxn_summary:
    MS = round(sum(key),1)
    for item in final_product_masses:
        #if MS == item then add key and its value to RS
        if MS == final_product_masses[item]:
            RS.append((item, rxn_summary[key]))
# Convert RS to dictionary
RS_dict = {}
for item in RS:
    RS_dict[item[0]] = item[1]

rxn_summary_df = pandas.DataFrame.from_dict(rxn_summary, orient='index', columns=['Moles'])
print(rxn_summary_df)

#Convert rxn_summary to dataframe
rxn_summary_df = pandas.DataFrame.from_dict(RS_dict, orient='index', columns=['Moles'])
rxn_summary_df.loc[f"{b.sn}"] = [unreacted]

#Multiple datafrome rows by final product molar masses and add to new column
rxn_summary_df['Final Product Mass'] = rxn_summary_df.index.map(final_product_masses.get)
#multiply the values in each row
rxn_summary_df['Final Product Mass Amount'] = rxn_summary_df['Final Product Mass'] * rxn_summary_df['Moles']
#divide final product mass amount bu sum of final product mass amount
rxn_summary_df['Final Product Percent'] = rxn_summary_df['Final Product Mass Amount'] / rxn_summary_df['Final Product Mass Amount'].sum() * 100


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