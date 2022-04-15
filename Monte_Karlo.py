from ChemData import ChemData
import random
from Database import *
from Reactions import *

#specify reaction chemicals and reaction type
a = Glycerol()
b = C181()
rt = Esterification()

#Starting material mass and moles
a.mass = 92.1
b.mass = 282.5
mol_a = round(a.mass / a.mw, 3)
mol_b = round(b.mass / b.mw, 3)

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
starting_molar_amounts = ({a.sn: [mol_a], b.sn: [mol_b]})
starting_molar_amounts.update({f"{a.sn}({1})_{b.sn}({str(i)})": [0] for i in range(1, species + 1)})

print(final_product_names)
print(final_product_masses)
print(starting_molar_amounts)



#Specifty rate constants
k1 = 1
k2 = .5

#randomly select mass from final product molar masses
reactant_mw = random.choice(list(final_product_masses.values()))



def monte_karlo():
    random_dist = []
    for i in range(0, a.comp.count(a.prg)):
        mc = random.randint(0, int(1/k1))
        if mc == 1:
            random_dist.append(1)
        else:
            random_dist.append(0)
    for i in range(0, a.comp.count(a.srg)):
        mc = random.randint(0, int(1/k2))
        if mc == int(1/k2):
            random_dist.append(1)
        else:
            random_dist.append(0)
    print(random_dist)


monte_karlo()



























