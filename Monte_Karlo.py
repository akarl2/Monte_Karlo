import random
from Database import *
from Reactions import *
import sys
import itertools

#Converts string name to class name
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

#specify reaction chemicals and reaction type
a = Glycerol()
b = C181()
rt = Esterification()

#Starting material mass and moles
a.mass = 92.1
b.mass = 282.5
a.mol = round(a.mass / a.mw, 3) * 3
b.mol = round(b.mass / b.mw, 3) * 3

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

final_molar_amounts = ({a.sn: [0], b.sn: [0]})
final_molar_amounts.update({f"{a.sn}({1})_{b.sn}({str(i)})": [0] for i in range(1, species + 1)})


#Specifty rate constants
k1 = 1
k2 = .5

#Creats starting composition list
composition = []
for i in range(0, int(a.mol)):
    composition.extend(group.__name__ for group in a.comp)

#Reacts away b.mol until gone.  Still need to add different rate constants
while b.mol != 0:
    MC = random.choices(list(enumerate(composition)), weights=[1 for rg in range(len(composition))], k=1)[0]
    if MC[1] != rt.rp.__name__:
        composition[MC[0]] = rt.rp.__name__
        b.mol -= 1
    else:
        pass
    print(b.mol)

composition = [composition[x:x+3] for x in range(0, len(composition), 3)]
print(composition)

for chemical in composition:
    if chemical == [a.comp[i].__name__ for i in range(0, len(a.comp))]:
        final_molar_amounts[a.sn][0] += 1
    else:
        pass
print(final_molar_amounts)





# for i in range(1, int(b.mol) + 1 * 100000):
#     choice = random.choices(a.comp, weights=[k1, k2, k1], k=1)[0].__name__
#     if choice == a.prg.__name__:
#         P_Hydroxyl += 1
#     elif choice == a.srg.__name__:
#         S_Hydroxyl += 1
#     #print(str_to_class(choice).__name__)
# print(f"P_Hydroxyl: {P_Hydroxyl}, S_hydroxyl: {S_Hydroxyl}")
# print(P_Hydroxyl / S_Hydroxyl)


# def monte_karlo():
#     random_dist = []
#     for i in range(0, a.comp.count(a.prg)):
#         mc = random.randint(0, int(1/k1))
#         if mc == 1:
#             random_dist.append(1)
#         else:
#             random_dist.append(0)
#     for i in range(0, a.comp.count(a.srg)):
#         mc = random.randint(0, int(1/k2))
#         if mc == int(1/k2):
#             random_dist.append(1)
#         else:
#             random_dist.append(0)
#     print(random_dist)
#
#
# monte_karlo()



























