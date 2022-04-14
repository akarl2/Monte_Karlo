from ChemData import ChemData
import random
from Database import *
from Reactions import *

#specify reaction chemicals and reaction type
a = Glycerol()
b = C181()
rt = Esterification()

#Starting material mass and moles
a_mass = 92.09
b_mass = 564.94
mol_a = round(a_mass / a.mw, 3)
mol_b = round(b_mass / b.mw, 3)

if a.tg >= b.tg:
    species = a.tg
else:
    species = b.tg

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
    for i in range(0, a.pct):
        mc = random.randint(0, int(1/k1))
        if mc == 1:
            random_dist.append(1)
        else:
            random_dist.append(0)
    for i in range(0, a.sct):
        mc = random.randint(0, int(1/k2))
        if mc == int(1/k2):
            random_dist.append(1)
        else:
            random_dist.append(0)
    print(random_dist)


monte_karlo()










# #Deterimine concentrations of each species
# while rxn_species_dict["COOH"][-1] >= .01:
#     rate1 = k1 * rxn_species_dict["Primary"][-1] * rxn_species_dict["COOH"][-1]
#     rate2 = k2 * rxn_species_dict["Secondary"][-1] * rxn_species_dict["COOH"][-1]
#     for key in rxn_species_dict:
#         if key == "PrimaryE":
#             rxn_species_dict[key].append(round(rxn_species_dict[key][-1] + rate1, 6))
#             rxn_species_dict["Primary"].append(round(rxn_species_dict["Primary"][-1] - rate1, 6))
#             rxn_species_dict["COOH"].append(round(rxn_species_dict["COOH"][-1] - rate1, 6))
#         elif key == "SecondaryE":
#             rxn_species_dict[key].append(round(rxn_species_dict[key][-1] + rate2, 6))
#             rxn_species_dict["Secondary"].append(round(rxn_species_dict["Secondary"][-1] - rate2, 6))
#             rxn_species_dict["COOH"].append(round(rxn_species_dict["COOH"][-1] - rate2, 6))
#         else:
#             pass
#
# finished_rxn = {"PrimaryE": [rxn_species_dict["PrimaryE"][-1]], "SecondaryE": [rxn_species_dict["SecondaryE"][-1]], "Primary": [rxn_species_dict["Primary"][-1]], "Secondary": [rxn_species_dict["Secondary"][-1]],"COOH": [rxn_species_dict["COOH"][-1]]}
#
# rxn = ["PrimaryE" for i in range(int(finished_rxn["PrimaryE"][-1] * 10000))]
# rxn += ["SecondaryE" for i in range(int(finished_rxn["SecondaryE"][-1] * 10000))]
# rxn += ["Primary" for i in range(int(finished_rxn["Primary"][-1] * 10000))]
# rxn += ["Secondary" for i in range(int(finished_rxn["Secondary"][-1] * 10000))]
# #rxn += ["COOH" for i in range(int(finished_rxn["COOH"][-1] * 10000))]
#
#
# def monte_karlo():
#     while len(rxn) > 100 :
#         a = random.choice(rxn)
#         b = random.choice(rxn)
#         c = random.choice(rxn)
#         chemical = [a, b, c]
#         if chemical == ["Primary", "Secondary", "Primary"]:
#             rxn.remove(a)
#             rxn.remove(b)
#             rxn.remove(c)
#             final_products["Alcohol"].append(1)
#         elif chemical == ["PrimaryE", "Secondary", "Primary"]:
#             rxn.remove(a)
#             rxn.remove(b)
#             rxn.remove(c)
#             final_products["Gly1_Ole1"].append(1)
#         elif chemical == ["PrimaryE", "Secondary", "PrimaryE"]:
#             rxn.remove(a)
#             rxn.remove(b)
#             rxn.remove(c)
#             final_products["Gly1_Ole2"].append(1)
#         elif chemical == ["PrimaryE", "SecondaryE", "PrimaryE"]:
#             rxn.remove(a)
#             rxn.remove(b)
#             rxn.remove(c)
#             final_products["Gly1_Ole3"].append(1)
#         elif chemical == ["PrimaryE", "SecondaryE", "Primary"]:
#             rxn.remove(a)
#             rxn.remove(b)
#             rxn.remove(c)
#             final_products["Gly1_Ole2"].append(1)
#         elif chemical == ["Primary", "SecondaryE", "Primary"]:
#             rxn.remove(a)
#             rxn.remove(b)
#             rxn.remove(c)
#             final_products["Gly1_Ole1"].append(1)
#         else:
#             pass


















