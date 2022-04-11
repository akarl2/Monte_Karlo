from ChemData import ChemData
import random

#Starting material names
Alcohol = "Glycerol"
COOH = "Oleic Acid"

#Starting material masses
Alcoholg = 92.09
COOHg = 564.94

species = 3
rxn_mass_dict = {"Gly1_Ole" + str(i): 0 for i in range(1, species + 1)}
final_products = {"Gly1_Ole" + str(i): [0] for i in range(1, species + 1)}
final_products.update({"Alcohol": [0], "COOH": [0]})

molesA = Alcoholg / ChemData[Alcohol]["MW"]
molesB = COOHg / ChemData[COOH]["MW"]
rxn_moles_dict = {Alcohol: [round(molesA, 6)], COOH: [round(molesB, 6)]}
rxn_species_dict = {"PrimaryE": [0], "SecondaryE": [0]}
rxn_molar_mass = {"Gly1_0le" + str(i): ChemData[Alcohol]["MW"] + i * ChemData[COOH]["MW"] - (i*18.01) for i in range(1, species + 1)}
p_mass_dict = {"Gly1_0le" + str(i): [] for i in range(1, species + 1)}

rxn_species_dict["Primary"] = [rxn_moles_dict[Alcohol][-1] * ChemData[Alcohol]["OHp"]]
rxn_species_dict["Secondary"] = [rxn_moles_dict[Alcohol][-1] * ChemData[Alcohol]["OHs"]]
rxn_species_dict["COOH"] = [rxn_moles_dict[COOH][-1] * ChemData[COOH]["RG"]]
print(rxn_species_dict)

#Reaction constants
k1 = .001
k2 = k1 / 2

#Deterimine concentrations of each species
while rxn_species_dict["COOH"][-1] >= .01:
    rate1 = k1 * rxn_species_dict["Primary"][-1] * rxn_species_dict["COOH"][-1]
    rate2 = k2 * rxn_species_dict["Secondary"][-1] * rxn_species_dict["COOH"][-1]
    for key in rxn_species_dict:
        if key == "PrimaryE":
            rxn_species_dict[key].append(round(rxn_species_dict[key][-1] + rate1, 6))
            rxn_species_dict["Primary"].append(round(rxn_species_dict["Primary"][-1] - rate1, 6))
            rxn_species_dict["COOH"].append(round(rxn_species_dict["COOH"][-1] - rate1, 6))
        elif key == "SecondaryE":
            rxn_species_dict[key].append(round(rxn_species_dict[key][-1] + rate2, 6))
            rxn_species_dict["Secondary"].append(round(rxn_species_dict["Secondary"][-1] - rate2, 6))
            rxn_species_dict["COOH"].append(round(rxn_species_dict["COOH"][-1] - rate2, 6))
        else:
            pass

finished_rxn = {"PrimaryE": [rxn_species_dict["PrimaryE"][-1]], "SecondaryE": [rxn_species_dict["SecondaryE"][-1]], "Primary": [rxn_species_dict["Primary"][-1]], "Secondary": [rxn_species_dict["Secondary"][-1]],"COOH": [rxn_species_dict["COOH"][-1]]}

rxn = ["PrimaryE" for i in range(int(finished_rxn["PrimaryE"][-1] * 10000))]
rxn += ["SecondaryE" for i in range(int(finished_rxn["SecondaryE"][-1] * 10000))]
rxn += ["Primary" for i in range(int(finished_rxn["Primary"][-1] * 10000))]
rxn += ["Secondary" for i in range(int(finished_rxn["Secondary"][-1] * 10000))]
rxn += ["COOH" for i in range(int(finished_rxn["COOH"][-1] * 10000))]
print(finished_rxn)

#Randomly distribute elements among final_products
def monte_karlo():
    element = random.choice(rxn)
    if element == "PrimaryE":















