from ChemData import ChemData
import random

Alcohol = "Glycerol"
COOH = "Oleic Acid"

Alcoholg = 100
COOHg = 100
species = 3
rxn_mass_dict = {"Gly1_Ole" + str(i): 0 for i in range(1, species + 1)}

molesA = Alcoholg / ChemData[Alcohol]["MW"]
molesB = COOHg / ChemData[COOH]["MW"]
rxn_moles_dict = {"Gly1_0le" + str(i): [0] for i in range(1, species + 1)}
rxn_molar_mass = {"Gly1_0le" + str(i): ChemData[Alcohol]["MW"] + i * ChemData[COOH]["MW"] - (i*18.01) for i in range(1, species + 1)}
p_mass_dict = {"Gly1_0le" + str(i): [] for i in range(1, species + 1)}

rxn_moles_dict[Alcohol] = [round(molesA, 6)]
rxn_moles_dict[COOH] = [round(molesB, 6)]

k1 = 1
k2 = .5

while rxn_moles_dict[COOH][-1] >= 0.001:
    rate1 = k2 * rxn_moles_dict[Alcohol][-1] * rxn_moles_dict[COOH][-1]
    rate2 = k2 * rxn_moles_dict[Alcohol][-1] * rxn_moles_dict[COOH][-1]

    for key in rxn_moles_dict:
        if key == Alcohol or key == COOH:
            update = rxn_moles_dict[key][-1] - rate1
            rxn_moles_dict[key].append(round(update, 6))
        else:
            update = rxn_moles_dict[key][-1] + rate1
            rxn_moles_dict[key].append(round(update, 6))
    print(rxn_moles_dict[Alcohol][-1], rxn_moles_dict[COOH][-1])







