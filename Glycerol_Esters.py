import matplotlib.pyplot as plt
from ChemData import ChemData

Alcohol = "Glycerol"
COOH = "Oleic Acid"

Alcoholg = 100
COOHg = 50
species = 3
rxn_mass_dict = {"Gly1_Ole" + str(i): 0 for i in range(1, species + 1)}

molesA = Alcoholg / ChemData[Alcohol]["MW"]
molesB = COOHg / ChemData[COOH]["MW"]
rxn_moles_dict = {"Gly1_0le" + str(i): [0] for i in range(1, species + 1)}
rxn_molar_mass = {"Gly1_0le" + str(i): ChemData[Alcohol]["MW"] + i * ChemData[COOH]["MW"] - (i*18.01) for i in range(1, species + 1)}
p_mass_dict = {"Gly1_0le" + str(i): [] for i in range(1, species + 1)}

rxn_moles_dict[Alcohol] = [round(molesA, 4)]
rxn_moles_dict[COOH] = [round(molesB, 4)]

k1 = .0001
k2 = k1 / 2

while molesB >= 0.001:
    rate1 = k1 * molesA * molesB
    rate2 = k2 * molesA * molesB

    for key in rxn_moles_dict:
        if key != Alcohol or COOH:
            update = rxn_moles_dict[key][-1] + rate1
            rxn_moles_dict[key].append(round(update, 4))
        else:
            update = rxn_moles_dict[key][-1] - rate1
            rxn_moles_dict[key].append(round(update, 4))
    print(rxn_moles_dict)




