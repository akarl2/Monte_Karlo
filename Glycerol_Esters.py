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
rxn_moles_dict = {"Gly1_0le" + str(i): 0 for i in range(1, species + 1)}
rxn_molar_mass = {"Gly1_0le" + str(i): ChemData[Alcohol]["MW"] + i * ChemData[COOH]["MW"] - (i*18.01) for i in range(1, species + 1)}
p_mass_dict = {"Gly1_0le" + str(i): [] for i in range(1, species + 1)}

print(rxn_mass_dict)
print(rxn_moles_dict)
print(rxn_molar_mass)
print(p_mass_dict)

k1 = .0001
k2 = k1 / 2

while molesB >= 0.001:
    rate1 = k1 * molesA * molesB
    rate2 = k2 * molesA * molesB

    #itterate through each item in the rxn_moles_dict and multiply by the rate and multply by the key in rxn_molar_mass and append to p_mass_dict
    for key in rxn_moles_dict:
        rxn_moles_dict[key] = rxn_moles_dict[key] + rate1 * rxn_molar_mass[key]
        p_mass_dict[key].append(rxn_moles_dict[key])


    print(rxn_moles_dict)

    p_mol_dict = {"A1_0" + str(i): 0 for i in range(1, species + 1)}

    gA = molesA * ChemData["Butanol"]["MW"]
    gb = molesB * ChemData["Epichlorohydrin"]["MW"]
    gC = molesC * ChemData["Butanol AC"]["MW"]
    gC2 = molesC2 * ChemData["Butanol ACBRXN"]["MW"]
    gD = molesD * ChemData["Butanol AC2"]["MW"]
    gE = molesE * ChemData["Butanol AC3"]["MW"]
    gF = molesF * ChemData["Butanol AC4"]["MW"]

    total = gA + gb + gC + gD + gE + gF + gC2
    p_A.append(round((gA / total) * 100, 2))
    p_B.append(round((gB / total) * 100, 2))
    p_C.append(round((gC / total) * 100, 2))
    p_C2.append(round((gC2 / total) * 100, 2))
    p_D.append(round((gD / total) * 100, 2))
    p_E.append(round((gE / total) * 100, 2))
    p_F.append(round((gF / total) * 100, 2))
    print(round(molesB, 4))


def plot_conc():
    plt.figure("Percent composition")
    plt.xlabel("Reactions")
    plt.ylabel("Percent (%)")
    plt.ylim(0, 85)
    plt.xlim(0, len(p_A))
    plt.plot(p_A, label="Butanol")
    plt.plot(p_C)
    plt.plot(p_C2)
    plt.plot(p_D)
    plt.plot(p_E)
    plt.plot(p_F)
    plt.show()

print(p_A[-1], p_C[-1], p_C2[-1], p_D[-1], p_E[-1], p_F[-1])
plot_conc()
