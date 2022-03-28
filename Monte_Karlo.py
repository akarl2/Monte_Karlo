import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

composition = []
Butanol = []
Butanol_AC = []
Butanol_AC_X1 = []
Butanol_AC_X2 = []
Butanol_AC_X3 = []


def monte_karlo(n):
    for reaction in range(n):
        composition.clear()
        print(reaction)
        for butanol in range(100):
            composition.append("Butanol")
        for epi in range(60):
            reactant = random.choice(composition)
            if reactant == "Butanol":
                composition.remove(reactant)
                composition.append("Butanol AC")
            elif reactant == "Butanol AC":
                reactant = random.choice(composition)
                if reactant == "Butanol AC":
                    reactant = random.choice(composition)
                    if reactant == "Butanol AC":
                        reactant = random.choice(composition)
                        if reactant == "Butanol AC":
                            composition.remove(reactant)
                            composition.append("Butanol AC X+1")
                        else:
                            composition.remove(reactant)
                            composition.append("Butanol AC")
                    else:
                        composition.remove(reactant)
                        composition.append("Butanol AC")
                else:
                    composition.remove(reactant)
                    composition.append("Butanol AC")

            elif reactant == "Butanol AC X+1":
                reactant = random.choice(composition)
                if reactant == "Butanol AC X+1":
                    reactant = random.choice(composition)
                    composition.remove(reactant)
                    composition.append("Butanol AC X+2")
                else:
                    composition.remove(reactant)
                    composition.append("Butanol AC X+1")
            else:
                composition.remove(reactant)
                composition.append("Butanol AC X+3")
        x = "Butanol"
        d = Counter(composition)
        Butanol.append(d[x])
        x = "Butanol AC"
        d = Counter(composition)
        Butanol_AC.append(d[x])
        x = "Butanol AC X+1"
        d = Counter(composition)
        Butanol_AC_X1.append(d[x])
        x = "Butanol AC X+2"
        d = Counter(composition)
        Butanol_AC_X2.append(d[x])
        x = "Butanol AC X+3"
        d = Counter(composition)
        Butanol_AC_X3.append(d[x])


monte_karlo(10001)
average_Butanol = sum(Butanol) / len(Butanol)
average_ButanolAC = sum(Butanol_AC) / len(Butanol_AC)
average_ButanolACX1 = sum(Butanol_AC_X1) / len(Butanol_AC_X1)
average_ButanolACX2 = sum(Butanol_AC_X2) / len(Butanol_AC_X2)
average_ButanolACX3 = sum(Butanol_AC_X3) / len(Butanol_AC_X3)

wt_average_butanol = average_Butanol * 74.14
wt_average_butanolAC = average_ButanolAC * 166.64
wt_average_butanolACX1 = average_ButanolACX1 * 259.16
wt_average_butanolACX2 = average_ButanolACX2 * 351.68
wt_average_butanolACX3 = average_ButanolACX3 * 444.2

total = wt_average_butanol + wt_average_butanolAC + wt_average_butanolACX1 + wt_average_butanolACX2 + wt_average_butanolACX3

print(f"Butanol: {round(wt_average_butanol / total * 100, 2)}")
print(f"Butanol AC: {round(wt_average_butanolAC / total * 100, 2)}")
print(f"Butanol AC X+1: {round(wt_average_butanolACX1 / total * 100, 2)}")
print(f"Butanol AC X+2: {round(wt_average_butanolACX2 / total * 100, 2)}")
print(f"Butanol AC X+3+: {round(wt_average_butanolACX3 / total * 100, 2)}")

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
Species = ['Butanol', 'AC', 'X1', 'X2', 'X3']
Count = [wt_average_butanolAC, wt_average_butanolAC, wt_average_butanolACX1, wt_average_butanolACX2,
         wt_average_butanolACX3]
ax.bar(Species, Count)
# plt.show()
