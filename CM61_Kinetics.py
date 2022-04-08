import math
import matplotlib.pyplot as plt
from ChemData import ChemData

gA = 285.52
gB = 359.39
gC2 = 0
gC = 0
gD = 0
gE = 0
gF = 0

molesA = gA / ChemData["Butanol"]["MW"]
molesB = gB / ChemData["Epichlorohydrin"]["MW"]
molesC = 0
molesC2 = 0
molesD = 0
molesE = 0
molesF = 0

k1 = .00001
k2, k3, k4 = .0000006, .0000006, .0000006
k5 = .00000018

p_A = []
p_B = []
p_C = []
p_C2 = []
p_D = []
p_E = []
p_F = []

while molesB >= 0.01:
    rate1 = k1 * molesA * molesB
    rate2 = k2 * molesC * molesB
    rate3 = k3 * molesD * molesB
    rate4 = k4 * molesE * molesB
    rate5 = k5 * molesA * molesB

    molesA = molesA - rate1
    molesB = molesB - rate1 - rate2 - rate3 - rate4 - rate5
    molesC2 = molesC2 + rate5
    molesC = molesC + rate1 - rate2
    molesD = molesD + rate2 - rate3
    molesE = molesE + rate3 - rate4
    molesF = molesF + rate4

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
