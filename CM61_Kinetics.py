import math
import matplotlib.pyplot as plt
from ChemData import ChemData

gA = 200
gB = 140
gC = 0
gD = 0

molesA = gA / ChemData["Butanol"]["MW"]
molesB = gB / ChemData["Epichlorohydrin"]["MW"]
molesC = 0
molesD = 0
molesE = 0
molesF = 0

volume = ((gA * ChemData["Butanol"]["DEN"]) + (gB * ChemData["Epichlorohydrin"]["DEN"])) / 1000

minutes = 1000
s_conA = molesA / volume
s_conB = molesB / volume
s_conC = 0
s_conD = 0
s_conE = 0
s_conF = 0


cat_con = 0
temperature = 325
k1 = .0001
k2 = 0.00001
k3 = 0.00001
k4 = 0.00001

conA = s_conA
conB = s_conB
conC = 0
conD = 0
conE = 0
conF = 0

seconds = 1
moles_A = [molesA]
moles_B = [molesB]
moles_C = [molesC]
moles_D = [molesD]
moles_E = [molesE]
moles_F = [molesF]
watts_list = [0]
energy = 0

std_enthalpy = -93000

while molesB >= 0.01:
    rate1 = k1 * molesA * molesB
    rate2 = k2 * molesC * molesB
    rate3 = k3 * molesD * molesB
    rate4 = k4 * molesE * molesB

    molesA = molesA - rate1
    molesB = molesB - rate1
    molesC = molesC + rate1 - rate2
    molesD = molesD + rate2 - rate3
    molesE = molesE + rate3 - rate4
    molesF = molesF + rate4

    gA = molesA * ChemData["Butanol"]["MW"]
    gb = molesB * ChemData["Epichlorohydrin"]["MW"]
    gC = molesC * ChemData["Butanol AC"]["MW"]
    gD = molesD * ChemData["Butanol AC2"]["MW"]
    gE = molesE * ChemData["Butanol AC3"]["MW"]
    gF = molesF * ChemData["Butanol AC4"]["MW"]


    total = gA + gb + gC + gD + gE + gF

    pA = gA / total * 100    # Percentages
    pB = gb / total * 100
    pC = gC / total * 100
    pD = gD / total * 100
    pE = gE / total * 100
    pF = gF / total * 100



    moles_A.append(molesA), moles_B.append(molesB), moles_C.append(molesC), moles_D.append(molesD), moles_E.append(molesE), moles_F.append(molesF)
    print(round(pA,2), round(pB,2), round(pC,2), round(pD,2), round(pE,2), round(pF,2))
    #print(molesA, molesB, molesC, molesD, molesE, molesF)
    seconds += 1


def plot_conc():
    plt.figure("Concentration")
    plt.xlabel("Time (seconds)")
    plt.ylabel("moles")
    plt.xlim(0, minutes * 60), plt.ylim(0, 100)
    plt.plot(pA,pB,pC,pD,pE,pF)
    #plt.plot(moles_A, label="A"), plt.plot(moles_B, label="B"), plt.plot(moles_C, label="C"), plt.plot(moles_D, label="D"), plt.plot(moles_E, label="E"), plt.plot(moles_F, label="F")
    plt.show()


plot_conc()

