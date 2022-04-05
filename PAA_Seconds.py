import math
import matplotlib.pyplot as plt
from scipy import integrate
from ChemData import ChemData

gA = 500
gB = 500
gC = 0
gD = 250

ngA = gA
ngB = gB * .7
ngC = 0
ngD = gD + (gB*.3)

molesA = ngA / ChemData["Glacial Acetic Acid"]["MW"]
molesB = ngB / ChemData["70% H2O2"]["MW"]
molesC = ngC / ChemData["Peracetic Acid"]["MW"]
molesD = ngD / ChemData["Water"]["MW"]

volume = (gA * ChemData["Glacial Acetic Acid"]["DEN"] + gB * ChemData["70% H2O2"]["DEN"] + gC *
          ChemData["Peracetic Acid"]["DEN"] + gD * ChemData["Water"]["DEN"]) / 1000

MA = molesA / volume
MB = molesB / volume
MC = molesC / volume
MD = molesD / volume

minutes = 300
s_conA = MA
s_conB = MB
s_conC = MC
s_conD = MD
cat_con = .314
temperature = 320
k1 = (6.83 * 10 ** 8) * (math.sqrt(s_conA * (1.75 * 10 ** -5)) + cat_con + math.sqrt(cat_con * (1.3 * 10 ** -2))) * (
    math.exp((-57846.15) / (8.134 * temperature)))
k2 = (6.73 * 10 ** 8) * (math.sqrt(s_conA * (1.75 * 10 ** -5)) + cat_con + math.sqrt(cat_con * (1.3 * 10 ** -2))) * (
    math.exp((-60407.78) / (8.134 * temperature)))
conA = s_conA
conB = s_conB
conC = s_conC
conD = s_conD
seconds = 1
con_A = [conA]
con_B = [conB]
con_C = [conC]
con_D = [conD]
watts_list = []
energy = 0
std_enthalpy = -13700
rxn_enthalpy = std_enthalpy + (((ChemData["Peracetic Acid"]["Cp"] + ChemData["Water"]["Cp"]) - (
        ChemData["70% H2O2"]["Cp"]+ ChemData["Glacial Acetic Acid"]["Cp"])) * (temperature - 298.15))


while seconds < minutes * 60:
    rate = (((k1 * conA * conB) - (k2 * conC * conD)) / 60) / 60
    conA = conA - rate
    conB = conB - rate
    conC = conC + rate
    conD = conD + rate
    con_A.append(conA), con_B.append(conB), con_C.append(conC), con_D.append(conD)
    keq = (conC * conD) / (conA * conB)
    molesC = rate * volume
    print(molesC)
    energy = energy + (rxn_enthalpy * molesC)
    watts = molesC * rxn_enthalpy * -1
    watts_list.append(watts)
    seconds += 1


def plot_conc():
    plt.xlabel("Time (seconds)")
    plt.ylabel("Concentration (mol/L)")
    plt.xlim(0, minutes * 60), plt.ylim(0, 8)
    plt.plot(con_A), plt.plot(con_B), plt.plot(con_C), plt.plot(con_D)
    plt.show()


def plot_watts():
    plt.xlabel("Time (seconds)")
    plt.ylabel("Watts")
    plt.xlim(0, minutes * 60), plt.ylim(0, max(watts_list) * 1.1)
    plt.plot(watts_list)
    plt.show()

plot_conc()

