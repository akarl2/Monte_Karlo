import math
import matplotlib.pyplot as plt
from ChemData import ChemData

gA = 100
gB = 200
gC = 0
gD = 0

ngA = gA
ngB = gB * .7
ngC = 0
ngD = gD + (gB*.3)

molesA = ngA / ChemData["Formic Acid"]["MW"]
molesB = ngB / ChemData["70% H2O2"]["MW"]
molesC = ngC / ChemData["Peraformic Acid"]["MW"]
molesD = ngD / ChemData["Water"]["MW"]

volume = (gA * ChemData["Formic Acid"]["DEN"] + gB * ChemData["70% H2O2"]["DEN"] + gC *
          ChemData["Peraformic Acid"]["DEN"] + gD * ChemData["Water"]["DEN"]) / 1000
print(volume)

MA = molesA / volume
MB = molesB / volume
MC = molesC / volume
MD = molesD / volume

minutes = 1000
s_conA = MA
s_conB = MB
s_conC = MC
s_conD = MD
cat_con = 0
temperature = 315
k1 = (1.20 * 10 ** -3) * math.exp((-55304/8.314)*(1/temperature - 1/323))
print(k1)
k2 = (1.60 * 10 ** -4) * math.exp((-105073/8.314)*(1/temperature - 1/323))
Ke = 1.60 * math.exp((-10000/8.314)*(1/298 - 1/temperature))

conA = s_conA
conB = s_conB
conC = s_conC
conD = s_conD
seconds = 1
con_A = [conA]
con_B = [conB]
con_C = [conC]
con_D = [conD]
watts_list = [0]
energy = 0
H_concentration = math.sqrt(s_conA * (1.8 * 10 ** -4))

decomp_enthalpy = -100000
std_enthalpy = -4840
rxn_enthalpy = std_enthalpy + (((ChemData["Peracetic Acid"]["Cp"] + ChemData["Water"]["Cp"]) - (
        ChemData["70% H2O2"]["Cp"]+ ChemData["Glacial Acetic Acid"]["Cp"])) * (temperature - 298.15))

while seconds < minutes * 60:
    rate1 = (k1 * conA * conB * H_concentration) * (1-(((conC * conD) / (conA * conB))*(1/Ke)))
    rate2 = (k2 * conC)
    conA = conA - rate1
    conB = conB - rate1
    conC = conC + rate1 - rate2
    conD = conD + rate1 + rate2
    con_A.append(conA), con_B.append(conB), con_C.append(conC), con_D.append(conD)
    keq = (conC * conD) / (conA * conB)
    moles_PfA_Formed = rate1 * volume
    moles_PFA_Decomp = rate2 * volume
    energy = energy + (rxn_enthalpy * molesC)
    watts = ((moles_PfA_Formed * rxn_enthalpy) + (moles_PFA_Decomp * decomp_enthalpy)) * -1
    watts_list.append(watts)
    print(seconds, conC)
    seconds += 1


def plot_conc():
    plt.figure("Concentration")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Concentration (mol/L)")
    plt.xlim(0, minutes * 60), plt.ylim(0, 8)
    plt.plot(con_A), plt.plot(con_B), plt.plot(con_C), plt.plot(con_D)
    plt.show()


def plot_watts():
    plt.figure("Reaction Watts")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Watts")
    plt.xlim(0, minutes * 60), plt.ylim(0, max(watts_list) * 1.1)
    plt.plot(watts_list)
    plt.show()


plot_conc(), plot_watts()
