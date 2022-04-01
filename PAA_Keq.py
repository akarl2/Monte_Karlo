import math
import matplotlib.pyplot as plt
from scipy import integrate

s_conA = 20
s_conB = 3
s_conC = 0
s_conD = 5
cat_con = .314
temperature = 325
k1 = (6.83*10**8)*(math.sqrt(s_conA*(1.75*10**-5)) + cat_con + math.sqrt(cat_con*(1.3*10**-2)))*(math.exp((-57846.15)/(8.134*temperature)))
k2 = (6.73*10**8)*(math.sqrt(s_conA*(1.75*10**-5)) + cat_con + math.sqrt(cat_con*(1.3*10**-2)))*(math.exp((-60407.78)/(8.134*temperature)))
conA = s_conA
conB = s_conB
conC = s_conC
conD = s_conD
minutes = 1
con_A = [conA]
con_B = [conB]
con_C = [conC]
con_D = [conD]
watts_list = [0]
energy = 0
enthalpy = -13.6

while minutes < 1000:
    rate = ((k1 * conA * conB) - (k2 * conC * conD)) / 60
    conA = conA - rate
    conB = conB - rate
    conC = conC + rate
    energy = energy + (enthalpy * rate)
    conD = conD + rate
    con_A.append(conA), con_B.append(conB), con_C.append(conC), con_D.append(conD)
    keq = (conC * conD) / (conA * conB)
    watts = energy * rate * -60
    watts_list.append(watts)
    print(minutes, energy, watts)
    minutes = minutes + 1

def plot_conc():
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (mol/L)")
    plt.xlim(0, 600), plt.ylim(0, 8)
    plt.plot(con_A), plt.plot(con_B), plt.plot(con_C), plt.plot(con_D)
    plt.show()

def plot_watts():
    plt.xlabel("Time (min)")
    plt.ylabel("Watts")
    plt.xlim(0, 600), plt.ylim(0, 50)
    plt.plot(watts_list)
    plt.show()

plot_watts()


