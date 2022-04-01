import math
import matplotlib.pyplot as plt

s_conA = 20
s_conB = 3
s_conC = 0
s_conD = 5
cat_con = 0
temperature = 315
k1 = (3.99*10**10)*(math.sqrt(s_conA*(1.78*10**-4)) + cat_con + math.sqrt(cat_con*(1.3*10**-2)))*(math.exp((-75200)/(8.134*temperature)))
k2 = (1.10*10**5)*(math.sqrt(s_conA*(1.78*10**-4)) + cat_con + math.sqrt(cat_con*(1.3*10**-2)))*(math.exp((-40400)/(8.134*temperature)))
conA = s_conA
conB = s_conB
conC = s_conC
conD = s_conD
minutes = 0
con_A = [conA]
con_B = [conB]
con_C = [conC]
con_D = [conD]
energy = 0

while minutes < 1000:
    # rate is in minutes
    rate = ((k1 * conA * conB) - (k2 * conC * conD)) / 60
    conA = conA - rate
    conB = conB - rate
    conC = conC + rate
    conD = conD + rate
    con_A.append(conA), con_B.append(conB), con_C.append(conC), con_D.append(conD)
    keq = (conC * conD) / (conA * conB)
    print(minutes)
    minutes = minutes + 1

def show_plot():
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (mol/L)")
    plt.xlim(0, 600), plt.ylim(0, 8)
    plt.plot(con_A), plt.plot(con_B), plt.plot(con_C), plt.plot(con_D)
    plt.show()

show_plot()
