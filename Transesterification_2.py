import math
import matplotlib.pyplot as plt
import numpy as np

# Molecular weights (g/mol)
MW_C16_ALCOHOL = 242.45    # C16 alcohol (hexadecanol
MW_C18_ALCOHOL = 268.48    # C18 alcohol (octadecanol)
MW_C20_ALCOHOL = 296.53   # C20 alcohol (eicosanol)
MW_C22_ALCOHOL = 342.6   # C22 alcohol (docosanol)
MW_C24_ALCOHOL = 352.65   # C24 alcohol (tetracosanol)

MW_C16_ACID = 256.43
MW_C18_ACID = 282.47
MW_C20_ACID = 310.51
MW_C22_ACID = 338.58
MW_C24_ACID = 366.63

ester_mw = {
    'C16_OH_C16_COOH_ESTER': MW_C16_ALCOHOL + MW_C16_ACID - 18.01,
    'C16_OH_C18_COOH_ESTER': MW_C16_ALCOHOL + MW_C18_ACID - 18.01,
    'C16_OH_C20_COOH_ESTER': MW_C16_ALCOHOL + MW_C20_ACID - 18.01,
    'C16_OH_C22_COOH_ESTER': MW_C16_ALCOHOL + MW_C22_ACID - 18.01,
    'C16_OH_C24_COOH_ESTER': MW_C16_ALCOHOL + MW_C24_ACID - 18.01,

    'C18_OH_C16_COOH_ESTER': MW_C18_ALCOHOL + MW_C16_ACID - 18.01,
    'C18_OH_C18_COOH_ESTER': MW_C18_ALCOHOL + MW_C18_ACID - 18.01,
    'C18_OH_C20_COOH_ESTER': MW_C18_ALCOHOL + MW_C20_ACID - 18.01,
    'C18_OH_C22_COOH_ESTER': MW_C18_ALCOHOL + MW_C22_ACID - 18.01,
    'C18_OH_C24_COOH_ESTER': MW_C18_ALCOHOL + MW_C24_ACID - 18.01,

    'C20_OH_C16_COOH_ESTER': MW_C20_ALCOHOL + MW_C16_ACID - 18.01,
    'C20_OH_C18_COOH_ESTER': MW_C20_ALCOHOL + MW_C18_ACID - 18.01,
    'C20_OH_C20_COOH_ESTER': MW_C20_ALCOHOL + MW_C20_ACID - 18.01,
    'C20_OH_C22_COOH_ESTER': MW_C20_ALCOHOL + MW_C22_ACID - 18.01,
    'C20_OH_C24_COOH_ESTER': MW_C20_ALCOHOL + MW_C24_ACID - 18.01,

    'C22_OH_C16_COOH_ESTER': MW_C22_ALCOHOL + MW_C16_ACID - 18.01,
    'C22_OH_C18_COOH_ESTER': MW_C22_ALCOHOL + MW_C18_ACID - 18.01,
    'C22_OH_C20_COOH_ESTER': MW_C22_ALCOHOL + MW_C20_ACID - 18.01,
    'C22_OH_C22_COOH_ESTER': MW_C22_ALCOHOL + MW_C22_ACID - 18.01,
    'C22_OH_C24_COOH_ESTER': MW_C22_ALCOHOL + MW_C24_ACID - 18.01,

    'C24_OH_C16_COOH_ESTER': MW_C24_ALCOHOL + MW_C16_ACID - 18.01,
    'C24_OH_C18_COOH_ESTER': MW_C24_ALCOHOL + MW_C18_ACID - 18.01,
    'C24_OH_C20_COOH_ESTER': MW_C24_ALCOHOL + MW_C20_ACID - 18.01,
    'C24_OH_C22_COOH_ESTER': MW_C24_ALCOHOL + MW_C22_ACID - 18.01,
    'C24_OH_C24_COOH_ESTER': MW_C24_ALCOHOL + MW_C24_ACID - 18.01,
}

# Create dictionary of alcohol molecular weights for easy lookup
ALCOHOL_MW = {
    'C16_OH': MW_C16_ALCOHOL,
    'C18_OH': MW_C18_ALCOHOL,
    'C20_OH': MW_C20_ALCOHOL,
    'C22_OH': MW_C22_ALCOHOL,
    'C24_OH': MW_C24_ALCOHOL
}

ACID_MW = {
    'C16_OH': MW_C16_ACID,
    'C18_OH': MW_C18_ACID,
    'C20_OH': MW_C20_ACID,
    'C22_OH': MW_C22_ACID,
    'C24_OH': MW_C24_ACID
}

Cetearyl_Alcohol = {
    'C16_OH': 0.50,
    'C18_OH': 0.50,
    'C20_OH': 0.00,
    'C22_OH': 0.00,
    'C24_OH': 0.00,
}

jojoba_oil = {
    'C16_OH_C16_COOH_ESTER': 0.00,
    'C16_OH_C18_COOH_ESTER': 0.00,
    'C16_OH_C20_COOH_ESTER': 0.00,
    'C16_OH_C22_COOH_ESTER': 0.16,
    'C16_OH_C24_COOH_ESTER': 0.56,
    'C18_OH_C16_COOH_ESTER': 0.00,
    'C18_OH_C18_COOH_ESTER': 0.05,
    'C18_OH_C20_COOH_ESTER': 0.80,
    'C18_OH_C22_COOH_ESTER': 1.40,
    'C18_OH_C24_COOH_ESTER': 1.60,
    'C20_OH_C16_COOH_ESTER': 0.92,
    'C20_OH_C18_COOH_ESTER': 4.31,
    'C20_OH_C20_COOH_ESTER': 22.67,
    'C20_OH_C22_COOH_ESTER': 11.20,
    'C20_OH_C24_COOH_ESTER': 0.95,
    'C22_OH_C16_COOH_ESTER': 0.16,
    'C22_OH_C18_COOH_ESTER': 3.36,
    'C22_OH_C20_COOH_ESTER': 39.47,
    'C22_OH_C22_COOH_ESTER': 2.23,
    'C22_OH_C24_COOH_ESTER': 0.00,
    'C24_OH_C16_COOH_ESTER': 0.28,
    'C24_OH_C18_COOH_ESTER': 1.07,
    'C24_OH_C20_COOH_ESTER': 7.42,
    'C24_OH_C22_COOH_ESTER': 1.39,
    'C24_OH_C24_COOH_ESTER': 0.00,
}

GRAMS_CETEARYL_ALCOHOL = 100
DENSITY_CETEARYL_ALCOHOL = 0.811
GRAMS_JOJOBA_OIL = 20
DENSITY_JOJOBA_OIL = 0.789

Total_Volume_ML = GRAMS_CETEARYL_ALCOHOL / DENSITY_CETEARYL_ALCOHOL + GRAMS_JOJOBA_OIL / DENSITY_JOJOBA_OIL
Total_Volume_L = Total_Volume_ML / 1000


#Data for Cetearly Alcohol
Initial_Average_Alcohol_MW = 0.00

for key in Cetearyl_Alcohol:
    Initial_Average_Alcohol_MW += Cetearyl_Alcohol[key] * ALCOHOL_MW[key]

Cetearly_Moles = GRAMS_CETEARYL_ALCOHOL / Initial_Average_Alcohol_MW

Cetearyl_Concentration = Cetearly_Moles / Total_Volume_L

for alcohol in Cetearyl_Alcohol:
    Cetearyl_Alcohol[alcohol] = Cetearyl_Alcohol[alcohol] * Cetearyl_Concentration

# Define kinetics parameters for each alcohol
base_Ea = 68000  # J/mol (activation energy)
base_A = 1.2e7  # Pre-exponential factor
alcohol_kinetics = {}
for alcohol in Cetearyl_Alcohol:
    alcohol_kinetics[alcohol] = {
        'Ea_forward': base_Ea,  # J/mol (activation energy)
        'A_forward': base_A,  # Pre-exponential factor
    }

#Data for Jojoba_Oil
Initial_Average_Oil_MW = 0.00

for key in jojoba_oil:
    Initial_Average_Oil_MW += (jojoba_oil[key] * ester_mw[key]) / 100

Jojoba_Moles = GRAMS_JOJOBA_OIL / Initial_Average_Oil_MW

Jojoba_Concentration = Jojoba_Moles / Total_Volume_L

for ester in jojoba_oil:
    jojoba_oil[ester] = jojoba_oil[ester] * Jojoba_Concentration / 100

print(jojoba_oil)

print(Initial_Average_Oil_MW)


Rxn_conditions = {
    'Jojoba_Oil': jojoba_oil,
    'Cetearly_Alcohol': Cetearyl_Alcohol,
    'Alcohol_Kinetics': alcohol_kinetics,
    'temp': 298.15,
    'pressure': 101325,
    'catalyst_conc' : 0.001,
    'Rxn_Time_MIN' : 1000,
}

# Simulate simple forward transesterification: alcohol from ester is released, new ester is formed
# Assume pseudo-first-order kinetics based on alcohol (limiting reagent) with excess ester
import pandas as pd

R = 8.314  # J/mol·K
T = Rxn_conditions['temp']
time_steps = 500
dt = Rxn_conditions['Rxn_Time_MIN'] * 60 / time_steps  # seconds per step

time_array = np.linspace(0, Rxn_conditions['Rxn_Time_MIN'] * 60, time_steps)
alcohol_profile = {alc: [] for alc in Cetearyl_Alcohol}
ester_profile = {ester: [] for ester in jojoba_oil}
released_alcohols = {alc: [] for alc in ALCOHOL_MW}

# Copy concentrations to avoid mutation
current_alc_conc = Cetearyl_Alcohol.copy()
current_jojoba = jojoba_oil.copy()
released_conc = {alc: 0.0 for alc in ALCOHOL_MW}

for t in time_array:
    # Store profiles
    for alc in current_alc_conc:
        alcohol_profile[alc].append(current_alc_conc[alc])
    for ester in current_jojoba:
        ester_profile[ester].append(current_jojoba[ester])
    for alc in released_conc:
        released_alcohols[alc].append(released_conc[alc])

    # Kinetic step
    delta_conc = {}
    for alc in current_alc_conc:
        A = alcohol_kinetics[alc]['A_forward']
        Ea = alcohol_kinetics[alc]['Ea_forward']
        k = A * np.exp(-Ea / (R * T))
        delta = k * current_alc_conc[alc] * dt
        delta_conc[alc] = delta

    # Apply concentration changes
    for ester in current_jojoba:
        est_parts = ester.split("_")
        formed_alc = est_parts[0] + "_OH"
        used_alc = est_parts[2] + "_OH"
        if used_alc in delta_conc:
            delta = delta_conc[used_alc]
            current_jojoba[ester] += delta
            current_alc_conc[used_alc] -= delta
            released_conc[formed_alc] += delta

# Plot both alcohol and ester concentrations (no legend to reduce clutter)
plt.figure(figsize=(12, 6))
for alc in alcohol_profile:
    plt.plot(time_array / 60, alcohol_profile[alc])
for alc in released_alcohols:
    plt.plot(time_array / 60, released_alcohols[alc], '--')
for ester in ester_profile:
    plt.plot(time_array / 60, ester_profile[ester], ':')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (mol/L)")
plt.title("Alcohol and Ester Concentration Profiles")
plt.grid(True)
plt.tight_layout()
plt.show()

# Create combined final concentration table
final_conc = {}
for alc in alcohol_profile:
    final_conc[alc] = alcohol_profile[alc][-1] + released_alcohols[alc][-1]
for ester in ester_profile:
    final_conc[ester] = ester_profile[ester][-1]


final_df = pd.DataFrame.from_dict(final_conc, orient='index', columns=['Final Concentration (mol/L)'])
final_df.index.name = 'Species'
final_df = final_df.sort_values(by='Final Concentration (mol/L)', ascending=False)

# Create initial concentration table
initial_conc = {}
for alc in Cetearyl_Alcohol:
    initial_conc[alc] = Cetearyl_Alcohol[alc]
for ester in jojoba_oil:
    initial_conc[ester] = jojoba_oil[ester]

initial_df = pd.DataFrame.from_dict(initial_conc, orient='index', columns=['Initial Concentration (mol/L)'])
initial_df.index.name = 'Species'
initial_df = initial_df.sort_values(by='Initial Concentration (mol/L)', ascending=False)

# Display initial concentrations in a table
fig, ax = plt.subplots(figsize=(10, min(25, 0.4 * len(initial_df))))
ax.axis('off')
table = ax.table(cellText=initial_df.values,
                 rowLabels=initial_df.index,
                 colLabels=initial_df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
plt.title("Initial Species Concentrations")
plt.tight_layout()
plt.show()

# Display final concentrations in a table
fig, ax = plt.subplots(figsize=(10, min(25, 0.4 * len(final_df))))
ax.axis('off')
table = ax.table(cellText=final_df.values,
                 rowLabels=final_df.index,
                 colLabels=final_df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
plt.title("Final Species Concentrations")
plt.tight_layout()
plt.show()






