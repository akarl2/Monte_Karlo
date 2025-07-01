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
    'C16_COOH': MW_C16_ACID,
    'C18_COOH': MW_C18_ACID,
    'C20_COOH': MW_C20_ACID,
    'C22_COOH': MW_C22_ACID,
    'C24_COOH': MW_C24_ACID
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

GRAMS_CETEARYL_ALCOHOL = 178
DENSITY_CETEARYL_ALCOHOL = 0.811
GRAMS_JOJOBA_OIL = 22
DENSITY_JOJOBA_OIL = 0.789

Total_Volume_ML = GRAMS_CETEARYL_ALCOHOL / DENSITY_CETEARYL_ALCOHOL + GRAMS_JOJOBA_OIL / DENSITY_JOJOBA_OIL
Total_Volume_L = Total_Volume_ML / 1000


#Data for Cetearly Alcohol
Initial_Average_Alcohol_MW = 0.00

for key in Cetearyl_Alcohol:
    Initial_Average_Alcohol_MW += Cetearyl_Alcohol[key] * ALCOHOL_MW[key]

Cetearly_Moles = GRAMS_CETEARYL_ALCOHOL / Initial_Average_Alcohol_MW

Cetearyl_Concentration = Cetearly_Moles / Total_Volume_L

# Convert mass fraction directly to mol/L using original gram weights
for alc in Cetearyl_Alcohol:
    grams = Cetearyl_Alcohol[alc] * GRAMS_CETEARYL_ALCOHOL
    mols = grams / ALCOHOL_MW[alc]
    Cetearyl_Alcohol[alc] = mols / Total_Volume_L

# Define kinetics parameters for each alcohol
base_Ea = 68000  # J/mol (activation energy)
base_A = 1.2e7  # Pre-exponential factor
alcohol_kinetics = {}
for alcohol in ALCOHOL_MW:
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

Rxn_conditions = {
    'Jojoba_Oil': jojoba_oil,
    'Cetearly_Alcohol': Cetearyl_Alcohol,
    'Alcohol_Kinetics': alcohol_kinetics,
    'temp': 318.15,
    'catalyst_conc' : 0.001,
    'Rxn_Time_MIN' : 500,
}

# Simulate simple forward transesterification: alcohol from ester is released, new ester is formed
# Assume pseudo-first-order kinetics based on alcohol (limiting reagent) with excess ester
import pandas as pd

R = 8.314  # J/mol·K
T = Rxn_conditions['temp']
dt = 1  # seconds per step
total_time_sec = Rxn_conditions['Rxn_Time_MIN'] * 60  # total time in seconds
time_steps = total_time_sec


time_array = np.arange(0, total_time_sec + 1, dt)
alcohol_profile = {alc: [] for alc in Cetearyl_Alcohol}
ester_profile = {ester: [] for ester in jojoba_oil}

# Initialize alcohol concentrations with all possible alcohols
current_alc_conc = {alc: Cetearyl_Alcohol.get(alc, 0.0) for alc in ALCOHOL_MW}
current_jojoba = jojoba_oil.copy()

for t in time_array:
    # Store profiles
    for alc in current_alc_conc:
        alcohol_profile[alc].append(current_alc_conc[alc])
    for ester in current_jojoba:
        ester_profile[ester].append(current_jojoba[ester])

    # Kinetic step: For each ester, allow all alcohols to attack and form new esters
    for ester in list(current_jojoba.keys()):
        try:
            parts = ester.split("_")
            donor_core = parts[0]
            donor_alc = donor_core + "_OH"
            acid_part = "_".join(parts[2:-1])
        except ValueError:
            print(f"Skipping malformed ester: {ester}")
            continue

        if ester not in ester_profile:
            ester_profile[ester] = []

        for attacking_alc in list(current_alc_conc.keys()):
            if not attacking_alc.endswith("_OH") or attacking_alc not in alcohol_kinetics:
                continue
            if attacking_alc == donor_alc:
                continue  # Skip self-swap

            A = alcohol_kinetics[attacking_alc]['A_forward']
            Ea = alcohol_kinetics[attacking_alc]['Ea_forward']
            k = A * np.exp(-Ea / (R * T))
            delta = k * current_alc_conc[attacking_alc] * dt

            if current_jojoba[ester] >= delta and current_alc_conc[attacking_alc] >= delta:
                # consume current ester and attacking alcohol
                current_jojoba[ester] -= delta
                current_alc_conc[attacking_alc] -= delta

                # release the displaced alcohol
                current_alc_conc[donor_alc] += delta

                # form the new ester
                attacker = attacking_alc.replace("_OH", "")
                new_ester = f"{attacker}_OH_{acid_part}_ESTER"
                if new_ester not in current_jojoba:
                    current_jojoba[new_ester] = 0.0
                current_jojoba[new_ester] += delta

                if new_ester not in ester_profile:
                    ester_profile[new_ester] = []

# Plot both alcohol and ester concentrations (no legend to reduce clutter)
plt.figure(figsize=(12, 6))
for alc in alcohol_profile:
    plt.plot(time_array / 60, alcohol_profile[alc])
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
    final_conc[alc] = alcohol_profile[alc][-1]
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
table = ax.table(cellText=np.round(initial_df.values, decimals=4),
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
table = ax.table(cellText=np.round(final_df.values, decimals=4),
                 rowLabels=final_df.index,
                 colLabels=final_df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
plt.title("Final Species Concentrations")
plt.tight_layout()
plt.show()


# Calculate molecular weights for all species
species_mw = {**ALCOHOL_MW, **ester_mw}

# Convert concentrations to grams
initial_weight = {spec: initial_conc[spec] * species_mw[spec] for spec in initial_conc if spec in species_mw}
final_weight = {spec: final_conc[spec] * species_mw[spec] for spec in final_conc if spec in species_mw}

total_initial_weight = sum(initial_weight.values())
total_final_weight = sum(final_weight.values())

initial_wt_percent = {spec: (wt / total_initial_weight) * 100 for spec, wt in initial_weight.items()}
final_wt_percent = {spec: (wt / total_final_weight) * 100 for spec, wt in final_weight.items()}

# Create DataFrames for weight percentages
initial_wt_df = pd.DataFrame.from_dict(initial_wt_percent, orient='index', columns=['Initial Weight %'])
initial_wt_df.index.name = 'Species'
initial_wt_df = initial_wt_df.sort_values(by='Initial Weight %', ascending=False)

final_wt_df = pd.DataFrame.from_dict(final_wt_percent, orient='index', columns=['Final Weight %'])
final_wt_df.index.name = 'Species'
final_wt_df = final_wt_df.sort_values(by='Final Weight %', ascending=False)

# Display initial weight percentages
fig, ax = plt.subplots(figsize=(10, min(25, 0.4 * len(initial_wt_df))))
ax.axis('off')
table = ax.table(cellText=np.round(initial_wt_df.values, decimals=4),
                 rowLabels=initial_wt_df.index,
                 colLabels=initial_wt_df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
plt.title("Initial Species Weight Percentages")
plt.tight_layout()
plt.show()

# Display final weight percentages
fig, ax = plt.subplots(figsize=(10, min(25, 0.4 * len(final_wt_df))))
ax.axis('off')
table = ax.table(cellText=np.round(final_wt_df.values, decimals=4),
                 rowLabels=final_wt_df.index,
                 colLabels=final_wt_df.columns,
                 loc='center',
                 cellLoc='center')
table.scale(1, 1.5)
plt.title("Final Species Weight Percentages")
plt.tight_layout()
plt.show()

# Summarize alcohol and ester weight % totals
final_alcohol_wt_pct = sum([final_wt_percent[spec] for spec in final_wt_percent if spec in ALCOHOL_MW])
final_ester_wt_pct = sum([final_wt_percent[spec] for spec in final_wt_percent if spec in ester_mw])

print(f"\nFinal Total Alcohol Weight %: {final_alcohol_wt_pct:.2f}%")
print(f"Final Total Ester Weight %: {final_ester_wt_pct:.2f}%")
