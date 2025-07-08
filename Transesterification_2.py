import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    'C18_OH_C20_COOH_ESTER': 0.82,
    'C18_OH_C22_COOH_ESTER': 1.41,
    'C18_OH_C24_COOH_ESTER': 1.58,
    'C20_OH_C16_COOH_ESTER': 0.92,
    'C20_OH_C18_COOH_ESTER': 4.45,
    'C20_OH_C20_COOH_ESTER': 22.77,
    'C20_OH_C22_COOH_ESTER': 11.05,
    'C20_OH_C24_COOH_ESTER': 0.96,
    'C22_OH_C16_COOH_ESTER': 0.16,
    'C22_OH_C18_COOH_ESTER': 3.37,
    'C22_OH_C20_COOH_ESTER': 38.95,
    'C22_OH_C22_COOH_ESTER': 2.24,
    'C22_OH_C24_COOH_ESTER': 0.00,
    'C24_OH_C16_COOH_ESTER': 0.28,
    'C24_OH_C18_COOH_ESTER': 1.05,
    'C24_OH_C20_COOH_ESTER': 7.46,
    'C24_OH_C22_COOH_ESTER': 1.30,
    'C24_OH_C24_COOH_ESTER': 0.19,
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
base_Ea = 67500  # J/mol (activation energy)
base_A = 1.0e7  # Pre-exponential factor

C16_OH_Ea_Factor = 1.0
C18_OH_Ea_Factor = 1.01
C20_OH_Ea_Factor = 1.0125
C22_OH_Ea_Factor = 1.015
C24_OH_Ea_Factor = 1.02

C16_OH_A_Factor = 1.0
C18_OH_A_Factor = 1.0
C20_OH_A_Factor = 1
C22_OH_A_Factor = 1
C24_OH_A_Factor = 1


# Individual kinetics parameters for each alcohol (can be customized)
individual_kinetics = {
    'C16_OH': {'Ea_forward': base_Ea * C16_OH_Ea_Factor, 'A_forward': C16_OH_A_Factor * base_A},
    'C18_OH': {'Ea_forward': base_Ea * C18_OH_Ea_Factor, 'A_forward': C18_OH_A_Factor * base_A},
    'C20_OH': {'Ea_forward': base_Ea * C20_OH_Ea_Factor, 'A_forward': C20_OH_A_Factor * base_A},
    'C22_OH': {'Ea_forward': base_Ea * C22_OH_Ea_Factor, 'A_forward': C22_OH_A_Factor * base_A},
    'C24_OH': {'Ea_forward': base_Ea * C24_OH_Ea_Factor, 'A_forward': C24_OH_A_Factor * base_A},
}

# Flag to control whether to use individual values or master values
use_individual_kinetics = True  # Set to True to use individual values, False to use master values

alcohol_kinetics = {}
for alcohol in ALCOHOL_MW:
    if use_individual_kinetics and alcohol in individual_kinetics:
        # Use individual values if flag is True and values are provided
        alcohol_kinetics[alcohol] = individual_kinetics[alcohol].copy()
    else:
        # Use master values otherwise
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
    'temp': 338.15,
    'catalyst_conc' : 0.09,
    'Rxn_Time_MIN' : 1440,
    'base_Ea': base_Ea,
    'base_A': base_A,
    'individual_kinetics': individual_kinetics,
    'use_individual_kinetics': use_individual_kinetics,
}

R = 8.314  # J/mol·K
T = Rxn_conditions['temp']
dt = 1  # seconds per step
total_time_sec = Rxn_conditions['Rxn_Time_MIN'] * 60  # total time in seconds
time_steps = total_time_sec
catalyst_conc = Rxn_conditions['catalyst_conc']

# Calculate reaction rate constants for each alcohol using Arrhenius equation
for alcohol, params in alcohol_kinetics.items():
    params['k_forward'] = params['A_forward'] * math.exp(-params['Ea_forward'] / (R * T))

time_array = np.arange(0, total_time_sec, dt)
alcohol_profile = {alc: [] for alc in Cetearyl_Alcohol}
ester_profile = {ester: [] for ester in jojoba_oil}

# Initialize alcohol concentrations with all possible alcohols
current_alc_conc = {alc: Cetearyl_Alcohol.get(alc, 0.0) for alc in ALCOHOL_MW}
current_jojoba = jojoba_oil.copy()

# Store initial profiles
for alc in current_alc_conc:
    alcohol_profile[alc].append(current_alc_conc[alc])
for ester in current_jojoba:
    ester_profile[ester].append(current_jojoba[ester])

# Simulation loop
for t in range(total_time_sec):
    # Calculate individual reaction rates for each alcohol and ester
    individual_rates = {}
    total_net_rate = 0

    # For each ester in the pool
    for ester_name, ester_concentration in current_jojoba.items():
        individual_rates[ester_name] = {}

        # Extract donor alcohol from ester name
        try:
            parts = ester_name.split("_")
            donor_core = parts[0]
            donor_alc = donor_core + "_OH"
            acid_part = "_".join(parts[2:-1])

        except ValueError:
            print(f"Skipping malformed ester: {ester_name}")
            continue

        for alcohol in current_alc_conc:
            if not alcohol.endswith("_OH") or alcohol not in alcohol_kinetics:
                continue
            if alcohol == donor_alc:
                continue  # Skip self-swap

            # Calculate forward rate for this alcohol and ester
            forward_rate = alcohol_kinetics[alcohol]['k_forward'] * ester_concentration * current_alc_conc[alcohol] * catalyst_conc

            # For reverse rate, we use the forward rates of the ester alcohol, This implements the concept that "one alcohol's reverse is another alcohol's forward"
            reverse_ester = f"{alcohol.replace('_OH', '')}_OH_{acid_part}_COOH_ESTER"
            reverse_conc = current_jojoba.get(reverse_ester, 0.0)
            reverse_rate = alcohol_kinetics[alcohol]['k_forward'] * reverse_conc * current_alc_conc[donor_alc] * catalyst_conc

            # Calculate net rate for this alcohol and ester
            net_rate = (forward_rate - reverse_rate) * dt

            individual_rates[ester_name][alcohol] = net_rate
            total_net_rate += net_rate

    # Update ester concentrations based on individual rates
    new_jojoba = current_jojoba.copy()
    for ester_name, ester_rates in individual_rates.items():
        ester_net_rate = sum(ester_rates.values())
        new_jojoba[ester_name] = max(0, current_jojoba[ester_name] - ester_net_rate)

    # Update alcohol concentrations based on reactions
    new_alc_conc = current_alc_conc.copy()

    # Process each ester's reactions
    for ester_name, ester_rates in individual_rates.items():
        try:
            parts = ester_name.split("_")
            donor_core = parts[0]
            donor_alc = donor_core + "_OH"
            acid_part = "_".join(parts[2:-1])
        except ValueError:
            continue

        # Sum the rates for this ester across all alcohols
        ester_total_rate = sum(ester_rates.values())

        # Add the released alcohol
        if donor_alc in new_alc_conc:
            new_alc_conc[donor_alc] += ester_total_rate
        else:
            new_alc_conc[donor_alc] = ester_total_rate

        # Process each attacking alcohol
        # Process each attacking alcohol
        for alcohol, rate in ester_rates.items():
            # Subtract consumed alcohol
            new_alc_conc[alcohol] -= rate

            # Parse ester to get acid part for constructing new ester name
            parts = ester_name.split("_")
            try:
                # Validate known pattern: <Donor>_OH_<Acid>_COOH_ESTER
                if parts[-2:] != ['COOH', 'ESTER']:
                    print(f"Malformed ester name: {ester_name}")
                    continue
                acid_part = parts[-3]  # The acid name, just before "COOH"
            except IndexError:
                print(f"Error parsing ester: {ester_name}")
                continue

            # Create the new ester from attacking alcohol and the acid
            new_donor = alcohol.replace('_OH', '')
            new_ester = f"{new_donor}_OH_{acid_part}_COOH_ESTER"

            # Optional: catch malformed new ester names
            if new_ester.count("COOH") > 1:
                print(f"Skipping malformed new ester: {new_ester}")
                continue
            new_jojoba[new_ester] = new_jojoba.get(new_ester, 0.0) + rate

    # Apply non-negative constraint to all alcohols
    for alcohol_name in new_alc_conc:
        new_alc_conc[alcohol_name] = max(0, new_alc_conc[alcohol_name])

    # Update current concentrations
    current_alc_conc = new_alc_conc
    current_jojoba = new_jojoba

    # Store profiles
    for alc in current_alc_conc:
        if alc in alcohol_profile:
            alcohol_profile[alc].append(current_alc_conc[alc])
        else:
            # Initialize with zeros for previous time steps and add current value
            alcohol_profile[alc] = [0.0] * len(time_array[:t]) + [current_alc_conc[alc]]

    for ester in current_jojoba:
        if ester in ester_profile:
            ester_profile[ester].append(current_jojoba[ester])
        else:
            # Initialize with zeros for previous time steps and add current value
            ester_profile[ester] = [0.0] * len(time_array[:t]) + [current_jojoba[ester]]

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

# Summarize alcohol and ester weight % totals
final_alcohol_wt_pct = sum([final_wt_percent[spec] for spec in final_wt_percent if spec in ALCOHOL_MW])
final_ester_wt_pct = sum([final_wt_percent[spec] for spec in final_wt_percent if spec in ester_mw])

# Calculate summed weight percent for each total chain length
chain_length_totals = {}

# Process all esters in final weight percentages
for ester, weight_pct in final_wt_percent.items():
    # Only process ester species
    if '_ESTER' in ester:
        # Extract alcohol and acid chain lengths from ester name
        parts = ester.split('_')
        alcohol_length = int(parts[0].replace('C', ''))
        acid_length = int(parts[2].replace('C', ''))

        # Calculate total chain length
        total_length = alcohol_length + acid_length

        # Add to the appropriate chain length total
        chain_key = f"C{total_length}"
        if chain_key in chain_length_totals:
            chain_length_totals[chain_key] += weight_pct
        else:
            chain_length_totals[chain_key] = weight_pct

# Sort chain lengths numerically
sorted_chain_lengths = sorted(chain_length_totals.keys(), key=lambda x: int(x.replace('C', '')))

# Calculate the ratio of (C36+C38) / (C40+C42)
numerator = chain_length_totals.get('C36', 0) + chain_length_totals.get('C38', 0)
denominator = chain_length_totals.get('C40', 0) + chain_length_totals.get('C42', 0)

if denominator > 0:
    ratio = numerator / denominator

# Create a function to display all data in a tabular notebook
def display_results():
    # Trim alcohol and ester profile arrays to match the length of time_array
    for alc in alcohol_profile:
        if len(alcohol_profile[alc]) > len(time_array):
            alcohol_profile[alc] = alcohol_profile[alc][:len(time_array)]

    for ester in ester_profile:
        if len(ester_profile[ester]) > len(time_array):
            ester_profile[ester] = ester_profile[ester][:len(time_array)]

    # Create the main window
    root = tk.Tk()
    root.title("Transesterification Simulation Results")
    root.geometry("1200x800")

    # Create the notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Tab 1: Concentration Profiles Plot
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Concentration Profiles")

    # Create the concentration profiles plot
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    for alc in alcohol_profile:
        ax1.plot(time_array / 60, alcohol_profile[alc], label=alc)
    for ester in ester_profile:
        ax1.plot(time_array / 60, ester_profile[ester], ':', label=ester)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Concentration (mol/L)")
    ax1.set_title("Alcohol and Ester Concentration Profiles")
    ax1.grid(True)
    fig1.tight_layout()

    # Embed the plot in the tab
    canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 2: Initial Concentrations Table
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Initial Concentrations")

    # Create the initial concentrations table
    fig2 = plt.figure(figsize=(10, min(25, 0.4 * len(initial_df))))
    ax2 = fig2.add_subplot(111)
    ax2.axis('off')
    table2 = ax2.table(cellText=np.round(initial_df.values, decimals=4),
                     rowLabels=initial_df.index,
                     colLabels=initial_df.columns,
                     loc='center',
                     cellLoc='center')
    table2.scale(1, 1.5)
    ax2.set_title("Initial Species Concentrations")
    fig2.tight_layout()

    # Embed the table in the tab
    canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 3: Final Concentrations Table
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Final Concentrations")

    # Create the final concentrations table
    fig3 = plt.figure(figsize=(10, min(25, 0.4 * len(final_df))))
    ax3 = fig3.add_subplot(111)
    ax3.axis('off')
    table3 = ax3.table(cellText=np.round(final_df.values, decimals=4),
                     rowLabels=final_df.index,
                     colLabels=final_df.columns,
                     loc='center',
                     cellLoc='center')
    table3.scale(1, 1.5)
    ax3.set_title("Final Species Concentrations")
    fig3.tight_layout()

    # Embed the table in the tab
    canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 4: Initial Weight Percentages Table
    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text="Initial Weight %")

    # Create the initial weight percentages table
    fig4 = plt.figure(figsize=(10, min(25, 0.4 * len(initial_wt_df))))
    ax4 = fig4.add_subplot(111)
    ax4.axis('off')
    table4 = ax4.table(cellText=np.round(initial_wt_df.values, decimals=4),
                     rowLabels=initial_wt_df.index,
                     colLabels=initial_wt_df.columns,
                     loc='center',
                     cellLoc='center')
    table4.scale(1, 1.5)
    ax4.set_title("Initial Species Weight Percentages")
    fig4.tight_layout()

    # Embed the table in the tab
    canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
    canvas4.draw()
    canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 5: Final Weight Percentages Table
    tab5 = ttk.Frame(notebook)
    notebook.add(tab5, text="Final Weight %")

    # Create the final weight percentages table
    fig5 = plt.figure(figsize=(10, min(25, 0.4 * len(final_wt_df))))
    ax5 = fig5.add_subplot(111)
    ax5.axis('off')
    table5 = ax5.table(cellText=np.round(final_wt_df.values, decimals=4),
                     rowLabels=final_wt_df.index,
                     colLabels=final_wt_df.columns,
                     loc='center',
                     cellLoc='center')
    table5.scale(1, 1.5)
    ax5.set_title("Final Species Weight Percentages")
    fig5.tight_layout()

    # Embed the table in the tab
    canvas5 = FigureCanvasTkAgg(fig5, master=tab5)
    canvas5.draw()
    canvas5.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 6: Ester Weight Percentages Over Time Plot
    tab6 = ttk.Frame(notebook)
    notebook.add(tab6, text="Ester Weight % Over Time")

    # Calculate weight percentages for esters at each time step
    ester_wt_percent_over_time = []
    # Also calculate chain length distributions at each time step
    chain_length_over_time = []

    # For each time step
    for t in range(len(time_array)):
        # Get concentrations at this time step
        current_conc = {}
        for alc in alcohol_profile:
            if t < len(alcohol_profile[alc]):
                current_conc[alc] = alcohol_profile[alc][t]
            else:
                current_conc[alc] = 0.0

        for ester in ester_profile:
            if t < len(ester_profile[ester]):
                current_conc[ester] = ester_profile[ester][t]
            else:
                current_conc[ester] = 0.0

        # Convert concentrations to weights
        current_weight = {spec: current_conc[spec] * species_mw[spec] for spec in current_conc if spec in species_mw}
        total_weight = sum(current_weight.values())

        # Calculate weight percentages
        if total_weight > 0:
            current_wt_percent = {spec: (wt / total_weight) * 100 for spec, wt in current_weight.items()}
            # Filter to include only esters
            current_ester_wt_percent = {spec: wt for spec, wt in current_wt_percent.items() if spec in ester_mw}
            ester_wt_percent_over_time.append(current_ester_wt_percent)

            # Calculate chain length distributions for this time step
            current_chain_lengths = {}

            # Process all esters in current weight percentages
            for ester, weight_pct in current_ester_wt_percent.items():
                # Extract alcohol and acid chain lengths from ester name
                parts = ester.split('_')
                alcohol_length = int(parts[0].replace('C', ''))
                acid_length = int(parts[2].replace('C', ''))

                # Calculate total chain length
                total_length = alcohol_length + acid_length

                # Add to the appropriate chain length total
                chain_key = f"C{total_length}"
                if chain_key in current_chain_lengths:
                    current_chain_lengths[chain_key] += weight_pct
                else:
                    current_chain_lengths[chain_key] = weight_pct

            # Normalize chain length percentages to 100%
            total_chain_pct = sum(current_chain_lengths.values())
            if total_chain_pct > 0:
                normalized_chain_lengths = {chain: (pct / total_chain_pct) * 100 
                                           for chain, pct in current_chain_lengths.items()}
                chain_length_over_time.append(normalized_chain_lengths)
            else:
                chain_length_over_time.append({})
        else:
            ester_wt_percent_over_time.append({})
            chain_length_over_time.append({})

    # Create the plot
    fig6 = plt.figure(figsize=(12, 8))
    ax6 = fig6.add_subplot(111)

    # Get all unique esters that appear in the simulation
    all_esters = set()
    for time_point in ester_wt_percent_over_time:
        all_esters.update(time_point.keys())

    # Plot weight percentage over time for each ester
    for ester in all_esters:
        # Extract data for this ester
        wt_pct_values = [time_point.get(ester, 0) for time_point in ester_wt_percent_over_time]
        ax6.plot(time_array / 60, wt_pct_values, label=ester)

    # Add labels and title
    ax6.set_xlabel("Time (min)")
    ax6.set_ylabel("Weight Percentage (%)")
    ax6.set_title("Ester Weight Percentages During Transesterification")
    ax6.grid(True)

    # Add legend (outside the plot to avoid overcrowding)
    ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    fig6.tight_layout()

    # Embed the plot in the tab
    canvas6 = FigureCanvasTkAgg(fig6, master=tab6)
    canvas6.draw()
    canvas6.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 7: Chain Length Distribution Over Time
    tab7 = ttk.Frame(notebook)
    notebook.add(tab7, text="Chain Length % Over Time")

    # Create the plot for chain length distributions
    fig7 = plt.figure(figsize=(12, 8))
    ax7 = fig7.add_subplot(111)

    # Get all unique chain lengths that appear in the simulation
    all_chain_lengths = set()
    for time_point in chain_length_over_time:
        all_chain_lengths.update(time_point.keys())

    # Sort chain lengths numerically
    sorted_all_chain_lengths = sorted(all_chain_lengths, key=lambda x: int(x.replace('C', '')))

    # Plot normalized percentage over time for each chain length
    for chain in sorted_all_chain_lengths:
        # Extract data for this chain length
        chain_pct_values = [time_point.get(chain, 0) for time_point in chain_length_over_time]
        ax7.plot(time_array / 60, chain_pct_values, label=chain, linewidth=2)

    # Add labels and title
    ax7.set_xlabel("Time (min)")
    ax7.set_ylabel("Normalized Percentage (%)")
    ax7.set_title("Total Chain Length Distribution During Transesterification (Normalized to 100%)")
    ax7.grid(True)

    # Add legend
    ax7.legend(loc='best', fontsize='medium')

    # Add a text annotation for cursor position
    cursor_text = ax7.text(0.02, 0.02, '', transform=ax7.transAxes, 
                          bbox=dict(facecolor='white', alpha=0.7),
                          verticalalignment='bottom')

    # Function to update cursor position text
    def update_cursor_text(event):
        if event.inaxes == ax7:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                cursor_text.set_text(f'Time: {x:.2f} min, Percentage: {y:.2f}%')
                fig7.canvas.draw_idle()

    # Connect the motion_notify_event
    fig7.canvas.mpl_connect('motion_notify_event', update_cursor_text)

    fig7.tight_layout()

    # Embed the plot in the tab
    canvas7 = FigureCanvasTkAgg(fig7, master=tab7)
    canvas7.draw()
    canvas7.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 8: Summary Statistics
    tab8 = ttk.Frame(notebook)
    notebook.add(tab8, text="Summary Statistics")

    # Create a text widget to display the summary statistics
    text_widget = tk.Text(tab8, wrap=tk.WORD, font=('Arial', 12))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Add the summary statistics to the text widget
    text_widget.insert(tk.END, f"Final Total Alcohol Weight %: {final_alcohol_wt_pct:.2f}%\n\n")
    text_widget.insert(tk.END, f"Final Total Ester Weight %: {final_ester_wt_pct:.2f}%\n\n")
    text_widget.insert(tk.END, "Summed Weight Percent by Total Chain Length:\n")
    for chain in sorted_chain_lengths:
        text_widget.insert(tk.END, f"{chain}: {chain_length_totals[chain]:.2f}%\n")
    text_widget.insert(tk.END, f"\nRatio of (C36+C38) / (C40+C42): {ratio:.4f}")

    # Make the text widget read-only
    text_widget.configure(state='disabled')

    # Start the main event loop
    root.mainloop()

# Call the function to display the results
display_results()
