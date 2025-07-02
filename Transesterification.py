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
    'C18_OH': 0.50
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

GRAMS_CETEARYL_ALCOHOL = 356
DENSITY_CETEARYL_ALCOHOL = 0.811
GRAMS_JOJOBA_OIL = 44
DENSITY_JOJOBA_OIL = 0.789


def simulate_transesterification(
        initial_ester_conc=None,
        initial_ester_distribution=None,
        alcohol_distribution=None,
        initial_product_ester_conc=None,
        initial_product_alcohol_conc=None,
        catalyst_conc=None,
        reaction_time_minutes=None,
        temperature=None,
        equilibrium_tolerance=1e-7,
        check_equilibrium=True,
        alcohol_kinetics=None
):

    if (initial_ester_conc is None and initial_ester_distribution is None) or alcohol_distribution is None or catalyst_conc is None or reaction_time_minutes is None:
        raise ValueError("Initial ester concentration or distribution, alcohol distribution, catalyst concentration, and reaction time must be provided")

    if temperature is None:
        raise ValueError("Temperature must be provided for Arrhenius calculations")

    # Set default values for product concentrations if not provided
    if initial_product_ester_conc is None:
        # Initialize product esters as a dictionary with zero concentration for each possible product
        initial_product_ester_conc = {}
        for alcohol in alcohol_distribution:
            alcohol_prefix = alcohol[0:3]  # e.g., 'C16'
            initial_product_ester_conc[f'{alcohol_prefix}-{alcohol_prefix}'] = 0.0
    elif not isinstance(initial_product_ester_conc, dict):
        # Convert single value to dictionary for backward compatibility
        single_value = initial_product_ester_conc  # Store the single value
        initial_product_ester_conc = {}  # Create a new empty dictionary
        for alcohol in alcohol_distribution:
            alcohol_prefix = alcohol[0:3]  # e.g., 'C16'
            initial_product_ester_conc[f'{alcohol_prefix}-{alcohol_prefix}'] = single_value

    if initial_product_alcohol_conc is None:
        initial_product_alcohol_conc = 0.0

    # Convert minutes to seconds
    seconds = reaction_time_minutes * 60

    # Handle both single ester and ester distribution
    if initial_ester_distribution is not None:
        # For backward compatibility in calculations
        total_ester_conc = sum(initial_ester_distribution.values())
        ester_conc = [total_ester_conc]

        # Initialize all esters in a single dictionary (big pool approach)
        all_esters = {ester: [conc] for ester, conc in initial_ester_distribution.items()}
    else:
        # For backward compatibility
        ester_conc = [initial_ester_conc]
        all_esters = {'C16-C18': [initial_ester_conc]}  # Use a default name

    # Add product esters to the big pool of esters
    for ester, conc in initial_product_ester_conc.items():
        if ester in all_esters:
            all_esters[ester][0] += conc
        else:
            all_esters[ester] = [conc]

    # Initialize all alcohols in a single dictionary (big pool approach)
    all_alcohols = {alcohol: [conc] for alcohol, conc in alcohol_distribution.items()}

    # Add product alcohols to the big pool of alcohols
    if isinstance(initial_product_alcohol_conc, dict):
        for alcohol, conc in initial_product_alcohol_conc.items():
            if alcohol in all_alcohols:
                all_alcohols[alcohol][0] += conc
            else:
                all_alcohols[alcohol] = [conc]
    else:
        # For backward compatibility, add default product alcohol
        if 'C18_OH' in all_alcohols:
            all_alcohols['C18_OH'][0] += initial_product_alcohol_conc
        else:
            all_alcohols['C18_OH'] = [initial_product_alcohol_conc]

    # Calculate reaction rate constants for each alcohol using Arrhenius equation
    R = 8.314  # Gas constant
    k_forwards = {}

    for alcohol, params in alcohol_kinetics.items():
        k_forwards[alcohol] = params['A_forward'] * math.exp(-params['Ea_forward'] / (R * temperature))
        # Note: We don't need to calculate reverse rates separately
        # One alcohol's reverse reaction is another alcohol's forward reaction

    # Time step (1 second)
    dt = 1.0

    # Variables to track equilibrium
    equilibrium_reached = False
    equilibrium_time = seconds

    # Simulation loop
    for t in range(seconds):
        # Calculate individual reaction rates for each alcohol and ester
        individual_rates = {}
        total_net_rate = 0

        # For each ester in the big pool
        for ester_name, ester_concentration in {k: v[-1] for k, v in all_esters.items()}.items():
            individual_rates[ester_name] = {}

            for alcohol in alcohol_distribution:
                # Calculate forward rate for this alcohol and ester
                forward_rate = k_forwards[alcohol] * ester_concentration * all_alcohols[alcohol][-1] * catalyst_conc

                # For reverse rate, we use the forward rates of other alcohols
                # This implements the concept that "one alcohol's reverse is another alcohol's forward"
                reverse_contribution = 0
                for other_alcohol in alcohol_distribution:
                    if other_alcohol != alcohol:
                        # Each other alcohol's forward reaction contributes to this alcohol's reverse reaction
                        other_forward = k_forwards[other_alcohol] * ester_concentration * all_alcohols[other_alcohol][-1] * catalyst_conc
                        reverse_contribution += other_forward / (len(alcohol_distribution) - 1)

                # Calculate net rate for this alcohol and ester
                net_rate = (forward_rate - reverse_contribution) * dt
                individual_rates[ester_name][alcohol] = net_rate
                total_net_rate += net_rate

        # Update ester concentrations based on individual rates
        new_ester_concs = {}
        for ester_name, ester_rates in individual_rates.items():
            ester_net_rate = sum(ester_rates.values())
            new_ester_concs[ester_name] = max(0, all_esters[ester_name][-1] - ester_net_rate)

        # Update total ester concentration for backward compatibility
        new_ester = max(0, ester_conc[-1] - total_net_rate)

        # Update alcohol concentrations based on reactions
        new_alcohol_concs = {}

        # First, copy current concentrations
        for alcohol in all_alcohols:
            new_alcohol_concs[alcohol] = all_alcohols[alcohol][-1]

        # Then update based on reactions
        for ester_name, ester_rates in individual_rates.items():
            # Extract the alcohol parts of the ester name (e.g., 'C18' from 'C16-C18')
            ester_parts = ester_name.split('-')
            if len(ester_parts) == 2:
                # The alcohol released from the ester
                released_alcohol = f"{ester_parts[1]}_OH"

                # Sum the rates for this ester across all alcohols
                ester_total_rate = sum(ester_rates.values())

                # Add the released alcohol
                if released_alcohol in new_alcohol_concs:
                    new_alcohol_concs[released_alcohol] += ester_total_rate
                else:
                    new_alcohol_concs[released_alcohol] = ester_total_rate

        # Subtract consumed alcohols
        for alcohol in alcohol_distribution:
            # Sum the rates for this alcohol across all esters
            alcohol_total_rate = sum(ester_rates[alcohol] for ester_rates in individual_rates.values())
            new_alcohol_concs[alcohol] -= alcohol_total_rate

        # Apply non-negative constraint to all alcohols
        for alcohol_name in new_alcohol_concs:
            new_alcohol_concs[alcohol_name] = max(0, new_alcohol_concs[alcohol_name])

        # Update ester concentrations based on reactions
        for alcohol in alcohol_distribution:
            # Get the corresponding product ester name
            alcohol_prefix = alcohol[0:3]  # e.g., 'C16'
            product_ester_name = f'{alcohol_prefix}-{alcohol_prefix}'

            # Sum the rates for this alcohol across all esters
            alcohol_total_rate = sum(ester_rates[alcohol] for ester_rates in individual_rates.values())

            # Add the formed ester
            if product_ester_name in new_ester_concs:
                new_ester_concs[product_ester_name] += alcohol_total_rate
            else:
                new_ester_concs[product_ester_name] = alcohol_total_rate

        # Check for equilibrium if requested
        if check_equilibrium and t > 0:
            # Calculate relative changes in concentrations for all esters
            ester_changes = []
            for ester_name, new_conc in new_ester_concs.items():
                if ester_name in all_esters:
                    change = abs(new_conc - all_esters[ester_name][-1]) / max(all_esters[ester_name][-1], 1e-10)
                    ester_changes.append(change)

            # Calculate changes for each alcohol
            alcohol_changes = []
            for alcohol_name, new_conc in new_alcohol_concs.items():
                if alcohol_name in all_alcohols:
                    change = abs(new_conc - all_alcohols[alcohol_name][-1]) / max(all_alcohols[alcohol_name][-1], 1e-10)
                    alcohol_changes.append(change)

            # Check if all changes are below tolerance
            if (all(change < equilibrium_tolerance for change in ester_changes) and
                all(change < equilibrium_tolerance for change in alcohol_changes)):

                equilibrium_reached = True
                equilibrium_time = t
                break

        # Append new concentrations
        ester_conc.append(new_ester)  # For backward compatibility

        # Append new ester concentrations to the big pool
        for ester_name, new_conc in new_ester_concs.items():
            if ester_name in all_esters:
                all_esters[ester_name].append(new_conc)
            else:
                # Initialize with zeros for previous time steps and add current value
                all_esters[ester_name] = [0.0] * len(ester_conc[:-1]) + [new_conc]

        # Append new alcohol concentrations to the big pool
        for alcohol_name, new_conc in new_alcohol_concs.items():
            if alcohol_name in all_alcohols:
                all_alcohols[alcohol_name].append(new_conc)
            else:
                # Initialize with zeros for previous time steps and add current value
                all_alcohols[alcohol_name] = [0.0] * len(ester_conc[:-1]) + [new_conc]

    # Return appropriate values based on check_equilibrium flag
    if check_equilibrium:
        # Return final equilibrium concentrations and time to equilibrium
        equilibrium_time_minutes = equilibrium_time / 60.0  # Convert seconds to minutes

        # Get final alcohol concentrations (all alcohols in the big pool)
        final_alcohol_concs = {alcohol: concs[-1] for alcohol, concs in all_alcohols.items()}

        # Get final ester concentrations (all esters in the big pool)
        final_ester_concs = {ester: concs[-1] for ester, concs in all_esters.items()}

        # For backward compatibility, separate alcohols into initial and product
        # Initial alcohols are those in the original alcohol_distribution
        initial_alcohol_concs = {alcohol: final_alcohol_concs[alcohol] 
                                for alcohol in alcohol_distribution if alcohol in final_alcohol_concs}

        # Product alcohols are those not in the original alcohol_distribution
        product_alcohol_concs = {alcohol: conc for alcohol, conc in final_alcohol_concs.items() 
                                if alcohol not in alcohol_distribution}

        # For backward compatibility, separate esters into initial and product
        # Product esters are those with the same alcohol prefix for both parts (e.g., 'C16-C16')
        product_ester_concs = {}
        for ester_name, conc in final_ester_concs.items():
            ester_parts = ester_name.split('-')
            if len(ester_parts) == 2 and ester_parts[0] == ester_parts[1]:
                product_ester_concs[ester_name] = conc

        # For backward compatibility, use the first product alcohol or 0.0 if none exist
        backward_compat_product_alcohol = next(iter(product_alcohol_concs.values())) if product_alcohol_concs else 0.0

        return (
            ester_conc[-1],  # For backward compatibility
            initial_alcohol_concs,  # For backward compatibility
            product_ester_concs,  # For backward compatibility
            backward_compat_product_alcohol,  # For backward compatibility
            equilibrium_time_minutes,
            equilibrium_reached,
            final_ester_concs,  # All esters in the big pool
            product_alcohol_concs  # For backward compatibility
        )
    else:
        # Return full concentration profiles over time
        time_points = np.arange(len(ester_conc))

        # Ensure all alcohol concentration lists have the same length as time_points
        for alcohol_name, concs in list(all_alcohols.items()):
            if len(concs) < len(time_points):
                # Pad with zeros at the beginning
                all_alcohols[alcohol_name] = [0.0] * (len(time_points) - len(concs)) + concs

        # For backward compatibility, separate alcohols into initial and product
        # Initial alcohols are those in the original alcohol_distribution
        initial_alcohol_concs = {alcohol: all_alcohols[alcohol] 
                                for alcohol in alcohol_distribution if alcohol in all_alcohols}

        # Product alcohols are those not in the original alcohol_distribution
        product_alcohol_concs = {alcohol: concs for alcohol, concs in all_alcohols.items() 
                                if alcohol not in alcohol_distribution}

        # For backward compatibility, separate esters into initial and product
        # Product esters are those with the same alcohol prefix for both parts (e.g., 'C16-C16')
        product_ester_concs = {}
        for ester_name, concs in all_esters.items():
            ester_parts = ester_name.split('-')
            if len(ester_parts) == 2 and ester_parts[0] == ester_parts[1]:
                product_ester_concs[ester_name] = concs

        return time_points, ester_conc, initial_alcohol_concs, product_ester_concs, product_alcohol_concs, all_esters


def plot_transesterification_results(time_points, ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc, ester_concs=None):
    """
    Plot the concentration profiles and weight percentages for all species

    Parameters:
    -----------
    time_points : array
        Time points in seconds
    ester_conc : array
        Concentration of C16-C18 ester (mol/L) over time (for backward compatibility)
    alcohol_concs : dict or array
        Dictionary mapping alcohol names to their concentrations (mol/L) over time,
        or a single array for backward compatibility
    product_ester_conc : dict or array
        Dictionary mapping product ester names to their concentrations (mol/L) over time,
        or a single array for backward compatibility
    product_alcohol_conc : dict or array
        Dictionary mapping product alcohol names to their concentrations (mol/L) over time,
        or a single array for backward compatibility
    ester_concs : dict, optional
        Dictionary mapping ester names to their concentrations (mol/L) over time
    """
    # Convert time points from seconds to minutes
    time_points_minutes = time_points / 60.0

    # Create figure with two subplots
    fig, ax1 = plt.subplots(figsize=(12, 8))  # or whatever size you want

    # Check if we're using the new dictionary format for alcohols
    if isinstance(alcohol_concs, dict):
        # Plot molar concentrations
        colors = ['b', 'g', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']  # Colors for different species

        # Plot all esters
        if ester_concs is not None:
            for i, (ester_name, concs) in enumerate(ester_concs.items()):
                color = colors[i % len(colors)]
                ax1.plot(time_points_minutes, concs, color=color, linestyle='-', label=f'{ester_name}')

        # Plot all alcohols (both initial and product)
        all_alcohols = {}
        all_alcohols.update(alcohol_concs)
        if isinstance(product_alcohol_conc, dict):
            all_alcohols.update(product_alcohol_conc)

        for i, (alcohol, concs) in enumerate(all_alcohols.items()):
            color = colors[(i + len(ester_concs or {})) % len(colors)]
            ax1.plot(time_points_minutes, concs, color=color, linestyle='-', label=f'{alcohol}')


    # Set labels and titles
    ax1.set_ylabel('Concentration (mol/L)')
    ax1.set_title('Molar Concentrations Over Time')
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    plt.show()


def display_equilibrium_results(ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc,
                           equilibrium_time_minutes, equilibrium_reached, initial_ester_conc=None, ester_concs=None,
                           product_alcohol_concs=None):
    """
    Display the equilibrium concentrations, weight percentages, and time to equilibrium

    Parameters:
    -----------
    ester_conc : float
        Final concentration of C16-C18 ester (mol/L) (for backward compatibility)
    alcohol_concs : dict or float
        Dictionary mapping alcohol names to their final concentrations (mol/L),
        or a single concentration value for backward compatibility
    product_ester_conc : dict or float
        Dictionary mapping product ester names to their final concentrations (mol/L),
        or a single concentration value for backward compatibility
    product_alcohol_conc : float
        Final concentration of product alcohol (C18_OH) (mol/L) (for backward compatibility)
    product_alcohol_concs : dict, optional
        Dictionary mapping product alcohol names to their final concentrations (mol/L)
    equilibrium_time_minutes : float
        Time to reach equilibrium in minutes
    equilibrium_reached : bool
        Whether equilibrium was reached within the simulation time
    initial_ester_conc : float, optional
        Initial concentration of C16-C18 ester (mol/L), used to calculate conversion
    ester_concs : dict, optional
        Dictionary mapping ester names to their final concentrations (mol/L)
    """
    print("=" * 60)
    print("EQUILIBRIUM RESULTS")
    print("=" * 60)

    if equilibrium_reached:
        print(f"Equilibrium reached after {equilibrium_time_minutes:.2f} minutes")
    else:
        print(f"Equilibrium NOT reached within simulation time ({equilibrium_time_minutes:.2f} minutes)")

    print("\nFinal Concentrations and Weight Percentages:")
    print("-" * 60)
    print(f"{'Species':<20} {'Concentration (mol/L)':<25} {'Weight %':<15}")
    print("-" * 60)

    # Combine all alcohols for display
    all_alcohols = {}
    if isinstance(alcohol_concs, dict):
        all_alcohols.update(alcohol_concs)
    if product_alcohol_concs:
        all_alcohols.update(product_alcohol_concs)
    elif isinstance(product_alcohol_conc, (int, float)):
        all_alcohols['C18_OH'] = product_alcohol_conc

    # Display all alcohols
    for alcohol_name, conc in all_alcohols.items():
        # Calculate weight percentage
        weight_percent = (conc * ALCOHOL_MW.get(alcohol_name, 0)) / (ester_conc * 600 + conc * ALCOHOL_MW.get(alcohol_name, 0)) * 100
        print(f"{alcohol_name:<20} {conc:<25.6f} {weight_percent:<15.2f}")

    # Display all esters if available
    if ester_concs:
        for ester_name, conc in ester_concs.items():
            # Calculate weight percentage (simplified)
            weight_percent = 100 * conc / (ester_conc + sum(all_alcohols.values()))
            print(f"{ester_name:<20} {conc:<25.6f} {weight_percent:<15.2f}")

    # Calculate conversion percentage if initial concentration is provided
    if initial_ester_conc is not None:
        conversion = (1 - ester_conc / initial_ester_conc) * 100
        print(f"\nConversion: {conversion:.2f}%")

    print("=" * 60)


# Example usage:
if __name__ == "__main__":
    # Example: Using Arrhenius equation with jojoba oil and cetearly alcohol
    print("\nUSING ARRHENIUS EQUATION WITH JOJOBA OIL AND CETEARLY ALCOHOL")
    print("-----------------------------------------------------------")

    print("Simulating transesterification between jojoba oil and cetearly alcohol:")

    # Calculate total volume in liters
    total_volume_L = ((GRAMS_JOJOBA_OIL / 1000) / DENSITY_JOJOBA_OIL) + (
                (GRAMS_CETEARYL_ALCOHOL / 1000) / DENSITY_CETEARYL_ALCOHOL)

    alcohol_distribution = Cetearyl_Alcohol.copy()

    # Calculate average molecular weight of cetearyl alcohol
    avg_alcohol_mw = 0
    for alcohol, percentage in Cetearyl_Alcohol.items():
        avg_alcohol_mw += percentage * ALCOHOL_MW[alcohol]

    # Calculate total moles
    cetearyl_moles = GRAMS_CETEARYL_ALCOHOL / avg_alcohol_mw

    # Calculate concentration in mol/L using total volume
    cetearyl_concentration = cetearyl_moles / total_volume_L

    # Convert distribution to concentrations
    for alcohol in alcohol_distribution:
        alcohol_distribution[alcohol] = Cetearyl_Alcohol[alcohol] * cetearyl_concentration

    # Define kinetics parameters for each alcohol
    base_Ea = 68000  # J/mol (activation energy)
    base_A = 1.2e7   # Pre-exponential factor
    alcohol_kinetics = {}
    for alcohol in alcohol_distribution:
        alcohol_kinetics[alcohol] = {
            'Ea_forward': base_Ea,  # J/mol (activation energy)
            'A_forward': base_A,    # Pre-exponential factor
        }

    # Calculate total concentration of jojoba oil esters from hardcoded mass and density
    total_jojoba_percentage = sum(jojoba_oil.values())

    # Calculate average molecular weight of jojoba oil
    avg_mw = 0
    for ester, percentage in jojoba_oil.items():
        if ester in ester_mw:
            avg_mw += (percentage / total_jojoba_percentage) * ester_mw[ester]

    # Calculate moles
    jojoba_moles = GRAMS_JOJOBA_OIL / avg_mw

    # Calculate concentration in mol/L using total volume
    jojoba_concentration = jojoba_moles / total_volume_L

    # Create a normalized distribution of jojoba oil esters
    normalized_jojoba_oil = {ester: (percentage / total_jojoba_percentage) * jojoba_concentration 
                            for ester, percentage in jojoba_oil.items()}

    print(normalized_jojoba_oil)

    # Map jojoba oil esters to the format expected by the simulation
    initial_product_ester_conc = {}
    for alcohol in alcohol_distribution:
        alcohol_prefix = alcohol[0:3]  # e.g., 'C16'
        initial_product_ester_conc[f'{alcohol_prefix}-{alcohol_prefix}'] = 0.0  # Initialize to zero

    # Initial conditions with temperature for Arrhenius calculation
    # We're now using the full distribution of jojoba oil esters
    initial_conditions = {
        'initial_ester_distribution': normalized_jojoba_oil,  # Dictionary of ester concentrations
        'alcohol_distribution': alcohol_distribution,
        'initial_product_ester_conc': initial_product_ester_conc,  # Dictionary of product ester concentrations
        'initial_product_alcohol_conc': 0.0,  # mol/L
        'temperature': 423.15,  # K (150°C)
        'catalyst_conc': 0.05,  # mol/L
        'reaction_time_minutes': 500,  # minutes
        'alcohol_kinetics': alcohol_kinetics
    }


    # Run simulation with equilibrium detection
    print("\nRunning simulation with equilibrium detection...")
    equilibrium_results = simulate_transesterification(**initial_conditions, check_equilibrium=True)

    # Display equilibrium results
    ester_final, alcohol_finals, product_ester_finals, product_alcohol_final, equil_time, equil_reached, ester_finals, product_alcohol_finals = equilibrium_results
    # For conversion calculation, use the sum of all initial ester concentrations
    initial_ester = sum(normalized_jojoba_oil.values())
    display_equilibrium_results(ester_final, alcohol_finals, product_ester_finals, product_alcohol_final,
                               equil_time, equil_reached, initial_ester, ester_finals, product_alcohol_finals)

    # Run simulation without equilibrium detection to get full concentration profiles
    print("\nRunning simulation for full concentration profiles...")
    time_results = simulate_transesterification(**initial_conditions, check_equilibrium=False)

    # Plot full concentration profiles and weight percentages
    print("\nPlotting concentration profiles and weight percentages...")
    plot_transesterification_results(*time_results)
