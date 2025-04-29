import math
import matplotlib.pyplot as plt
import numpy as np
from Database import C16_OH, C18_OH, C20_OH, C22_OH, C24_OH


# Molecular weights (g/mol)
MW_C16_C18_ESTER = 480.0  # C16-C18 ester (approximate)
MW_C16_ALCOHOL = 242.0    # C16 alcohol (hexadecanol)
MW_C16_C16_ESTER = 450.0  # C16-C16 ester (approximate)
MW_C18_ALCOHOL = 270.0    # C18 alcohol (octadecanol)
MW_C20_ALCOHOL = 298.50   # C20 alcohol (eicosanol)
MW_C22_ALCOHOL = 326.56   # C22 alcohol (docosanol)
MW_C24_ALCOHOL = 354.62   # C24 alcohol (tetracosanol)

# Create dictionary of alcohol molecular weights for easy lookup
ALCOHOL_MW = {
    'C16_OH': MW_C16_ALCOHOL,
    'C18_OH': MW_C18_ALCOHOL,
    'C20_OH': MW_C20_ALCOHOL,
    'C22_OH': MW_C22_ALCOHOL,
    'C24_OH': MW_C24_ALCOHOL
}

def calculate_weight_percentages(ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc):
    """
    Calculate weight percentages from molar concentrations

    Parameters:
    -----------
    ester_conc : float or array
        Concentration of C16-C18 ester (mol/L)
    alcohol_concs : dict or float/array
        Dictionary mapping alcohol names to their concentrations (mol/L),
        or a single concentration value for backward compatibility
    product_ester_conc : dict or float/array
        Dictionary mapping product ester names to their concentrations (mol/L),
        or a single concentration value for backward compatibility
    product_alcohol_conc : float or array
        Concentration of product alcohol (C18_OH) (mol/L)

    Returns:
    --------
    dict
        Dictionary containing weight percentages for all species
    """
    # Handle both the new dictionary format and the old single alcohol format
    if isinstance(alcohol_concs, dict):
        # New format with multiple alcohols
        # Convert inputs to numpy arrays
        ester_conc = np.asarray(ester_conc)
        product_alcohol_conc = np.asarray(product_alcohol_conc)

        # Convert alcohol concentrations to numpy arrays
        for alcohol in alcohol_concs:
            alcohol_concs[alcohol] = np.asarray(alcohol_concs[alcohol])

        # Define molecular weights for product esters
        product_ester_mw = {
            'C16-C16': 450.0,  # C16-C16 ester (approximate)
            'C16-C18': 480.0,  # C16-C18 ester (approximate)
            'C16-C20': 510.0,  # C16-C20 ester (approximate)
            'C16-C22': 540.0,  # C16-C22 ester (approximate)
            'C16-C24': 570.0   # C16-C24 ester (approximate)
        }

        # Convert concentrations to mass concentrations (g/L)
        ester_mass = ester_conc * MW_C16_C18_ESTER
        product_alcohol_mass = product_alcohol_conc * MW_C18_ALCOHOL

        # Calculate mass for each alcohol
        alcohol_masses = {}
        for alcohol, conc in alcohol_concs.items():
            alcohol_masses[alcohol] = conc * ALCOHOL_MW[alcohol]

        # Calculate mass for each product ester
        product_ester_masses = {}
        if isinstance(product_ester_conc, dict):
            for ester_name, conc in product_ester_conc.items():
                conc_array = np.asarray(conc)
                product_ester_masses[ester_name] = conc_array * product_ester_mw.get(ester_name, MW_C16_C16_ESTER)
        else:
            # Backward compatibility for single product ester
            product_ester_masses = {'C16-C16': np.asarray(product_ester_conc) * MW_C16_C16_ESTER}

        # Calculate total mass
        total_mass = ester_mass
        total_mass += product_alcohol_mass

        for alcohol, mass in alcohol_masses.items():
            total_mass += mass

        for ester_name, mass in product_ester_masses.items():
            total_mass += mass

        # Calculate weight percentages
        result = {
            'ester': (ester_mass / total_mass) * 100,
            'product_alcohol': (product_alcohol_mass / total_mass) * 100
        }

        # Add weight percentages for each alcohol
        for alcohol, mass in alcohol_masses.items():
            result[alcohol] = (mass / total_mass) * 100

        # Add weight percentages for each product ester
        for ester_name, mass in product_ester_masses.items():
            result[ester_name] = (mass / total_mass) * 100

        return result
    else:
        # Old format with single alcohol (for backward compatibility)
        # Convert inputs to numpy arrays
        ester_conc = np.asarray(ester_conc)
        alcohol_conc = np.asarray(alcohol_concs)  # Rename for clarity
        product_ester_conc = np.asarray(product_ester_conc)
        product_alcohol_conc = np.asarray(product_alcohol_conc)

        # Convert concentrations to mass concentrations (g/L)
        ester_mass = ester_conc * MW_C16_C18_ESTER
        alcohol_mass = alcohol_conc * MW_C16_ALCOHOL
        product_ester_mass = product_ester_conc * MW_C16_C16_ESTER
        product_alcohol_mass = product_alcohol_conc * MW_C18_ALCOHOL

        # Calculate total mass
        total_mass = ester_mass + alcohol_mass + product_ester_mass + product_alcohol_mass

        # Calculate weight percentages
        ester_wt_pct = (ester_mass / total_mass) * 100
        alcohol_wt_pct = (alcohol_mass / total_mass) * 100
        product_ester_wt_pct = (product_ester_mass / total_mass) * 100
        product_alcohol_wt_pct = (product_alcohol_mass / total_mass) * 100

        # Return in old format for backward compatibility
        return ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct


def simulate_transesterification(
        initial_ester_conc=None,
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
    """
    Simulates transesterification reaction between a mono-ester and multiple alcohols

    Parameters:
    -----------
    initial_ester_conc : float
        Initial concentration of C16-C18 ester (mol/L)
    alcohol_distribution : dict
        Dictionary mapping alcohol names (e.g., 'C16_OH') to their initial concentrations (mol/L)
    initial_product_ester_conc : float or dict, optional
        Initial concentration of product ester (mol/L) or dictionary mapping product ester names
        to their initial concentrations
    initial_product_alcohol_conc : float, optional
        Initial concentration of product alcohol (C18_OH) (mol/L)
    catalyst_conc : float
        Catalyst concentration (mol/L)
    reaction_time_minutes : int
        Maximum reaction time in minutes (simulation will stop earlier if equilibrium is reached)
    temperature : float
        Reaction temperature in Kelvin
    equilibrium_tolerance : float, optional
        Tolerance for determining when equilibrium is reached (default: 1e-7)
    check_equilibrium : bool, optional
        Whether to check for equilibrium during simulation (default: True)
    alcohol_kinetics : dict, optional
        Dictionary mapping alcohol names to dictionaries containing 'Ea_forward', 'Ea_reverse',
        'A_forward', and 'A_reverse' values. If not provided, default values will be used.

    Returns:
    --------
    tuple
        If check_equilibrium=True: Final equilibrium concentrations and time to equilibrium
        If check_equilibrium=False: Lists of concentrations over time for all species
    """
    # Ensure required parameters are provided
    if initial_ester_conc is None or alcohol_distribution is None or catalyst_conc is None or reaction_time_minutes is None:
        raise ValueError("Initial ester concentration, alcohol distribution, catalyst concentration, and reaction time must be provided")

    if temperature is None:
        raise ValueError("Temperature must be provided for Arrhenius calculations")

    # Set default values for product concentrations if not provided
    if initial_product_ester_conc is None:
        # Initialize product esters as a dictionary with zero concentration for each possible product
        initial_product_ester_conc = {f'C16-{alcohol[1:3]}': 0.0 for alcohol in alcohol_distribution}
    elif not isinstance(initial_product_ester_conc, dict):
        # Convert single value to dictionary for backward compatibility
        initial_product_ester_conc = {f'C16-{alcohol[1:3]}': initial_product_ester_conc for alcohol in alcohol_distribution}

    if initial_product_alcohol_conc is None:
        initial_product_alcohol_conc = 0.0

    # Set default kinetics parameters if not provided
    if alcohol_kinetics is None:
        alcohol_kinetics = {}
        for alcohol in alcohol_distribution:
            alcohol_kinetics[alcohol] = {
                'Ea_forward': 70000,  # J/mol (activation energy)
                'A_forward': 1.0e7,   # Pre-exponential factor
            }
    else:
        # Ensure all alcohols in the distribution have kinetics parameters
        for alcohol in alcohol_distribution:
            if alcohol not in alcohol_kinetics:
                alcohol_kinetics[alcohol] = {
                    'Ea_forward': 70000,  # J/mol (activation energy)
                    'A_forward': 1.0e7,   # Pre-exponential factor
                }
            # Remove any reverse parameters if they exist (for backward compatibility)
            if 'Ea_reverse' in alcohol_kinetics[alcohol]:
                del alcohol_kinetics[alcohol]['Ea_reverse']
            if 'A_reverse' in alcohol_kinetics[alcohol]:
                del alcohol_kinetics[alcohol]['A_reverse']

    # Convert minutes to seconds
    seconds = reaction_time_minutes * 60

    # Initialize concentration arrays
    ester_conc = [initial_ester_conc]

    # Initialize product ester concentrations (one for each alcohol)
    product_ester_conc = {ester: [conc] for ester, conc in initial_product_ester_conc.items()}

    # Initialize product alcohol concentration (C18_OH)
    product_alcohol_conc = [initial_product_alcohol_conc]

    # Initialize alcohol concentrations
    alcohol_concs = {alcohol: [conc] for alcohol, conc in alcohol_distribution.items()}

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
        # Calculate individual reaction rates for each alcohol
        individual_rates = {}
        total_net_rate = 0

        for alcohol in alcohol_distribution:
            # Calculate forward rate for this alcohol
            forward_rate = k_forwards[alcohol] * ester_conc[-1] * alcohol_concs[alcohol][-1] * catalyst_conc

            # For reverse rate, we use the forward rates of other alcohols
            # This implements the concept that "one alcohol's reverse is another alcohol's forward"
            reverse_contribution = 0
            for other_alcohol in alcohol_distribution:
                if other_alcohol != alcohol:
                    # Each other alcohol's forward reaction contributes to this alcohol's reverse reaction
                    other_forward = k_forwards[other_alcohol] * ester_conc[-1] * alcohol_concs[other_alcohol][-1] * catalyst_conc
                    reverse_contribution += other_forward / (len(alcohol_distribution) - 1)

            # Calculate net rate for this alcohol
            net_rate = (forward_rate - reverse_contribution) * dt
            individual_rates[alcohol] = net_rate
            total_net_rate += net_rate

        # Update ester concentration based on total net rate
        new_ester = ester_conc[-1] - total_net_rate

        # Update product alcohol concentration (C18_OH)
        new_product_alcohol = product_alcohol_conc[-1] + total_net_rate

        # Update product ester concentrations based on individual rates
        new_product_ester_concs = {}
        for alcohol in alcohol_distribution:
            # Get the corresponding product ester name
            product_ester_name = f'C16-{alcohol[1:3]}'

            # Update this product ester's concentration
            new_conc = product_ester_conc[product_ester_name][-1] + individual_rates[alcohol]
            new_product_ester_concs[product_ester_name] = max(0, new_conc)

        # Update alcohol concentrations based on their individual rates
        new_alcohol_concs = {}
        for alcohol in alcohol_distribution:
            new_alcohol_concs[alcohol] = max(0, alcohol_concs[alcohol][-1] - individual_rates[alcohol])

        # Apply non-negative constraint
        new_ester = max(0, new_ester)
        new_product_alcohol = max(0, new_product_alcohol)

        # Check for equilibrium if requested
        if check_equilibrium and t > 0:
            # Calculate relative changes in concentrations
            ester_change = abs(new_ester - ester_conc[-1]) / max(ester_conc[-1], 1e-10)
            product_alcohol_change = abs(new_product_alcohol - product_alcohol_conc[-1]) / max(product_alcohol_conc[-1], 1e-10)

            # Calculate changes for each product ester
            product_ester_changes = []
            for ester_name, new_conc in new_product_ester_concs.items():
                change = abs(new_conc - product_ester_conc[ester_name][-1]) / max(product_ester_conc[ester_name][-1], 1e-10)
                product_ester_changes.append(change)

            # Calculate changes for each alcohol
            alcohol_changes = []
            for alcohol in alcohol_distribution:
                change = abs(new_alcohol_concs[alcohol] - alcohol_concs[alcohol][-1]) / max(alcohol_concs[alcohol][-1], 1e-10)
                alcohol_changes.append(change)

            # Check if all changes are below tolerance
            if (ester_change < equilibrium_tolerance and
                product_alcohol_change < equilibrium_tolerance and
                all(change < equilibrium_tolerance for change in product_ester_changes) and
                all(change < equilibrium_tolerance for change in alcohol_changes)):

                equilibrium_reached = True
                equilibrium_time = t
                break

        # Append new concentrations
        ester_conc.append(new_ester)
        product_alcohol_conc.append(new_product_alcohol)

        # Append new product ester concentrations
        for ester_name, new_conc in new_product_ester_concs.items():
            product_ester_conc[ester_name].append(new_conc)

        # Append new alcohol concentrations
        for alcohol in alcohol_distribution:
            alcohol_concs[alcohol].append(new_alcohol_concs[alcohol])

    # Return appropriate values based on check_equilibrium flag
    if check_equilibrium:
        # Return final equilibrium concentrations and time to equilibrium
        equilibrium_time_minutes = equilibrium_time / 60.0  # Convert seconds to minutes

        # Get final alcohol concentrations
        final_alcohol_concs = {alcohol: concs[-1] for alcohol, concs in alcohol_concs.items()}

        # Get final product ester concentrations
        final_product_ester_concs = {ester: concs[-1] for ester, concs in product_ester_conc.items()}

        return (
            ester_conc[-1],
            final_alcohol_concs,
            final_product_ester_concs,
            product_alcohol_conc[-1],
            equilibrium_time_minutes,
            equilibrium_reached
        )
    else:
        # Return full concentration profiles over time
        time_points = np.arange(len(ester_conc))
        return time_points, ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc


def plot_transesterification_results(time_points, ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc):
    """
    Plot the concentration profiles and weight percentages for all species

    Parameters:
    -----------
    time_points : array
        Time points in seconds
    ester_conc : array
        Concentration of C16-C18 ester (mol/L) over time
    alcohol_concs : dict or array
        Dictionary mapping alcohol names to their concentrations (mol/L) over time,
        or a single array for backward compatibility
    product_ester_conc : dict or array
        Dictionary mapping product ester names to their concentrations (mol/L) over time,
        or a single array for backward compatibility
    product_alcohol_conc : array
        Concentration of product alcohol (C18_OH) (mol/L) over time
    """
    # Convert time points from seconds to minutes
    time_points_minutes = time_points / 60.0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Check if we're using the new dictionary format for alcohols
    if isinstance(alcohol_concs, dict):
        # Calculate weight percentages at each time point
        weight_percentages = []
        for i in range(len(time_points)):
            # Extract concentrations at this time point
            ester_at_t = ester_conc[i]
            product_alcohol_at_t = product_alcohol_conc[i]

            # Extract alcohol concentrations at this time point
            alcohols_at_t = {}
            for alcohol, concs in alcohol_concs.items():
                alcohols_at_t[alcohol] = concs[i]

            # Extract product ester concentrations at this time point
            if isinstance(product_ester_conc, dict):
                product_esters_at_t = {}
                for ester_name, concs in product_ester_conc.items():
                    product_esters_at_t[ester_name] = concs[i]
            else:
                # Backward compatibility
                product_esters_at_t = product_ester_conc[i]

            # Calculate weight percentages at this time point
            wt_pct = calculate_weight_percentages(
                ester_at_t, alcohols_at_t, product_esters_at_t, product_alcohol_at_t
            )
            weight_percentages.append(wt_pct)

        # Plot molar concentrations on top subplot
        ax1.plot(time_points_minutes, ester_conc, 'b-', label='C16-C18 Ester')
        ax1.plot(time_points_minutes, product_alcohol_conc, 'c-', label='C18 Alcohol (Product)')

        # Plot each alcohol concentration
        colors = ['g', 'm', 'y', 'k', 'orange']  # Colors for different alcohols
        for i, (alcohol, concs) in enumerate(alcohol_concs.items()):
            color = colors[i % len(colors)]
            ax1.plot(time_points_minutes, concs, color=color, linestyle='-', label=f'{alcohol}')

        # Plot each product ester concentration
        if isinstance(product_ester_conc, dict):
            for i, (ester_name, concs) in enumerate(product_ester_conc.items()):
                color = colors[i % len(colors)]
                ax1.plot(time_points_minutes, concs, color=color, linestyle='--', label=f'{ester_name} (Product)')

        # Plot weight percentages on bottom subplot
        ax2.plot(time_points_minutes, [wp['ester'] for wp in weight_percentages], 'b-', label='C16-C18 Ester')
        ax2.plot(time_points_minutes, [wp['product_alcohol'] for wp in weight_percentages], 'c-', label='C18 Alcohol (Product)')

        # Plot each alcohol weight percentage
        for i, alcohol in enumerate(alcohol_concs.keys()):
            color = colors[i % len(colors)]
            ax2.plot(time_points_minutes, [wp[alcohol] for wp in weight_percentages], color=color, linestyle='-', label=f'{alcohol}')

        # Plot each product ester weight percentage
        if isinstance(product_ester_conc, dict):
            for i, ester_name in enumerate(product_ester_conc.keys()):
                color = colors[i % len(colors)]
                ax2.plot(time_points_minutes, [wp.get(ester_name, 0) for wp in weight_percentages], 
                         color=color, linestyle='--', label=f'{ester_name} (Product)')
    else:
        # Old format with single alcohol (for backward compatibility)
        # Calculate weight percentages
        ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct = calculate_weight_percentages(
            ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc
        )

        # Plot molar concentrations on top subplot
        ax1.plot(time_points_minutes, ester_conc, 'b-', label='C16-C18 Ester')
        ax1.plot(time_points_minutes, alcohol_concs, 'g-', label='C16 Alcohol')
        ax1.plot(time_points_minutes, product_ester_conc, 'r-', label='C16-C16 Ester')
        ax1.plot(time_points_minutes, product_alcohol_conc, 'c-', label='C18 Alcohol')

        # Plot weight percentages on bottom subplot
        ax2.plot(time_points_minutes, ester_wt_pct, 'b-', label='C16-C18 Ester')
        ax2.plot(time_points_minutes, alcohol_wt_pct, 'g-', label='C16 Alcohol')
        ax2.plot(time_points_minutes, product_ester_wt_pct, 'r-', label='C16-C16 Ester')
        ax2.plot(time_points_minutes, product_alcohol_wt_pct, 'c-', label='C18 Alcohol')

    # Set labels and titles
    ax1.set_ylabel('Concentration (mol/L)')
    ax1.set_title('Molar Concentrations Over Time')
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Weight Percentage (%)')
    ax2.set_title('Weight Percentages Over Time')
    ax2.grid(True)
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    plt.show()


def display_equilibrium_results(ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc,
                           equilibrium_time_minutes, equilibrium_reached, initial_ester_conc=None):
    """
    Display the equilibrium concentrations, weight percentages, and time to equilibrium

    Parameters:
    -----------
    ester_conc : float
        Final concentration of C16-C18 ester (mol/L)
    alcohol_concs : dict or float
        Dictionary mapping alcohol names to their final concentrations (mol/L),
        or a single concentration value for backward compatibility
    product_ester_conc : dict or float
        Dictionary mapping product ester names to their final concentrations (mol/L),
        or a single concentration value for backward compatibility
    product_alcohol_conc : float
        Final concentration of product alcohol (C18_OH) (mol/L)
    equilibrium_time_minutes : float
        Time to reach equilibrium in minutes
    equilibrium_reached : bool
        Whether equilibrium was reached within the simulation time
    initial_ester_conc : float, optional
        Initial concentration of C16-C18 ester (mol/L), used to calculate conversion
    """
    # Calculate weight percentages
    weight_percentages = calculate_weight_percentages(
        ester_conc, alcohol_concs, product_ester_conc, product_alcohol_conc
    )

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

    # Check if we're using the new dictionary format for alcohols
    if isinstance(alcohol_concs, dict):
        # New format with multiple alcohols
        print(f"C16-C18 Ester:      {ester_conc:.6f} mol/L{' ' * 10}{weight_percentages['ester']:.2f}%")

        # Print each alcohol
        for alcohol, conc in alcohol_concs.items():
            print(f"{alcohol}:           {conc:.6f} mol/L{' ' * 10}{weight_percentages[alcohol]:.2f}%")

        # Print each product ester
        if isinstance(product_ester_conc, dict):
            for ester_name, conc in product_ester_conc.items():
                print(f"{ester_name} (Product): {conc:.6f} mol/L{' ' * 10}{weight_percentages.get(ester_name, 0):.2f}%")
        else:
            # Backward compatibility
            print(f"Product Ester:      {product_ester_conc:.6f} mol/L{' ' * 10}{weight_percentages.get('product_ester', 0):.2f}%")

        print(f"C18 Alcohol (Product): {product_alcohol_conc:.6f} mol/L{' ' * 10}{weight_percentages['product_alcohol']:.2f}%")
    else:
        # Old format with single alcohol (for backward compatibility)
        ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct = weight_percentages
        print(f"C16-C18 Ester:      {ester_conc:.6f} mol/L{' ' * 10}{ester_wt_pct:.2f}%")
        print(f"C16 Alcohol:        {alcohol_concs:.6f} mol/L{' ' * 10}{alcohol_wt_pct:.2f}%")
        print(f"C16-C16 Ester:      {product_ester_conc:.6f} mol/L{' ' * 10}{product_ester_wt_pct:.2f}%")
        print(f"C18 Alcohol:        {product_alcohol_conc:.6f} mol/L{' ' * 10}{product_alcohol_wt_pct:.2f}%")

    # Calculate conversion percentage if initial concentration is provided
    if initial_ester_conc is not None:
        conversion = (1 - ester_conc / initial_ester_conc) * 100
        print(f"\nConversion: {conversion:.2f}%")

    print("=" * 60)


# Example usage:
if __name__ == "__main__":
    # Example: Using Arrhenius equation with a blend of alcohols ranging from C16-C24
    print("\nUSING ARRHENIUS EQUATION WITH A BLEND OF ALCOHOLS RANGING FROM C16-C24")
    print("-----------------------------------------------------------")

    print("Simulating transesterification with a blend of alcohols (C16-C24):")

    # Define a blend of alcohols with different initial concentrations
    alcohol_distribution = {
        'C16_OH': 1.0,  # mol/L
        'C18_OH': 1.0,  # mol/L
        'C20_OH': 1.0,  # mol/L
        'C22_OH': 1.0,  # mol/L
        'C24_OH': 1.0   # mol/L
    }

    # Define different kinetics parameters for each alcohol
    # Note: We only need to specify forward parameters
    # One alcohol's reverse reaction is another alcohol's forward reaction
    alcohol_kinetics = {
        'C16_OH': {
            'Ea_forward': 68000,  # J/mol (activation energy)
            'A_forward': 1.4e7,   # Pre-exponential factor
        },
        'C18_OH': {
            'Ea_forward': 68000,  # J/mol (activation energy)
            'A_forward': 1.0e7,   # Pre-exponential factor
        },
        'C20_OH': {
            'Ea_forward': 68000,  # J/mol (activation energy)
            'A_forward': 1.2e7,   # Pre-exponential factor
        },
        'C22_OH': {
            'Ea_forward': 68000,  # J/mol (activation energy)
            'A_forward': 1.2e7,   # Pre-exponential factor
        },
        'C24_OH': {
            'Ea_forward': 68000,  # J/mol (activation energy)
            'A_forward': 1.0e7,   # Pre-exponential factor
        }
    }

    # Initialize product ester concentrations (one for each alcohol)
    initial_product_ester_conc = {f'C16-{alcohol[1:3]}': 0.0 for alcohol in alcohol_distribution}

    # Initial conditions with temperature for Arrhenius calculation
    initial_conditions = {
        'initial_ester_conc': 2.0,  # mol/L
        'alcohol_distribution': alcohol_distribution,
        'initial_product_ester_conc': initial_product_ester_conc,  # Dictionary of product ester concentrations
        'initial_product_alcohol_conc': 0.0,  # mol/L
        'temperature': 423.15,  # K (150°C)
        'catalyst_conc': 0.005,  # mol/L
        'reaction_time_minutes': 100,  # minutes
        'alcohol_kinetics': alcohol_kinetics
    }

    # Run simulation with equilibrium detection
    print("\nRunning simulation with equilibrium detection...")
    equilibrium_results = simulate_transesterification(**initial_conditions, check_equilibrium=True)

    # Display equilibrium results
    ester_final, alcohol_finals, product_ester_finals, product_alcohol_final, equil_time, equil_reached = equilibrium_results
    initial_ester = initial_conditions['initial_ester_conc']
    display_equilibrium_results(ester_final, alcohol_finals, product_ester_finals, product_alcohol_final,
                               equil_time, equil_reached, initial_ester)

    # Run simulation without equilibrium detection to get full concentration profiles
    print("\nRunning simulation for full concentration profiles...")
    time_results = simulate_transesterification(**initial_conditions, check_equilibrium=False)

    # Plot full concentration profiles and weight percentages
    print("\nPlotting concentration profiles and weight percentages...")
    plot_transesterification_results(*time_results)
