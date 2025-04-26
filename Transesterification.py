import math
import matplotlib.pyplot as plt
import numpy as np


# Molecular weights (g/mol)
MW_C16_C18_ESTER = 480.0  # C16-C18 ester (approximate)
MW_C16_ALCOHOL = 242.0    # C16 alcohol (hexadecanol)
MW_C16_C16_ESTER = 450.0  # C16-C16 ester (approximate)
MW_C18_ALCOHOL = 270.0    # C18 alcohol (octadecanol)


def convert_grams_to_molar(ester_grams, alcohol_grams, product_ester_grams, product_alcohol_grams, catalyst_grams=0.0, catalyst_mw=1.0):
    """
    Convert grams directly to molar concentrations (mol/L)

    Parameters:
    -----------
    ester_grams : float
        Grams of C16-C18 ester
    alcohol_grams : float
        Grams of C16 alcohol
    product_ester_grams : float
        Grams of C16-C16 ester
    product_alcohol_grams : float
        Grams of C18 alcohol
    catalyst_grams : float, optional
        Grams of catalyst (default: 0.0)
    catalyst_mw : float, optional
        Molecular weight of catalyst in g/mol (default: 1.0)
        This is a placeholder - in a real application, you should use the actual molecular weight

    Returns:
    --------
    tuple
        Molar concentrations for each species (mol/L) and catalyst concentration
    """
    # Convert grams to moles
    ester_moles = ester_grams / MW_C16_C18_ESTER
    alcohol_moles = alcohol_grams / MW_C16_ALCOHOL
    product_ester_moles = product_ester_grams / MW_C16_C16_ESTER
    product_alcohol_moles = product_alcohol_grams / MW_C18_ALCOHOL

    # Convert catalyst grams to moles using provided molecular weight
    catalyst_moles = catalyst_grams / catalyst_mw

    # Calculate total moles
    total_moles = ester_moles + alcohol_moles + product_ester_moles + product_alcohol_moles + catalyst_moles

    # Calculate concentrations directly from the amounts (no solvent)
    # Here we assume the concentrations are simply the molar amounts
    ester_conc = ester_moles
    alcohol_conc = alcohol_moles
    product_ester_conc = product_ester_moles
    product_alcohol_conc = product_alcohol_moles
    catalyst_conc = catalyst_moles

    return ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc, catalyst_conc


def convert_weight_percentages_to_molar(ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct, total_concentration=None):
    """
    Convert weight percentages to molar concentrations (mol/L)

    Parameters:
    -----------
    ester_wt_pct : float
        Weight percentage of C16-C18 ester
    alcohol_wt_pct : float
        Weight percentage of C16 alcohol
    product_ester_wt_pct : float
        Weight percentage of C16-C16 ester
    product_alcohol_wt_pct : float
        Weight percentage of C18 alcohol
    total_concentration : float, optional
        Total concentration in mol/L to scale the results. 
        If None, concentrations are determined directly from the amounts of esters and alcohols.

    Returns:
    --------
    tuple
        Molar concentrations for each species (mol/L)
    """
    # Ensure percentages sum to 100%
    total_pct = ester_wt_pct + alcohol_wt_pct + product_ester_wt_pct + product_alcohol_wt_pct
    if abs(total_pct - 100.0) > 1e-6:
        # Normalize if not exactly 100%
        scale_factor = 100.0 / total_pct
        ester_wt_pct *= scale_factor
        alcohol_wt_pct *= scale_factor
        product_ester_wt_pct *= scale_factor
        product_alcohol_wt_pct *= scale_factor

    # Convert weight percentages to mass fractions
    ester_mass_fraction = ester_wt_pct / 100.0
    alcohol_mass_fraction = alcohol_wt_pct / 100.0
    product_ester_mass_fraction = product_ester_wt_pct / 100.0
    product_alcohol_mass_fraction = product_alcohol_wt_pct / 100.0

    # Calculate moles of each component in 1 kg of mixture
    ester_moles = ester_mass_fraction * 1000 / MW_C16_C18_ESTER
    alcohol_moles = alcohol_mass_fraction * 1000 / MW_C16_ALCOHOL
    product_ester_moles = product_ester_mass_fraction * 1000 / MW_C16_C16_ESTER
    product_alcohol_moles = product_alcohol_mass_fraction * 1000 / MW_C18_ALCOHOL

    # Calculate total moles
    total_moles = ester_moles + alcohol_moles + product_ester_moles + product_alcohol_moles

    if total_concentration is not None:
        # Scale to get the desired total concentration (using solvent)
        scale = total_concentration / total_moles

        # Calculate molar concentrations
        ester_conc = ester_moles * scale
        alcohol_conc = alcohol_moles * scale
        product_ester_conc = product_ester_moles * scale
        product_alcohol_conc = product_alcohol_moles * scale
    else:
        # Calculate concentrations directly from the amounts (no solvent)
        # Here we assume the concentrations are simply the molar amounts
        ester_conc = ester_moles
        alcohol_conc = alcohol_moles
        product_ester_conc = product_ester_moles
        product_alcohol_conc = product_alcohol_moles

    return ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc


def calculate_weight_percentages(ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc):
    """
    Calculate weight percentages from molar concentrations

    Parameters:
    -----------
    ester_conc : float or array
        Concentration of C16-C18 ester (mol/L)
    alcohol_conc : float or array
        Concentration of C16 alcohol (mol/L)
    product_ester_conc : float or array
        Concentration of C16-C16 ester (mol/L)
    product_alcohol_conc : float or array
        Concentration of C18 alcohol (mol/L)

    Returns:
    --------
    tuple
        Weight percentages for each species
    """
    # Convert inputs to numpy arrays to handle both scalar and array inputs
    ester_conc = np.asarray(ester_conc)
    alcohol_conc = np.asarray(alcohol_conc)
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

    return ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct


def simulate_transesterification(
        initial_ester_conc=None,
        initial_alcohol_conc=None,
        initial_product_ester_conc=None,
        initial_product_alcohol_conc=None,
        catalyst_conc=None,
        reaction_time_minutes=None,
        k_forward=None,
        k_reverse=None,
        temperature=None,
        equilibrium_tolerance=1e-5,
        check_equilibrium=True,
        initial_ester_wt_pct=None,
        initial_alcohol_wt_pct=None,
        initial_product_ester_wt_pct=None,
        initial_product_alcohol_wt_pct=None,
        total_concentration=None,
        use_weight_percent=False,
        use_solvent=True,
        initial_ester_grams=None,
        initial_alcohol_grams=None,
        initial_product_ester_grams=None,
        initial_product_alcohol_grams=None,
        catalyst_grams=None,
        catalyst_mw=1.0,
        use_grams=False
):
    """
    Simulates transesterification reaction between a mono-ester and alcohol

    Parameters:
    -----------
    initial_ester_conc : float, optional
        Initial concentration of C16-C18 ester (mol/L)
    initial_alcohol_conc : float, optional
        Initial concentration of C16 alcohol (mol/L)
    initial_product_ester_conc : float, optional
        Initial concentration of C16-C16 ester (mol/L)
    initial_product_alcohol_conc : float, optional
        Initial concentration of C18 alcohol (mol/L)
    catalyst_conc : float, optional
        Catalyst concentration (mol/L)
    reaction_time_minutes : int
        Maximum reaction time in minutes (simulation will stop earlier if equilibrium is reached)
    k_forward : float, optional
        Forward reaction rate constant. If not provided, will be calculated using Arrhenius equation.
    k_reverse : float, optional
        Reverse reaction rate constant. If not provided, will be calculated using Arrhenius equation.
    temperature : float, optional
        Reaction temperature in Kelvin. Required if k_forward or k_reverse is not provided.
    equilibrium_tolerance : float, optional
        Tolerance for determining when equilibrium is reached (default: 1e-10)
    check_equilibrium : bool, optional
        Whether to check for equilibrium during simulation (default: True)
    initial_ester_wt_pct : float, optional
        Initial weight percentage of C16-C18 ester
    initial_alcohol_wt_pct : float, optional
        Initial weight percentage of C16 alcohol
    initial_product_ester_wt_pct : float, optional
        Initial weight percentage of C16-C16 ester
    initial_product_alcohol_wt_pct : float, optional
        Initial weight percentage of C18 alcohol
    total_concentration : float, optional
        Total concentration in mol/L to scale the weight percentage results. 
        If None and use_solvent is False, concentrations are determined directly from the amounts.
    use_weight_percent : bool, optional
        Whether to use weight percentages instead of molar concentrations (default: False)
    use_solvent : bool, optional
        Whether to use a solvent when calculating concentrations (default: True).
        If False, concentrations are determined directly from the amounts of esters and alcohols.
    initial_ester_grams : float, optional
        Initial grams of C16-C18 ester
    initial_alcohol_grams : float, optional
        Initial grams of C16 alcohol
    initial_product_ester_grams : float, optional
        Initial grams of C16-C16 ester
    initial_product_alcohol_grams : float, optional
        Initial grams of C18 alcohol
    catalyst_grams : float, optional
        Grams of catalyst
    catalyst_mw : float, optional
        Molecular weight of catalyst in g/mol (default: 1.0).
        This is a placeholder - in a real application, you should use the actual molecular weight.
    use_grams : bool, optional
        Whether to use grams instead of molar concentrations or weight percentages (default: False).
        If True, concentrations are determined directly from the grams of each component.

    Returns:
    --------
    tuple
        If check_equilibrium=True: Final equilibrium concentrations and time to equilibrium
        If check_equilibrium=False: Lists of concentrations over time for all species
    """
    # Check if we're using grams
    if use_grams:
        if None in (initial_ester_grams, initial_alcohol_grams, initial_product_ester_grams, initial_product_alcohol_grams, catalyst_grams):
            raise ValueError("All initial grams must be provided when use_grams is True")

        # Convert grams to molar concentrations
        initial_ester_conc, initial_alcohol_conc, initial_product_ester_conc, initial_product_alcohol_conc, catalyst_conc = convert_grams_to_molar(
            initial_ester_grams, initial_alcohol_grams, initial_product_ester_grams, initial_product_alcohol_grams, catalyst_grams, catalyst_mw
        )
    # Check if we're using weight percentages
    elif use_weight_percent:
        if None in (initial_ester_wt_pct, initial_alcohol_wt_pct, initial_product_ester_wt_pct, initial_product_alcohol_wt_pct):
            raise ValueError("All weight percentages must be provided when use_weight_percent is True")

        # Convert weight percentages to molar concentrations
        if use_solvent:
            # Use total_concentration for scaling if using solvent
            initial_ester_conc, initial_alcohol_conc, initial_product_ester_conc, initial_product_alcohol_conc = convert_weight_percentages_to_molar(
                initial_ester_wt_pct, initial_alcohol_wt_pct, initial_product_ester_wt_pct, initial_product_alcohol_wt_pct, total_concentration
            )
        else:
            # Calculate concentrations directly from amounts if not using solvent
            initial_ester_conc, initial_alcohol_conc, initial_product_ester_conc, initial_product_alcohol_conc = convert_weight_percentages_to_molar(
                initial_ester_wt_pct, initial_alcohol_wt_pct, initial_product_ester_wt_pct, initial_product_alcohol_wt_pct, None
            )
    else:
        # Ensure all required molar concentrations are provided
        if None in (initial_ester_conc, initial_alcohol_conc, initial_product_ester_conc, initial_product_alcohol_conc):
            raise ValueError("All initial concentrations must be provided when use_weight_percent is False and use_grams is False")

    # Ensure catalyst concentration and reaction time are provided
    if catalyst_conc is None or reaction_time_minutes is None:
        raise ValueError("Catalyst concentration and reaction time must be provided")

    # Convert minutes to seconds
    seconds = reaction_time_minutes * 60

    # Initialize concentration arrays
    ester_conc = [initial_ester_conc]
    alcohol_conc = [initial_alcohol_conc]
    product_ester_conc = [initial_product_ester_conc]
    product_alcohol_conc = [initial_product_alcohol_conc]

    # Determine reaction rate constants
    if k_forward is None or k_reverse is None:
        if temperature is None:
            raise ValueError("Temperature must be provided if k_forward or k_reverse is not specified")

        # Reaction rate constants (Arrhenius equation)
        # These values are approximate and should be adjusted based on experimental data
        Ea_forward = 65000  # J/mol (activation energy)
        Ea_reverse = 65000  # J/mol
        R = 8.314  # Gas constant
        A_forward = 5.0e7  # Pre-exponential factor
        A_reverse = 5.0e7

        if k_forward is None:
            k_forward = A_forward * math.exp(-Ea_forward / (R * temperature))
        if k_reverse is None:
            k_reverse = A_reverse * math.exp(-Ea_reverse / (R * temperature))

    # Time step (1 second)
    dt = 1.0

    # Variables to track equilibrium
    equilibrium_reached = False
    equilibrium_time = seconds

    # Simulation loop
    for t in range(seconds):
        # Calculate reaction rates
        forward_rate = k_forward * ester_conc[-1] * alcohol_conc[-1] * catalyst_conc
        reverse_rate = k_reverse * product_ester_conc[-1] * product_alcohol_conc[-1] * catalyst_conc

        net_rate = (forward_rate - reverse_rate) * dt

        # Update concentrations
        new_ester = ester_conc[-1] - net_rate
        new_alcohol = alcohol_conc[-1] - net_rate
        new_product_ester = product_ester_conc[-1] + net_rate
        new_product_alcohol = product_alcohol_conc[-1] + net_rate

        # Apply non-negative constraint
        new_ester = max(0, new_ester)
        new_alcohol = max(0, new_alcohol)
        new_product_ester = max(0, new_product_ester)
        new_product_alcohol = max(0, new_product_alcohol)

        # Check for equilibrium if requested
        if check_equilibrium and t > 0:
            # Calculate relative changes in concentrations
            ester_change = abs(new_ester - ester_conc[-1]) / max(ester_conc[-1], 1e-10)
            alcohol_change = abs(new_alcohol - alcohol_conc[-1]) / max(alcohol_conc[-1], 1e-10)
            product_ester_change = abs(new_product_ester - product_ester_conc[-1]) / max(product_ester_conc[-1], 1e-10)
            product_alcohol_change = abs(new_product_alcohol - product_alcohol_conc[-1]) / max(product_alcohol_conc[-1], 1e-10)

            # Check if all changes are below tolerance
            if (ester_change < equilibrium_tolerance and
                alcohol_change < equilibrium_tolerance and
                product_ester_change < equilibrium_tolerance and
                product_alcohol_change < equilibrium_tolerance):

                equilibrium_reached = True
                equilibrium_time = t
                break

        # Append new concentrations
        ester_conc.append(new_ester)
        alcohol_conc.append(new_alcohol)
        product_ester_conc.append(new_product_ester)
        product_alcohol_conc.append(new_product_alcohol)

    # Return appropriate values based on check_equilibrium flag
    if check_equilibrium:
        # Return final equilibrium concentrations and time to equilibrium
        equilibrium_time_minutes = equilibrium_time / 60.0  # Convert seconds to minutes
        return (
            ester_conc[-1],
            alcohol_conc[-1],
            product_ester_conc[-1],
            product_alcohol_conc[-1],
            equilibrium_time_minutes,
            equilibrium_reached
        )
    else:
        # Return full concentration profiles over time
        time_points = np.arange(len(ester_conc))
        return time_points, ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc


def plot_transesterification_results(time_points, ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc):
    """
    Plot the concentration profiles and weight percentages for all species
    """
    # Calculate weight percentages
    ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct = calculate_weight_percentages(
        ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc
    )

    # Convert time points from seconds to minutes
    time_points_minutes = time_points / 60.0

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    # Plot molar concentrations on left y-axis
    ax1.plot(time_points_minutes, ester_conc, 'b-', label='C16-C18 Ester (mol/L)')
    ax1.plot(time_points_minutes, alcohol_conc, 'g-', label='C16 Alcohol (mol/L)')
    ax1.plot(time_points_minutes, product_ester_conc, 'r-', label='C16-C16 Ester (mol/L)')
    ax1.plot(time_points_minutes, product_alcohol_conc, 'c-', label='C18 Alcohol (mol/L)')

    # Plot weight percentages on right y-axis
    ax2.plot(time_points_minutes, ester_wt_pct, 'b--', label='C16-C18 Ester (wt%)')
    ax2.plot(time_points_minutes, alcohol_wt_pct, 'g--', label='C16 Alcohol (wt%)')
    ax2.plot(time_points_minutes, product_ester_wt_pct, 'r--', label='C16-C16 Ester (wt%)')
    ax2.plot(time_points_minutes, product_alcohol_wt_pct, 'c--', label='C18 Alcohol (wt%)')

    # Set labels and title
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Concentration (mol/L)')
    ax2.set_ylabel('Weight Percentage (%)')
    plt.title('Transesterification Reaction Progress')

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_equilibrium_results(ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc,
                           equilibrium_time_minutes, equilibrium_reached, initial_ester_conc=None):
    """
    Display the equilibrium concentrations, weight percentages, and time to equilibrium

    Parameters:
    -----------
    ester_conc : float
        Final concentration of C16-C18 ester (mol/L)
    alcohol_conc : float
        Final concentration of C16 alcohol (mol/L)
    product_ester_conc : float
        Final concentration of C16-C16 ester (mol/L)
    product_alcohol_conc : float
        Final concentration of C18 alcohol (mol/L)
    equilibrium_time_minutes : float
        Time to reach equilibrium in minutes
    equilibrium_reached : bool
        Whether equilibrium was reached within the simulation time
    initial_ester_conc : float, optional
        Initial concentration of C16-C18 ester (mol/L), used to calculate conversion
    """
    # Calculate weight percentages
    ester_wt_pct, alcohol_wt_pct, product_ester_wt_pct, product_alcohol_wt_pct = calculate_weight_percentages(
        ester_conc, alcohol_conc, product_ester_conc, product_alcohol_conc
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
    print(f"{'Species':<15} {'Concentration (mol/L)':<25} {'Weight %':<15}")
    print("-" * 60)
    print(f"C16-C18 Ester:  {ester_conc:.6f} mol/L{' ' * 10}{ester_wt_pct:.2f}%")
    print(f"C16 Alcohol:    {alcohol_conc:.6f} mol/L{' ' * 10}{alcohol_wt_pct:.2f}%")
    print(f"C16-C16 Ester:  {product_ester_conc:.6f} mol/L{' ' * 10}{product_ester_wt_pct:.2f}%")
    print(f"C18 Alcohol:    {product_alcohol_conc:.6f} mol/L{' ' * 10}{product_alcohol_wt_pct:.2f}%")

    # Calculate conversion percentage if initial concentration is provided
    if initial_ester_conc is not None:
        conversion = (1 - ester_conc / initial_ester_conc) * 100
        print(f"\nConversion: {conversion:.2f}%")

    print("=" * 60)


# Example usage:
if __name__ == "__main__":
    # Example 1: Using direct rate constants with molar concentrations
    print("\n1. USING DIRECT RATE CONSTANTS WITH MOLAR CONCENTRATIONS")
    print("------------------------------------------------------")

    # Initial conditions with direct rate constants
    initial_conditions_direct = {
        'initial_ester_conc': 2.0,  # mol/L
        'initial_alcohol_conc': 5,  # mol/L
        'initial_product_ester_conc': 0.0,  # mol/L
        'initial_product_alcohol_conc': 0.0,  # mol/L
        'catalyst_conc': 0.05,  # mol/L
        'reaction_time_minutes': 400,  # minutes
        'k_forward': 0.001,  # Directly specified rate constant
        'k_reverse': 0.0005  # Directly specified rate constant
    }

    # Run simulation with equilibrium detection
    equilibrium_results = simulate_transesterification(**initial_conditions_direct, check_equilibrium=True)

    # Display equilibrium results
    ester_final, alcohol_final, product_ester_final, product_alcohol_final, equil_time, equil_reached = equilibrium_results
    initial_ester = initial_conditions_direct['initial_ester_conc']
    display_equilibrium_results(ester_final, alcohol_final, product_ester_final, product_alcohol_final,
                               equil_time, equil_reached, initial_ester)


    # Example 2: Using direct grams input (no solvent)
    print("\n2. USING DIRECT GRAMS INPUT (NO SOLVENT)")
    print("--------------------------------------")

    # Initial conditions with grams
    initial_conditions_grams = {
        'initial_ester_grams': 70.0,  # grams of C16-C18 ester
        'initial_alcohol_grams': 30.0,  # grams of C16 alcohol
        'initial_product_ester_grams': 0.0,  # grams of C16-C16 ester
        'initial_product_alcohol_grams': 0.0,  # grams of C18 alcohol
        'catalyst_grams': 0.2,  # grams of catalyst
        'reaction_time_minutes': 400,  # minutes
        'k_forward': 0.01,  # Directly specified rate constant
        'k_reverse': 0.005,  # Directly specified rate constant
        'use_grams': True  # Flag to use grams input
        # Note: You can also specify 'catalyst_mw' to provide the actual molecular weight of your catalyst
        # For example: 'catalyst_mw': 180.0  # g/mol for a specific catalyst
    }

    # Run simulation with equilibrium detection
    equilibrium_results = simulate_transesterification(**initial_conditions_grams, check_equilibrium=True)

    # Display equilibrium results
    ester_final, alcohol_final, product_ester_final, product_alcohol_final, equil_time, equil_reached = equilibrium_results

    # Convert initial grams to molar concentrations to get initial ester concentration
    # Use the same catalyst_mw as in the simulation (default is 1.0)
    catalyst_mw = initial_conditions_grams.get('catalyst_mw', 1.0)
    initial_ester, _, _, _, _ = convert_grams_to_molar(
        initial_conditions_grams['initial_ester_grams'],
        initial_conditions_grams['initial_alcohol_grams'],
        initial_conditions_grams['initial_product_ester_grams'],
        initial_conditions_grams['initial_product_alcohol_grams'],
        initial_conditions_grams['catalyst_grams'],
        catalyst_mw
    )

    display_equilibrium_results(ester_final, alcohol_final, product_ester_final, product_alcohol_final,
                               equil_time, equil_reached, initial_ester)

    # Example 3: Using Arrhenius equation (backward compatibility)
    print("\n3. USING ARRHENIUS EQUATION (BACKWARD COMPATIBILITY)")
    print("---------------------------------------------------")

    # Initial conditions with temperature for Arrhenius calculation
    initial_conditions_arrhenius = {
        'initial_ester_conc': 2.0,  # mol/L
        'initial_alcohol_conc': 5,  # mol/L
        'initial_product_ester_conc': 0.0,  # mol/L
        'initial_product_alcohol_conc': 0.0,  # mol/L
        'temperature': 363.15,  # K
        'catalyst_conc': 0.05,  # mol/L
        'reaction_time_minutes': 400  # minutes
    }

    # Run simulation with equilibrium detection
    equilibrium_results = simulate_transesterification(**initial_conditions_arrhenius, check_equilibrium=True)

    # Display equilibrium results
    ester_final, alcohol_final, product_ester_final, product_alcohol_final, equil_time, equil_reached = equilibrium_results
    initial_ester = initial_conditions_arrhenius['initial_ester_conc']
    display_equilibrium_results(ester_final, alcohol_final, product_ester_final, product_alcohol_final,
                               equil_time, equil_reached, initial_ester)

    # Example 4: Running full time-dependent simulation with weight percentages
    print("\n4. RUNNING FULL TIME-DEPENDENT SIMULATION WITH WEIGHT PERCENTAGES")
    print("--------------------------------------------------------------")
    # Run simulation without equilibrium detection to get full concentration profiles
    time_results = simulate_transesterification(**initial_conditions_grams, check_equilibrium=False)

    # Plot full concentration profiles
    plot_transesterification_results(*time_results)
