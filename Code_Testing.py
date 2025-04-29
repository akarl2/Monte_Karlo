import numpy as np
import matplotlib.pyplot as plt

# Define rate constants
k_forward = 0.1  # L/mol/s
k_reverse = 0.1  # L/mol/s

keq = k_forward / k_reverse  # Equilibrium constant
print(f"Equilibrium constant (keq): {keq:.4f}")


# Initial concentrations (mol/L)
# [Ester1, Alcohol2, Ester2, Alcohol1]
initial_conc = [1, 1, 0.01,0.02 ]  # Example: 1M ester and 2M alcohol to start

# Time parameters
t_start = 0
t_end = 5000  # 5000 seconds (~1.4 hours)
dt = 1  # Time step (s)
time_points = np.arange(t_start, t_end, dt)  # Array of time points

# Arrays to store concentrations over time
ester1_conc = np.zeros(len(time_points))
alcohol2_conc = np.zeros(len(time_points))
ester2_conc = np.zeros(len(time_points))
alcohol1_conc = np.zeros(len(time_points))

# Initial concentrations
ester1_conc[0], alcohol2_conc[0], ester2_conc[0], alcohol1_conc[0] = initial_conc

# Euler method for solving ODEs
for i in range(1, len(time_points)):
    ester1 = ester1_conc[i-1]
    alcohol2 = alcohol2_conc[i-1]
    ester2 = ester2_conc[i-1]
    alcohol1 = alcohol1_conc[i-1]

    # Compute rates for forward and reverse reactions
    rate_forward = k_forward * ester1 * alcohol2
    rate_reverse = k_reverse * ester2 * alcohol1

    # Update concentrations using Euler's method
    d_ester1 = -rate_forward + rate_reverse
    d_alcohol2 = -rate_forward + rate_reverse  # 2 moles of Alcohol2 involved
    d_ester2 = rate_forward - rate_reverse
    d_alcohol1 = rate_forward - rate_reverse  # 2 moles of Alcohol1 involved

    # Update concentrations
    ester1_conc[i] = ester1_conc[i-1] + d_ester1 * dt
    alcohol2_conc[i] = alcohol2_conc[i-1] + d_alcohol2 * dt
    ester2_conc[i] = ester2_conc[i-1] + d_ester2 * dt
    alcohol1_conc[i] = alcohol1_conc[i-1] + d_alcohol1 * dt

# Plotting results
plt.plot(time_points, ester1_conc, label='Ester1')
plt.plot(time_points, alcohol2_conc, label='Alcohol2')
plt.plot(time_points, ester2_conc, label='Ester2')
plt.plot(time_points, alcohol1_conc, label='Alcohol1')

plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.title('Transesterification Reaction Progress (2:1:1:2)')
plt.grid(True)
plt.show()

# Calculate K_eq at the end of the simulation
ester1_final = ester1_conc[-1]
alcohol2_final = alcohol2_conc[-1]
ester2_final = ester2_conc[-1]
alcohol1_final = alcohol1_conc[-1]

K_eq = (ester2_final * alcohol1_final) / (ester1_final * alcohol2_final)
print(f"Equilibrium constant (K_eq) at t={time_points[-1]} s: {K_eq:.4f}")