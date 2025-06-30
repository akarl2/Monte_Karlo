import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Define rate constants
k_forward = 0.001  # L/mol/s
k_reverse = 0.0005# L/mol/s

keq = k_forward / k_reverse  # Equilibrium constant
print(f"Equilibrium constant (keq): {keq:.4f}")


# Initial concentrations (mol/L)
# [Ester1, Alcohol2, Ester2, Alcohol1]
initial_conc = [1, 2, 0.0,0.0 ]  # Example: 1M ester and 2M alcohol to start

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

# Plotting results with interactive cursor
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(num='Transesterification Reaction Progress')
lines = []
lines.append(ax.plot(time_points, ester1_conc, label='Ester1')[0])
lines.append(ax.plot(time_points, alcohol2_conc, label='Alcohol2')[0])
lines.append(ax.plot(time_points, ester2_conc, label='Ester2')[0])
lines.append(ax.plot(time_points, alcohol1_conc, label='Alcohol1')[0])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (mol/L)')
ax.set_title('Transesterification Reaction Progress (2:1:1:2)')
ax.grid(True)
ax.legend()

# Add cursor tracking
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(x, y):
    annot.xy = (x, y)
    text = f'Time: {x:.1f}s\nEster1: {np.interp(x, time_points, ester1_conc):.4f}\n' \
           f'Alcohol2: {np.interp(x, time_points, alcohol2_conc):.4f}\n' \
           f'Ester2: {np.interp(x, time_points, ester2_conc):.4f}\n' \
           f'Alcohol1: {np.interp(x, time_points, alcohol1_conc):.4f}'
    annot.set_text(text)


def hover(event):
    if event.inaxes == ax:
        annot.set_visible(True)
        update_annot(event.xdata, event.ydata)
        fig.canvas.draw_idle()
    else:
        annot.set_visible(False)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

# Calculate K_eq at the end of the simulation
ester1_final = ester1_conc[-1]
alcohol2_final = alcohol2_conc[-1]
ester2_final = ester2_conc[-1]
alcohol1_final = alcohol1_conc[-1]

K_eq = (ester2_final * alcohol1_final) / (ester1_final * alcohol2_final)
print(f"Equilibrium constant (K_eq) at t={time_points[-1]} s: {K_eq:.4f}")

# Create a table of final concentrations
import pandas as pd

# Create data for the table
data = {
    'Component': ['Ester1', 'Alcohol2', 'Ester2', 'Alcohol1'],
    'Final Concentration (mol/L)': [ester1_final, alcohol2_final, ester2_final, alcohol1_final]
}

# Create and display the DataFrame
df = pd.DataFrame(data)
df.set_index('Component', inplace=True)
print("\nFinal Concentrations:")
print(df)

keq_upper = (ester2_final * alcohol2_final)
keq_lower = (alcohol1_final * ester1_final)

print(keq_upper, keq_lower)

# Display the plot in a window and keep it open
plt.ioff()  # Turn off interactive mode
plt.show(block=True)  # Keep the window open until manually closed
