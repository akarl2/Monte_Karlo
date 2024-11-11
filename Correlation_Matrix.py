import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandasgui import show
from ydata_profiling import ProfileReport
from matplotlib.widgets import Button
import webbrowser
import os


def calc_corr_matrix(df):
    # Clean up the dataframe by removing non-breaking spaces and converting to numeric
    df = df.applymap(lambda x: x.replace('\xa0', ' ') if isinstance(x, str) else x)
    df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values
    df_numeric = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coercing errors to NaN
    df_numeric = df_numeric.dropna(axis=1, how='all')  # Drop columns that are still non-numeric

    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr()

    # Create a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create the heatmap plot
    plt.figure(figsize=(14, 8), dpi=100)
    sns.heatmap(corr_matrix, mask=mask, annot=True, vmin=-1, vmax=1, cmap='vlag', linewidths=.5)

    # Get the current figure and set the window title
    fig = plt.gcf()
    fig.canvas.manager.set_window_title("Correlation Matrix")

    # Add a button to the plot
    ax_button = plt.axes([0.81, 0.02, 0.1, 0.075])  # Position for the button
    button = Button(ax_button, 'Run Full Report')

    # Define a callback function to generate the profiling report
    def run_report(event):
        print("Generating profiling report...")
        profile = ProfileReport(df_numeric, title="Pandas Profiling Report")
        output_file = "output.html"
        profile.to_file(output_file)  # Save the report as an HTML file

        # Get the absolute path and add 'file://' prefix
        abs_path = os.path.abspath(output_file)
        file_url = f"file://{abs_path}"

        # Open the HTML report in the default web browser
        webbrowser.open(file_url)

    # Connect the button click event to the function
    button.on_clicked(run_report)

    # Show the plot
    plt.show()