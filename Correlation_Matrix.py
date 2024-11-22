import tkinter as tk
from tkinter import Toplevel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from pandas_profiling import ProfileReport
import webbrowser
import os


def calc_corr_matrix(df, parent_window):
    # Clean up the dataframe by removing non-breaking spaces and converting to numeric
    df = df.map(lambda x: x.replace('\xa0', ' ') if isinstance(x, str) else x)
    df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values
    df_numeric = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coercing errors to NaN
    df_numeric = df_numeric.dropna(axis=1, how='all')  # Drop columns that are still non-numeric

    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr()

    # Create a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a new Tkinter Toplevel window for the heatmap
    corr_window = Toplevel(parent_window)
    corr_window.title("Correlation Matrix")

    # Create a Matplotlib figure for the heatmap
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    sns.heatmap(corr_matrix, mask=mask, annot=True, vmin=-1, vmax=1, cmap='vlag', linewidths=.5, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=16)

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=corr_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Add a button to the Tkinter window for generating the report
    def run_report():
        profile = ProfileReport(df_numeric, title="Pandas Profiling Report")
        output_file = "output.html"
        profile.to_file(output_file)  # Save the report as an HTML file

        # Get the absolute path and add 'file://' prefix
        abs_path = os.path.abspath(output_file)
        file_url = f"file://{abs_path}"

        # Open the HTML report in the default web browser
        webbrowser.open(file_url)

    report_button = tk.Button(corr_window, text="Run Full Report", command=run_report)
    report_button.pack(pady=10)