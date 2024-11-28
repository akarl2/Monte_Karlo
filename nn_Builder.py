import tkinter as tk
from contextlib import redirect_stdout
from tkinter import ttk, Toplevel
from tkinter.scrolledtext import ScrolledText
import threading
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import NN_Optimal_Settings
import matplotlib.pyplot as plt
from keras.src.utils.module_utils import tensorflow
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
import sys
import io
import re
from sys import platform
import random
from functools import partial
import subprocess
import webbrowser
import datetime
from tkinter import Toplevel, Text, Scrollbar, Button, Label, Canvas, Frame, Entry, messagebox, Checkbutton, BooleanVar
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, InputLayer, Conv3D, MaxPooling3D
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from io import BytesIO
from PIL import Image, ImageTk
from pandastable import Table

from NN_Optimal_Settings import LOSS_METRICS_DICT


class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass  # Compatibility with sys.stdout

class TrainingStatusCallback(Callback):
    def __init__(self, text_redirector):
        super().__init__()
        self.text.redirector = text_redirector

    def on_epoch_end(self, epoch, logs=None):
        message = f"Epoch {epoch + 1} - " + " - ".join([f"{key}: {value:.4f}" for key, value in logs.items()]) + "\n"
        self.text_redirector.write(message)

class NeuralNetworkArchitectureBuilder:
    def __init__(self, master, X_data=None, y_data=None, NN_PD_DATA_X=None, NN_PD_DATA_Y=None, train_test_split_var=None):
        self.master = master
        self.X_data = X_data
        self.y_data = y_data
        self.NN_PD_DATA_X = NN_PD_DATA_X
        self.NN_PD_DATA_Y = NN_PD_DATA_Y
        self.train_test_split_var = train_test_split_var

        # Initialize lists for layers and components
        self.layer_fields = []
        self.layer_types = []
        self.layer_nodes_vars = []
        self.layer_activations = []
        self.layer_kernel_sizes = []
        self.layer_type_widgets = []
        self.layer_node_widgets = []
        self.layer_activation_widgets = []
        self.layer_kernel_widgets = []
        self.layer_kernel_labels = []
        self.layer_remove_buttons = []
        self.layer_info_labels = []
        self.layer_nodes_labels = []
        self.layer_activations_labels = []
        self.layer_regularizer_labels = []
        self.layer_regularizer_widgets = []
        self.layer_regularizer_type = []
        self.layer_regularizer_vars = []

        #Results
        self.results_tab_count = 0

    def configure_nn_popup(self):
        self.master.title("Neural Network Architecture Builder")

        # Configure the parent window's grid
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Create notebook with modern styling
        self.notebook = ttk.Notebook(self.master)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Data Preview")

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Neural Network Builder")

        # Configure resizing behavior for tabs
        self.tab1.rowconfigure(0, weight=1)
        self.tab1.columnconfigure(0, weight=1)
        self.tab2.rowconfigure(0, weight=1)
        self.tab2.columnconfigure(0, weight=1)

        # Delay creation of results tabs
        self.results_tabs = {}

        # Create custom style for modern design
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="white")
        style.configure("TLabel", font=("Arial", 10), background="#f4f4f4")
        style.configure("TCombobox", fieldbackground="white", background="#f4f4f4", relief="flat")

        # Frame for training parameters
        self.params_frame = ttk.LabelFrame(self.tab2, text="Training Parameters")
        self.params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Store variables for retrieving input later
        self.epochs_var = tk.StringVar(value="50")
        self.batch_var = tk.StringVar(value="32")
        self.loss_var = tk.StringVar(value="Binary Cross-Entropy")
        self.optimizer_var = tk.StringVar(value="Adam")
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.validation_split_var = tk.StringVar(value="0.2")

        # Add parameters to the frame
        # Row 1: Epochs and Batch Size
        self.epochs_label = ttk.Label(self.params_frame, text="Number of Epochs:")
        self.epochs_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.epochs_entry = ttk.Entry(self.params_frame, textvariable=self.epochs_var)
        self.epochs_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        self.batch_label = ttk.Label(self.params_frame, text="Batch Size:")
        self.batch_label.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.batch_entry = ttk.Entry(self.params_frame, textvariable=self.batch_var)
        self.batch_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        # Row 2: Loss Function and Optimizer
        self.loss_label = ttk.Label(self.params_frame, text="Loss Function:")
        self.loss_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.loss_dropdown = ttk.Combobox(self.params_frame, textvariable=self.loss_var,
                                          values=["Binary Cross-Entropy", "Categorical Cross-Entropy",
                                                  "Sparse Categorical Cross-Entropy", "Mean Squared Error"])
        self.loss_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.optimizer_label = ttk.Label(self.params_frame, text="Optimizer:")
        self.optimizer_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.optimizer_entry = ttk.Entry(self.params_frame, textvariable=self.optimizer_var)
        self.optimizer_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # Row 3: Learning Rate and Validation Split
        self.learning_rate_label = ttk.Label(self.params_frame, text="Learning Rate:")
        self.learning_rate_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.learning_rate_entry = ttk.Entry(self.params_frame, textvariable=self.learning_rate_var)
        self.learning_rate_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.validation_split_label = ttk.Label(self.params_frame, text="Validation Split:")
        self.validation_split_label.grid(row=2, column=2, sticky="w", padx=5, pady=5)
        self.validation_split_entry = ttk.Entry(self.params_frame, textvariable=self.validation_split_var)
        self.validation_split_entry.grid(row=2, column=3, sticky="w", padx=5, pady=5)

        # Dropdown for selecting Random Starts (1 to 100)
        self.random_starts_label = ttk.Label(self.params_frame, text="Random Starts:")
        self.random_starts_label.grid(row=0, column=4, sticky="w", padx=5, pady=5)
        self.random_starts_options = [str(i) for i in range(1, 101)]  # List of values from 1 to 100 as strings
        self.random_starts_combobox = ttk.Combobox(self.params_frame, values=self.random_starts_options, state="readonly")
        self.random_starts_combobox.set("1")  # Set the default value to "1"
        self.random_starts_combobox.grid(row=0, column=5, sticky="w", padx=5, pady=5)

        #toggle for displaying training status
        self.training_status_var = BooleanVar()
        self.training_status_var.set(False)
        self.training_status_check = ttk.Checkbutton(self.params_frame, text="Display Training Status", variable=self.training_status_var)
        self.training_status_check.grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        self.early_stop_var = BooleanVar()
        self.early_stop_var.set(True)
        self.early_stop_check = ttk.Checkbutton(self.params_frame, text="Early Stopping", variable=self.early_stop_var)
        self.early_stop_check.grid(row=2, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        # Frame for layer configuration (move below the params frame)
        self.layers_frame = ttk.Frame(self.tab2)
        self.layers_frame.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")

        # Add Layer button
        add_layer_button = ttk.Button(self.layers_frame, text="Add Layer", command=self.add_layer_fields)
        add_layer_button.grid(row=1000, column=0, pady=5, padx=10)

        # Frame for visualization
        self.visualization_frame = ttk.Frame(self.tab2)
        self.visualization_frame.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")

        # Start training button
        start_training_button = ttk.Button(self.tab2, text="Start Training", command=lambda: self.run_training())
        start_training_button.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

        if self.train_test_split_var is not None:
            self.NN_PD_DATA_X_train, self.NN_PD_DATA_X_test, self.NN_PD_DATA_Y_train, self.NN_PD_DATA_Y_test = train_test_split(self.NN_PD_DATA_X, self.NN_PD_DATA_Y, test_size=self.train_test_split_var)
            self.X_train = self.NN_PD_DATA_X_train.to_numpy()
            self.y_train = self.NN_PD_DATA_Y_train.to_numpy()
            self.X_test = self.NN_PD_DATA_X_test.to_numpy()
            self.y_test = self.NN_PD_DATA_Y_test.to_numpy()
        else:
            self.NN_PD_DATA_X_train = self.NN_PD_DATA_X
            self.NN_PD_DATA_Y_train = self.NN_PD_DATA_Y
            self.X_train = self.NN_PD_DATA_X.to_numpy()
            self.y_train = self.NN_PD_DATA_Y.to_numpy()

        #apply standardscaler to the data
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)

        if self.train_test_split_var is not None:
            self.X_test = self.sc.transform(self.X_test)

        # Display the data preview in the first tab
        self.display_data_preview()
        self.notebook.select(self.tab2)

        # Initialize with a single layer
        self.add_layer_fields()

    def display_data_preview(self):
        """Display the NN_PD_DATA_X and NN_PD_DATA_Y as separate tables using pandastable."""
        if self.NN_PD_DATA_X_train is not None and self.NN_PD_DATA_Y_train is not None:
            # Create a frame for the X data table
            x_table_frame = tk.Frame(self.tab1)
            x_table_frame.place(relx=0.25, rely=0.5, anchor='center', relwidth=0.45, relheight=0.8)

            x_label = ttk.Label(self.tab1, text="X Data (Features):")
            x_label.place(relx=0.25, rely=0.1, anchor='center')

            # Initialize and display the pandastable for X data
            x_table = Table(x_table_frame, dataframe=self.NN_PD_DATA_X_train, showtoolbar=False, showstatusbar=False)
            x_table.show()

            # Create a frame for the Y data table
            y_table_frame = tk.Frame(self.tab1)
            y_table_frame.place(relx=0.75, rely=0.5, anchor='center', relwidth=0.45, relheight=0.8)

            y_label = ttk.Label(self.tab1, text="Y Data (Target):")
            y_label.place(relx=0.75, rely=0.1, anchor='center')

            # Initialize and display the pandastable for Y data
            y_table = Table(y_table_frame, dataframe=self.NN_PD_DATA_Y_train, showtoolbar=False, showstatusbar=False)
            y_table.show()
        else:
            # If no data is available, display a message
            no_data_label = ttk.Label(self.tab1, text="No data available to preview.")
            no_data_label.place(relx=0.5, rely=0.5, anchor='center')

    def on_layer_type_change(self, index):
        """ Handle the layer type change event. """
        layer_type = self.layer_types[index].get()

        # Disable activation dropdown if the layer type is Dropout, Flatten, 2D Pooling, or 3D Pooling
        if layer_type == "Dropout" or layer_type == "Flatten" or layer_type == "2D Pooling" or layer_type == "3D Pooling":
            self.layer_activation_widgets[index].config(state="disabled")
            self.layer_activations[index].set("None")  # Set activation to None for these layers
            self.layer_regularizer_widgets[index].config(state="disabled")  # Disable regularizer dropdown
            self.layer_regularizer_type[index].set("None")  # Set regularizer to None for these layers
        elif layer_type == "Dense":
            # Show regularizer for Dense layer
            self.layer_regularizer_widgets[index].grid(row=index, column=9, padx=10, pady=5)
            self.layer_activations[index].set("relu")  # Default to ReLU for Dense layers
        elif layer_type in ["2D Convolutional", "3D Convolutional"]:
            # Show regularizer for 2D and 3D convolutional layers
            self.layer_regularizer_widgets[index].grid(row=index, column=9, padx=10, pady=5)
            self.layer_activation_widgets[index].config(state="normal")  # Enable activation for these layers
            self.layer_activations[index].set("relu")  # Default to ReLU for Conv layers
        else:
            # Hide the regularizer for other layer types
            self.layer_regularizer_widgets[index].grid_forget()

            # Enable activation for non-Dropout and non-Pooling layers
            self.layer_activation_widgets[index].config(state="normal")

        # Update kernel size field
        self.update_kernel_size_field(index)
        self.show_visual_key()  # This runs after updating the kernel size

    def add_layer_fields(self):
        """ Adds fields for configuring a new layer with the specified column layout, ensuring consistent alignment. """
        layer_index = len(self.layer_fields)

        style = ttk.Style()
        style.configure("Red.TButton", foreground="red")

        # Create the remove button using the custom style
        remove_button = ttk.Button(self.layers_frame, text="X", style="Red.TButton",
                                   command=lambda b=layer_index: self.remove_layer_fields(b))  # Use dynamic index

        remove_button.grid(row=layer_index, column=0, padx=10, pady=5)
        self.layer_remove_buttons.append(remove_button)

        # Column 1: Layer Information Label
        layer_info_label = ttk.Label(self.layers_frame, text=f"Layer {layer_index + 1}:")
        layer_info_label.grid(row=layer_index, column=1, padx=10, pady=5, sticky="w")
        self.layer_info_labels.append(layer_info_label)

        # Column 2: Layer Type Dropdown
        layer_type_var = tk.StringVar(value="Dense")
        layer_type_dropdown = ttk.Combobox(self.layers_frame, textvariable=layer_type_var,
                                           values=["Dense", "2D Convolutional", "3D Convolutional", "2D Pooling", "3D Pooling", "Flatten", "Dropout"], width=20)
        layer_type_dropdown.grid(row=layer_index, column=2, padx=10, pady=5, sticky="w")
        layer_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.on_layer_type_change(layer_index))

        self.layer_types.append(layer_type_var)
        self.layer_type_widgets.append(layer_type_dropdown)

        # Column 3: Nodes, Kernels, or Dropout Rate Entry
        nodes_label = ttk.Label(self.layers_frame, text="Nodes/Filters/Rate:")
        nodes_label.grid(row=layer_index, column=3, sticky="e", padx=(5, 2), pady=5)
        self.layer_nodes_labels.append(nodes_label)

        nodes_var = tk.DoubleVar(value=10)
        nodes_entry = ttk.Entry(self.layers_frame, textvariable=nodes_var, width=5)
        nodes_entry.grid(row=layer_index, column=4, padx=10, pady=5, sticky="w")
        nodes_entry.bind("<FocusOut>", lambda e: self.show_visual_key())  # Trigger visualization update
        self.layer_nodes_vars.append(nodes_var)
        self.layer_node_widgets.append(nodes_entry)

        # Column 4: Activation Function Dropdown
        activation_label = ttk.Label(self.layers_frame, text="Activation:")
        activation_label.grid(row=layer_index, column=5, sticky="e", padx=(5, 2), pady=5)
        self.layer_activations_labels.append(activation_label)

        activation_var = tk.StringVar(value="relu" if layer_index < 1 else "relu")  # Default to ReLU for hidden layers
        activation_dropdown = ttk.Combobox(self.layers_frame, textvariable=activation_var,
                                           values=["relu", "sigmoid", "tanh", "linear", "softmax", "None"], width=10)  # Add "None"
        activation_dropdown.grid(row=layer_index, column=6, padx=10, pady=5, sticky="w")
        activation_dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_visual_key())  # Trigger visualization update

        # Disable activation dropdown for Dropout layers
        if layer_type_var.get() == "Dropout":
            activation_dropdown.config(state="disabled")  # Grey out the dropdown

        self.layer_activations.append(activation_var)
        self.layer_activation_widgets.append(activation_dropdown)

        # Columns 5 and 6: Kernel Size label and entry (conditionally displayed)
        kernel_size_x_var = tk.IntVar(value=3)
        kernel_size_y_var = tk.IntVar(value=3)
        kernel_size_label = ttk.Label(self.layers_frame, text="Kernel Size:")
        kernel_size_entry_x = ttk.Entry(self.layers_frame, textvariable=kernel_size_x_var, width=5)
        kernel_size_entry_y = ttk.Entry(self.layers_frame, textvariable=kernel_size_y_var, width=5)
        kernel_size_entry_x.bind("<FocusOut>", lambda e: self.show_visual_key())  # Trigger visualization update
        kernel_size_entry_y.bind("<FocusOut>", lambda e: self.show_visual_key())  # Trigger visualization update

        # Store kernel size widgets and variables for independent control
        self.layer_kernel_labels.append(kernel_size_label)
        self.layer_kernel_widgets.append([kernel_size_entry_x, kernel_size_entry_y])

        # Regularizer Dropdown (initially hidden unless Dense is selected)
        regularizer_var = tk.StringVar(value="None")
        regularizer_label = ttk.Label(self.layers_frame, text="Regularizer:")
        regularizer_label.grid(row=layer_index, column=8, sticky="e", padx=(5, 2), pady=5)
        regularizer_dropdown = ttk.Combobox(self.layers_frame, textvariable=regularizer_var, values=["None", "l1", "l2"], width=10)
        regularizer_dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_visual_key())  # Trigger visualization update
        regularizer_dropdown.grid(row=layer_index, column=9, padx=10, pady=5)
        self.layer_regularizer_widgets.append(regularizer_dropdown)
        self.layer_regularizer_labels.append(regularizer_label)
        self.layer_regularizer_type.append(regularizer_var)

        #regulizer entry
        regularizer_entry = ttk.Entry(self.layers_frame, width=10)
        regularizer_entry.grid(row=layer_index, column=10, padx=10, pady=5)
        regularizer_entry.insert(0, "0.001")  # Default value
        regularizer_entry.bind("<FocusOut>", lambda e: self.show_visual_key())  # Trigger visualization update
        self.layer_regularizer_vars.append(regularizer_entry)

        self.layer_fields.append((layer_type_var, nodes_var, activation_var, kernel_size_x_var, kernel_size_y_var, regularizer_var, regularizer_entry))

        # Show/hide kernel size fields based on the selected layer type
        self.update_kernel_size_field(layer_index)

        self.show_visual_key()

    def update_kernel_size_field(self, index):
        """ Adds the kernel size fields at the end (columns 7 and 8) when needed, without rearranging other columns. """
        layer_type = self.layer_types[index].get()

        if layer_type in ["2D Convolutional", "3D Convolutional", "2D Pooling", "3D Pooling"]:
            # Show kernel size label and entry in columns 7 and 8
            self.layer_kernel_labels[index].grid(row=index, column=11, padx=10, pady=5, sticky="e")
            self.layer_kernel_widgets[index][0].grid(row=index, column=12, padx=10, pady=5, sticky="w")
            self.layer_kernel_widgets[index][1].grid(row=index, column=13, padx=10, pady=5, sticky="w")

            # Adjust second kernel entry behavior based on layer type
            if layer_type in ["2D Convolutional", "2D Pooling"]:
                # Set second entry to 1 and disable it
                self.layer_kernel_widgets[index][1].config(state="normal")  # Temporarily enable to set value
                self.layer_kernel_widgets[index][1].delete(0, tk.END)  # Clear current value
                self.layer_kernel_widgets[index][1].insert(0, "1")  # Set to 1
                self.layer_kernel_widgets[index][1].config(state="disabled")  # Disable again
            else:
                # Enable second entry for 3D types
                self.layer_kernel_widgets[index][1].config(state="normal")

        else:
            # Hide kernel size label and both entries if not needed
            self.layer_kernel_labels[index].grid_forget()
            self.layer_kernel_widgets[index][0].grid_forget()
            self.layer_kernel_widgets[index][1].grid_forget()

    def remove_layer_fields(self, index):
        # Remove the widgets from the grid
        for widget in self.layers_frame.grid_slaves(row=index):
            widget.grid_forget()

        # Remove the configuration from lists
        del self.layer_fields[index]
        del self.layer_types[index]
        del self.layer_nodes_vars[index]
        del self.layer_activations[index]
        del self.layer_regularizer_type[index]
        del self.layer_regularizer_vars[index]

        # Remove associated widgets and labels from lists
        del self.layer_type_widgets[index]
        del self.layer_node_widgets[index]
        del self.layer_kernel_widgets[index]
        del self.layer_kernel_labels[index]
        del self.layer_remove_buttons[index]
        del self.layer_info_labels[index]  # Update Layer info label list
        del self.layer_nodes_labels[index]  # Remove Nodes/Rate label
        del self.layer_activations_labels[index]  # Remove Activation label
        del self.layer_activation_widgets[index]
        del self.layer_regularizer_labels[index]  # Remove Regularizer label
        del self.layer_regularizer_widgets[index]  # Remove Regularizer widget

        # Reorder rows and reassign labels
        for i in range(len(self.layer_fields)):
            # Explicitly set the label text for each layer
            self.layer_info_labels[i].config(text=f"Layer {i + 1}:")  # This renumbers each Layer label correctly
            # Update positions for all widgets in the row
            self.layer_remove_buttons[i].grid(row=i, column=0, padx=10, pady=5)
            self.layer_info_labels[i].grid(row=i, column=1, padx=10, pady=5, sticky="w")
            self.layer_type_widgets[i].grid(row=i, column=2, padx=10, pady=5)
            self.layer_nodes_labels[i].grid(row=i, column=3, sticky="e", padx=(5, 2), pady=5)  # Nodes/Rate label
            self.layer_node_widgets[i].grid(row=i, column=4, padx=10, pady=5)
            self.layer_activations_labels[i].grid(row=i, column=5, sticky="e", padx=(5, 2), pady=5)  # Activation label
            self.layer_activation_widgets[i].grid(row=i, column=6, padx=10, pady=5, sticky="w")
            self.layer_regularizer_labels[i].grid(row=i, column=8, sticky="e", padx=(5, 2), pady=5)  # Regularizer label
            self.layer_regularizer_widgets[i].grid(row=i, column=9, padx=10, pady=5)
            self.layer_kernel_widgets[i][0].grid(row=i, column=11, padx=10, pady=5, sticky="w")  # Kernel size entry X
            self.layer_kernel_widgets[i][1].grid(row=i, column=12, padx=10, pady=5, sticky="w")  # Kernel size entry Y
            self.layer_kernel_labels[i].grid(row=i, column=13, padx=10, pady=5, sticky="e")  # Kernel size label
            self.layer_regularizer_vars[i].grid(row=i, column=10, padx=10, pady=5)

            # Rebind combobox to capture updated index `i`
            self.layer_type_widgets[i].unbind("<<ComboboxSelected>>")
            self.layer_type_widgets[i].bind("<<ComboboxSelected>>", lambda event, idx=i: self.on_layer_type_change(idx))
            self.layer_activation_widgets[i].unbind("<<ComboboxSelected>>")
            self.layer_activation_widgets[i].bind("<<ComboboxSelected>>", lambda event, idx=i: self.on_layer_type_change(idx))
            self.layer_regularizer_widgets[i].unbind("<<ComboboxSelected>>")
            self.layer_regularizer_widgets[i].bind("<<ComboboxSelected>>", lambda event, idx=i: self.on_layer_type_change(idx))
            self.layer_remove_buttons[i].config(command=lambda b=i: self.remove_layer_fields(b))

            # Update kernel size field visibility based on new index
            self.update_kernel_size_field(i)

        # After the rows are adjusted, refresh the visual key
        self.show_visual_key()

    def show_visual_key(self):
        """ Show a visual representation of the neural network architecture, including data shapes and layer parameters. """
        # Collect current layer configurations
        layer_val_entry = [
            nodes_var.get() if layer_type.get() == "Dropout" else int(nodes_var.get())
            if layer_type.get() in ["Dense", "2D Convolutional", "3D Convolutional"] else None
            for layer_type, nodes_var in zip(self.layer_types, self.layer_nodes_vars)]
        layer_types = [layer_type.get() for layer_type in self.layer_types]
        activations = [activation.get() for activation in self.layer_activations]
        kernel_sizes = [
            (kernel_size[0].get(), kernel_size[1].get()) if layer_type in ["2D Convolutional", "3D Convolutional",
                                                                           "2D Pooling", "3D Pooling"]
            else None
            for kernel_size, layer_type in zip(self.layer_kernel_widgets, layer_types)]
        regularizer_types = [regularizer.get() for regularizer in self.layer_regularizer_type]
        regularizer_values = [regularizer.get() for regularizer in self.layer_regularizer_vars]
        num_layers = len(layer_val_entry)
        total_params = 0

        # Create a new figure for the visualization
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('off')

        # Display X_data and y_data shapes at the top of the plot
        x_shape = self.X_train.shape if self.X_train is not None else "N/A"
        y_shape = self.y_train.shape if self.y_train is not None else "N/A"
        ax.text(0.5, 0.9, f"Train Data Shape (X): {x_shape}    Train Target Shape (y): {y_shape}", ha="center",
                va="center",
                fontsize=12, weight="bold")

        # Define positions for each layer
        x_positions = np.linspace(0.1, 0.9, num_layers)
        y_offset = 0.5

        for i, (x, layer_val, layer_type, activation, kernel_size, regularizer_type, regularizer_value) in enumerate(
                zip(x_positions, layer_val_entry, layer_types, activations, kernel_sizes, regularizer_types,
                    regularizer_values)):
            # Start with basic layer text
            layer_text = f"Layer {i + 1}: {layer_type}\n"

            # Calculate the number of parameters based on layer type
            layer_params = 0

            # Find the previous valid node count for Dense layers, ignoring Dropout and Flatten layers
            if layer_type == "Dense":
                prev_nodes = None
                for j in range(i - 1, -1, -1):  # Traverse backward to find the previous valid node count
                    if isinstance(layer_val_entry[j], int):  # Check if it's a valid integer node count
                        prev_nodes = layer_val_entry[j]
                        break
                if prev_nodes is None:  # Default to input shape if no previous valid layer was found
                    prev_nodes = x_shape[1]

                layer_params = int(layer_val) * (int(prev_nodes) + 1)  # Including bias term
                layer_text += f" Nodes: {layer_val}\n"
            elif layer_type == "2D Convolutional":
                filters = int(layer_val)
                kernel_height, kernel_width = int(kernel_size[0]), int(kernel_size[1])
                layer_params = filters * (kernel_height * kernel_width + 1)  # +1 for bias
                layer_text += f" Filters: {filters}\nKernel Size: {kernel_height} x {kernel_width}\n"
            elif layer_type == "2D Pooling":
                kernel_height, kernel_width = int(kernel_size[0]), int(kernel_size[1])
                layer_text += f"Kernel Size: {kernel_height} x {kernel_width}\n"
            elif layer_type == "Dropout":
                dropout_rate = float(layer_val)
                layer_text += f"Dropout Rate: {dropout_rate}\n"
            elif layer_type == "Flatten":
                layer_text += "Flatten layer\n"

            # Add activation, excluding certain layer types
            if layer_type not in ["Dropout", "Flatten", "2D Pooling", "3D Pooling"]:
                layer_text += f"Activation: {activation}\n"

            # Add regularizer if defined
            if regularizer_type and regularizer_value:  # If both regularizer type and value are defined
                layer_text += f"Regularizer: {regularizer_type} ({regularizer_value})\n"

            # Add the number of parameters to the text
            total_params += layer_params
            layer_text += f"Parameters: {layer_params}\n"

            # Add the layer text to the plot
            ax.text(x, y_offset, layer_text, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.5", edgecolor="black", facecolor="lightblue"))

            # Add arrows between layers (except the last layer)
            if i < num_layers - 1:
                ax.annotate("", xy=(x + 0.15, y_offset), xytext=(x + 0.1, y_offset),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

        # Display total parameters at the bottom
        ax.text(0.5, 0.1, f"Total Parameters: {total_params}", ha="center", va="center",
                fontsize=12, weight="bold")

        # Clear previous visualization
        for widget in self.visualization_frame.winfo_children():
            widget.grid_forget()  # Clear the grid (instead of destroy)

        # Create a new canvas for the updated visualization
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    def run_training(self):
        # Give a warning if the last layer nodes are not equal to the number of classes
        if self.layer_fields[-1][1].get() != self.y_data.shape[1]:
            tk.messagebox.showwarning("Warning",
                                      "The number of nodes in the last layer should be equal to the number of classes in the target data.")
            return

        # Give a warning if the last layer activation is not suitable for the target data. E.g., softmax for multi-class classification, sigmoid for binary classification, etc.
        if self.layer_fields[-1][2].get() == "relu" and np.unique(self.y_data).shape[0] == 2:
            tk.messagebox.showwarning("Warning",
                                      "Binary classification should use sigmoid activation in the last layer.")
            return

        # Get the number of epochs and batch size
        epochs = int(self.epochs_entry.get())
        batch_size = int(self.batch_entry.get())
        loss_function = self.loss_dropdown.get()
        optimizer = self.optimizer_entry.get()
        learning_rate = float(self.learning_rate_entry.get())
        metrics = LOSS_METRICS_DICT[loss_function]["metrics"]

        # Define a class to store layer information
        class Layer:
            def __init__(self, layer_type, nodes, activation, kernel_size, regularizer_type, regularizer_value):
                self.layer_type = layer_type
                self.nodes = nodes
                self.activation = activation
                self.kernel_size = kernel_size
                self.regularizer_type = regularizer_type
                self.regularizer_value = regularizer_value

        # Collect current layer configurations
        layers = []
        for layer_type_var, nodes_var, activation_var, kernel_size_x_var, kernel_size_y_var, regularizer_var, regularizer_entry in self.layer_fields:
            # Get kernel size (handle cases where kernel size is None or not applicable)
            kernel_size = (
                (kernel_size_x_var.get(), kernel_size_y_var.get())
                if layer_type_var.get() in ["2D Convolutional", "3D Convolutional", "2D Pooling", "3D Pooling"]
                else None)

            # Handle regularizer type and value (check if the regularizer is "None")
            regularizer_type = None if regularizer_var.get() == "None" else regularizer_var.get()

            # Ensure proper types for layer configurations
            nodes = int(nodes_var.get())  # Ensure nodes are integers
            if layer_type_var.get() == "Dropout":
                nodes = float(nodes_var.get())  # Dropout rate should be float

            # Create and append layer object
            layers.append(
                Layer(
                    layer_type_var.get(),
                    nodes,
                    activation_var.get(),
                    kernel_size,
                    regularizer_type,
                    regularizer_entry.get()
                )
            )

        # Define a function to build the neural network model
        def build_model(layers, input_shape):
            model = Sequential()
            # Loop through the layers
            for i, layer in enumerate(layers):
                if layer.layer_type == "Dense":
                    if i == 0:  # Add input shape only to the first layer
                        # First layer - explicit input layer
                        model.add(InputLayer(shape=input_shape))
                        model.add(Dense(units=layer.nodes, activation=layer.activation,
                                        kernel_regularizer=l1(float(layer.regularizer_value)) if layer.regularizer_type == "l1"
                                        else l2(float(layer.regularizer_value)) if layer.regularizer_type == "l2"
                                        else None))
                    else:
                        model.add(Dense(units=layer.nodes, activation=layer.activation,
                                        kernel_regularizer=l1(float(layer.regularizer_value)) if layer.regularizer_type == "l1"
                                        else l2(float(layer.regularizer_value)) if layer.regularizer_type == "l2"
                                        else None))

                elif layer.layer_type == "2D Convolutional":
                    model.add(Conv2D(filters=layer.nodes, kernel_size=layer.kernel_size, activation=layer.activation,
                                     kernel_regularizer=l1(float(layer.regularizer_value)) if layer.regularizer_type == "l1"
                                     else l2(float(layer.regularizer_value)) if layer.regularizer_type == "l2"
                                     else None))

                elif layer.layer_type == "3D Convolutional":
                    model.add(Conv3D(filters=layer.nodes, kernel_size=layer.kernel_size, activation=layer.activation,
                                     kernel_regularizer=l1(float(layer.regularizer_value)) if layer.regularizer_type == "l1"
                                     else l2(float(layer.regularizer_value)) if layer.regularizer_type == "l2"
                                     else None))

                elif layer.layer_type == "2D Pooling":
                    model.add(MaxPooling2D(pool_size=layer.kernel_size))

                elif layer.layer_type == "Dropout":
                    model.add(Dropout(rate=layer.nodes))  # Dropout rate is a float

                elif layer.layer_type == "Flatten":
                    model.add(Flatten())

            return model

        def create_master_popup():
            if not hasattr(create_master_popup, "popup"):  # Ensure only one instance exists
                popup = tk.Toplevel()  # Create an instance of Toplevel
                popup.title("Training Progress")
                popup.geometry("1200x400")
                text_widget = ScrolledText(popup, wrap="none", height=20, width=80)
                text_widget.pack(fill="both", expand=True)

                create_master_popup.popup = popup
                create_master_popup.text_widget = text_widget

            return create_master_popup.text_widget

        def build_and_train_model(seed, text_redirector=None):
            tf.random.set_seed(seed)
            input_shape = self.X_train.shape[1:]
            model = build_model(layers, input_shape)

            # Compile the model
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy() if loss_function == "Binary Cross-Entropy"
                else tf.keras.losses.CategoricalCrossentropy() if loss_function == "Categorical Cross-Entropy"
                else tf.keras.losses.SparseCategoricalCrossentropy() if loss_function == "Sparse Categorical Cross-Entropy"
                else tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer == "Adam" else None,
                metrics=metrics
            )

            # Start TensorBoard logging for the current seed
            log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{seed}"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Define callbacks
            callbacks = [tensorboard_callback]

            if self.early_stop_var.get():
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    min_delta=10 ** -4,
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)

            # Train the model
            if text_redirector:
                with redirect_stdout(text_redirector):
                    history = model.fit(
                        self.X_train, self.y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,  # Enable verbose output for progress updates
                        callbacks=callbacks,
                        validation_split=float(self.validation_split_entry.get())
                    )
            else:
                history = model.fit(
                    self.X_train, self.y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,  # Enable verbose output for progress updates
                    callbacks=callbacks,
                    validation_split=float(self.validation_split_entry.get())
                )

            # Return the model and its final validation loss
            final_val_loss = min(history.history['val_loss'])
            return model, final_val_loss, history

        def run_training_in_thread():
            def train():
                # Run training with randomly selected seeds
                best_model = None
                best_val_loss = float('inf')
                best_history = None

                # Get the random starts value
                random_starts = int(self.random_starts_combobox.get())

                # Conditionally create a text redirector
                if self.training_status_var:
                    text_redirector = TextRedirector(create_master_popup())
                else:
                    text_redirector = None

                for _ in range(random_starts):  # Run for the number of specified random starts
                    seed = random.randint(1, 1000)  # Randomly select a seed
                    model, val_loss, history = build_and_train_model(seed, text_redirector)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        best_history = history

                # Once training is done, transfer the results to the main thread using self.after
                self.master.after(0, self.display_training_results, best_history, best_model)

            # Start the training process in a new thread
            training_thread = threading.Thread(target=train)
            training_thread.start()

        run_training_in_thread()

    def display_training_results(self, history, model):
        self.results_tab_count += 1
        tab_name = f"Result {self.results_tab_count}"
        new_tab = ttk.Frame(self.notebook)
        self.notebook.add(new_tab, text=tab_name)
        self.notebook.select(new_tab)

        # Add results content to the new tab
        canvas = Canvas(new_tab)
        scrollable_frame = Frame(canvas)
        scrollbar = Scrollbar(new_tab, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        #Create a window inside the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", update_scroll_region)

        def on_mouse_wheel(event):
            system = platform
            if system.startswith("win"):
                canvas.yview_scroll(-1 * int(event.delta / 120), "units")
            else:
                canvas.yview_scroll(-1 * int(event.delta), "units")

        canvas.bind("<MouseWheel>", on_mouse_wheel)  # For macOS trackpad gesture scrolling

        def on_key_scroll(event):
            if event.keysym == "Up":
                canvas.yview_scroll(-1, "units")
            elif event.keysym == "Down":
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<KeyPress-Up>", on_key_scroll)
        canvas.bind_all("<KeyPress-Down>", on_key_scroll)

        new_tab.columnconfigure(0, weight=1)
        new_tab.rowconfigure(0, weight=1)
        canvas.columnconfigure(0, weight=1)
        canvas.rowconfigure(0, weight=1)
        scrollable_frame.columnconfigure(0, weight=1)
        scrollable_frame.rowconfigure(0, weight=1)

        # Display loss and metrics from training history
        if history.history:
            results_text = Text(scrollable_frame, wrap="word", height=15, width=80)
            results_text.pack(pady=10)
            results_text.insert("end", "Training Metrics:\n")
            for key, values in history.history.items():
                results_text.insert("end", f"{key}: {values[-1]:.4f}\n")
            results_text.insert("end", "\n")

        #add results from the test data
        if self.train_test_split_var is not None:
            test_results = model.evaluate(self.X_test, self.y_test, verbose=0)
            results_text.insert("end", "Test Metrics:\n")
            for i, metric in enumerate(model.metrics_names):
                results_text.insert("end", f"{metric}: {test_results[i]:.4f}\n")
            results_text.insert("end", "\n")
            #calcuate R2 score if regression
            if "MeanSquaredError" in model.loss.__class__.__name__:
                r2 = r2_score(self.y_test, model.predict(self.X_test))
                results_text.insert("end", f"r2_Score: {r2:.4f}\n\n")

        # Determine the type of model (classification or regression)
        loss_name = model.loss.__class__.__name__
        if "Crossentropy" in loss_name or "Hinge" in loss_name:  # Classification tasks
            # Evaluate on the test or train data
            if self.train_test_split_var is not None:
                y_pred = model.predict(self.X_test)
                y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
                cm = confusion_matrix(self.y_test, y_pred_classes)
                cr = classification_report(self.y_test, y_pred_classes)
                results_text.insert("end", "Classification Report for test set:\n" + cr + "\n\n")
            else:
                y_pred = model.predict(self.X_train)
                y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
                cm = confusion_matrix(self.y_data, y_pred_classes)
                cr = classification_report(self.y_data, y_pred_classes)
                results_text.insert("end", "Classification Report for train set:\n" + cr + "\n\n")

            # Generate and display the confusion matrix heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=np.unique(self.y_test if self.train_test_split_var else self.y_data),
                        yticklabels=np.unique(self.y_test if self.train_test_split_var else self.y_data))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix Heatmap')

            img_buf = BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img = Image.open(img_buf)
            img_tk = ImageTk.PhotoImage(img)
            plt.close()

            heatmap_label = Label(scrollable_frame, image=img_tk)
            heatmap_label.image = img_tk  # Keep a reference to avoid garbage collection
            heatmap_label.pack(pady=10)

        elif loss_name == "MeanSquaredError":  # Regression tasks
            if self.train_test_split_var is not None:
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
            else:
                y_pred = model.predict(self.X_train)
                mse = mean_squared_error(self.y_data, y_pred)
            results_text.insert("end", f"Mean Squared Error (MSE): {mse:.4f}\n")

        x_feature_var = tk.StringVar(value=self.NN_PD_DATA_X_train.columns[0])
        x_feature_label = Label(scrollable_frame, text="Select X Feature:")
        x_feature_label.pack(pady=5)
        x_feature_dropdown = ttk.Combobox(scrollable_frame, textvariable=x_feature_var, values=list(self.NN_PD_DATA_X_train.columns))
        x_feature_dropdown.pack(pady=5)

        if len(self.NN_PD_DATA_X_train.columns) > 1:
            y_feature_var = tk.StringVar(value=self.NN_PD_DATA_X_train.columns[1])
            y_feature_label = Label(scrollable_frame, text="Select Y Feature:")
            y_feature_label.pack(pady=5)
            y_feature_dropdown = ttk.Combobox(scrollable_frame, textvariable=y_feature_var,
                                              values=list(self.NN_PD_DATA_X_train.columns))
            y_feature_dropdown.pack(pady=5)
        else:
            y_feature_var = None

        # Add a new notebook inside the new tab for sub-tabs
        sub_notebook = ttk.Notebook(scrollable_frame)
        sub_notebook.pack(fill="both", expand=True, pady=10)

        def plot_surface_response():
            # Get selected features
            x_feature = x_feature_var.get()
            y_feature = y_feature_var.get() if y_feature_var is not None else None
            z_feature = self.NN_PD_DATA_Y_train.columns[0]

            # Create a new sub-tab for this plot
            plot_tab_name = f"{x_feature} vs {z_feature}" if not y_feature else f"{x_feature}, {y_feature} vs {z_feature}"
            plot_tab = ttk.Frame(sub_notebook)
            sub_notebook.add(plot_tab, text=plot_tab_name)
            sub_notebook.select(plot_tab)

            # Create left and right frames for layout
            left_frame = Frame(plot_tab)
            left_frame.pack(side="left", fill="y", padx=10, pady=10)

            right_frame = Frame(plot_tab)
            right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)


            # Add dropdowns and button to the left frame
            Label(left_frame, text="Select X Feature:").pack(pady=5)
            x_feature_dropdown = ttk.Combobox(left_frame, textvariable=x_feature_var, values=list(self.NN_PD_DATA_X_train.columns))
            x_feature_dropdown.pack(pady=5)

            if len(self.NN_PD_DATA_X_train.columns) > 1:
                Label(left_frame, text="Select Y Feature:").pack(pady=5)
                y_feature_dropdown = ttk.Combobox(left_frame, textvariable=y_feature_var,
                                                  values=list(self.NN_PD_DATA_X_train.columns))
                y_feature_dropdown.pack(pady=5)
            else:
                Label(left_frame, text="Only one feature available. Y feature not required.").pack(pady=5)

            # Button to replot
            Button(left_frame, text="Replot", command=plot_surface_response).pack(pady=10)

            # Add Checkbutton for toggling scatter points
            show_training = tk.BooleanVar(value=False)
            training_checkbox = ttk.Checkbutton(left_frame, text="Show Training Data", variable=show_training)
            training_checkbox.pack(pady=5)

            if self.train_test_split_var is not None:
                show_test = tk.BooleanVar(value=False)
                test_checkbox = ttk.Checkbutton(left_frame, text="Show Test Data", variable=show_test)
                test_checkbox.pack(pady=5)

            # Generate grid values for the selected feature(s)
            x_min, x_max = self.NN_PD_DATA_X_train[x_feature].min(), self.NN_PD_DATA_X_train[x_feature].max()
            x_vals = np.linspace(x_min, x_max, 100)

            fig, ax = plt.subplots(figsize=(6, 4))  # Make the plot smaller

            if y_feature:  # 3D Plot
                # Remove all ticks and labels
                ax.set_xticks([])  # Clear X-axis ticks
                ax.set_yticks([])  # Clear Y-axis ticks

                # remove the box around the plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                y_min, y_max = self.NN_PD_DATA_X_train[y_feature].min(), self.NN_PD_DATA_X_train[y_feature].max()
                y_vals = np.linspace(y_min, y_max, 100)
                x_grid, y_grid = np.meshgrid(x_vals, y_vals)

                grid_data = pd.DataFrame({x_feature: x_grid.ravel(), y_feature: y_grid.ravel()})

                for col in self.NN_PD_DATA_X_train.columns:
                    if col != x_feature and col != y_feature:
                        grid_data[col] = self.NN_PD_DATA_X_train[col].mean()

                grid_data_scaled = self.sc.transform(grid_data)

                # Predict probabilities or class labels
                if hasattr(model, "predict_proba"):  # Use probabilities if available
                    z_vals = model.predict_proba(grid_data_scaled)[:, 1].reshape(x_grid.shape)
                else:  # Otherwise, use class predictions
                    z_vals = model.predict(grid_data_scaled).reshape(x_grid.shape)

                ax = plt.axes(projection="3d")
                ax.plot_surface(x_grid, y_grid, z_vals, cmap="viridis", alpha=0.8)

                if "Crossentropy" in loss_name or "Hinge" in loss_name:
                    ax.set_xlabel(f"X: {x_feature}")
                    ax.set_ylabel(f"Y: {y_feature}")
                    ax.set_zlabel(f"Probability, Z: {z_feature}")
                    ax.set_title(f"Probability Surface: {z_feature}")
                else:
                    ax.set_xlabel(f"X: {x_feature}")
                    ax.set_ylabel(f"Y: {y_feature}")
                    ax.set_zlabel(f"Value, Z: {z_feature}")
                    ax.set_title(f"Value Surface: {z_feature}")

            else:  # 2D Plot
                grid_data = pd.DataFrame({x_feature: x_vals})
                for col in self.NN_PD_DATA_X_train.columns:
                    if col != x_feature:
                        grid_data[col] = self.NN_PD_DATA_X_train[col].mean()

                grid_data_scaled = self.sc.transform(grid_data)

                # Predict probabilities or class labels
                if hasattr(model, "predict_proba"):  # Use probabilities if available
                    z_vals = model.predict_proba(grid_data_scaled)[:, 1]
                else:  # Otherwise, use class predictions
                    z_vals = model.predict(grid_data_scaled)

                ax.plot(x_vals, z_vals, label=f"Probability: {z_feature}", color="blue")

                if "Crossentropy" in loss_name or "Hinge" in loss_name:
                    ax.ax.set_xlabel(f"X: {x_feature}")
                    ax.set_ylabel(f"Probability, Z: {z_feature}")
                    ax.set_title(f"Probability Curve: {z_feature}")
                    ax.legend()
                else:
                    ax.set_xlabel(f"X: {x_feature}")
                    ax.set_ylabel(f"Value, Z: {z_feature}")
                    ax.set_title(f"Value Curve: {z_feature}")
                    ax.legend()


            # Initialize scatter points
            scatter_training = None
            scatter_test = None

            def toggle_scatter():
                nonlocal scatter_training
                nonlocal scatter_test
                if scatter_training:
                    scatter_training.set_visible(show_training.get())
                if scatter_test:
                    scatter_test.set_visible(show_test.get())
                canvas.draw()


            if len(self.NN_PD_DATA_X_train) > 100:
                num_points_train = 100
            else:
                num_points_train = len(self.NN_PD_DATA_X_train)

            train_indices = np.random.choice(self.NN_PD_DATA_X_train.index, size=num_points_train, replace=False)

            if self.train_test_split_var is not None:
                if len(self.NN_PD_DATA_X_test) > 100:
                    num_points_test = 100
                else:
                    num_points_test = len(self.NN_PD_DATA_X_test)

            if self.train_test_split_var is not None:
                test_indices = np.random.choice(self.NN_PD_DATA_X_test.index, size=num_points_test, replace=False)


            if y_feature:  # Scatter for 3D plot
                scatter_training = ax.scatter(
                    self.NN_PD_DATA_X_train.loc[train_indices, x_feature],
                    self.NN_PD_DATA_X_train.loc[train_indices, y_feature],
                    self.NN_PD_DATA_Y_train.loc[train_indices].values.flatten(),  # Align with actual class (0 or 1)
                    color="blue",
                    label="Training Data",
                    visible=False
                )
                if self.train_test_split_var is not None:
                    scatter_test = ax.scatter(
                        self.NN_PD_DATA_X_test.loc[test_indices, x_feature],
                        self.NN_PD_DATA_X_test.loc[test_indices, y_feature],
                        self.NN_PD_DATA_Y_test.loc[test_indices].values.flatten(),  # Align with actual class (0 or 1)
                        color="red",
                        label="Test Data",
                        visible=False)
            else:  # Scatter for 2D plot
                scatter_training = ax.scatter(
                    self.NN_PD_DATA_X_train.loc[train_indices, x_feature],
                    self.NN_PD_DATA_Y_train.loc[train_indices].values.flatten(),  # Align with actual class (0 or 1)
                    color="blue",
                    label="Training Data",
                    visible=False
                )
                if self.train_test_split_var is not None:
                    scatter_test = ax.scatter(
                        self.NN_PD_DATA_X_test.loc[test_indices, x_feature],
                        self.NN_PD_DATA_Y_test.loc[test_indices].values.flatten(),  # Align with actual class (0 or 1)
                        color="red",
                        label="Test Data",
                        visible=False)

            # Add the plot to the right frame
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Attach toggle functionality
            training_checkbox.config(command=toggle_scatter)
            if self.train_test_split_var is not None:
                test_checkbox.config(command=toggle_scatter)

        plot_button = Button(scrollable_frame, text="Plot Surface Response", command=plot_surface_response)
        plot_button.pack(pady=10)

        # Add fields for input values for predictions
        entries = []
        for i in range(self.NN_PD_DATA_X_train.shape[1]):
            label = Label(scrollable_frame, text=f"{self.NN_PD_DATA_X_train.columns[i]}:")
            label.pack(pady=5)
            entry = Entry(scrollable_frame)
            entry.pack(pady=5)
            entries.append(entry)

        def on_nn_predict():
            input_values = []
            for entry in entries:
                try:
                    value = float(entry.get())  # Convert to float
                    input_values.append(value)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
                    return

            # Prepare the input array
            X_new = np.array(input_values).reshape(1, -1)

            X_new = self.sc.transform(X_new)

            # Get predictions from the model
            y_pred = model.predict(X_new)

            # Handle predictions based on the loss function
            if loss_name == "CategoricalCrossentropy":
                predicted_class = np.argmax(y_pred, axis=1)[0]
                probabilities = y_pred[0]
                result_label.config(
                    text=f"Predicted Class: {predicted_class}\nProbabilities: {probabilities}")

            elif loss_name == "SparseCategoricalCrossentropy":
                predicted_class = np.argmax(y_pred, axis=1)[0]
                probabilities = y_pred[0]
                result_label.config(
                    text=f"Predicted Class: {predicted_class}\nProbabilities: {probabilities}")

            elif loss_name == "BinaryCrossentropy":
                predicted_value = y_pred[0][0]
                predicted_class = 1 if predicted_value > 0.5 else 0
                result_label.config(
                    text=f"Predicted Class: {predicted_class} (Probability: {predicted_value:.4f})")

            elif loss_name == "MeanSquaredError":
                predicted_value = y_pred[0][0]
                result_label.config(text=f"Predicted Value: {predicted_value:.4f}")

            else:
                result_label.config(text=f"Unsupported loss function: {loss_name}")

        # Button for predictions
        predict_button = Button(scrollable_frame, text="Predict with NN", command=on_nn_predict)
        predict_button.pack(pady=10)

        # Label to display prediction results
        result_label = Label(scrollable_frame, text="Enter inputs to predict.", fg="white")
        result_label.pack(pady=10)
        sub_notebook2 = ttk.Notebook(scrollable_frame)
        sub_notebook2.pack(fill="both", expand=True, pady=10)

        def plot_training_history():
            metric_groups = {}
            # Group the metrics by their base name (e.g., 'loss', 'accuracy')
            for key in history.history.keys():
                base_name = key.replace('val_', '')
                if base_name not in metric_groups:
                    metric_groups[base_name] = {'train': None, 'val': None}
                if key.startswith('val_'):
                    metric_groups[base_name]['val'] = key
                else:
                    metric_groups[base_name]['train'] = key

            # Create a tab for each metric group (e.g., loss, accuracy)
            for base_name, metrics in metric_groups.items():
                tab_name = f"Epoch vs {base_name.capitalize()}"
                plot_tab = ttk.Frame(sub_notebook2)
                sub_notebook2.add(plot_tab, text=tab_name)
                sub_notebook2.select(plot_tab)

                # Get training and validation data
                train_data = history.history.get(metrics['train'], [])
                val_data = history.history.get(metrics['val'], [])
                epochs = range(1, len(train_data) + 1)

                # Create the plot
                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot training and validation metrics
                if train_data:
                    ax.plot(epochs, train_data, label=f"Train {base_name.capitalize()}", marker='o')
                if val_data:
                    ax.plot(epochs, val_data, label=f"Validation {base_name.capitalize()}", marker='o')

                # Set title and labels
                ax.set_title(f"Model Training History: Epoch vs {base_name.capitalize()}")
                ax.set_xlabel('Epoch')
                ax.set_ylabel(base_name.capitalize())

                # Add a legend
                ax.legend()

                # Add a text box for displaying cursor coordinates
                text_box = ax.text(0.95, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=10,
                                   bbox=dict(facecolor='white', alpha=0.7))

                def on_move(event, ax, text_box, train_data, val_data, epochs, base_name):
                    if event.inaxes != ax:
                        return# Debugging output
                    x = event.xdata  # Get x (epoch)

                    if x is None:
                        return  # Ignore invalid x values

                    closest_epoch = round(x)
                    if closest_epoch < 1:
                        closest_epoch = 1
                    elif closest_epoch > len(epochs):
                        closest_epoch = len(epochs)

                    train_value = train_data[closest_epoch - 1] if closest_epoch <= len(train_data) else None
                    val_value = val_data[closest_epoch - 1] if closest_epoch <= len(val_data) else None

                    text_box.set_text(
                        f"Epoch: {closest_epoch}\n"
                        f"Train {base_name.capitalize()}: {train_value:.4f}\n"
                        f"Val {base_name.capitalize()}: {val_value:.4f}" if val_value is not None else "")
                    ax.figure.canvas.draw()

                # For each metric plot
                fig.canvas.mpl_connect('motion_notify_event',
                    partial(on_move, ax=ax, text_box=text_box, train_data=train_data, val_data=val_data, epochs=epochs,
                            base_name=base_name))

                # Embed the plot in Tkinter window
                canvas = FigureCanvasTkAgg(fig, master=plot_tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add a button to plot the training history dynamically
        plot_history_button = Button(scrollable_frame, text="Plot Training History",
                                     command=plot_training_history)
        plot_history_button.pack(pady=10)





















