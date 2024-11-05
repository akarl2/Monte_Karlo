import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NeuralNetworkArchitectureBuilder:
    def __init__(self, master, X_data=None, y_data=None):
        self.master = master
        self.layer_fields = []
        self.layer_types = []
        self.layer_nodes_vars = []
        self.layer_activations = []
        self.layer_kernel_sizes = []
        self.layer_type_widgets = []  # Store the ComboBox widgets
        self.layer_node_widgets = []  # Store the Entry widgets for nodes
        self.layer_activation_widgets = []  # Store the ComboBox widgets for activations
        self.layer_kernel_widgets = []  # Store the Entry widgets for kernel sizes
        self.layer_kernel_labels = []  # Store kernel size labels
        self.layer_remove_buttons = []

    def configure_nn_popup(self):
        self.master.title("Neural Network Architecture Builder")

        # Frame for layer configuration
        self.layers_frame = tk.Frame(self.master)
        self.layers_frame.pack(pady=10, fill="x")

        # Add Layer button
        add_layer_button = tk.Button(self.master, text="Add Layer", command=self.add_layer_fields)
        add_layer_button.pack(pady=5)

        # Frame for visualization
        self.visualization_frame = tk.Frame(self.master)
        self.visualization_frame.pack(pady=10, fill="both", expand=True)

        # Initialize with two layers by default
        self.add_layer_fields()
        self.add_layer_fields()

        # Start training button
        start_training_button = tk.Button(self.master, text="Start Training", command=self.run_training)
        start_training_button.pack(pady=10)

    def add_layer_fields(self):
        """ Adds fields for configuring a new layer with a 'Remove' button. """
        layer_index = len(self.layer_fields)

        # Layer type selection dropdown
        layer_type_var = tk.StringVar(value="Dense")
        layer_type_dropdown = ttk.Combobox(self.layers_frame, textvariable=layer_type_var,
                                           values=["Dense", "Convolutional", "Pooling", "Flatten", "Dropout"])
        layer_type_dropdown.grid(row=layer_index, column=0, padx=10, pady=5)
        layer_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_kernel_size_field(layer_index))

        self.layer_types.append(layer_type_var)
        self.layer_type_widgets.append(layer_type_dropdown)

        # Nodes or Dropout Rate Label
        nodes_label = tk.Label(self.layers_frame, text="Nodes:")
        nodes_label.grid(row=layer_index, column=1, padx=10)

        # Nodes Entry
        nodes_var = tk.IntVar(value=10)
        nodes_entry = tk.Entry(self.layers_frame, textvariable=nodes_var, width=5)
        nodes_entry.grid(row=layer_index, column=2, padx=10)
        self.layer_nodes_vars.append(nodes_var)
        self.layer_node_widgets.append(nodes_entry)

        # Activation function dropdown
        activation_var = tk.StringVar(value="relu" if layer_index < 1 else "softmax")
        activation_dropdown = ttk.Combobox(self.layers_frame, textvariable=activation_var,
                                           values=["relu", "sigmoid", "tanh", "linear", "softmax"])
        activation_dropdown.grid(row=layer_index, column=3, padx=10)
        self.layer_activations.append(activation_var)
        self.layer_activation_widgets.append(activation_dropdown)

        # Kernel size label and entry will be added conditionally
        kernel_size_var = tk.IntVar(value=3)
        kernel_size_entry = tk.Entry(self.layers_frame, textvariable=kernel_size_var, width=5)
        kernel_size_label = tk.Label(self.layers_frame, text="Kernel Size:")

        # Store the kernel size widgets for conditional updates
        self.layer_kernel_sizes.append(kernel_size_var)
        self.layer_kernel_widgets.append(kernel_size_entry)
        self.layer_kernel_labels.append(kernel_size_label)

        # Remove Layer button
        remove_button = tk.Button(self.layers_frame, text="X", fg="red",
                                  command=lambda: self.remove_layer_fields(layer_index))
        remove_button.grid(row=layer_index, column=5, padx=10)
        self.layer_remove_buttons.append(remove_button)

        # Append to fields for tracking
        self.layer_fields.append((layer_type_var, nodes_var, activation_var, kernel_size_var))

        # Refresh the visualization
        self.show_visual_key()

        # Update kernel size field visibility
        self.update_kernel_size_field(layer_index)

    def update_kernel_size_field(self, index):
        """ Updates the visibility of the kernel size field based on the selected layer type. """
        layer_type = self.layer_types[index].get()

        if layer_type in ["Convolutional", "Pooling"]:
            # Show kernel size label and entry
            self.layer_kernel_labels[index].grid(row=index, column=4, padx=10)
            self.layer_kernel_widgets[index].grid(row=index, column=4, padx=10)
        else:
            # Hide kernel size label and entry
            self.layer_kernel_labels[index].grid_forget()
            self.layer_kernel_widgets[index].grid_forget()

    def remove_layer_fields(self, index):
        """ Removes fields for the specified layer. """
        # Remove the widgets from the grid
        for widget in self.layers_frame.grid_slaves(row=index):
            widget.grid_forget()

        # Remove the configuration from lists
        del self.layer_fields[index]
        del self.layer_types[index]
        del self.layer_nodes_vars[index]
        del self.layer_activations[index]
        del self.layer_kernel_sizes[index]

        # Also remove the corresponding widgets from their respective lists
        del self.layer_type_widgets[index]
        del self.layer_node_widgets[index]
        del self.layer_activation_widgets[index]
        del self.layer_kernel_widgets[index]
        del self.layer_kernel_labels[index]
        del self.layer_remove_buttons[index]

        # Reorder rows after deletion
        for i in range(len(self.layer_fields)):
            self.layer_type_widgets[i].grid(row=i, column=0, padx=10, pady=5)
            self.layer_node_widgets[i].grid(row=i, column=2, padx=10)
            self.layer_activation_widgets[i].grid(row=i, column=3, padx=10)
            # Update kernel size field visibility
            self.update_kernel_size_field(i)
            self.layer_remove_buttons[i].grid(row=i, column=5, padx=10)

        # Refresh the visualization
        self.show_visual_key()

    def show_visual_key(self):
        """ Show a visual representation of the neural network architecture. """
        # Collect current layer configurations
        layers = [nodes_var.get() for nodes_var in self.layer_nodes_vars]
        layer_types = [layer_type.get() for layer_type in self.layer_types]
        activations = [activation.get() for activation in self.layer_activations]
        kernel_sizes = [kernel_size.get() if layer_type in ["Convolutional", "Pooling"] else None
                        for layer_type, kernel_size in zip(layer_types, self.layer_kernel_sizes)]

        num_layers = len(layers)

        # Create a new figure for the visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')

        # Define positions for each layer
        x_positions = np.linspace(0.1, 0.9, num_layers)
        y_offset = 0.5

        for i, (x, nodes, layer_type, activation, kernel_size) in enumerate(
                zip(x_positions, layers, layer_types, activations, kernel_sizes)):
            layer_text = f"Layer {i + 1} ({layer_type})\n"
            if layer_type == "Dense":
                layer_text += f"{nodes} Nodes\n"
            elif layer_type == "Convolutional":
                layer_text += f"{nodes} Filters\nKernel Size: {kernel_size}\n"
            elif layer_type == "Pooling":
                layer_text += f"Kernel Size: {kernel_size}\n"
            layer_text += f"{activation} Activation"

            ax.text(x, y_offset, layer_text, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.5", edgecolor="black", facecolor="lightblue"))

            if i < num_layers - 1:
                ax.annotate("", xy=(x + 0.15, y_offset), xytext=(x + 0.05, y_offset),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

        # Clear previous visualization
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        # Create a new canvas for the updated visualization
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
