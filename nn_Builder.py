import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NeuralNetworkArchitectureBuilder:
    def __init__(self, master, X_data=None, y_data=None):
        self.master = master
        self.layer_nodes_vars = []
        self.layer_activations = []
        self.layer_types = []
        self.layer_kernel_sizes = []
        self.layer_dropout_rates = []

    def configure_nn_popup(self):
        # Popup for neural network architecture builder
        config_popup = tk.Toplevel(self.master)
        config_popup.title("Neural Network Architecture Builder")

        # Select the number of layers
        tk.Label(config_popup, text="Number of Layers:").pack(anchor=tk.W, padx=10, pady=5)
        self.num_layers_var = tk.IntVar(value=1)
        num_layers_spinbox = tk.Spinbox(config_popup, from_=1, to=10, textvariable=self.num_layers_var,
                                         command=self.update_layer_fields)
        num_layers_spinbox.pack(anchor=tk.W, padx=10)

        # Frame to hold layer configuration fields
        self.layers_frame = tk.Frame(config_popup)
        self.layers_frame.pack(pady=10, fill="x")

        # Frame to hold the visualization
        self.visualization_frame = tk.Frame(config_popup)
        self.visualization_frame.pack(pady=10, fill="both", expand=True)

        # Initial update for layer fields and visual key
        self.update_layer_fields()

        # Button to run training (placeholder for functionality)
        start_training_button = tk.Button(config_popup, text="Start Training", command=self.run_training)
        start_training_button.pack(pady=10, anchor="s")

    def update_layer_fields(self):
        # Clear existing layer configuration fields
        for widget in self.layers_frame.winfo_children():
            widget.destroy()

        # Reset configuration details for each layer
        self.layer_nodes_vars.clear()
        self.layer_activations.clear()
        self.layer_types.clear()
        self.layer_kernel_sizes.clear()
        self.layer_dropout_rates.clear()

        for i in range(self.num_layers_var.get()):
            # Layer type selection dropdown
            tk.Label(self.layers_frame, text=f"Layer {i + 1} Type:").grid(row=i, column=0, padx=10, pady=5)

            layer_type_var = tk.StringVar(value="Dense")
            layer_type_dropdown = ttk.Combobox(self.layers_frame, textvariable=layer_type_var,
                                               values=["Dense", "Convolutional", "Pooling", "Flatten", "Dropout"])
            layer_type_dropdown.grid(row=i, column=1, padx=10)
            layer_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_visual_key())
            self.layer_types.append(layer_type_var)

            # Nodes or Dropout Rate Label
            nodes_label = tk.Label(self.layers_frame, text="Nodes:")
            nodes_label.grid(row=i, column=2, padx=10)

            # Nodes or Dropout Rate Entry
            nodes_var = tk.IntVar(value=10)
            nodes_entry = tk.Entry(self.layers_frame, textvariable=nodes_var, width=5)
            nodes_entry.grid(row=i, column=3, padx=10)
            nodes_entry.bind("<KeyRelease>", lambda e: self.show_visual_key())
            self.layer_nodes_vars.append(nodes_var)

            # Activation function dropdown
            activation_label = tk.Label(self.layers_frame, text="Activation:")
            activation_label.grid(row=i, column=4, padx=10)

            activation_var = tk.StringVar(value="relu" if i < self.num_layers_var.get() - 1 else "softmax")
            activation_dropdown = ttk.Combobox(self.layers_frame, textvariable=activation_var,
                                               values=["relu", "sigmoid", "tanh", "linear", "softmax"])
            activation_dropdown.grid(row=i, column=5, padx=10)
            activation_dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_visual_key())
            self.layer_activations.append(activation_var)

            # Kernel size entry for Conv and Pooling layers
            kernel_label = tk.Label(self.layers_frame, text="Kernel Size:")
            kernel_label.grid(row=i, column=6, padx=10)

            kernel_size_var = tk.IntVar(value=3)  # Default kernel size
            kernel_size_entry = tk.Entry(self.layers_frame, textvariable=kernel_size_var, width=5)
            kernel_size_entry.grid(row=i, column=7, padx=10)
            kernel_size_entry.bind("<KeyRelease>", lambda e: self.show_visual_key())
            self.layer_kernel_sizes.append(kernel_size_var)

            # Dropout rate entry (initially hidden)
            dropout_rate_var = tk.DoubleVar(value=0.5)
            self.layer_dropout_rates.append(dropout_rate_var)

            # Dropout rate entry; initially hidden, visible only for Dropout layers
            dropout_rate_entry = tk.Entry(self.layers_frame, textvariable=dropout_rate_var, width=5)

            # Dynamic option updates based on selected layer type
            layer_type_var.trace("w", lambda *args, i=i, nodes_label=nodes_label, kernel_label=kernel_label,
                                         activation_label=activation_label,
                                         activation_dropdown=activation_dropdown,
                                         nodes_entry=nodes_entry, kernel_size_entry=kernel_size_entry,
                                         dropout_rate_entry=dropout_rate_entry:
            self.update_layer_options(i, nodes_label, kernel_label, activation_label,
                                      activation_dropdown, dropout_rate_entry,
                                      nodes_entry, kernel_size_entry))

        # Initial visualization of the updated layers
        self.show_visual_key()

    def update_layer_options(self, layer_index, nodes_label, kernel_label, activation_label,
                             activation_dropdown, dropout_rate_entry, nodes_entry, kernel_size_entry):
        """
        Update the layer configuration options dynamically based on the selected layer type.
        """
        layer_type = self.layer_types[layer_index].get()

        # Adjust the widget visibility and labels based on layer type
        if layer_type == "Dense":
            nodes_label.config(text="Nodes:")
            nodes_entry.grid()  # Show nodes for Dense layers
            kernel_label.grid_remove()  # Hide kernel size for Dense layers
            kernel_size_entry.grid_remove()
            activation_label.grid()  # Show activation
            activation_dropdown.grid()
            dropout_rate_entry.grid_remove()  # Hide dropout rate for Dense

        elif layer_type == "Convolutional":
            nodes_label.config(text="Filters:")
            nodes_entry.grid()  # Show filters for Conv layers
            kernel_label.grid()  # Show kernel size for Conv layers
            kernel_size_entry.grid()
            activation_label.grid()  # Show activation
            activation_dropdown.grid()
            dropout_rate_entry.grid_remove()  # Hide dropout rate for Conv

        elif layer_type == "Pooling":
            nodes_label.grid_remove()  # Hide nodes for Pooling layers
            nodes_entry.grid_remove()  # Hide nodes for Pooling layers
            kernel_label.grid()  # Show kernel size for Pooling layers
            kernel_size_entry.grid()
            activation_label.grid_remove()  # Pooling layers don’t have activation
            activation_dropdown.grid_remove()
            dropout_rate_entry.grid_remove()  # Hide dropout rate for Pooling

        elif layer_type == "Dropout":
            nodes_label.config(text="Dropout Rate:")  # Label becomes Dropout Rate
            nodes_entry.grid_remove()  # Hide nodes entry for Dropout
            kernel_label.grid_remove()  # Hide kernel size for Dropout
            kernel_size_entry.grid_remove()
            activation_label.grid_remove()  # Dropout layers don’t have activation
            activation_dropdown.grid_remove()
            dropout_rate_entry.grid()  # Show dropout rate for Dropout layers

        elif layer_type == "Flatten":
            nodes_entry.grid_remove()  # Hide nodes for Flatten layers
            kernel_label.grid_remove()  # Hide kernel size for Flatten layers
            kernel_size_entry.grid_remove()
            activation_label.grid_remove()  # Flatten layers don’t have activation
            activation_dropdown.grid_remove()
            dropout_rate_entry.grid_remove()  # Hide dropout rate for Flatten

    def show_visual_key(self):
        """
        Show a visual representation of the neural network architecture in a horizontal layout.
        Each layer box displays the layer number, layer type, nodes/filters, kernel size, and activation function.
        """
        # Gather details for each layer
        layers = [int(nodes_var.get()) for nodes_var in self.layer_nodes_vars]
        layer_types = [layer_type.get() for layer_type in self.layer_types]
        activations = [activation_var.get() for activation_var in self.layer_activations]
        kernel_sizes = [int(kernel_size.get()) for kernel_size in self.layer_kernel_sizes]
        num_layers = len(layers)

        # Create a new figure for the visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')  # Turn off the axis

        # Define positions for each layer
        x_positions = np.linspace(0.1, 0.9, num_layers)  # Horizontal spacing
        y_offset = 0.5  # Vertical center position for the layers

        for i, (x, nodes, layer_type, activation, kernel_size) in enumerate(
                zip(x_positions, layers, layer_types, activations, kernel_sizes)):
            # Display type, nodes, kernel size, and activation for each layer
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

            # Draw arrows to the next layer
            if i < num_layers - 1:
                ax.annotate("", xy=(x + 0.15, y_offset), xytext=(x + 0.05, y_offset),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

            # Embed the figure in the Tkinter interface
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()  # Clear previous visualization
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_training(self):
        # Placeholder for the training functionality
        print("Training started with the following architecture:")
        for i in range(len(self.layer_types)):
            layer_type = self.layer_types[i].get()
            nodes = self.layer_nodes_vars[i].get()
            activation = self.layer_activations[i].get()
            kernel_size = self.layer_kernel_sizes[i].get() if layer_type in ["Convolutional", "Pooling"] else None
            dropout_rate = self.layer_dropout_rates[i].get() if layer_type == "Dropout" else None

            print(
                f"Layer {i + 1}: Type={layer_type}, Nodes/Filters={nodes}, Kernel Size={kernel_size}, Activation={activation}, Dropout Rate={dropout_rate}")

        # Example usage of the class

    if __name__ == "__main__":
        root = tk.Tk()
        root.title("Neural Network Builder")
        nn_builder = NeuralNetworkArchitectureBuilder(root)
        root.mainloop()