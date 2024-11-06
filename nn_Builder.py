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
        self.layer_type_widgets = []
        self.layer_node_widgets = []
        self.layer_activation_widgets = []
        self.layer_kernel_widgets = []
        self.layer_kernel_labels = []
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

    def on_layer_type_change(self, index):
        """ Handle the layer type change event. """
        layer_type = self.layer_types[index].get()

        # Disable activation dropdown if the layer type is Dropout
        if layer_type == "Dropout":
            self.layer_activation_widgets[index].config(state="disabled")
            # Set the activation to None (since Dropout doesn't have an activation function)
            self.layer_activations[index].set(None)
        else:
            self.layer_activation_widgets[index].config(state="normal")

        # Update kernel size field
        self.update_kernel_size_field(index)
        self.show_visual_key()  # This runs after updating the kernel size

    def add_layer_fields(self):
        """ Adds fields for configuring a new layer with the specified column layout, ensuring consistent alignment. """
        layer_index = len(self.layer_fields)

        # Column 0: Remove Layer button (red "X")
        remove_button = tk.Button(self.layers_frame, text="X", fg="red",
                                  command=lambda: self.remove_layer_fields(layer_index))
        remove_button.grid(row=layer_index, column=0, padx=10, pady=5)
        self.layer_remove_buttons.append(remove_button)

        # Column 1: Layer Information Label
        layer_info_label = tk.Label(self.layers_frame, text=f"Layer {layer_index + 1}:")
        layer_info_label.grid(row=layer_index, column=1, padx=10, pady=5, sticky="w")

        # Column 2: Layer Type Dropdown
        layer_type_var = tk.StringVar(value="Dense")
        layer_type_dropdown = ttk.Combobox(self.layers_frame, textvariable=layer_type_var,
                                           values=["Dense", "Convolutional", "Pooling", "Flatten", "Dropout"])
        layer_type_dropdown.grid(row=layer_index, column=2, padx=10, pady=5, sticky="w")
        layer_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.on_layer_type_change(layer_index))

        self.layer_types.append(layer_type_var)
        self.layer_type_widgets.append(layer_type_dropdown)

        # Column 3: Nodes, Kernels, or Dropout Rate Entry
        nodes_label = tk.Label(self.layers_frame, text="Nodes/Rate:")
        nodes_label.grid(row=layer_index, column=3, sticky="e", padx=(5, 2), pady=5)

        nodes_var = tk.IntVar(value=10)
        nodes_entry = tk.Entry(self.layers_frame, textvariable=nodes_var, width=5)
        nodes_entry.grid(row=layer_index, column=4, padx=10, pady=5, sticky="w")
        nodes_entry.bind("<FocusOut>", lambda e: self.show_visual_key())  # Trigger visualization update
        self.layer_nodes_vars.append(nodes_var)
        self.layer_node_widgets.append(nodes_entry)

        # Column 4: Activation Function Dropdown
        activation_label = tk.Label(self.layers_frame, text="Activation:")
        activation_label.grid(row=layer_index, column=5, sticky="e", padx=(5, 2), pady=5)

        activation_var = tk.StringVar(value="relu" if layer_index < 1 else "softmax")
        activation_dropdown = ttk.Combobox(self.layers_frame, textvariable=activation_var,
                                           values=["relu", "sigmoid", "tanh", "linear", "softmax",
                                                   "None"])  # Add "None"
        activation_dropdown.grid(row=layer_index, column=6, padx=10, pady=5, sticky="w")
        activation_dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_visual_key())  # Trigger visualization update

        # Disable activation dropdown for Dropout layers
        if layer_type_var.get() == "Dropout":
            activation_dropdown.config(state="disabled")  # Grey out the dropdown

        self.layer_activations.append(activation_var)
        self.layer_activation_widgets.append(activation_dropdown)

        # Columns 5 and 6: Kernel Size label and entry (conditionally displayed)
        kernel_size_intvar = tk.IntVar(value=3)
        kernel_size_label = tk.Label(self.layers_frame, text="Kernel Size:")
        kernel_size_entry = tk.Entry(self.layers_frame, textvariable=kernel_size_intvar, width=5)

        # Add kernel size entry to the list for later reference
        self.layer_kernel_sizes.append(kernel_size_intvar)

        # Store kernel size widgets for later display control
        self.layer_kernel_labels.append(kernel_size_label)
        self.layer_kernel_widgets.append(kernel_size_entry)

        # Add kernel size widgets to lists for tracking
        self.layer_fields.append((layer_type_var, nodes_var, activation_var, kernel_size_label))

        # Show/hide kernel size fields based on the selected layer type
        self.update_kernel_size_field(layer_index)

        # Refresh the visualization
        self.show_visual_key()

    def update_kernel_size_field(self, index):
        """ Adds the kernel size fields at the end (columns 7 and 8) when needed, without rearranging other columns. """
        layer_type = self.layer_types[index].get()

        if layer_type in ["Convolutional", "Pooling"]:
            # Show kernel size label and entry in columns 7 and 8
            self.layer_kernel_labels[index].grid(row=index, column=7, padx=10, pady=5, sticky="e")
            self.layer_kernel_widgets[index].grid(row=index, column=8, padx=10, pady=5, sticky="w")
        else:
            # Hide kernel size label and entry if not needed
            self.layer_kernel_labels[index].grid_forget()
            self.layer_kernel_widgets[index].grid_forget()

    def remove_layer_fields(self, index):
        """ Removes fields for the specified layer and updates the layout. """
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

        # Reorder rows after deletion to maintain correct indexing and reassign labels
        for i in range(len(self.layer_fields)):
            # Reassign layer labels (Layer 1, Layer 2, etc.)
            self.layer_remove_buttons[i].grid(row=i, column=0, padx=10)
            self.layer_type_widgets[i].grid(row=i, column=2, padx=10, pady=5)
            self.layer_node_widgets[i].grid(row=i, column=3, padx=10)
            self.layer_activation_widgets[i].grid(row=i, column=4, padx=10)

            # Update kernel size field visibility
            self.update_kernel_size_field(i)
            self.layer_remove_buttons[i].grid(row=i, column=0, padx=10)

            # Reassign layer info labels (Layer 1, Layer 2, etc

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

        print(f"Layers: {layers}")
        print(f"Layer Types: {layer_types}")
        print(f"Activations: {activations}")
        print(f"Kernel Sizes: {kernel_sizes}")


        # Create a new figure for the visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')

        # Define positions for each layer
        x_positions = np.linspace(0.1, 0.9, num_layers)
        y_offset = 0.5

        for i, (x, nodes, layer_type, activation, kernel_size) in enumerate(
                zip(x_positions, layers, layer_types, activations, kernel_sizes)):
            # Start with basic layer text
            layer_text = f"Layer {i + 1}: {layer_type}\n"

            # Handle different layer types
            if layer_type == "Dense":
                layer_text += f" Nodes: {nodes}\n"
            elif layer_type == "Convolutional":
                if kernel_size:  # Only add kernel size if it's valid
                    layer_text += f" Filters: {nodes}\nKernel Size: {kernel_size}\n"
                else:
                    layer_text += f"Kernel Size: Not Defined\n"
            elif layer_type == "Pooling":
                if kernel_size:  # Only add kernel size if it's valid
                    layer_text += f"Kernel Size: {kernel_size}\n"
                else:
                    layer_text += f"Kernel Size: Not Defined\n"
            elif layer_type == "Dropout":
                layer_text += f"Dropout Rate: {nodes}\n"

            # Skip activation for Dropout layers
            if layer_type != "Dropout":
                layer_text += f"Activation: {activation}"  # Add activation for non-Dropout layerse

            # Add the layer text to the plot
            ax.text(x, y_offset, layer_text, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.5", edgecolor="black", facecolor="lightblue"))

            # Add arrows between layers (except the last layer)
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


