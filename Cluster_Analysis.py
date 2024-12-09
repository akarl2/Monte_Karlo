import tkinter
from tkinter import ttk, Frame

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans, DBSCAN
from matplotlib.patches import Circle


class ClusterAnalysis:
    def __init__(self, master, data, n_clusters, random_starts, data_PD, full_dataset):
        self.data = data
        self.master = master
        self.n_clusters = n_clusters
        self.random_starts = random_starts
        self.notebook = None
        self.data_PD = data_PD
        self.min_samples = 2
        self.full_dataset = full_dataset

    def Configure_Cluster_Popup(self):
        self.master.title("Cluster Analysis")
        self.master.geometry("1800x1000")  # Adjust for visualization
        self.master.resizable(True, True)

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # Create notebook with modern styling
        self.notebook = ttk.Notebook(self.master)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def DBSCAN_Clustering(self, tab):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.data)
        self.display_clustering(self.data, dbscan, tab)

    def KMeans_Clustering(self, tab):
        #valid rows
        valid_rows = ~np.isnan(self.data).any(axis=1)
        self.data = self.data[valid_rows]

        #remove rows with NA in data_PD
        self.data_PD = self.data_PD.dropna()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.random_starts, random_state=0).fit(self.data)
        self.display_clustering(kmeans, tab)

    def display_clustering(self, kmeans, tab):

        n_features = self.data.shape[1]
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        cmap = plt.cm.get_cmap('viridis', len(centroids))

        # Adjust cluster labels to start from 1
        adjusted_labels = labels + 1

        # Add cluster column to full dataset
        self.full_dataset.loc[:, "Cluster"] = adjusted_labels  # Use .loc to avoid the warning

        # Move the Cluster column to the first position
        cols = ["Cluster"] + [col for col in self.full_dataset.columns if col != "Cluster"]
        self.full_dataset = self.full_dataset[cols]  # Reorder columns

        # Create a resizable layout
        paned_window = ttk.PanedWindow(tab, orient="horizontal")
        paned_window.pack(fill="both", expand=True)

        # Left frame for the plot
        plot_frame = ttk.Frame(paned_window)
        paned_window.add(plot_frame, weight=1)

        # Right frame for the pandas table and dropdowns
        control_frame = ttk.Frame(paned_window)
        paned_window.add(control_frame, weight=1)

        # Create the Treeview widget to display the table
        tree = ttk.Treeview(control_frame, show="headings")
        tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Set up the columns in the Treeview
        columns = self.full_dataset.columns.tolist()
        tree["columns"] = columns

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)

        # Maintain a mapping of Treeview indices to the original dataset indices
        self.full_dataset["OriginalIndex"] = self.full_dataset.index  # Store original indices
        sorted_dataset = self.full_dataset.sort_values(by="Cluster").reset_index(drop=True)

        # Insert rows into the Treeview
        for _, row in sorted_dataset.iterrows():
            tree.insert("", "end", values=row.tolist())

        use_dropdowns = False

        # Function to update the plot
        def update_plot(selected_row=None):
            """Update the plot based on the selected features."""

            if use_dropdowns:
                x_index = self.data_PD.columns.tolist().index(x_var.get())
                y_index = self.data_PD.columns.tolist().index(y_var.get())
                z_index = self.data_PD.columns.tolist().index(z_var.get())
            else:
                feature_var_x = tkinter.StringVar(value=self.data_PD.columns[0])
                feature_var_y = tkinter.StringVar(value=self.data_PD.columns[1])
                x_index = self.data_PD.columns.tolist().index(feature_var_x.get())
                y_index = self.data_PD.columns.tolist().index(feature_var_y.get())

                # Only define z_var if there are more than 2 features
                if n_features > 2:
                    feature_var_z = tkinter.StringVar(value=self.data_PD.columns[2])
                    z_index = self.data_PD.columns.tolist().index(feature_var_z.get())
                else:
                    feature_var_z = None
                    z_index = None

            fig = plt.figure(figsize=(6, 6))
            if n_features == 2 or z_index is None:
                # 2D plot
                ax = fig.add_subplot(111)
                for i, centroid in enumerate(centroids, start=1):
                    cluster_points = self.data[adjusted_labels == i]
                    color = cmap(i - 1)

                    # Calculate the average distance of points in the cluster to the centroid
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    avg_distance = np.mean(distances)

                    ax.scatter(cluster_points[:, x_index], cluster_points[:, y_index], color=color, alpha=0.7,
                               label=f'Cluster {i}')
                    ax.scatter(centroid[x_index], centroid[y_index], color=color, marker='x', s=100,
                               label=f'Centroid {i}')

                    # Add circle around centroid
                    circle = Circle((centroid[x_index], centroid[y_index]), avg_distance, color=color, fill=False)
                    ax.add_artist(circle)

                if selected_row is not None:
                    selected_point = self.data[int(selected_row)]
                    ax.scatter(selected_point[x_index], selected_point[y_index], color="red", marker="o", s=150,
                               label="Selected Point", edgecolors="black", linewidths=1.5)

                ax.set_xlabel(self.full_dataset.columns[x_index])
                ax.set_ylabel(self.full_dataset.columns[y_index])
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
            else:
                # 3D plot
                ax = fig.add_subplot(111, projection='3d')
                for i, centroid in enumerate(centroids, start=1):
                    cluster_points = self.data[adjusted_labels == i]
                    color = cmap(i - 1)

                    ax.scatter(cluster_points[:, x_index], cluster_points[:, y_index], cluster_points[:, z_index],
                               color=color, alpha=0.7, label=f'Cluster {i}')
                    ax.scatter(centroid[x_index], centroid[y_index], centroid[z_index], color=color, marker='x', s=100,
                               label=f'Centroid {i}')

                if selected_row is not None:
                    selected_point = self.data[int(selected_row)]
                    ax.scatter(selected_point[x_index], selected_point[y_index], selected_point[z_index],
                               color="red", marker="o", s=150, label="Selected Point", edgecolors="black",
                               linewidths=1.5)

                ax.set_xlabel(self.data_PD.columns[x_index])
                ax.set_ylabel(self.data_PD.columns[y_index])
                ax.set_zlabel(self.data_PD.columns[z_index])
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)

            ax.set_title(f"K-Means Clustering with {len(centroids)} Clusters")
            plt.subplots_adjust(bottom = 0.25)

            for widget in plot_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        if n_features > 3:
            use_dropdowns = True
            # Add dropdowns for feature selection
            axis_selection_frame = ttk.Frame(control_frame)
            axis_selection_frame.pack(fill="x", padx=5, pady=5)

            # Initialize tkinter StringVars with the first three features
            feature_options = self.data_PD.columns.tolist()  # Use headers from the DataFrame
            x_var = tkinter.StringVar(value=feature_options[0])  # Default to the first feature
            y_var = tkinter.StringVar(value=feature_options[1])  # Default to the second feature
            z_var = tkinter.StringVar(value=feature_options[2])  # Default to the third feature

            # Create dropdowns for axis selection
            ttk.Label(axis_selection_frame, text="X Axis:").grid(row=0, column=0, sticky="w", padx=5)
            x_dropdown = ttk.Combobox(axis_selection_frame, textvariable=x_var, values=feature_options)
            x_dropdown.grid(row=0, column=1, sticky="ew", padx=5)

            ttk.Label(axis_selection_frame, text="Y Axis:").grid(row=1, column=0, sticky="w", padx=5)
            y_dropdown = ttk.Combobox(axis_selection_frame, textvariable=y_var, values=feature_options)
            y_dropdown.grid(row=1, column=1, sticky="ew", padx=5)

            ttk.Label(axis_selection_frame, text="Z Axis:").grid(row=2, column=0, sticky="w", padx=5)
            z_dropdown = ttk.Combobox(axis_selection_frame, textvariable=z_var, values=feature_options)
            z_dropdown.grid(row=2, column=1, sticky="ew", padx=5)

            # Bind the dropdown changes to update the plot
            x_dropdown.bind("<<ComboboxSelected>>", lambda _: update_plot())
            y_dropdown.bind("<<ComboboxSelected>>", lambda _: update_plot())
            z_dropdown.bind("<<ComboboxSelected>>", lambda _: update_plot())

        # Initial plot
        update_plot()

        def on_table_select(event):
            """Highlight the selected row on the plot."""
            selected_item = tree.selection()
            if selected_item:
                treeview_index = tree.index(selected_item[0])  # Get Treeview index
                original_index = sorted_dataset.iloc[treeview_index]["OriginalIndex"]
                update_plot(original_index)

        # Bind the Treeview selection event to the function
        tree.bind("<<TreeviewSelect>>", on_table_select)




