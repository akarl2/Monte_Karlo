import tkinter
from tkinter import ttk, Frame

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import centroid
from sklearn.cluster import KMeans, DBSCAN
from matplotlib.patches import Circle


class ClusterAnalysis:
    def __init__(self, master, data, n_clusters, random_starts, data_PD, full_dataset, cluster_method):
        self.creating_plots = False
        self.plot_frames = None
        self.sorted_dataset = None
        self.adjusted_labels = None
        self.centroids = None
        self.master_dropdown_frame = None
        self.cmap = None
        self.plot_frame = None
        self.ax = None
        self.fig = None
        self.z_var = None
        self.y_var = None
        self.x_var = None
        self.use_dropdowns = None
        self.data = data
        self.master = master
        self.n_clusters = n_clusters
        self.random_starts = random_starts
        self.notebook = None
        self.data_PD = data_PD
        self.min_samples = 2
        self.full_dataset = full_dataset
        self.cluster_method = cluster_method
        self.wcss = []
        self.plot_frames = {}
        self.tabs = {}  # Store tabs for each cluster count
        self.plots = {}  # Store plots for each cluster count
        self.cluster_data = {}
        self.feature_options = self.data_PD.columns.tolist()
        self.use_dropdowns = False
        self.n_features = self.data.shape[1]

        if self.n_features > 3:
            self.use_dropdowns = True

        self.x_var = tkinter.StringVar(value=self.feature_options[0])
        self.y_var = tkinter.StringVar(value=self.feature_options[1])
        self.z_var = tkinter.StringVar(value=self.feature_options[2]) if len(self.feature_options) > 2 else None

        self.configure_cluster_popup()


    def configure_cluster_popup(self):
        self.master.title("Cluster Analysis")
        self.master.geometry("1800x700")  # Adjust for visualization
        self.master.resizable(True, True)

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # Create notebook with modern styling
        self.notebook = ttk.Notebook(self.master)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        if self.n_features > 3:
            self.create_master_dropdown()

        self.create_tabs()

    def create_master_dropdown(self):
        """Create master dropdowns for feature selection."""
        self.master_dropdown_frame = ttk.Frame(self.master)
        self.master_dropdown_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        # X Axis dropdown
        ttk.Label(self.master_dropdown_frame, text="X Axis:").grid(row=0, column=0, sticky="w", padx=5)
        x_dropdown = ttk.Combobox(self.master_dropdown_frame, textvariable=self.x_var, values=self.feature_options)
        x_dropdown.grid(row=0, column=1, sticky="ew", padx=5)
        x_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_plot())

        # Y Axis dropdown
        ttk.Label(self.master_dropdown_frame, text="Y Axis:").grid(row=0, column=2, sticky="w", padx=5)
        y_dropdown = ttk.Combobox(self.master_dropdown_frame, textvariable=self.y_var, values=self.feature_options)
        y_dropdown.grid(row=0, column=3, sticky="ew", padx=5)
        y_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_plot())

        # Z Axis dropdown (if 3D plotting is available)
        if len(self.feature_options) > 2:
            ttk.Label(self.master_dropdown_frame, text="Z Axis:").grid(row=0, column=4, sticky="w", padx=5)
            z_dropdown = ttk.Combobox(self.master_dropdown_frame, textvariable=self.z_var, values=self.feature_options)
            z_dropdown.grid(row=0, column=5, sticky="ew", padx=5)
            z_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_plot())

    def create_tabs(self):
        # Drop rows with missing values
        valid_indices = self.data_PD.dropna().index
        self.data = self.data[valid_indices]
        self.data_PD = self.data_PD.loc[valid_indices]
        self.full_dataset = self.full_dataset.loc[valid_indices]

        """Create tabs for cluster counts from 2 to 10."""
        for n_clusters in range(2, 11):
            # Create a new tab
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"{n_clusters} Clusters")
            self.notebook.select(tab)
            self.tabs[n_clusters] = tab

            if self.n_features == 1 or self.n_features == 2:
                fig, ax = plt.subplots(figsize=(6, 6))
            else:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='3d')

            self.plots[n_clusters] = {"fig": fig, "ax": ax}

            self.n_clusters = n_clusters

            # Perform clustering and visualization
            if self.cluster_method == "KMeans":
                self.KMeans_Clustering(tab)
            elif self.cluster_method == "DBSCAN":
                self.DBSCAN_Clustering(tab)

        self.notebook.select(self.tabs[2])

    def DBSCAN_Clustering(self, tab):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.data)
        self.display_clustering(self.data, dbscan, tab)

    def KMeans_Clustering(self, tab):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.random_starts, random_state=0).fit(self.data)
        self.wcss.append(kmeans.inertia_)
        self.display_clustering(kmeans, tab)

    def display_clustering(self, kmeans, tab):
        labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
        self.cmap = plt.cm.get_cmap('viridis', len(self.centroids))

        # Adjust cluster labels to start from 1
        self.adjusted_labels = labels + 1

        self.plots[self.n_clusters]["centroids"] = self.centroids
        self.plots[self.n_clusters]["labels"] = self.adjusted_labels
        self.plots[self.n_clusters]["cmap"] = self.cmap

        # Add cluster column to full dataset
        self.full_dataset = self.full_dataset.copy()  # Avoid SettingWithCopyWarning
        self.full_dataset.loc[:, "Cluster"] = self.adjusted_labels  # Use .loc to avoid the warning

        # Move the Cluster column to the first position
        cols = ["Cluster"] + [col for col in self.full_dataset.columns if col != "Cluster"]
        self.full_dataset = self.full_dataset[cols]  # Reorder columns

        # Create a resizable layout
        paned_window = ttk.PanedWindow(tab, orient="horizontal")
        paned_window.pack(fill="both", expand=True)

        # Left frame for the plot
        plot_frame = ttk.Frame(paned_window)
        paned_window.add(plot_frame, weight=1)
        self.plot_frames[self.n_clusters] = plot_frame

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

        self.sorted_dataset = self.full_dataset.sort_values(by="Cluster").reset_index(drop=True)

        self.plots[self.n_clusters]["sorted_dataset"] = self.sorted_dataset

        # Insert rows into the Treeview
        for _, row in self.sorted_dataset.iterrows():
            tree.insert("", "end", values=row.tolist())

        # Bind the Treeview selection event to the function
        tree.bind("<<TreeviewSelect>>", lambda event: self.on_table_select(event, tree))

        def plot_elbow():
            """Create the elbow plot."""
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(range(2, len(self.wcss) + 2), self.wcss, marker="o", linestyle="-")
            ax.set_title("Elbow Method")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
            ax.grid()

            canvas = FigureCanvasTkAgg(fig, master=elbow_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        if hasattr(self, "wcss") and len(self.wcss) >= 9:
            elbow_tab = ttk.Frame(self.notebook)
            self.notebook.add(elbow_tab, text="Elbow Method")
            elbow_frame = ttk.Frame(elbow_tab)
            elbow_frame.pack(fill="both", expand=True)
            plot_elbow()

        self.update_plot()

    # Function to update the plot
    def update_plot(self, selected_row_data=None):

        current_selected_cluster = self.notebook.index(self.notebook.select()) + 2

        plot_frame = self.plot_frames[current_selected_cluster]
        plot_data = self.plots[current_selected_cluster]
        fig = plot_data["fig"]
        ax = plot_data["ax"]

        centroids = self.plots[current_selected_cluster]["centroids"]
        adjusted_labels = self.plots[current_selected_cluster]["labels"]
        cmap = self.plots[current_selected_cluster]["cmap"]

        ax.clear()

        if self.use_dropdowns:
            x_index = self.data_PD.columns.tolist().index(self.x_var.get())
            y_index = self.data_PD.columns.tolist().index(self.y_var.get())
            z_index = self.data_PD.columns.tolist().index(self.z_var.get())
        else:
            feature_var_x = tkinter.StringVar(value=self.data_PD.columns[0])
            feature_var_y = tkinter.StringVar(value=self.data_PD.columns[1])
            x_index = self.data_PD.columns.tolist().index(feature_var_x.get())
            y_index = self.data_PD.columns.tolist().index(feature_var_y.get())
            if self.n_features > 2:
                feature_var_z = tkinter.StringVar(value=self.data_PD.columns[2])
                z_index = self.data_PD.columns.tolist().index(feature_var_z.get())

        def plot2d():
            for i, centroid in enumerate(centroids, start=1):
                cluster_points = self.data[adjusted_labels == i]
                color = cmap(i - 1)

                # Calculate the average distance of points in the cluster to the centroid and add a circle
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                avg_distance = np.mean(distances)
                ax.scatter(cluster_points[:, x_index], cluster_points[:, y_index], color=color, alpha=0.7,
                                label=f'Cluster {i}')
                ax.scatter(centroid[x_index], centroid[y_index], color=color, marker='x', s=100,
                                label=f'Centroid {i}')
                circle = Circle((centroid[x_index], centroid[y_index]), avg_distance, color=color, fill=False)
                ax.add_artist(circle)

            if selected_row_data is not None:
                ax.scatter(selected_row_data[0], selected_row_data[1], color="yellow", marker="o",
                                s=150, label="Selected Point", edgecolors="black", linewidths=1.5)

            ax.set_xlabel(self.data_PD.columns[x_index])
            ax.set_ylabel(self.data_PD.columns[y_index])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)

        def plot3d():
            for i, centroid in enumerate(centroids, start=1):
                cluster_points = self.data[adjusted_labels == i]
                color = cmap(i - 1)
                ax.scatter(cluster_points[:, x_index], cluster_points[:, y_index], cluster_points[:, z_index],
                                color=color, alpha=0.7, label=f'Cluster {i}')
                ax.scatter(centroid[x_index], centroid[y_index], centroid[z_index], color=color, marker='x',
                                s=100, label=f'Centroid {i}')

            if selected_row_data is not None:
                ax.scatter(selected_row_data[0], selected_row_data[1], selected_row_data[2],
                                color="yellow", marker="o", s=150, label="Selected Point", edgecolors="black",
                                linewidths=1.5)

            ax.set_xlabel(self.data_PD.columns[x_index])
            ax.set_ylabel(self.data_PD.columns[y_index])
            ax.set_zlabel(self.data_PD.columns[z_index])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)

        if self.n_features == 2:
            plot2d()
        elif self.n_features >= 3:
            plot3d()

        ax.set_title(f"K-Means Clustering with {len(centroids)} Clusters")
        plt.subplots_adjust(bottom=0.25)

        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def on_table_select(self, event, treeview):
        selected_item = treeview.selection()
        #get current selected cluster
        current_tab = self.notebook.index(self.notebook.select()) + 2
        sorted_dataset = self.plots[current_tab]["sorted_dataset"]

        if selected_item:
            treeview_index = treeview.index(selected_item[0])  # Get Treeview index

            x_value = sorted_dataset.iloc[treeview_index][self.x_var.get()]
            y_value = sorted_dataset.iloc[treeview_index][self.y_var.get()]
            z_value = sorted_dataset.iloc[treeview_index][self.z_var.get()] if self.z_var else None

            selected_row_data = [x_value, y_value, z_value] if self.z_var else [x_value, y_value]
            print(selected_row_data)
            self.update_plot(selected_row_data)











