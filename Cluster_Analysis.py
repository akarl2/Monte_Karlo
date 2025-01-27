import tkinter
from itertools import combinations
from pyexpat import features
from tkinter import ttk, Frame, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import centroid
from sklearn.cluster import KMeans, DBSCAN
from matplotlib.patches import Circle
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class ClusterAnalysis:
    def __init__(self, master, data, n_clusters, random_starts, data_PD, full_dataset, cluster_method, optimal_cluster):
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
        self.silo_score = []

        if self.n_features > 3:
            self.use_dropdowns = True

        self.x_var = tkinter.StringVar(value=self.feature_options[0])
        self.y_var = tkinter.StringVar(value=self.feature_options[1])
        self.z_var = tkinter.StringVar(value=self.feature_options[2]) if len(self.feature_options) > 2 else None

        # Drop rows with missing values
        valid_indices = self.data_PD.dropna().index
        self.data = self.data[valid_indices]
        self.data_PD = self.data_PD.loc[valid_indices]
        self.full_dataset = self.full_dataset.loc[valid_indices]

        if optimal_cluster:
            #run find_best_features_and_clusters function
            best_result = self.find_best_features_and_clusters(self.data_PD)
            if best_result != "Canceled":
                messagebox.showinfo("Best Features and Clusters",
                                    f"Best Features: \n{', '.join(best_result['feature_names'])}\n\n"
                                    f"Best Clusters: {best_result['n_clusters']}\n\n"
                                    f"Silhouette Score: {best_result['silhouette_score']}")
            else:
                messagebox.showinfo("Canceled", "The process has been canceled")

        else:
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

    def find_best_features_and_clusters(self, data, cluster_range=(2, 10)):
        """
        Find the best subset of features and cluster count that maximizes the silhouette score.

        Args:
            data (np.ndarray): The dataset as a NumPy array (samples x features).
            cluster_range (tuple): Range of cluster numbers to test (default: 2 to 10).

        Returns:
            dict or str: Best result with feature indices, cluster count, and silhouette score,
                         or "Canceled" if the process is interrupted.
        """
        n_samples, n_features = self.data.shape
        best_result = {'feature_names': None, 'n_clusters': None, 'silhouette_score': -1}

        total_combinations = sum(
            len(range(cluster_range[0], cluster_range[1] + 1)) * len(
                list(combinations(range(n_features), n_sub_features)))
            for n_sub_features in range(2, n_features + 1)
        )

        # Create the Tkinter popup window
        popup = tkinter.Tk()
        popup.title("Progress")
        popup.geometry("400x225")
        label = tkinter.Label(popup, text="Optimizing features and clusters...")
        label.pack(pady=10)
        progress = ttk.Progressbar(popup, orient="horizontal", length=250, mode="determinate")
        progress.pack(pady=10)
        percent_label = tkinter.Label(popup, text="0.00% Complete")
        percent_label.pack(pady=10)
        progress["maximum"] = total_combinations

        cancel_flag = tkinter.BooleanVar(value=False)

        def cancel():
            cancel_flag.set(True)
            try:
                popup.destroy()
            except tkinter.TclError:
                pass  # Ignore errors if the window is already destroyed

        cancel_button = ttk.Button(popup, text="Cancel", command=cancel)
        cancel_button.pack(pady=10)
        popup.update()

        # Iterate through all subsets of features (2 to all features)
        current_combination = 0
        for n_sub_features in range(2, n_features + 1):
            for feature_indices in combinations(range(n_features), n_sub_features):
                if cancel_flag.get():  # Check if cancel was pressed
                    try:
                        popup.destroy()
                    except tkinter.TclError:
                        pass  # Ignore errors if the window is already destroyed
                    return "Canceled"  # Exit and indicate cancellation

                feature_data = self.data[:, feature_indices]  # Select subset of features

                # Test clustering for different cluster counts
                for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
                    if cancel_flag.get():  # Check if cancel was pressed
                        try:
                            popup.destroy()
                        except tkinter.TclError:
                            pass  # Ignore errors if the window is already destroyed
                        return "Canceled"  # Exit and indicate cancellation

                    try:
                        # Perform K-Means clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(feature_data)
                        labels = kmeans.labels_

                        # Compute silhouette score
                        score = silhouette_score(feature_data, labels)

                        # Check if this result is the best so far
                        if score > best_result['silhouette_score']:
                            best_result['feature_indices'] = feature_indices
                            best_result['feature_names'] = [self.data_PD.columns[i].replace('\xa0', ' ') for i in
                                                            feature_indices]
                            best_result['n_clusters'] = n_clusters
                            best_result['silhouette_score'] = score
                    except Exception as e:
                        # Handle exceptions (e.g., cluster number too high for the number of samples)
                        print(f"Error with features {feature_indices} and {n_clusters} clusters: {e}")
                    finally:
                        current_combination += 1
                        percent_complete = current_combination / total_combinations * 100
                        percent_label.config(text=f"{percent_complete:.2f}% Complete")
                        progress["value"] = current_combination
                        popup.update()

        try:
            popup.destroy()
        except tkinter.TclError:
            pass  # Ignore errors if the window is already destroyed

        return best_result


    def create_tabs(self):

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

        #defult to select the last tab in the notebook by -1
        self.notebook.select(self.notebook.index(self.notebook.tabs()[-1]))

    def DBSCAN_Clustering(self, tab):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.data)
        self.display_clustering(self.data, dbscan, tab)

    def KMeans_Clustering(self, tab):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.random_starts, random_state=0).fit(self.data)
        self.wcss.append(kmeans.inertia_)
        self.silo_score.append(silhouette_score(self.data, kmeans.labels_))
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
            rounded_row = [round(value, 4) if isinstance(value, (int, float)) else value for value in row.tolist()]
            tree.insert("", "end", values=rounded_row)

        # Bind the Treeview selection event to the function
        tree.bind("<<TreeviewSelect>>", lambda event: self.on_table_select(event, tree))

        def display_cluster_metrics():
            """Display elbow and silhouette plots side by side on a single tab."""

            # Create a single tab for both plots
            metrics_tab = ttk.Frame(self.notebook)
            self.notebook.add(metrics_tab, text="Cluster Metrics")

            # Create a PanedWindow to organize the two plots horizontally
            paned_window = ttk.PanedWindow(metrics_tab, orient="horizontal")
            paned_window.pack(fill="both", expand=True)

            # Create frames for each plot
            elbow_frame = ttk.Frame(paned_window, width=300, height=300)
            silhouette_frame = ttk.Frame(paned_window, width=300, height=300)
            paned_window.add(elbow_frame, weight=1)
            paned_window.add(silhouette_frame, weight=1)


            # Elbow Plot
            def plot_elbow():
                """Create the elbow plot."""
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.plot(range(2, len(self.wcss) + 2), self.wcss, marker="o", linestyle="-")
                ax.set_title("Elbow Method")
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
                ax.grid()

                canvas = FigureCanvasTkAgg(fig, master=elbow_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

            # Silhouette Score Plot
            def plot_silhouette():
                """Create the silhouette score plot."""
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.plot(range(2, len(self.silo_score) + 2), self.silo_score, marker="o", linestyle="-")
                ax.set_title("Silhouette Score")
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel("Silhouette Score")
                ax.grid()

                # Add labels indicating silhouette score thresholds
                ax.axhline(y=0.70, color='r', linestyle='--', label='Excellent Silhouette Score')
                ax.axhline(y=0.5, color='y', linestyle='--', label='Good Silhouette Score')
                ax.axhline(y=0.25, color='g', linestyle='--', label='Weak Silhouette Score')
                ax.legend()

                canvas = FigureCanvasTkAgg(fig, master=silhouette_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

            # Generate the plots
            plot_elbow()
            plot_silhouette()


        if hasattr(self, "wcss") and len(self.wcss) >= 9:
            display_cluster_metrics()

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
                ax.scatter(selected_row_data[0], selected_row_data[1], facecolor="none", marker="o",
                                s=150, label="Selected Point", edgecolors="red", linewidths=1.5)

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
                ax.scatter(selected_row_data[0], selected_row_data[1], selected_row_data[2], facecolor='none',
                                marker="o", s=150, label="Selected Point", edgecolors="red",
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
        current_tab = self.notebook.index(self.notebook.select()) + 2
        sorted_dataset = self.plots[current_tab]["sorted_dataset"]

        if selected_item:
            treeview_index = treeview.index(selected_item[0])  # Get Treeview index

            x_value = sorted_dataset.iloc[treeview_index][self.x_var.get()]
            y_value = sorted_dataset.iloc[treeview_index][self.y_var.get()]
            z_value = sorted_dataset.iloc[treeview_index][self.z_var.get()] if self.z_var else None

            selected_row_data = [x_value, y_value, z_value] if self.z_var else [x_value, y_value]

            self.update_plot(selected_row_data)











