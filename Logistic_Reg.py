import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, r2_score, classification_report
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_model(X_train, y_train, model, df, parent):
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    model_loss = model.loss  # Get the loss method from the model

    # Create a grid of values for the two variables (1st and 2nd columns)
    x1_vals = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
    x2_vals = np.linspace(np.min(X_train[:, 1]), np.max(X_train[:, 1]), 100)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

    # Compute the surface (sigmoid for log_loss or linear for hinge)
    if model_loss == "log_loss":
        # Logistic Regression case (sigmoid surface)
        if X_train.shape[1] == 2:  # Linear terms only (e.g., x1, x2)
            z = 1 / (1 + np.exp(-(coef[0] * x1_grid + coef[1] * x2_grid + intercept)))
        elif X_train.shape[1] == 5:  # Quadratic terms (e.g., x1^2, x2^2)
            z = 1 / (1 + np.exp(-(coef[0] * x1_grid + coef[1] * x2_grid +
                                  coef[2] * x1_grid ** 2 + coef[3] * x1_grid * x2_grid +
                                  coef[4] * x2_grid ** 2 + intercept)))
        elif X_train.shape[1] == 9:  # Cubic terms (e.g., x1^3, x2^3, etc.)
            z = 1 / (1 + np.exp(-(coef[0] * x1_grid + coef[1] * x2_grid +
                                  coef[2] * x1_grid ** 2 + coef[3] * x1_grid * x2_grid +
                                  coef[4] * x2_grid ** 2 +
                                  coef[5] * x1_grid ** 3 + coef[6] * (x1_grid ** 2) * x2_grid +
                                  coef[7] * x1_grid * (x2_grid ** 2) + coef[8] * x2_grid ** 3 + intercept)))
        else:
            raise ValueError("Degree higher than 3 is not implemented in this plot function.")

        # Create the 3D plot for logistic regression
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_grid, x2_grid, z, cmap='coolwarm', alpha=0.6)
        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color="red", label="Training Data")

    elif model_loss == "hinge":
        # SVM case (linear decision boundary)
        if X_train.shape[1] == 2:  # Linear terms only
            z = coef[0] * x1_grid + coef[1] * x2_grid + intercept
        elif X_train.shape[1] == 5:  # Quadratic terms (x1^2, x2^2)
            z = coef[0] * x1_grid + coef[1] * x2_grid + coef[2] * x1_grid ** 2 + coef[3] * x1_grid * x2_grid + coef[4] * x2_grid ** 2 + intercept
        elif X_train.shape[1] == 9:  # Cubic terms
            z = (coef[0] * x1_grid + coef[1] * x2_grid +
                coef[2] * x1_grid ** 2 + coef[3] * x1_grid * x2_grid +
                coef[4] * x2_grid ** 2 +
                coef[5] * x1_grid ** 3 + coef[6] * (x1_grid ** 2) * x2_grid +
                coef[7] * x1_grid * (x2_grid ** 2) + coef[8] * x2_grid ** 3 + intercept)
        else:
            raise ValueError("Degree higher than 3 is not implemented for SVM decision boundary.")

        # Create the 3D plot for SVM
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_grid, x2_grid, z, cmap='coolwarm', alpha=0.6)

        # Map y_train to colors for different classes (red for class 0, green for class 1)
        colors = ['red' if label == 0 else 'green' for label in y_train]
        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color=colors, label="Training Data", edgecolor='k')
    else:
        raise ValueError("Unsupported loss function for 3D plotting.")

    # Set labels for the axes
    ax.set_xlabel(f"{df.columns[0]}")
    ax.set_ylabel(f"{df.columns[1]}")
    ax.set_zlabel("Probability" if model_loss == "log_loss" else "Score")
    ax.set_title(f"{model_loss.capitalize()} Model Decision Boundary")

    # Embed plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



def plot_2d_model(X_train, y_train, model, df, parent):
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    model_loss = model.loss  # Get the loss method from the model

    # Create a grid of values for the original X feature (first column of X_train before transformation)
    x_vals = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)

    # Handle logistic regression (sigmoid) and SVM (hinge) loss cases
    if model_loss == "log_loss":
        # Logistic Regression case (sigmoid function)
        if X_train.shape[1] == 1:
            z = 1 / (1 + np.exp(-(coef[0] * x_vals + intercept)))
        elif X_train.shape[1] == 2:
            z = 1 / (1 + np.exp(-(coef[0] * x_vals + coef[1] * x_vals + intercept)))
        elif X_train.shape[1] == 3:
            z = 1 / (1 + np.exp(-(coef[0] * x_vals + coef[1] * x_vals + coef[2] * x_vals + intercept)))
        else:
            raise ValueError("Only 1, 2, or 3 features are supported for 2D plotting with sigmoid.")

        # Create the 2D plot
        fig, ax = plt.subplots()
        ax.plot(x_vals, z, color="blue", label="Sigmoid Curve")

        # Scatter plot for the original X_train[:, 0] (before polynomial expansion)
        ax.scatter(X_train[:, 0], y_train, color="red", label="Training Data")

    elif model_loss == "hinge":
        # SVM case (linear decision boundary)
        if X_train.shape[1] == 1:
            z = coef[0] * x_vals + intercept
        elif X_train.shape[1] == 2:
            z = coef[0] * x_vals + coef[1] * x_vals + intercept
        elif X_train.shape[1] == 3:
            z = coef[0] * x_vals + coef[1] * x_vals + coef[2] * x_vals + intercept
        else:
            raise ValueError("Only 1, 2, or 3 features are supported for 2D plotting with hinge loss.")

        # Create the 2D plot for linear boundary
        fig, ax = plt.subplots()
        ax.plot(x_vals, z, color="blue", label="SVM Decision Boundary")

        # Map y_train to colors for the different classes (red for class 0, green for class 1)
        colors = ['red' if label == 0 else 'green' for label in y_train]
        ax.scatter(X_train[:, 0], y_train, color=colors, label="Training Data", edgecolor='k')
    else:
        raise ValueError("Unsupported loss function for 2D plotting.")

    # Scatter plot for the original X_train[:, 0] (before polynomial expansion)
    ax.set_xlabel(f"{df.columns[0]}")  # Label the original feature
    ax.set_ylabel("Probability" if model_loss == "log_loss" else "Score")
    ax.set_title(f"{model_loss.capitalize()} Model Decision Boundary")

    # Embed plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def display_results_in_window(X_train, y_train, model, df, train_accuracy, train_confusion_matrix,
                              train_r2, X_test=None, y_test=None, test_confusion_matrix=None,
                              feat_before_poly=None, degree=1, alpha=None, max_iter=None, loss=None, penalty=None):
    # Ensure degree is assigned a value, defaulting to 1 if not provided
    degree = degree if degree is not None else 1

    root = tk.Tk()
    root.title("Logistic Regression Results")

    # Set window size and make it resizable
    root.geometry("600x600")
    root.resizable(True, True)

    # Create a scrollable frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Display the model parameters in a table format using Treeview
    param_tree = ttk.Treeview(scrollable_frame, columns=("Parameter", "Value"), show="headings", height=6)
    param_tree.heading("Parameter", text="Parameter")
    param_tree.heading("Value", text="Value")

    # Insert the model parameters as rows in the table
    param_tree.insert("", "end", values=("Alpha (learning rate)", f"{alpha}"))
    param_tree.insert("", "end", values=("Max Iterations", f"{max_iter}"))
    param_tree.insert("", "end", values=("Loss Function", f"{loss}"))
    param_tree.insert("", "end", values=("Penalty", f"{penalty if penalty else 'None'}"))
    param_tree.insert("", "end", values=("Polynomial Degree", f"{degree}"))
    param_tree.insert("", "end", values=("Number of iterations", f"{model.n_iter_}"))

    # Pack the treeview at the top of the window
    param_tree.pack(pady=10, padx=10, fill=tk.X)

    # Plot based on the number of features
    if feat_before_poly == 1:  # 1 feature
        plot_2d_model(X_train, y_train, model, df, scrollable_frame)
    elif feat_before_poly == 2:  # 2 features
        plot_3d_model(X_train, y_train, model, df, scrollable_frame)

    # Display training accuracy and R^2 score
    accuracy_label = tk.Label(scrollable_frame, text=f"Training Accuracy: {train_accuracy:.4f}")
    accuracy_label.pack(pady=10)

    r2_label = tk.Label(scrollable_frame, text=f"Model R^2: {train_r2:.4f}")
    r2_label.pack(pady=10)

    # Display the labeled training confusion matrix
    fig_cm_train, ax_cm_train = plt.subplots()
    disp_train = ConfusionMatrixDisplay(confusion_matrix=train_confusion_matrix, display_labels=['0', '1'])
    disp_train.plot(ax=ax_cm_train, cmap=plt.cm.Blues)
    ax_cm_train.set_xlabel("Predicted")
    ax_cm_train.set_ylabel("Actual")

    # Embed training confusion matrix in the tkinter window
    canvas_cm_train = FigureCanvasTkAgg(fig_cm_train, master=scrollable_frame)
    canvas_cm_train.draw()
    canvas_cm_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # If test data is provided, display test accuracy, R^2 score, and confusion matrix
    if X_test is not None and y_test is not None and test_confusion_matrix is not None:
        # Display the labeled test confusion matrix
        fig_cm_test, ax_cm_test = plt.subplots()
        disp_test = ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix, display_labels=['0', '1'])
        disp_test.plot(ax=ax_cm_test, cmap=plt.cm.Blues)
        ax_cm_test.set_xlabel("Predicted")
        ax_cm_test.set_ylabel("Actual")

        # Embed test confusion matrix in the tkinter window
        canvas_cm_test = FigureCanvasTkAgg(fig_cm_test, master=scrollable_frame)
        canvas_cm_test.draw()
        canvas_cm_test.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        report = classification_report(y_test, model.predict(X_test))

        # Use a Text widget to display the report, with scrolling capabilities
        report_text = scrolledtext.ScrolledText(scrollable_frame, wrap=tk.NONE, height=10, width=70)
        report_text.insert(tk.END, report)  # Insert the classification report
        report_text.config(state=tk.DISABLED)  # Make the text box read-only
        report_text.pack(pady=10)

    # Display the model coefficients in a table
    coeffs_table = ttk.Treeview(scrollable_frame, columns=("Feature", "Coefficient"), show='headings')
    coeffs_table.heading("Feature", text="Feature")
    coeffs_table.heading("Coefficient", text="Coefficient")

    # Get model coefficients and add to table
    coefficients = model.coef_[0] if hasattr(model, 'coef_') else []
    intercept = model.intercept_[0] if hasattr(model, 'intercept_') else None

    # Fit PolynomialFeatures on the data to generate the names of the features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(df.iloc[:, :-1])  # Fit to the feature columns, excluding the target

    # Get the names of the polynomial features after fitting
    poly_feature_names = poly.get_feature_names_out(df.columns[:-1])  # Exclude target column

    # Add intercept as the first row in the table
    if intercept is not None:
        coeffs_table.insert("", tk.END, values=("Intercept", intercept))

    # Insert polynomial feature names and their corresponding coefficients into the table
    for feature, coef in zip(poly_feature_names, coefficients):
        coeffs_table.insert("", tk.END, values=(feature, coef))

    coeffs_table.pack(pady=10)

    # Input section for probability prediction based on new values
    input_label = tk.Label(scrollable_frame, text="Enter values for probability prediction or classification:")
    input_label.pack(pady=10)

    # Create input fields dynamically in a table-like format using pack (avoiding grid)
    entries = []
    input_frame = tk.Frame(scrollable_frame)  # Create a new frame for better organization
    input_frame.pack(pady=10)

    # Create input fields dynamically in a table-like format using pack
    entries = []
    for i in range(feat_before_poly):
        row_frame = tk.Frame(scrollable_frame)  # Create a new frame for each row
        row_frame.pack(fill=tk.X, pady=5)  # Pack the frame, fill horizontally
        label = tk.Label(row_frame, text=f"{df.columns[i]}:")
        label.pack(side=tk.LEFT, padx=10)  # Pack the label on the left side
        entry = tk.Entry(row_frame)
        entry.pack(side=tk.LEFT, padx=10)  # Pack the entry next to the label
        entries.append(entry)

    # Label to display the prediction result
    result_label = tk.Label(scrollable_frame, text="")
    result_label.pack(pady=10)

    def on_predict():
        # Get the input values from the entry fields
        input_values = []
        for entry in entries:
            try:
                value = float(entry.get())  # Convert to float
                input_values.append(value)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
                return

        # Convert the input values to a NumPy array
        X_new = np.array(input_values).reshape(1, -1)

        # If polynomial features were used, transform the input data
        if degree > 1:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_new = poly.fit_transform(X_new)

        # Check the loss function used and predict accordingly
        if loss == "log_loss":
            # Predict the probability using the trained model for log_loss
            prob = model.predict_proba(X_new)[0][1]  # Get the probability for class 1
            result_label.config(text=f"Predicted Probability: {prob:.4f}")
        elif loss == "hinge":
            # Predict the class using the trained model for hinge loss (SVM)
            predicted_class = model.predict(X_new)[0]  # Get the predicted class (0 or 1)
            result_label.config(text=f"Predicted Class: {predicted_class}")
        else:
            result_label.config(text="Unsupported loss function")

    # Button to trigger the prediction
    predict_button = tk.Button(scrollable_frame, text="Predict", command=on_predict)
    predict_button.pack(pady=10)

    root.mainloop()

# Function to run regression and plot the results
def run_regression(files, alpha=0.01, max_iter=10000000, loss="log_loss", penalty=None, test_size=None, degree=1, to_scale=False):
    df = files
    df = df.applymap(lambda x: x.replace('\xa0', ' ') if isinstance(x, str) else x)
    df = df.dropna(axis=0, how='any')
    df = df.apply(pd.to_numeric, errors='coerce')

    # Convert DataFrame to NumPy array
    data = df.to_numpy()

    # Select features and target variable
    X_data = data[:, :-1]  # All columns except the last (features)
    y_data = data[:, -1]   # Last column as the target variable (label)

    valid_indices = ~np.isnan(y_data) & ~np.any(np.isnan(X_data), axis=1)
    X_data = X_data[valid_indices]
    y_data = y_data[valid_indices]

    feat_before_poly = X_data.shape[1]

    # Transform features if necessary
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_data = poly.fit_transform(X_data)

    # Apply train/test split if test_size is provided
    if test_size:
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=42)
    else:
        X_train, y_train = X_data, y_data
        X_test, y_test = None, None

    # Initialize and train the model
    model = SGDClassifier(alpha=alpha, max_iter=max_iter, loss=loss, penalty=penalty)
    model.fit(X_train, y_train)

    # Make predictions on training data
    y_train_pred = model.predict(X_train)

    # Compute accuracy, confusion matrix, and R^2 score for training data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    if X_test is not None:
        # Make predictions on test data if test set exists
        y_test_pred = model.predict(X_test)

        # Compute accuracy, confusion matrix, and R^2 score for test data
        test_cm = confusion_matrix(y_test, y_test_pred)

        # Display results for both train and test sets
        display_results_in_window(X_train, y_train, model, df, train_accuracy, train_cm,
                                  train_r2, X_test=X_test, y_test=y_test,
                                  test_confusion_matrix=test_cm, feat_before_poly=feat_before_poly,
                                  degree=degree, alpha=alpha, max_iter=max_iter, loss=loss, penalty=penalty)
    else:
        # Display results for only the training set
        display_results_in_window(X_train, y_train, model, df, train_accuracy, train_cm, train_r2,
                                  feat_before_poly=feat_before_poly, degree=degree, alpha=alpha,
                                  max_iter=max_iter, loss=loss, penalty=penalty)