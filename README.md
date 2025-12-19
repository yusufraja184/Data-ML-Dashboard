import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
df = None
model = None
y_test_global = None
y_pred_global = None
problem_global = None

def load_dataset():
    global df
    path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if path:
        df = pd.read_csv(path)
        text.delete("1.0", tk.END)
        text.insert(tk.END, df.head().to_string())
        update_columns()
        messagebox.showinfo("Success", "Dataset Loaded Successfully")

def update_columns():
    cols = df.columns.tolist()
    x_combo["values"] = cols
    y_combo["values"] = cols
    target_combo["values"] = cols

# ---------------- VISUALIZATION ----------------
def plot_graph():
    plot = plot_type.get()
    x = x_combo.get()
    y = y_combo.get()

    plt.figure(figsize=(6,4))

    if plot == "Histogram":
        sns.histplot(df[x], kde=True, color="#38bdf8")

    elif plot == "Boxplot":
        sns.boxplot(y=df[x], color="#22c55e")

    elif plot == "Scatter Plot":
        sns.scatterplot(x=df[x], y=df[y], color="#facc15")

    elif plot == "Heatmap":
        sns.heatmap(df.select_dtypes(include=np.number).corr(),
                    annot=True, cmap="coolwarm")

    plt.title(plot)
    plt.tight_layout()
    plt.show()

# ---------------- TRAIN MODEL ----------------
def train_model():
    global model, y_test_global, y_pred_global, problem_global

    target = target_combo.get()
    model_name = model_combo.get()
    problem = problem_type.get()
    problem_global = problem

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem == "Regression":
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "KNN":
            model = KNeighborsRegressor()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        result_label.config(text=f"R² Score: {score:.4f}")

    else:
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        result_label.config(text=f"Accuracy: {acc:.4f}")

    y_test_global = y_test
    y_pred_global = y_pred

    # Actual vs Predicted
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, color="#38bdf8")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

# ---------------- CONFUSION MATRIX ----------------
def show_confusion():
    if problem_global != "Classification":
        messagebox.showwarning("Warning", "Confusion Matrix only for Classification")
        return

    cm = confusion_matrix(y_test_global, y_pred_global)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ---------------- EXPORT MODEL ----------------
def export_model():
    if model is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".pkl")
    joblib.dump(model, path)
    messagebox.showinfo("Export", "Model saved successfully")

# ---------------- EXPORT PREDICTIONS ----------------
def export_predictions():
    df_pred = pd.DataFrame({
        "Actual": y_test_global,
        "Predicted": y_pred_global
    })
    path = filedialog.asksaveasfilename(defaultextension=".csv")
    df_pred.to_csv(path, index=False)
    messagebox.showinfo("Export", "Predictions exported")

# ---------------- UI ----------------
root = tk.Tk()
root.title("ML Dashboard")
root.geometry("1150x720")
root.config(bg="#0f172a")

tk.Label(root, text="Machine Learning Dashboard",
         font=("Arial", 22, "bold"),
         fg="#38bdf8", bg="#0f172a").pack(pady=10)

tk.Button(root, text="Upload Dataset", command=load_dataset,
          bg="#2563eb", fg="white", width=20).pack()

text = tk.Text(root, height=8, width=130)
text.pack(pady=10)

# -------- Visualization --------
frame1 = tk.Frame(root, bg="#0f172a")
frame1.pack(pady=10)

plot_type = tk.StringVar(value="Histogram")

ttk.Combobox(frame1, textvariable=plot_type,
             values=["Histogram", "Boxplot", "Scatter Plot", "Heatmap"]).grid(row=0, column=0)

x_combo = ttk.Combobox(frame1)
x_combo.grid(row=0, column=1)

y_combo = ttk.Combobox(frame1)
y_combo.grid(row=0, column=2)

tk.Button(frame1, text="Plot",
          command=plot_graph, bg="#22c55e").grid(row=0, column=3, padx=10)

frame2 = tk.Frame(root, bg="#0f172a")
frame2.pack(pady=15)

problem_type = tk.StringVar(value="Regression")
ttk.Combobox(frame2, textvariable=problem_type,
             values=["Regression", "Classification"]).grid(row=0, column=0)

target_combo = ttk.Combobox(frame2)
target_combo.grid(row=0, column=1)

model_combo = ttk.Combobox(frame2,
    values=["Linear Regression", "Logistic Regression", "KNN", "Random Forest"])
model_combo.grid(row=0, column=2)

tk.Button(frame2, text="Train Model",
          command=train_model, bg="#facc15").grid(row=0, column=3, padx=10)

tk.Button(frame2, text="Confusion Matrix",
          command=show_confusion, bg="#38bdf8").grid(row=0, column=4)

tk.Button(frame2, text="Export Model",
          command=export_model, bg="#22c55e").grid(row=0, column=5)

tk.Button(frame2, text="Export Predictions",
          command=export_predictions, bg="#ec4899").grid(row=0, column=6)

result_label = tk.Label(root, text="", fg="#22c55e",
                        bg="#0f172a", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
