import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    if not os.path.exists("reports"):
        os.makedirs("reports")
        print("Created base reports directory: reports")

    try:
        df = pd.read_csv("data/raw/data.csv")
        print("Data loaded successfully from data/raw/data.csv")
    except FileNotFoundError:
        print("Error: The file at data/raw/data.csv was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    if "readmitted" not in df.columns:
        print("\nError: The target column 'readmitted' is not present in the dataset.")
        sys.exit(1)
    else:
        print("\nTarget Variable: 'readmitted'")

    if not os.path.exists("reports/plots"):
        os.makedirs("reports/plots")
        print("Created directory: reports/plots")

    print("\nPlotting Correlation Matrix...")
    plt.figure(figsize=(12, 10))

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["number"])

    # Compute the correlation matrix on numeric columns
    corr = numeric_cols.corr()

    # Plot the correlation heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("reports/plots/correlation_matrix.png")
    plt.close()
    print("Correlation matrix saved to reports/plots/correlation_matrix.png")

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numerical_cols:
        print(f"\nPlotting distribution for '{col}'...")
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"reports/plots/{col}_distribution.png")
        plt.close()
        print(f"Distribution plot for '{col}' saved to reports/plots/{col}_distribution.png")

    feature_cols = [col for col in df.columns if col != "readmitted"]
    for col in feature_cols:
        print(f"\nPlotting relationship between 'readmitted' and '{col}'...")
        plt.figure(figsize=(8, 6))
        if df[col].dtype in ["int64", "float64"]:
            sns.boxplot(x="readmitted", y=col, data=df)
        else:
            sns.countplot(x=col, hue="readmitted", data=df)
        plt.title(f"readmitted vs {col}")
        plt.xlabel(col)
        plt.ylabel("Count" if df[col].dtype == "object" else col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"reports/plots/{col}_vs_readmitted.png")
        plt.close()
        print(f"Target relationship plot for '{col}' saved to reports/plots/{col}_vs_readmitted.png")

    if not os.path.exists("reports/cleanlab"):
        os.makedirs("reports/cleanlab")
        print("Created directory: reports/cleanlab")

    print("\nDetecting label issues using Cleanlab...")

    le = LabelEncoder()
    if df["readmitted"].dtype == "object" or df["readmitted"].dtype.name == "category":
        df["readmitted"] = le.fit_transform(df["readmitted"])
        print("Encoded target variable 'readmitted' to numerical values.")

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    # Use get_dummies with sparse output
    X_encoded = pd.get_dummies(X, drop_first=True, sparse=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reduce the number of trees, limit the maximum depth, and use all CPU cores
    classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Reduce the number of cross-validation folds in Cleanlab
    clean_learning_model = CleanLearning(clf=classifier, cv_n_folds=3)
    clean_learning_model.fit(X_train, y_train)

    y_pred = clean_learning_model.predict(X_test)

    # Update the find_label_issues function call
    label_issues = find_label_issues(
        labels=y_test,
        pred_probs=clean_learning_model.predict_proba(X_test),
        return_indices_ranked_by="self_confidence",
        filter_by="prune_by_noise_rate",
    )

    problematic_indices = label_issues
    print(f"Number of potential label issues detected: {len(problematic_indices)}")

    label_issues_df = df.iloc[X_test.index[problematic_indices]].copy()
    label_issues_df["predicted_label"] = y_pred[problematic_indices]
    label_issues_df["cleanlab_confidence"] = clean_learning_model.predict_proba(X_test)[
        problematic_indices
    ].max(axis=1)

    label_issues_df.to_csv("reports/cleanlab/cleanlab_report.csv", index=False)
    print("Cleanlab report saved to reports/cleanlab/cleanlab_report.csv")

    plt.figure(figsize=(10, 6))
    sns.histplot(label_issues_df["cleanlab_confidence"], bins=30, kde=True)
    plt.title("Confidence Scores of Detected Label Issues")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/plots/cleanlab_confidence_scores.png")
    plt.close()
    print("Confidence scores plot saved to reports/plots/cleanlab_confidence_scores.png")

    print("\nEDA Completed Successfully! All reports and plots are saved in the 'reports/' directory.")

if __name__ == "__main__":
    main()