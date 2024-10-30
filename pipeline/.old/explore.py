import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Set the style and font settings
plt.style.use("seaborn-v0_8")
sns.set_theme()

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16


def save_plot(
    plt, filename: str, bbox_inches: str = "tight", pad_inches: float = 0.1
) -> None:
    """Save plot with consistent parameters."""
    plt.tight_layout()
    plt.savefig(
        filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=300, format="png"
    )
    plt.close()


def analyze_label_issues(label_issues_df: pd.DataFrame) -> Dict:
    """Analyze the label issues in detail and return insights."""
    total_issues = len(label_issues_df)

    # Calculate confidence statistics
    confidence_stats = {
        "mean_confidence": label_issues_df["cleanlab_confidence"].mean(),
        "median_confidence": label_issues_df["cleanlab_confidence"].median(),
        "min_confidence": label_issues_df["cleanlab_confidence"].min(),
        "max_confidence": label_issues_df["cleanlab_confidence"].max(),
    }

    # Analyze misclassification patterns
    misclassification_patterns = pd.crosstab(
        label_issues_df["actual_label"],
        label_issues_df["predicted_label"],
        margins=True,
    )

    # Calculate percentage of issues relative to dataset size
    dataset_size = len(label_issues_df)
    issue_percentage = (total_issues / dataset_size) * 100

    # Identify high confidence disagreements
    high_confidence_threshold = 0.9
    high_confidence_issues = label_issues_df[
        label_issues_df["cleanlab_confidence"] >= high_confidence_threshold
    ]

    return {
        "total_issues": total_issues,
        "confidence_stats": confidence_stats,
        "misclassification_patterns": misclassification_patterns,
        "issue_percentage": issue_percentage,
        "high_confidence_issues": len(high_confidence_issues),
    }


def create_cleanlab_visualizations(
    label_issues_df: pd.DataFrame, save_dir: str
) -> None:
    """Create detailed visualizations for Cleanlab results."""
    # Confidence distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=label_issues_df, x="cleanlab_confidence", bins=30, kde=True)
    plt.title("Distribution of Confidence Scores in Label Issues")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    save_plot(plt, f"{save_dir}/cleanlab_confidence_distribution.png")

    # Confusion matrix for label issues
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(
        label_issues_df["actual_label"], label_issues_df["predicted_label"]
    )
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix of Label Issues")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    save_plot(plt, f"{save_dir}/cleanlab_confusion_matrix.png")

    # Feature importance visualization (if available)
    if isinstance(label_issues_df.get("feature_importance", None), pd.Series):
        plt.figure(figsize=(12, 6))
        label_issues_df["feature_importance"].sort_values(ascending=True).plot(
            kind="barh"
        )
        plt.title("Feature Importance in Label Issues")
        plt.xlabel("Importance Score")
        save_plot(plt, f"{save_dir}/cleanlab_feature_importance.png")


def generate_cleanlab_report(analysis_results: Dict, save_path: str) -> None:
    """Generate a detailed report of the Cleanlab analysis."""
    with open(save_path, "w") as f:
        f.write("Cleanlab Analysis Report\n")
        f.write("=======================\n\n")

        f.write("1. Overall Statistics\n")
        f.write(f"Total label issues detected: {analysis_results['total_issues']}\n")
        f.write(
            f"Percentage of dataset: {analysis_results['issue_percentage']:.2f}%\n\n"
        )

        f.write("2. Confidence Statistics\n")
        for metric, value in analysis_results["confidence_stats"].items():
            f.write(f"{metric}: {value:.3f}\n")
        f.write("\n")

        f.write("3. Misclassification Patterns\n")
        f.write(str(analysis_results["misclassification_patterns"]))
        f.write("\n\n")

        f.write("4. High Confidence Issues\n")
        f.write(
            f"Number of high confidence issues: {analysis_results['high_confidence_issues']}\n"
        )

        f.write("\n5. Recommendations\n")
        f.write("- Review high confidence issues first\n")
        f.write("- Focus on patterns in misclassification matrix\n")
        f.write("- Consider data quality in areas with highest issue rates\n")


def main():
    # Create reports directories if they don't exist
    for dir_path in ["reports", "reports/plots", "reports/cleanlab"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # Load both raw and processed data
    try:
        raw_df = pd.read_parquet("data/raw/data.parquet")
        processed_df = pd.read_parquet("data/interim/data_cleaned.parquet")
        print("Data loaded successfully")

        # Compare class distributions
        print("\nRaw data class distribution:")
        print(raw_df["readmitted"].value_counts())
        print("\nProcessed data class distribution:")
        print(processed_df["readmitted"].value_counts())

        # If processed data has issues, use raw data
        if len(processed_df["readmitted"].unique()) < 2:
            print(
                "\nWARNING: Processed data has lost class information. Using raw data instead."
            )
            df = raw_df.copy()

            # Perform minimal preprocessing for exploration
            if df["readmitted"].dtype == object:
                df["readmitted"] = df["readmitted"].map({">30": 0, "<30": 1, "NO": 0})
        else:
            df = processed_df.copy()

    except FileNotFoundError as e:
        print(f"Error: Required data files not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        sys.exit(1)

    # Basic data info
    print(f"\nDataset shape: {df.shape}")
    print("\nData types of each variable:")
    print(df.dtypes)

    # Store original readmitted values
    original_readmitted = df["readmitted"].copy()

    # Convert 'readmitted' to categorical for plotting
    df["readmitted"] = df["readmitted"].map({1: "Readmitted", 0: "Not Readmitted"})

    # Distribution of Readmission
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x="readmitted", data=df)
    plt.title("Distribution of Readmission")
    plt.xlabel("Readmission Status")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )
    save_plot(plt, "reports/plots/readmitted_distribution.png")

    # Time in Hospital vs Readmission
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="readmitted", y="time_in_hospital", data=df)
    plt.title("Time in Hospital vs Readmission")
    plt.xlabel("Readmission Status")
    plt.ylabel("Time in Hospital (days)")
    save_plot(plt, "reports/plots/time_in_hospital_vs_readmission.png")

    # Age and Readmission
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(y="age", hue="readmitted", data=df)
    plt.title("Age Distribution by Readmission Status")
    plt.xlabel("Count")
    plt.ylabel("Age Group")
    for p in ax.patches:
        width = p.get_width()
        plt.text(
            width,
            p.get_y() + p.get_height() / 2.0,
            f"{int(width)}",
            ha="left",
            va="center",
        )
    save_plot(plt, "reports/plots/age_vs_readmission.png")

    # Service Utilization Analysis
    if "service_utilization" not in df.columns:
        df["service_utilization"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="readmitted", y="service_utilization", data=df)
    plt.title("Service Utilization by Readmission Status")
    plt.xlabel("Readmission Status")
    plt.ylabel("Total Service Utilization")
    save_plot(plt, "reports/plots/service_utilization_vs_readmission.png")

    # Correlation Matrix for Numeric Variables
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True, center=0
    )
    plt.title("Correlation Matrix of Numeric Variables")
    save_plot(plt, "reports/plots/correlation_matrix.png")

    # Cleanlab Analysis
    print("\nPerforming Cleanlab analysis...")

    # Prepare data for Cleanlab
    df_encoded = df.copy()
    df_encoded["readmitted"] = original_readmitted

    print("\nClass distribution before encoding:")
    print(df_encoded["readmitted"].value_counts())

    # Convert categorical variables to numeric
    categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop("readmitted", axis=1)
    y = df_encoded["readmitted"]

    print("\nUnique classes in target variable:", np.unique(y))
    print("Class distribution:")
    print(y.value_counts())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining set class distribution:")
    print(y_train.value_counts())
    print("\nTest set class distribution:")
    print(y_test.value_counts())

    # Initialize and train the model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Initialize Cleanlab
    cl = CleanLearning(clf=clf)
    cl.fit(X_train, y_train)

    # Find label issues
    pred_probs = cl.predict_proba(X_test)
    label_issues = find_label_issues(
        labels=y_test.values,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Create detailed label issues dataframe
    label_issues_df = pd.DataFrame(
        {
            "actual_label": y_test.iloc[label_issues],
            "predicted_label": cl.predict(X_test)[label_issues],
            "cleanlab_confidence": pred_probs[label_issues].max(axis=1),
        }
    )

    # Get feature importance
    feature_importance = pd.Series(cl.clf.feature_importances_, index=X.columns)

    # Add feature importance to analysis results
    analysis_results = analyze_label_issues(label_issues_df)
    analysis_results["feature_importance"] = feature_importance

    # Create visualizations
    create_cleanlab_visualizations(label_issues_df, "reports/cleanlab")

    # Save feature importance
    feature_importance.to_csv("reports/cleanlab/feature_importance.csv")

    # Generate detailed report
    generate_cleanlab_report(analysis_results, "reports/cleanlab/detailed_analysis.txt")

    # Save problematic cases
    label_issues_df.to_csv("reports/cleanlab/problematic_cases.csv", index=True)

    print("\nAnalysis completed! Check the reports directory for detailed results.")
    print(
        f"Found {len(label_issues)} potential label issues ({(len(label_issues)/len(y_test)*100):.2f}% of test set)"
    )
    print(f"Detailed analysis saved to reports/cleanlab/detailed_analysis.txt")
    print(f"Problematic cases saved to reports/cleanlab/problematic_cases.csv")
    print(f"Feature importance saved to reports/cleanlab/feature_importance.csv")


if __name__ == "__main__":
    main()
