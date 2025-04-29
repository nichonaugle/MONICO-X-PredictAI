
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def add_time_until_failure_from_folder(folder_path, selected_features, fault_col="Ramsey C4701E.Fault Relay", 
                                       time_col="Timestamp", interval=30):
    """
    Load and average CSV files from a folder, then add a Time_Until_Failure column.

    Args:
        folder_path (str): Folder containing CSV files.
        selected_features (list): List of features to average.
        fault_col (str): Column indicating failure (1 = fault).
        time_col (str): Timestamp column.
        interval (int): Averaging interval in seconds.

    Returns:
        pd.DataFrame: Combined DataFrame with Time_Until_Failure column.
    """
    all_data = pd.DataFrame()
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")])

    for file in files:
        df = pd.read_csv(file, encoding='utf-16', low_memory=False)
        if fault_col in df.columns:
            print(f"Processing file: {file}")
            df[time_col] = pd.to_datetime(df[time_col])
            df = df[selected_features + [fault_col, time_col]]
            df.set_index(time_col, inplace=True)

            agg_dict = {col: 'mean' for col in selected_features}
            agg_dict[fault_col] = 'max'

            df_resampled = df.resample(f"{interval}s").agg(agg_dict).reset_index()
            all_data = pd.concat([all_data, df_resampled], ignore_index=True)

    all_data.sort_values(time_col, inplace=True)
    all_data.reset_index(drop=True, inplace=True)

    # Add time until failure
    all_data["Time_Until_Failure"] = np.nan
    failure_indices = all_data[all_data[fault_col] == 1].index
    last_failure_time = None
    for idx in reversed(all_data.index):
        if idx in failure_indices:
            last_failure_time = all_data.loc[idx, time_col]
            all_data.loc[idx, "Time_Until_Failure"] = 0
        elif last_failure_time is not None:
            delta_hours = (last_failure_time - all_data.loc[idx, time_col]).total_seconds() / 3600
            all_data.loc[idx, "Time_Until_Failure"] = delta_hours

    return all_data


def train_random_forest_for_failure(df, selected_features, target_col="Time_Until_Failure", random_state=42, n_estimators=75):
    """
    Train and evaluate a Random Forest model to predict time until failure.

    Args:
        df (pd.DataFrame): DataFrame with features and target.
        selected_features (list): Columns to use as features.
        target_col (str): Target column for regression.
        random_state (int): Random seed for reproducibility.
        n_estimators (int): Number of trees.

    Returns:
        model (RandomForestRegressor), metrics (dict): Trained model and evaluation metrics.
    """
    df = df.dropna(subset=[target_col])
    df = df.dropna()

    split_index = int(len(df) * 0.8)
    X_train = df[selected_features].iloc[:split_index]
    X_test = df[selected_features].iloc[split_index:]
    y_train = df[target_col].iloc[:split_index]
    y_test = df[target_col].iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        bootstrap=True,
        random_state=random_state,
        verbose=2,
        max_features='log2',
        min_samples_split=12,
        min_samples_leaf=3,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    feature_importances = pd.DataFrame({"Feature": selected_features, "Importance": model.feature_importances_})
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(12, 36))
    sns.barplot(x=feature_importances["Importance"], y=feature_importances["Feature"])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top Features Affecting Time Until Failure")
    plt.xticks(rotation=0)
    plt.yticks(fontsize=8)
    plt.show()

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "R² Score": r2}
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    return model, metrics
