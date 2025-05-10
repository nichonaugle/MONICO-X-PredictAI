import shap

target_col="Time_Until_Failure"
df = df_combined_30sec_wTimes # Data frame of data 
# Define split index (80% for training)
split_index = int(len(df) * 0.8)

# Drop rows with NaN values in the target column
df = df.dropna(subset=[target_col])

# Corrected Train/Test split
X_train = df[cleaned_features].iloc[:split_index]  # Explicitly select features
X_test = df[cleaned_features].iloc[split_index:]  # Ensure proper split

# Create SHAP explainer for the trained Random Forest model
explainer = shap.TreeExplainer(model)  # Optimized for tree-based models
sample_size = 10  # Or 500, whatever your system can handle
shap_values = explainer.shap_values(X_test.iloc[:sample_size])
shap.summary_plot(shap_values, X_test.iloc[:sample_size])

shap.initjs()  # Ensure SHAP interactive visualizations work

# Select first row as test sample
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0]))
