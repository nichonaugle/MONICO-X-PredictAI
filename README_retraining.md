
# Monico Predictive Maintenance - Retraining Utilities

This package provides tools for retraining the predictive maintenance model used to estimate the time until compressor failure using historical sensor data.

## Files
- `retrain_model.py`: Contains functions for preparing data and training the Random Forest model.

## Usage

### Step 1: Prepare and Load Data
Call `add_time_until_failure_from_folder()` to:
- Load multiple `.csv` files from a folder
- Resample sensor data over a time interval (e.g., 30 seconds)
- Compute and append a `Time_Until_Failure` column

```python
from retrain_model import add_time_until_failure_from_folder

folder_path = "path/to/your/csv/folder"
selected_features = ["feature1", "feature2", ...]  # Replace with your sensor feature list

df = add_time_until_failure_from_folder(folder_path, selected_features)
```

### Step 2: Train Random Forest Model
Once your data has a `Time_Until_Failure` column, call `train_random_forest_for_failure()`:

```python
from retrain_model import train_random_forest_for_failure

model, metrics = train_random_forest_for_failure(df, selected_features)
```

## Output
- Displays model evaluation (MAE and R² score)
- Shows a feature importance plot
- Returns the trained model and evaluation metrics

## Requirements
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Make sure to install required packages via `pip install -r requirements.txt` if needed.
