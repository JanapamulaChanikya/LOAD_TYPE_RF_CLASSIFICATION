# Load Type Prediction using Random Forest Classifier

Overview:
-----------
This project predicts electricity load types (Light_Load, Medium_Load, Maximum_Load) from time-series power usage data using a Random Forest Classifier. The model is trained on a labeled dataset and evaluated using key performance metrics. It also predicts load types for the files in the `test/` folder.

Solution Approach:
--------------------
1. Data Loading:
   - Load the dataset using pandas.
   - Convert the 'Date_Time' column to datetime format.

2. Data Cleaning:
   - Fill missing values using the median value for each column.

3. Feature Engineering:
   - Label encode the 'Load_Type' column.
   - Extract 'Hour' and 'Month' features from 'Date_Time'.

4. Feature Scaling:
   - Standardize features using StandardScaler.

5. Train-Test Split:
   - Use the latest month as the test set and the remaining data as the training set.

6. Model Training:
   - Train a Random Forest Classifier using the training set.

7. Model Evaluation:
   - Evaluate the model on the test set using accuracy, precision, recall, and F1-score metrics.
   - Generate a confusion matrix for a detailed breakdown.

8. Predictions:
   - Predict the load types for the files in the `test/` folder and save the predictions.

Instructions to Run:
--------------------
1. Clone this repository or unzip the folder.
2. Ensure you have the required dependencies:
3. Navigate to the `src/` folder and run:
4. The script will:
- Train the model.
- Evaluate the model.
- Generate predictions for test files in the `test/` folder.
- Save the predictions to `test_predictions.csv`.

Per-Field Recall Metric Score:
------------------------------
| Class          | Recall |
|----------------|--------|
| Light_Load     | 0.91   |
| Maximum_Load   | 0.95   |
| Medium_Load    | 0.98   |

Overall Performance Metrics:
-----------------------------
- Accuracy: 0.9308
- Precision (macro): 0.9061
- Recall (macro): 0.9450
- F1 Score (macro): 0.9231

Files Included:
---------------
- `src/load_type_predictor.py` — Main Python script for data loading, preprocessing, training, evaluation, and predictions.
- `test/` — Folder containing the test dataset(s) (e.g., `sample_test_data.csv`).
- `requirements.txt` — List of dependencies required to run the code.
- `README.txt` — This file, containing the solution approach, instructions, and performance metrics.

Notes:
-------
- Please ensure the test data follows the same format as the training data.
- Predictions will be saved in the root directory as `test_predictions.csv`.
