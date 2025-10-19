import pandas as pd
import torch

from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import GridSearch, DemographicParity

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the dataset
file_path = 'input/employers_data.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Dataset '{file_path}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found. Please check the file path.")
    exit()

# Create the target variable based on the 75th Percentile of the 'Salary' column
if 'Salary' in df.columns:
    salary_threshold = df['Salary'].quantile(0.75)
    df['High_Earner_Target'] = (df['Salary'] > salary_threshold).astype(int)
    print(f"Target variable created. High salary threshold: {salary_threshold:.2f}")
else:
    print("ERROR: 'Salary' column not found in the dataset.")
    exit()

# Encode categorical data (e.g., Education_Level) into numerical format
le = LabelEncoder()
df['Education_Level_Code'] = le.fit_transform(df['Education_Level'])

# Informative message answering the question: which tokenizer does it use?
# This pipeline operates on structured/tabular data and does not use any tokenizer.
# For categorical encoding, it uses scikit-learn's LabelEncoder for 'Education_Level'.
print("Tokenizer: None. This script uses LabelEncoder (scikit-learn) for categorical encoding; no NLP tokenizer is involved.")

# Define features (X), target (y), and sensitive attribute (e.g., Gender)
try:
    X = df[['Age', 'Experience_Years', 'Education_Level_Code']]
    y = df['High_Earner_Target']
    sensitive_features = df['Gender']
except KeyError as e:
    print(f"ERROR: Missing required column: {e}")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test = \
    train_test_split(X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y)

print("Data prepared and split into training and testing sets.")

# Train a standard logistic regression model
print("\n--- Training Standard Model (Unmitigated) ---")
unmitigated_model = LogisticRegression(solver='liblinear', random_state=42)
unmitigated_model.fit(X_train, y_train)
y_pred_unmitigated = unmitigated_model.predict(X_test)

# Define evaluation metrics
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}

# Evaluate the standard model
metric_frame_unmitigated = MetricFrame(metrics=metrics,
                                       y_true=y_test,
                                       y_pred=y_pred_unmitigated,
                                       sensitive_features=sensitive_features_test)

print("\nStandard Model Results (By Group):")
print(metric_frame_unmitigated.by_group)

# Train a bias-mitigated model using Fairlearn's GridSearch
print("\n--- Training Bias-Mitigated Model (Fairlearn GridSearch) ---")
constraint = DemographicParity()
mitigator = GridSearch(LogisticRegression(solver='liblinear', random_state=42),
                       constraints=constraint,
                       grid_size=40)

mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
y_pred_mitigated = mitigator.predict(X_test)

# Evaluate the mitigated model
metric_frame_mitigated = MetricFrame(metrics=metrics,
                                     y_true=y_test,
                                     y_pred=y_pred_mitigated,
                                     sensitive_features=sensitive_features_test)

print("\nBias-Mitigated Model Results (By Group):")
print(metric_frame_mitigated.by_group)

# Compare results between the standard and mitigated models
print("\n" + "="*40)
print("--- COMPARISON OF RESULTS ---")
print("="*40)

print("\nOverall Accuracy:")
print(f"Standard Model              : {metric_frame_unmitigated.overall['accuracy']:.3f}")
print(f"Bias-Mitigated Model        : {metric_frame_mitigated.overall['accuracy']:.3f}")

print("\nSelection Rates (High Earner Prediction Rate):")
print("\nStandard Model:")
print(metric_frame_unmitigated.by_group['selection_rate'])
print("\nBias-Mitigated Model:")
print(metric_frame_mitigated.by_group['selection_rate'])

print("\nDisparity in Selection Rates:")
print(f"Standard Model              : {metric_frame_unmitigated.difference(method='between_groups')['selection_rate']:.3f}")
print(f"Bias-Mitigated Model        : {metric_frame_mitigated.difference(method='between_groups')['selection_rate']:.3f}")
print("="*40)

# Select the best model manually based on accuracy or fairness
print("\n--- Selecting Best Model from GridSearch ---")
best_model = None
best_accuracy = 0

for predictor in mitigator.predictors_:
    y_pred = predictor.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = predictor

if best_model is None:
    print("ERROR: No valid model found in GridSearch predictors.")
    exit()

print(f"Best model selected with accuracy: {best_accuracy:.3f}")

# Save the best model to a .pth file
output_model_pth_path = 'output/bias_mitigated_model.pth'
try:
    torch.save(best_model, output_model_pth_path)
    print(f"Best bias-mitigated model saved successfully to '{output_model_pth_path}'.")
except Exception as e:
    print(f"ERROR: Failed to save the model in .pth format. {e}")

# Export the bias-mitigated model to ONNX format
print("\n--- Exporting Bias-Mitigated Model to ONNX Format ---")
try:
    # Define the input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

    # Convert the best model to ONNX format
    onnx_model = convert_sklearn(best_model, initial_types=initial_type)
    onnx_output_path = 'output/bias_mitigated_model.onnx'

    # Save the ONNX model to a file
    with open(onnx_output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Bias-mitigated model successfully exported to ONNX format at '{onnx_output_path}'.")
except Exception as e:
    print(f"ERROR: Failed to export the model to ONNX format. {e}")