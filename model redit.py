"""
Credit Card Fraud Detection System
---------------------------------
This script implements a complete machine learning pipeline for credit card fraud detection.
It includes data preprocessing, model training, evaluation, hyperparameter tuning, and model explainability.

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBClassifier

#######################
# STEP 1: DATA LOADING AND EXPLORATION
#######################
print("="*50)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*50)

# Load the credit card transaction dataset
df = pd.read_csv("C:\\Users\\USER\\Downloads\\creditcard.csv")

# Display the first few rows of the dataset
print("\n📊 Dataset Preview:")
print(df.head())

# Generate statistical summary of the dataset
print("\n📊 Statistical Summary:")
print(df.describe())

# Display information about the dataset structure
print("\n📊 Dataset Information:")
print(df.info())

# Check for missing values
print(f"\n📊 Missing values: {df.isnull().sum().sum()}")

# Analyze class distribution (fraudulent vs legitimate transactions)
print(f"\n📊 Class distribution:")
print(df['Class'].value_counts())
print(f"Percentage of fraudulent transactions: {df['Class'].mean()*100:.4f}%")

#######################
# STEP 2: DATA PREPROCESSING
#######################
print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING")
print("="*50)

# Separate features and target variable
X = df.drop(columns=['Class'])  # Features (all columns except Class)
y = df['Class']                 # Target variable (fraud indicator: 1=fraud, 0=legitimate)

# Split data into training and testing sets with stratified sampling to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,          # 80% training, 20% testing
    stratify=y,             # Maintain the same class distribution in both sets
    random_state=42         # For reproducibility
)

# Standardize features (mean=0, std=1) to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform it
X_test = scaler.transform(X_test)        # Transform test data using the same parameters

print(f"\n📊 Training set shape: {X_train.shape}")
print(f"📊 Testing set shape: {X_test.shape}")
print(f"📊 Positive class (fraud) in training set: {sum(y_train)}")
print(f"📊 Positive class (fraud) in testing set: {sum(y_test)}")

#######################
# STEP 3: MODEL TRAINING
#######################
print("\n" + "="*50)
print("STEP 3: MODEL TRAINING")
print("="*50)

print("\n🔍 Training multiple models for comparison...")

# Train XGBoost model (gradient boosting)
print("\n🔍 Training XGBoost model...")
best_xgb = XGBClassifier(
    use_label_encoder=False,  # Avoid warning
    eval_metric='logloss',    # Evaluation metric during training
    random_state=42           # For reproducibility
)
best_xgb.fit(X_train, y_train)

# Train Random Forest model (ensemble of decision trees)
print("\n🔍 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    random_state=42    # For reproducibility
)
rf_model.fit(X_train, y_train)

# Train Logistic Regression model (linear model for classification)
print("\n🔍 Training Logistic Regression model...")
log_model = LogisticRegression(
    solver='liblinear',  # Efficient for small datasets
    random_state=42      # For reproducibility
)
log_model.fit(X_train, y_train)

#######################
# STEP 4: MODEL EVALUATION
#######################
print("\n" + "="*50)
print("STEP 4: MODEL EVALUATION")
print("="*50)

# Function to evaluate model performance using multiple metrics
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using precision, recall, F1-score, and AUC-PR.
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test labels
    
    Returns:
    - None (prints evaluation metrics)
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_pred)
    
    # Print metrics
    print(f'Precision: {precision:.4f}')  # Proportion of true positives among predicted positives
    print(f'Recall: {recall:.4f}')        # Proportion of true positives identified (sensitivity)
    print(f'F1-Score: {f1:.4f}')          # Harmonic mean of precision and recall
    print(f'AUC-PR: {auc_pr:.4f}')        # Area under Precision-Recall curve

# Evaluate each model
print("\n📌 XGBoost Performance:")
evaluate_model(best_xgb, X_test, y_test)

print("\n📌 Random Forest Performance:")
evaluate_model(rf_model, X_test, y_test)

print("\n📌 Logistic Regression Performance:")
evaluate_model(log_model, X_test, y_test)

#######################
# STEP 5: HYPERPARAMETER TUNING
#######################
print("\n" + "="*50)
print("STEP 5: HYPERPARAMETER TUNING")
print("="*50)

print("\n🔧 Tuning XGBoost hyperparameters using Grid Search...")

# Define hyperparameter search space for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],       # Number of gradient boosted trees
    'max_depth': [3, 5, 7],               # Maximum depth of each tree
    'learning_rate': [0.01, 0.1, 0.2]     # Step size shrinkage to prevent overfitting
}

# Perform grid search with cross-validation
grid_xgb = GridSearchCV(
    best_xgb,                # Model to tune
    param_grid_xgb,          # Hyperparameter grid
    scoring='f1',            # Optimization metric
    cv=3                     # Number of cross-validation folds
)
grid_xgb.fit(X_train, y_train)

# Get the best model from grid search
best_xgb = grid_xgb.best_estimator_

# Print best parameters and score
print(f"\n🔧 Best XGBoost parameters: {grid_xgb.best_params_}")
print(f"🔧 Best F1 score from cross-validation: {grid_xgb.best_score_:.4f}")

# Evaluate tuned model
print("\n📌 Tuned XGBoost Performance:")
evaluate_model(best_xgb, X_test, y_test)

#######################
# STEP 6: ENSEMBLE LEARNING
#######################
print("\n" + "="*50)
print("STEP 6: ENSEMBLE LEARNING")
print("="*50)

print("\n🔄 Creating weighted voting ensemble from all models...")

# Create a weighted voting ensemble of all models
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),       # Tuned XGBoost model
        ('rf', rf_model),        # Random Forest model
        ('log', log_model)       # Logistic Regression model
    ],
    voting='soft',               # Use predicted probabilities for voting
    weights=[3, 2, 1]            # Weight models by their relative performance
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate the ensemble model
print("\n📌 Ensemble Model Performance:")
evaluate_model(ensemble_model, X_test, y_test)

#######################
# STEP 7: MODEL EXPLAINABILITY WITH SHAP
#######################
print("\n" + "="*50)
print("STEP 7: MODEL EXPLAINABILITY WITH SHAP")
print("="*50)

print("\n🔎 Generating SHAP explanations for model interpretability...")

# Initialize JavaScript visualization
shap.initjs()

# Function to create and save SHAP plots
def generate_shap_plots(model, model_name, X_train, X_test, y_test):
    """
    Generate and save SHAP plots for model explainability.
    
    Parameters:
    - model: Trained model to explain
    - model_name: Name of the model for file naming
    - X_train: Training features
    - X_test: Test features
    - y_test: Test labels
    """
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    
    # Calculate SHAP values for test set
    shap_values = explainer(X_test)
    
    # Create and save summary plot (feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{model_name}.png")
    plt.close()
    
    # Find a fraudulent transaction to explain
    idx = y_test[y_test == 1].index[0]
    
    # Create and save waterfall plot (individual prediction explanation)
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_values[idx])
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{model_name}.png")
    plt.close()
    
    # Create and save force plot (interactive explanation)
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values[idx].values, 
        X_test[idx]
    )
    shap.save_html(f"shap_force_{model_name}.html", force_plot)
    
    print(f"✅ SHAP plots for {model_name} saved successfully.")

# Generate SHAP plots for each model
generate_shap_plots(best_xgb, "xgboost", X_train, X_test, y_test)
generate_shap_plots(rf_model, "random_forest", X_train, X_test, y_test)
generate_shap_plots(log_model, "logistic_regression", X_train, X_test, y_test)

#######################
# STEP 8: DEPLOYMENT CONSIDERATIONS
#######################
print("\n" + "="*50)
print("STEP 8: DEPLOYMENT CONSIDERATIONS")
print("="*50)

print("""
📋 Key deployment considerations for this fraud detection model:

1. Model Monitoring:
   - Implement monitoring for model drift as transaction patterns change over time
   - Set up automated retraining pipeline to periodically update the model
   - Track performance metrics in production environment

2. Explainability Framework:
   - Use SHAP values to explain why specific transactions were flagged as fraudulent
   - Provide explanations to fraud analysts for investigation
   - Create a dashboard for visualization of model decisions

3. Feedback Loop:
   - Collect feedback from fraud analysts on false positives/negatives
   - Incorporate this feedback into model retraining
   - Continuously improve model performance based on real outcomes

4. Real-time Scoring:
   - Optimize model for low-latency prediction to process transactions in real-time
   - Implement appropriate scaling infrastructure for high-volume processing
   - Consider implementing A/B testing for model updates

5. Regulatory Compliance:
   - Ensure model decisions are explainable for regulatory requirements
   - Document model development, validation, and monitoring processes
   - Implement appropriate data security and privacy controls
""")

#######################
# STEP 9: SAVE FINAL MODEL
#######################
print("\n" + "="*50)
print("STEP 9: SAVE FINAL MODEL")
print("="*50)

# Save the ensemble model to disk for later use
joblib.dump(ensemble_model, 'fraud_detection_model.pkl')
print("\n✅ Model saved successfully as 'fraud_detection_model.pkl'!")

# Save the scaler for preprocessing new data
joblib.dump(scaler, 'fraud_detection_scaler.pkl')
print("✅ Scaler saved successfully as 'fraud_detection_scaler.pkl'!")

print("\n" + "="*50)
print("CREDIT CARD FRAUD DETECTION PIPELINE COMPLETE")
print("="*50)