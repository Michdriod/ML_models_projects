## Step-by-Step Guide for Credit Card Fraud Detection Project

### 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
```

### 2. Load and Explore Data
```python
df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.describe())
print(df.shape)
print(df.isnull().sum())  # Check for missing values
```

### 3. Split Data into Training and Testing Sets
```python
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### 4. Train an Initial XGBoost Model
```python
best_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
best_xgb.fit(X_train, y_train)
```

### 5. Train an Initial Random Forest Model
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### 6. Train an Initial Logistic Regression Model
```python
log_model = LogisticRegression(solver='liblinear', random_state=42)
log_model.fit(X_train, y_train)
```

### 7. Evaluate Model Performance
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_pred)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC-PR: {auc_pr:.4f}')

print("\n📌 XGBoost Performance:")
evaluate_model(best_xgb, X_test, y_test)

print("\n📌 Random Forest Performance:")
evaluate_model(rf_model, X_test, y_test)

print("\n📌 Logistic Regression Performance:")
evaluate_model(log_model, X_test, y_test)
```

### 8. Optimize the Decision Threshold
```python
y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label="Precision", linestyle="--")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs. Decision Threshold")
plt.legend()
plt.grid()
plt.show()
```

### 9. Create an Ensemble Model
```python
ensemble_model = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', rf_model), ('log', log_model)], voting='soft'
)
ensemble_model.fit(X_train, y_train)
```

### 10. Apply Stacking Instead of Averaging
```python
stacking_features = np.column_stack([
    best_xgb.predict_proba(X_test)[:, 1],
    rf_model.predict_proba(X_test)[:, 1],
    log_model.predict_proba(X_test)[:, 1]
])
stacking_model = LogisticRegression(solver='liblinear', random_state=42)
stacking_model.fit(stacking_features, y_test)
y_pred_stacked = stacking_model.predict(stacking_features)
```

### 11. SHAP Analysis for Feature Importance
```python
explainer_xgb = shap.Explainer(best_xgb)
shap_values_xgb = explainer_xgb(X_test)
shap.summary_plot(shap_values_xgb, X_test)

explainer_rf = shap.Explainer(rf_model)
shap_values_rf = explainer_rf(X_test)
shap.summary_plot(shap_values_rf, X_test)

explainer_log = shap.Explainer(log_model)
shap_values_log = explainer_log(X_test)
shap.summary_plot(shap_values_log, X_test)
```

### 12. Save Models for Deployment
```python
joblib.dump(best_xgb, "xgb_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(log_model, "log_model.pkl")
joblib.dump(stacking_model, "stacking_model.pkl")
```

### 13. Deploy the Model
```python
def load_and_predict(model_path, input_data):
    model = joblib.load(model_path)
    return model.predict(input_data)

new_data = np.array([X_test.iloc[0]])  # Example input
y_pred_new = load_and_predict("stacking_model.pkl", new_data)
print("Predicted Fraud Label:", y_pred_new)
```
