import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
import joblib

# === Load and Enhance Dataset ===
df = pd.read_csv("behavior_features.csv")

# Temporal rolling features (simulated windowed behavior)
df['person_rolling_mean'] = df['num_person'].rolling(window=5, min_periods=1).mean()
df['bag_rolling_mean'] = df['num_bag'].rolling(window=5, min_periods=1).mean()
df['overlap_rolling_mean'] = df['overlap_count'].rolling(window=5, min_periods=1).mean()

# Drop NaN just in case
df = df.dropna()

# Features & Labels
features = [
    'num_person', 'num_bag', 'overlap_count', 'unique_classes',
    'person_rolling_mean', 'bag_rolling_mean', 'overlap_rolling_mean'
]
X = df[features]
y = df['is_shoplifting']

# === Handle Class Imbalance via Oversampling ===
df_balanced = pd.concat([X, y], axis=1)
majority = df_balanced[df_balanced.is_shoplifting == 0]
minority = df_balanced[df_balanced.is_shoplifting == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_upsampled])
X = df_balanced[features]
y = df_balanced['is_shoplifting']

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === XGBoost Classifier ===
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# === Evaluation ===
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# === Save Model & Scaler ===
joblib.dump(xgb_model, 'shoplifting_classifier_xgb.pkl')
joblib.dump(scaler, 'shoplifting_scaler_xgb.pkl')

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Normal", "Shoplifting"],
            yticklabels=["Normal", "Shoplifting"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix.png")
plt.show()

# === Feature Importance ===
importances = xgb_model.feature_importances_
plt.figure(figsize=(7, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - XGBoost")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
plt.show()
