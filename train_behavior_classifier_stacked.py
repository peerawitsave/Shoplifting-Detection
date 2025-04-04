import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# === Load Dataset and Generate Rolling Features ===
df = pd.read_csv("behavior_features.csv")

df['person_rolling_mean'] = df['num_person'].rolling(window=5, min_periods=1).mean()
df['bag_rolling_mean'] = df['num_bag'].rolling(window=5, min_periods=1).mean()
df['overlap_rolling_mean'] = df['overlap_count'].rolling(window=5, min_periods=1).mean()

df = df.dropna()

# === Features and Labels ===
features = [
    'num_person', 'num_bag', 'overlap_count', 'unique_classes',
    'person_rolling_mean', 'bag_rolling_mean', 'overlap_rolling_mean'
]
X = df[features]
y = df['is_shoplifting']

# === Class Balancing via Oversampling ===
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

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Define Base Models for Stacking ===
base_rf = RandomForestClassifier(random_state=42)
base_xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
base_lgbm = LGBMClassifier(random_state=42)

# === Stacked Ensemble Classifier ===
stacked_model = StackingClassifier(
    estimators=[
        ('rf', base_rf),
        ('xgb', base_xgb),
        ('lgbm', base_lgbm)
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=3,
    n_jobs=-1
)

# === Train and Evaluate Stacked Model ===
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)
y_proba = stacked_model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report: Stacked Model ===")
print(classification_report(y_test, y_pred))
print(f"Stacked ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# === Save Model and Scaler ===
joblib.dump(stacked_model, 'shoplifting_classifier_stacked.pkl')
joblib.dump(scaler, 'shoplifting_scaler_stacked.pkl')

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["Normal", "Shoplifting"],
            yticklabels=["Normal", "Shoplifting"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stacked Model")
plt.tight_layout()
plt.savefig("stacked_confusion_matrix.png")
plt.show()

# === Feature Importance (Averaged from Base Models) ===
# Fit each model independently to get importances
base_rf.fit(X_train, y_train)
base_xgb.fit(X_train, y_train)
base_lgbm.fit(X_train, y_train)

avg_importances = (
    base_rf.feature_importances_ +
    base_xgb.feature_importances_ +
    base_lgbm.feature_importances_
) / 3

plt.figure(figsize=(7, 5))
sns.barplot(x=avg_importances, y=features)
plt.title("Average Feature Importance - Stacked Model")
plt.tight_layout()
plt.savefig("stacked_feature_importance.png")
plt.show()
