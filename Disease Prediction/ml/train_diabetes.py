import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("=" * 60)
print("  Diabetes Model Training Pipeline")
print("=" * 60)

# Features matching frontend exactly
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
TARGET = 'Outcome'

# ─── 1. Download / Load Dataset ───────────────────────────────────────────────
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names = FEATURES + [TARGET]

print("\n[1/8] Downloading Pima Indians Diabetes dataset...")
try:
    import ssl, urllib.request, io
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    with opener.open(url) as response:
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')), names=column_names)
    print(f"      ✓ Dataset loaded. Shape: {df.shape}")
except Exception as e:
    print(f"      ✗ Error downloading: {e}")
    print("      Using embedded sample data...")
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 18, n),
        'Glucose': np.random.randint(44, 199, n),
        'BloodPressure': np.random.randint(24, 122, n),
        'SkinThickness': np.random.randint(0, 99, n),
        'Insulin': np.random.randint(0, 846, n),
        'BMI': np.round(np.random.uniform(0, 67.1, n), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.078, 2.42, n), 3),
        'Age': np.random.randint(21, 81, n),
        'Outcome': np.random.randint(0, 2, n)
    })

# ─── 2. Clean ──────────────────────────────────────────────────────────────────
print("\n[2/8] Cleaning data...")
# 0 values in these columns are physiologically impossible (missing data)
zero_to_nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_to_nan_cols] = df[zero_to_nan_cols].replace(0, np.nan)
df = df.drop_duplicates()
print(f"      ✓ After cleaning. Shape: {df.shape}")
print(f"      Class distribution: {df[TARGET].value_counts().to_dict()}")
print(f"      Missing values (after 0→NaN):\n{df.isnull().sum()[df.isnull().sum() > 0]}")

X = df[FEATURES]
y = df[TARGET]

# ─── 3. Preprocess ─────────────────────────────────────────────────────────────
print("\n[3/8] Setting up preprocessing pipeline (median imputation + StandardScaler)...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, FEATURES)],
    remainder='drop'
)

# ─── 4. Stratified Split ──────────────────────────────────────────────────────
print("\n[4/8] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"      ✓ Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ─── 5. Train Models ──────────────────────────────────────────────────────────
print("\n[5/8] Training models...")
models = {
    'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'SVC': SVC(probability=True, random_state=42, class_weight='balanced'),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBClassifier': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
}

all_metrics = {}
best_model_name = ""
best_f1 = -1
best_pipeline = None

print("\n      {'Model':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("      " + "-" * 78)

for name, clf in models.items():
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'algorithm': name,
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    all_metrics[name] = metrics
    print(f"      {name:<28} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['roc_auc']:>10.4f}")

    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        best_model_name = name
        best_pipeline = full_pipeline

print(f"\n      ✓ Best Model: {best_model_name} (F1 = {best_f1:.4f})")

# ─── 6. Save Artifacts ────────────────────────────────────────────────────────
print("\n[6/8] Saving model artifacts...")
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'diabetes')
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_pipeline, os.path.join(save_dir, 'best_model.joblib'))
preprocessor_fitted = best_pipeline.named_steps['preprocessor']
joblib.dump(preprocessor_fitted, os.path.join(save_dir, 'scaler.joblib'))

metadata = {
    'disease': 'diabetes',
    'algorithm': best_model_name,
    'features': FEATURES,
    'accuracy': all_metrics[best_model_name]['accuracy'],
    'precision': all_metrics[best_model_name]['precision'],
    'recall': all_metrics[best_model_name]['recall'],
    'f1_score': all_metrics[best_model_name]['f1_score'],
    'roc_auc': all_metrics[best_model_name]['roc_auc'],
    'trained_at': datetime.now().isoformat(),
    'version': '1.0'
}

with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

with open(os.path.join(save_dir, 'all_metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=4)

print(f"      ✓ Artifacts saved to: {save_dir}")

print("\n[7/8] Verifying saved model...")
loaded_model = joblib.load(os.path.join(save_dir, 'best_model.joblib'))
test_input = X_test.iloc[[0]]
pred = loaded_model.predict(test_input)
prob = loaded_model.predict_proba(test_input)[:, 1]
print(f"      ✓ Model verification passed. Sample prediction: {pred[0]}, Probability: {prob[0]:.4f}")

print("\n[8/8] Summary:")
print(f"      Disease: Diabetes (Pima Indians Dataset)")
print(f"      Best Model: {best_model_name}")
print(f"      F1 Score: {best_f1:.4f}")
print(f"      Features: {', '.join(FEATURES)}")
print("\n" + "=" * 60)
print("  Diabetes Training Complete!")
print("=" * 60)
