"""
MLflow Experiment Tracking for ML Pipeline
==========================================

This extends the original MLflow lab by:
- Using custom heart disease dataset
- Tracking multiple models with detailed metrics
- Adding custom visualizations and artifacts
- Implementing model registry integration
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ============================================
# CONFIGURATION
# ============================================

# Set MLflow tracking URI (local directory)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
EXPERIMENT_NAME = "heart-disease-classification"
mlflow.set_experiment(EXPERIMENT_NAME)

print("="*70)
print("MLFLOW EXPERIMENT TRACKING")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print("="*70)


# ============================================
# DATA LOADING
# ============================================

def create_sample_dataset():
    """Creates synthetic heart disease dataset"""
    print("\n[INFO] Creating synthetic dataset...")
    
    np.random.seed(42)
    n_samples = 500  # Increased from 300
    
    # Create synthetic features
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target with logic
    df['target'] = (
        ((df['age'] > 55) & (df['chol'] > 250)) | 
        (df['cp'] > 2) | 
        (df['thalach'] < 120)
    ).astype(int)
    
    # Save dataset
    df.to_csv('data/heart_disease.csv', index=False)
    print(f"[INFO] Dataset created: {df.shape}")
    
    return df


def load_data():
    """Loads dataset (creates if doesn't exist)"""
    try:
        df = pd.read_csv('data/heart_disease.csv')
        print(f"[INFO] Dataset loaded: {df.shape}")
    except FileNotFoundError:
        df = create_sample_dataset()
    
    return df


def preprocess_data(df):
    """Preprocess data and return train/test splits"""
    print("\n[INFO] Preprocessing data...")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================
# MLFLOW TRACKING FUNCTIONS
# ============================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add AUC if probabilities available
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['roc_auc'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Creates and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    filename = f'outputs/confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Create and save ROC curve plot"""
    if y_pred_proba is None:
        return None
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        filename = f'outputs/roc_curve_{model_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    except:
        return None


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, 
                        hyperparameters=None):
    """
    Train model and log everything to MLflow
    
    This is the CORE of MLflow tracking - we log:
    - Parameters (hyperparameters)
    - Metrics (accuracy, precision, etc.)
    - Artifacts (plots, models)
    - Tags (metadata)
    """
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        
        # ========================================
        # 1. LOG PARAMETERS
        # ========================================
        
        # Log model hyperparameters
        params = model.get_params()
        mlflow.log_params(params)
        
        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # ========================================
        # 2. TRAIN MODEL
        # ========================================
        
        import time
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # ========================================
        # 3. MAKE PREDICTIONS
        # ========================================
        
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # ========================================
        # 4. LOG METRICS
        # ========================================
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)
        
        # Print metrics
        print(f"\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:15} : {value:.4f}")
        
        # ========================================
        # 5. LOG ARTIFACTS (Plots)
        # ========================================
        
        # Confusion Matrix
        cm_file = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_file)
        
        # ROC Curve
        roc_file = plot_roc_curve(y_test, y_pred_proba, model_name)
        if roc_file:
            mlflow.log_artifact(roc_file)
        
        # ========================================
        # 6. LOG MODEL
        # ========================================
        
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=None  # We'll register best model later
        )
        
        # ========================================
        # 7. LOG TAGS (Metadata)
        # ========================================
        
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("dataset", "heart_disease")
        mlflow.set_tag("author", "Your Name")
        mlflow.set_tag("version", "1.0")
        
        print(f"\n[INFO] Run logged to MLflow")
        print(f"[INFO] Run ID: {mlflow.active_run().info.run_id}")
    
    return metrics


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def main():
    """Main training pipeline with MLflow tracking"""
    
    print("\n" + "="*70)
    print("STARTING MLFLOW EXPERIMENT")
    print("="*70)
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train all models and log to MLflow
    results = {}
    
    for model_name, model in models.items():
        metrics = train_and_log_model(
            model, model_name, X_train, X_test, y_train, y_test
        )
        results[model_name] = metrics
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    
    print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        marker = "" if model_name == best_model else "  "
        print(
            f"{marker} {model_name:<23} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['f1_score']:>10.4f} "
            f"{metrics.get('roc_auc', 0.0):>10.4f}"
        )
    
    print("-" * 70)
    print(f"\nBest Model: {best_model}")
    print(f"Accuracy: {results[best_model]['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("VIEW RESULTS IN MLFLOW UI")
    print("="*70)
    print("\nRun this command:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")
    print("="*70)


if __name__ == "__main__":
    main()