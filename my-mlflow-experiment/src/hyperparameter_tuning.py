"""
Hyperparameter Tuning with MLflow Tracking
===========================================

This file demonstrates:
- Grid search with MLflow
- Nested runs for organization
- Parent-child run relationships
- Best model selection
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# Set MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("heart-disease-hyperparameter-tuning")


def load_and_prep_data():
    """Load and preprocess data"""
    df = pd.read_csv('data/heart_disease.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def hyperparameter_tuning_random_forest():
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING: RANDOM FOREST")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Start PARENT run
    with mlflow.start_run(run_name="RF_Hyperparameter_Tuning") as parent_run:
        
        # Log parent run info
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("tuning_method", "GridSearch")
        mlflow.log_param("total_combinations", 
                        len(param_grid['n_estimators']) * 
                        len(param_grid['max_depth']) * 
                        len(param_grid['min_samples_split']) * 
                        len(param_grid['min_samples_leaf']))
        
        best_score = 0
        best_params = None
        
        # Manual grid search with MLflow tracking
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:
                    for min_leaf in param_grid['min_samples_leaf']:
                        
                        # Start CHILD run (nested)
                        with mlflow.start_run(nested=True) as child_run:
                            
                            # Create model with these params
                            model = RandomForestClassifier(
                                n_estimators=n_est,
                                max_depth=max_d,
                                min_samples_split=min_split,
                                min_samples_leaf=min_leaf,
                                random_state=42
                            )
                            
                            # Log parameters
                            mlflow.log_params({
                                'n_estimators': n_est,
                                'max_depth': max_d if max_d else 'None',
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf
                            })
                            
                            # Train
                            model.fit(X_train, y_train)
                            
                            # Evaluate
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            # Log metrics
                            mlflow.log_metrics({
                                'accuracy': accuracy,
                                'f1_score': f1
                            })
                            
                            # Track best
                            if accuracy > best_score:
                                best_score = accuracy
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_depth': max_d,
                                    'min_samples_split': min_split,
                                    'min_samples_leaf': min_leaf
                                }
                            
                            print(f"[INFO] n_est={n_est}, max_depth={max_d}, "
                                  f"min_split={min_split}, min_leaf={min_leaf} "
                                  f"→ Accuracy: {accuracy:.4f}")
        
        # Log best results to parent
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_accuracy", best_score)
        
        print(f"\n[SUCCESS] Best Accuracy: {best_score:.4f}")
        print(f"[SUCCESS] Best Params: {best_params}")
        
        return best_params, best_score


def hyperparameter_tuning_gradient_boosting():
    """Hyperparameter tuning for Gradient Boosting"""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING: GRADIENT BOOSTING")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    with mlflow.start_run(run_name="GB_Hyperparameter_Tuning") as parent_run:
        
        mlflow.log_param("model_type", "GradientBoosting")
        
        best_score = 0
        best_params = None
        
        for n_est in param_grid['n_estimators']:
            for lr in param_grid['learning_rate']:
                for max_d in param_grid['max_depth']:
                    for subsamp in param_grid['subsample']:
                        
                        with mlflow.start_run(nested=True):
                            
                            model = GradientBoostingClassifier(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_d,
                                subsample=subsamp,
                                random_state=42
                            )
                            
                            mlflow.log_params({
                                'n_estimators': n_est,
                                'learning_rate': lr,
                                'max_depth': max_d,
                                'subsample': subsamp
                            })
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            mlflow.log_metric('accuracy', accuracy)
                            
                            if accuracy > best_score:
                                best_score = accuracy
                                best_params = {
                                    'n_estimators': n_est,
                                    'learning_rate': lr,
                                    'max_depth': max_d,
                                    'subsample': subsamp
                                }
                            
                            print(f"[INFO] n_est={n_est}, lr={lr}, "
                                  f"max_depth={max_d}, subsample={subsamp} "
                                  f"→ Accuracy: {accuracy:.4f}")
        
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_accuracy", best_score)
        
        print(f"\n[SUCCESS] Best Accuracy: {best_score:.4f}")
        print(f"[SUCCESS] Best Params: {best_params}")
        
        return best_params, best_score


def main():
    """Run hyperparameter tuning"""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING WITH MLFLOW")
    print("="*70)
    
    # Tune Random Forest
    rf_params, rf_score = hyperparameter_tuning_random_forest()
    
    # Tune Gradient Boosting
    gb_params, gb_score = hyperparameter_tuning_gradient_boosting()
    
    # Summary
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)
    print(f"\nRandom Forest Best Accuracy: {rf_score:.4f}")
    print(f"Gradient Boosting Best Accuracy: {gb_score:.4f}")
    
    if rf_score > gb_score:
        print(f"\nWinner: Random Forest")
    else:
        print(f"\nWinner: Gradient Boosting")
    
    print("\n" + "="*70)
    print("View results in MLflow UI: mlflow ui")
    print("="*70)


if __name__ == "__main__":
    main()