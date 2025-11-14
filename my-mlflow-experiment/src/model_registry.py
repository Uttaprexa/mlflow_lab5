"""
MLflow Model Registry
=====================
Register and manage model versions
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")


def register_best_model():
    """Find best model and register it"""
    
    print("\n" + "="*70)
    print("MODEL REGISTRY")
    print("="*70)
    
    client = MlflowClient()
    
    # Search for best model
    experiment = mlflow.get_experiment_by_name("heart-disease-classification")
    
    if not experiment:
        print("[ERROR] Experiment not found")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        print("[ERROR] No runs found")
        return
    
    best_run = runs.iloc[0]
    run_id = best_run['run_id']
    accuracy = best_run['metrics.accuracy']
    
    print(f"\nBest Model:")
    print(f"  Run ID: {run_id}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Register model
    model_name = "heart-disease-classifier"
    model_uri = f"runs:/{run_id}/model"
    
    print(f"\nRegistering model: {model_name}")
    
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"[SUCCESS] Model registered!")
        print(f"  Name: {model_name}")
        print(f"  Version: {model_version.version}")
        
        # Add description
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Heart disease classifier with {accuracy:.2%} accuracy. "
                       f"Trained on synthetic heart disease dataset."
        )
        
        # Transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        print(f"[SUCCESS] Model transitioned to Staging")
        
        return model_version
        
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return None


def list_registered_models():
    """List all registered models"""
    
    print("\n" + "="*70)
    print("REGISTERED MODELS")
    print("="*70)
    
    client = MlflowClient()
    
    try:
        models = client.search_registered_models()
        
        if not models:
            print("[INFO] No registered models found")
            return
        
        for model in models:
            print(f"\nModel: {model.name}")
            print(f"  Description: {model.description}")
            print(f"  Latest Versions:")
            
            for version in model.latest_versions:
                print(f"    Version {version.version}: {version.current_stage}")
    
    except Exception as e:
        print(f"[ERROR] {e}")


def load_and_test_model():
    """Load registered model and test it"""
    
    print("\n" + "="*70)
    print("LOADING REGISTERED MODEL")
    print("="*70)
    
    model_name = "heart-disease-classifier"
    model_stage = "Staging"
    
    model_uri = f"models:/{model_name}/{model_stage}"
    
    try:
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        print(f"[SUCCESS] Model loaded from registry")
        
        # Test prediction
        import numpy as np
        sample = np.array([[55, 1, 2, 140, 250, 0, 1, 150, 0, 2.5, 1, 1, 2]])
        prediction = model.predict(sample)
        
        print(f"\nTest Prediction:")
        print(f"  Input shape: {sample.shape}")
        print(f"  Prediction: {prediction[0]}")
        print(f"  Result: {'Disease' if prediction[0] == 1 else 'No Disease'}")
        
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    """Main registry pipeline"""
    
    # Register best model
    register_best_model()
    
    # List registered models
    list_registered_models()
    
    # Load and test
    load_and_test_model()
    
    print("\n" + "="*70)
    print("MODEL REGISTRY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()