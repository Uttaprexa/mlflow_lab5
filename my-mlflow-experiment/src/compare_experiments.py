"""
Compare MLflow Experiments
==========================
Query MLflow tracking server and compare experiments
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("file:./mlruns")


def get_all_experiments():
    client = MlflowClient()
    experiments = client.search_experiments()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS")
    print("="*70)
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Artifact Location: {exp.artifact_location}")
    
    return experiments


def compare_runs_in_experiment(experiment_name):
    
    print("\n" + "="*70)
    print(f"COMPARING RUNS IN: {experiment_name}")
    print("="*70)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"[ERROR] Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    
    if len(runs) == 0:
        print("[INFO] No runs found")
        return
    
    print(f"\nTotal Runs: {len(runs)}")
    print("\nTop 10 Runs by Accuracy:")
    print("-" * 70)
    
    display_cols = ['run_id', 'metrics.accuracy', 'metrics.f1_score', 
                    'params.n_estimators', 'params.max_depth']
    
    available_cols = [col for col in display_cols if col in runs.columns]
    
    print(runs[available_cols].head(10).to_string())
    
    return runs


def plot_experiment_comparison(experiment_name):
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        return
    
    # Plot 1: Accuracy comparison
    plt.figure(figsize=(12, 6))
    
    if 'tags.mlflow.runName' in runs.columns and 'metrics.accuracy' in runs.columns:
        runs_sorted = runs.sort_values('metrics.accuracy', ascending=False).head(10)
        
        plt.bar(range(len(runs_sorted)), runs_sorted['metrics.accuracy'])
        plt.xlabel('Run')
        plt.ylabel('Accuracy')
        plt.title(f'Top 10 Runs by Accuracy - {experiment_name}')
        plt.xticks(range(len(runs_sorted)), 
                   runs_sorted['tags.mlflow.runName'], 
                   rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(f'outputs/comparison_{experiment_name.replace(" ", "_")}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[INFO] Plot saved: outputs/comparison_{experiment_name.replace(' ', '_')}.png")


def find_best_model():
    """Finds the best model across all experiments"""
    
    print("\n" + "="*70)
    print("FINDING BEST MODEL ACROSS ALL EXPERIMENTS")
    print("="*70)
    
    client = MlflowClient()
    experiments = client.search_experiments()
    
    all_runs = []
    
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if len(runs) > 0:
            runs['experiment_name'] = exp.name
            all_runs.append(runs)
    
    if not all_runs:
        print("[INFO] No runs found")
        return
    
    all_runs_df = pd.concat(all_runs, ignore_index=True)
    
    # Find best by accuracy
    if 'metrics.accuracy' in all_runs_df.columns:
        best_run = all_runs_df.loc[all_runs_df['metrics.accuracy'].idxmax()]
        
        print(f"\nBest Model Found:")
        print(f"  Experiment: {best_run['experiment_name']}")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Accuracy: {best_run['metrics.accuracy']:.4f}")
        
        if 'tags.mlflow.runName' in best_run:
            print(f"  Run Name: {best_run['tags.mlflow.runName']}")
        
        return best_run


def main():
    """Main comparison pipeline"""
    
    # Get all experiments
    experiments = get_all_experiments()
    
    # Compare runs in main experiment
    runs = compare_runs_in_experiment("heart-disease-classification")
    
    # Plot comparison
    if runs is not None and len(runs) > 0:
        plot_experiment_comparison("heart-disease-classification")
    
    # Find best model
    best = find_best_model()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()