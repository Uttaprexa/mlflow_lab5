# MLflow Experiment Tracking Lab

## Project Overview

In this lab, I have demonstrates comprehensive MLflow experiment tracking for machine learning pipelines, extending the original MLflow lab with advanced features including multi-model training, hyperparameter tuning, model comparison, and model registry integration.

---

## What I Built

A complete MLflow-based experiment tracking system featuring:

- **Multi-Model Training**: 7 different ML algorithms tracked simultaneously
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Automated Artifacts**: Confusion matrices, ROC curves, performance plots
- **Hyperparameter Tuning**: Grid search with nested MLflow runs
- **Model Comparison**: Cross-experiment analysis and visualization
- **Model Registry**: Version control and lifecycle management

---

## Original Lab vs My Implementation

### Original MLflow Lab

The original lab covered:
- Basic MLflow setup and configuration
- Simple parameter logging (`mlflow.log_param()`)
- Basic metric logging (`mlflow.log_metric()`)
- Model saving with MLflow
- Introduction to MLflow UI

**Scope**: Tutorial-style introduction with 1-2 models

---

### My Enhanced Implementation

#### Key Enhancements

**1. Comprehensive Model Training (`src/train.py`)**
- Trained **7 different ML models** (vs 1-2 in original):
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- Logged **5+ metrics per model** (accuracy, precision, recall, F1, ROC-AUC)
- **Automated artifact generation**:
  - Confusion matrices for each model
  - ROC curves with AUC scores
  - Performance comparison visualizations
- **Complete metadata tracking**:
  - Model hyperparameters
  - Dataset statistics
  - Training time
  - Custom tags

**2. Advanced Hyperparameter Tuning (`src/hyperparameter_tuning.py`)**
- **Parent-Child Run Relationships**: Organized grid search results hierarchically
- **Nested Run Architecture**: Parent run for experiment, child runs for each HP combination
- **Comprehensive Grid Search**:
  - Random Forest: 144 parameter combinations
  - Gradient Boosting: 81 parameter combinations
- **Best Model Identification**: Automatic selection based on performance

**3. Experiment Comparison System (`src/compare_experiments.py`)**
- **Cross-Experiment Analysis**: Compare models across multiple experiments
- **Automated Visualizations**: Generate comparison plots programmatically
- **Performance Rankings**: Identify top performers automatically
- **MLflow Query API**: Advanced search and filtering

**4. Model Registry Integration (`src/model_registry.py`)**
- **Model Versioning**: Register and version best models
- **Stage Transitions**: Move models through Staging → Production
- **Metadata Management**: Add descriptions and tags
- **Deployment-Ready**: Load registered models for inference

---

## Project Structure
```
my-mlflow-experiment/
├── src/
│   ├── train.py                    # Main training with MLflow (7 models)
│   ├── hyperparameter_tuning.py   # Grid search with nested runs
│   ├── compare_experiments.py     # Experiment comparison & analysis
│   └── model_registry.py          # Model versioning & lifecycle
├── data/
│   └── heart_disease.csv          # Synthetic dataset (500 samples, 14 features)
├── mlruns/                         # MLflow tracking data (auto-generated)
│   ├── 0/                         # Default experiment
│   ├── 1/                         # heart-disease-classification
│   └── 2/                         # heart-disease-hyperparameter-tuning
├── outputs/                        # Generated visualizations
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── comparison_*.png
├── models/                         # Saved model artifacts
├── notebooks/                      # Jupyter notebooks (optional)
├── requirements.txt               # Python dependencies
├── generate_data.py               # Dataset generation script
└── README.md                      # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Windows/Mac/Linux

### Step-by-Step Installation
```bash
# 1. Clone the repository
git clone https://github.com/Uttaprexa/mlflow_lab5.git 
cd my-mlflow-experiment-lab

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Generate dataset (if not present)
python generate_data.py

# 6. Verify installation
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

---

## Running the Lab

###  Basic Model Training

Train 7 models and log everything to MLflow:
```bash
python src/train.py
```

**What this does:**
- Creates synthetic heart disease dataset (500 samples)
- Trains 7 different ML models
- Logs parameters, metrics, and artifacts to MLflow
- Generates confusion matrices and ROC curves
- Saves models with MLflow

**Expected output:**
```
======================================================================
MLFLOW EXPERIMENT TRACKING
======================================================================
Experiment: heart-disease-classification
Tracking URI: file:./mlruns
======================================================================

[INFO] Creating synthetic dataset...
[INFO] Dataset created: (500, 14)

======================================================================
Training: Logistic Regression
======================================================================

Metrics:
  accuracy        : 0.8500
  precision       : 0.8528
  recall          : 0.8500
  f1_score        : 0.8511
  roc_auc         : 0.9100

[INFO] Run logged to MLflow
...
```

---

### Hyperparameter Tuning

Perform grid search with comprehensive tracking:
```bash
python src/hyperparameter_tuning.py
```

**What this does:**
- Searches 144 Random Forest parameter combinations
- Searches 81 Gradient Boosting parameter combinations
- Uses nested runs (parent experiment → child parameter sets)
- Identifies best hyperparameters automatically

**Expected output:**
```
======================================================================
HYPERPARAMETER TUNING: RANDOM FOREST
======================================================================

[INFO] n_est=50, max_depth=5, min_split=2, min_leaf=1 → Accuracy: 0.8700
[INFO] n_est=100, max_depth=10, min_split=5, min_leaf=2 → Accuracy: 0.9200
...

[SUCCESS] Best Accuracy: 0.9350
[SUCCESS] Best Params: {'n_estimators': 200, 'max_depth': 20, ...}
```

---

### Compare Experiments

Analyze and visualize all tracked experiments:
```bash
python src/compare_experiments.py
```

**What this does:**
- Lists all experiments in MLflow
- Compares runs within each experiment
- Finds best model across all experiments
- Generates comparison visualizations

**Expected output:**
```
======================================================================
ALL EXPERIMENTS
======================================================================

Experiment: heart-disease-classification
  ID: 1
  Artifact Location: file:///...

======================================================================
FINDING BEST MODEL ACROSS ALL EXPERIMENTS
======================================================================

Best Model Found:
  Experiment: heart-disease-classification
  Run ID: abc123...
  Accuracy: 0.9500
  Run Name: Gradient Boosting
```

---

###  Model Registry

Register and manage best model:
```bash
python src/model_registry.py
```

**What this does:**
- Identifies best performing model
- Registers it in MLflow Model Registry
- Transitions model to "Staging" stage
- Adds metadata and descriptions
- Demonstrates model loading for inference

**Expected output:**
```
======================================================================
MODEL REGISTRY
======================================================================

Best Model:
  Run ID: abc123...
  Accuracy: 0.9500

Registering model: heart-disease-classifier

[SUCCESS] Model registered!
  Name: heart-disease-classifier
  Version: 1

[SUCCESS] Model transitioned to Staging

Test Prediction:
  Input shape: (1, 13)
  Prediction: 1
  Result: Disease
```

---

###  Launch MLflow UI

View all experiments, runs, and models in the web interface:
```bash
mlflow ui
```

Then open your browser to: **http://localhost:5000**

**MLflow UI Features:**
- **Experiments Tab**: View all experiments and runs
- **Compare Runs**: Side-by-side metric comparison
- **Visualizations**: Interactive charts and plots
- **Artifacts**: Browse logged files and models
- **Model Registry**: Manage model versions and stages

---

##  Dataset Information

### Synthetic Heart Disease Dataset

**Source**: Programmatically generated for reproducibility

**Specifications:**
- **Total Samples**: 500
- **Features**: 13 clinical parameters
- **Target**: Binary classification (0 = No Disease, 1 = Disease)

**Features:**
1. `age`: Age in years (30-80)
2. `sex`: Sex (0 = female, 1 = male)
3. `cp`: Chest pain type (0-3)
4. `trestbps`: Resting blood pressure (90-200 mm Hg)
5. `chol`: Serum cholesterol (120-400 mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)
7. `restecg`: Resting ECG results (0-2)
8. `thalach`: Maximum heart rate (70-200)
9. `exang`: Exercise induced angina (0 = no, 1 = yes)
10. `oldpeak`: ST depression (0-6)
11. `slope`: Slope of peak exercise ST segment (0-2)
12. `ca`: Number of major vessels (0-3)
13. `thal`: Thalassemia (0-3)

**Target Distribution:**
- Class 0 (No Disease): ~60%
- Class 1 (Disease): ~40%

**Train/Test Split:** 80/20 (stratified)

---

##  Technical Implementation Details

### Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Programming language |
| MLflow | 2.8.0+ | Experiment tracking |
| scikit-learn | 1.3.0+ | ML algorithms |
| pandas | 2.0.0+ | Data manipulation |
| numpy | 1.24.0+ | Numerical computing |
| matplotlib | 3.7.0+ | Visualization |
| seaborn | 0.12.0+ | Statistical plots |

## Comparison: Original Lab vs My Implementation

| Feature | Original MLflow Lab | My Implementation | Improvement |
|---------|-------------------|-------------------|-------------|
| **Models Trained** | 1-2 | 7 | 3.5-7x more |
| **Experiments** | 1 | 3 | Multiple organized experiments |
| **Metrics per Model** | 1-2 | 5+ | Comprehensive evaluation |
| **Artifacts Logged** | None | 14+ plots | Visual analysis |
| **Hyperparameter Tuning** | Not covered | 225 combinations | Advanced optimization |
| **Nested Runs** | Not used | Parent-child hierarchy | Better organization |
| **Model Registry** | Not implemented | Full lifecycle | Production-ready |
| **Comparison Tools** | Basic | Automated system | Programmatic analysis |


---

## Learning Outcomes

### From Original Lab
- Understood MLflow architecture and concepts
- Learned basic parameter and metric logging
- Explored MLflow UI interface
- Saved models with MLflow

### My Additional Learning
- Implemented multi-model comparison system
- Built automated artifact generation pipeline
- Mastered nested runs for experiment organization
- Developed hyperparameter tuning with MLflow tracking
- Created model registry with lifecycle management
- Built programmatic experiment comparison tools
- Designed production-ready MLflow workflows
- Applied MLOps best practices

---

### Course Materials
- Original MLflow Lab: [[Link to course repository](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs)]

---


