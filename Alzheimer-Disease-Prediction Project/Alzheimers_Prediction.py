# Alzheimer's Disease Prediction
# Complete Machine Learning Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# FEATURES TO REMOVE (Non-medical and weak evidence features)
# ============================================================================
FEATURES_TO_REMOVE = [
    'PatientID',           # Administrative identifier
    'DoctorInCharge',      # Administrative data
    'Gender',              # Weak predictor
    'Ethnicity',           # May introduce bias, weak medical relevance
    'AlcoholConsumption',  # Inconsistent evidence
    'SleepQuality'         # Weak/inconsistent evidence
]

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load the Alzheimer's disease dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Remove non-medical and weak evidence features
    print(f"\n{'='*80}")
    print("REMOVING NON-MEDICAL AND WEAK EVIDENCE FEATURES")
    print(f"{'='*80}")
    features_removed = [col for col in FEATURES_TO_REMOVE if col in df.columns]
    if features_removed:
        print(f"\nRemoving features: {features_removed}")
        df = df.drop(columns=features_removed)
        print(f"New dataset shape: {df.shape}")
    else:
        print("\nNo features to remove (features not found in dataset)")
    
    return df

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df, target_column='Diagnosis'):
    """Perform exploratory data analysis"""
    
    # Target distribution
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    df[target_column].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Target Variable Distribution')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    df[target_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Target Variable Percentage')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(14, 10))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
            center=0, square=True, linewidths=1)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.show()
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df, target_column='Diagnosis'):
    """Preprocess the data for machine learning"""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"\nEncoding categorical variables: {list(categorical_cols)}")
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target variable if it's categorical
    if y.dtype == 'object':
        print(f"\nEncoding target variable")
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("\nHandling missing values...")
        X = X.fillna(X.median())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# ============================================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and evaluate their performance"""
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        # 'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        # 'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results

# ============================================================================
# 5. VISUALIZATION OF RESULTS
# ============================================================================

def visualize_results(results, y_test):
    """Visualize model performance"""
    
    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    precisions = [results[name]['precision'] for name in model_names]
    recalls = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    # Bar plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        ('Accuracy', accuracies),
        ('Precision', precisions),
        ('Recall', recalls),
        ('F1-Score', f1_scores)
    ]
    
    for idx, (metric_name, metric_values) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(model_names, metric_values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrices for top 3 models
    top_3_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, result) in enumerate(top_3_models):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # ROC curves for models with probability predictions
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            auc = roc_auc_score(y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Plot feature importance for tree-based models"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# ============================================================================
# 7. MAIN EXECUTION PIPELINE
# ============================================================================

def main(filepath, target_column='Diagnosis'):
    """Main execution pipeline"""
    
    print("="*80)
    print("ALZHEIMER'S DISEASE PREDICTION - ML PIPELINE")
    print("="*80)
    
    # Load data
    print("\n[1] Loading Data...")
    df = load_data(filepath)
    
    # EDA
    print("\n[2] Performing Exploratory Data Analysis...")
    df = perform_eda(df, target_column)
    
    # Preprocessing
    print("\n[3] Preprocessing Data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, target_column)
    
    # Train and evaluate models
    print("\n[4] Training and Evaluating Models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    print("\n[5] Visualizing Results...")
    visualize_results(results, y_test)
    
    # Feature importance for best model
    print("\n[6] Feature Importance Analysis...")
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, feature_names, best_model_name)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"Precision: {results[best_model_name]['precision']:.4f}")
    print(f"Recall: {results[best_model_name]['recall']:.4f}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    return results, best_model

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Replace with your actual file path
    FILEPATH = "alzheimers_disease_data.csv"
    TARGET_COLUMN = 'Diagnosis'  # Adjust based on your dataset
    
    # Run the pipeline
    results, best_model = main(FILEPATH, TARGET_COLUMN)
    
    # You can now use best_model for predictions on new data
    # Example: predictions = best_model.predict(new_data_scaled)
