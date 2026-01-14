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
from scipy.stats import f_oneway
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
    print(df)
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

    from scipy.stats import f_oneway

    print("\n--- ANOVA Test (Age vs Diagnosis) ---")

    group_0 = df[df['Diagnosis'] == 0]['Age']
    group_1 = df[df['Diagnosis'] == 1]['Age']

    f_stat, p_value = f_oneway(group_0, group_1)

    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value}")

    
    return df

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
def preprocess_data(df, target_column='Diagnosis'):
    """Preprocess the data for machine learning"""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target variable if it's categorical
    if y.dtype == 'object':
        print(f"\nEncoding target variable")
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # VISUALIZE MISSING VALUES
    print("\n--- Identifying and Handling Missing Values ---")
    print("\n1. Missing Values Count:")
    print(X.isnull().sum())
    
    print("\n2. Visualizing Missing Values:")
    plt.figure(figsize=(12, 6))
    sns.heatmap(X.isnull(), cbar=True, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Columns')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png', dpi=300)
    plt.show()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("\n3. Handling missing values...")
        X = X.fillna(X.median())
        print("Missing values after handling:")
        print(X.isnull().sum())
    else:
        print("\n3. No missing values detected!")
    
    # DATA FORMATTING
    print("\n--- Data Formatting ---")
    print("\n1. Current Data Types:")
    print(X.dtypes)
    
    print("\n2. Numerical Columns:")
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    print(list(numerical_cols))
    
    print("\n3. Categorical Columns:")
    categorical_cols_orig = X.select_dtypes(include=['object']).columns
    print(list(categorical_cols_orig))
    
    # DATA NORMALIZATION
    print("\n--- Data Normalization ---")
    
    # Import MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    
    print("\n1. Min-Max Scaling (Normalization)")
    print("   Formula: x_scaled = (x - x_min) / (x_max - x_min)")
    print("   Result: Values scaled to range [0, 1]")
    
    if 'Age' in X.columns:
        # Create a copy for demonstration
        min_max_scaler = MinMaxScaler()
        age_minmax = min_max_scaler.fit_transform(X[['Age']])
        
        print("\n   Example: Age Column")
        print(f"   Original range: [{X['Age'].min():.2f}, {X['Age'].max():.2f}]")
        print(f"   Scaled range: [{age_minmax.min():.2f}, {age_minmax.max():.2f}]")
        
        # Show comparison
        comparison_df = pd.DataFrame({
            'Original_Age': X['Age'].head(10),
            'MinMax_Scaled': age_minmax[:10].flatten()
        })
        print("\n   Sample comparison:")
        print(comparison_df)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(X['Age'], bins=20, color='skyblue', edgecolor='black')
        axes[0].set_title('Original Age Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(age_minmax, bins=20, color='salmon', edgecolor='black')
        axes[1].set_title('Min-Max Scaled Age [0,1]', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Scaled Age')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('minmax_scaling.png', dpi=300)
        plt.show()
    
    print("\n2. Z-Score Normalization (Standardization)")
    print("   Formula: z = (x - μ) / σ")
    print("   Note: This will be applied later using StandardScaler")
    
    # BINNING
    print("\n--- Binning ---")
    
    if 'Age' in X.columns:
        print("\n1. Age Binning:")
        X['Age_binned'] = pd.cut(
            X['Age'],
            bins=[0, 60, 75, 100],
            labels=['Young', 'Middle', 'Old']
        )
        
        # Show binning results
        print("\n   Sample of Age Binning:")
        print(X[['Age', 'Age_binned']].head(10))
        
        print("\n   Age Group Distribution:")
        print(X['Age_binned'].value_counts())
        
        # Visualize
        plt.figure(figsize=(10, 6))
        X['Age_binned'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Age Group Distribution After Binning', fontsize=14, fontweight='bold')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('age_binning.png', dpi=300)
        plt.show()

    # INDICATOR VARIABLES
    print("\n--- Indicator Variables (One-Hot Encoding) ---")
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        print(f"\n1. Categorical columns to encode: {list(categorical_cols)}")
        
        # Show before encoding
        print("\n2. Before encoding (first 5 rows):")
        print(X[categorical_cols].head())
        
        # Apply one-hot encoding
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        print(f"\n3. After one-hot encoding:")
        print(f"   Original features: {len(categorical_cols)}")
        print(f"   New indicator columns created: {len([col for col in X.columns if any(cat in col for cat in categorical_cols)])}")
        print(f"   Total columns now: {X.shape[1]}")
    
    # SPLIT AND SCALE
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features (Z-score normalization)
    print("\n--- Final Scaling (Z-Score Normalization) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETE")
    print("="*80)
    
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
    FILEPATH = "alzheimers_disease_data.csv"
    TARGET_COLUMN = 'Diagnosis'  
    
    # Run the pipeline
    results, best_model = main(FILEPATH, TARGET_COLUMN)
    
    # You can now use best_model for predictions on new data
    # Example: predictions = best_model.predict(new_data_scaled)