import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import joblib


df=pd.read_csv(r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\Chrun prediction\dataset\Churn_Modelling.csv')
#First look into data
print(df.head(10))
print("Dataset Shape:", df.shape)
#Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic preprocessing
def preprocess_bank_data(df):
    # Create a copy
    df_clean = df.copy()
    
    # Verify required columns exist
    required_columns = ['Geography', 'Gender', 'CreditScore', 'Age', 'Tenure', 
                       'Balance', 'NumOfProducts', 'EstimatedSalary']
    missing_cols = [col for col in required_columns if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    
    # Convert categorical variables
    geography_dummies = pd.get_dummies(df_clean['Geography'], prefix='Geography')
    df_clean = df_clean.drop('Geography', axis=1)
    df_clean = pd.concat([df_clean, geography_dummies], axis=1)
    
    # Convert Gender with error handling
    gender_map = {'Female': 0, 'Male': 1}
    if not df_clean['Gender'].isin(gender_map.keys()).all():
        raise ValueError("Unknown gender categories found in data")
    df_clean['Gender'] = df_clean['Gender'].map(gender_map)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                     'NumOfProducts', 'EstimatedSalary']
    df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])
    
    return df_clean

# Preprocess the data
df_processed = preprocess_bank_data(df)

# Basic statistics
print("\nBasic statistics:")
print(df_processed.describe())

# Calculate churn rate
churn_rate = (df_processed['Exited'].sum() / len(df_processed)) * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

def analyze_customer_segments(df):
    """Analyze customer segments based on key characteristics"""
    
    # Create a copy of the dataframe
    df_segments = df.copy()
    
    
    # Age segments
    df_segments['AgeGroup'] = pd.cut(df['Age'],
                                    bins=[0, 30, 40, 50, 60, 100],
                                    labels=['<30', '30-40', '40-50', '50-60', '>60'])
    
    # Balance segments - handle unique bins and duplicates
    balance_values = df['Balance'].sort_values().unique()
    if len(balance_values) >= 4:
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        balance_bins = [df['Balance'].quantile(q) for q in quantiles]
        # Ensure bins are unique
        balance_bins = sorted(list(set(balance_bins)))
        if len(balance_bins) >= 2:
            df_segments['BalanceGroup'] = pd.cut(df['Balance'],
                                               bins=balance_bins,
                                               labels=['Low', 'Medium', 'High', 'Very High'][:len(balance_bins)-1],
                                               duplicates='drop')
        else:
            # If not enough unique bins, create binary segmentation
            df_segments['BalanceGroup'] = pd.qcut(df['Balance'],
                                                q=2,
                                                labels=['Low', 'High'],
                                                duplicates='drop')
    else:
        # For very few unique values
        df_segments['BalanceGroup'] = pd.qcut(df['Balance'],
                                            q=2,
                                            labels=['Low', 'High'],
                                            duplicates='drop')
    
    # Credit score segments
    df_segments['CreditScoreGroup'] = pd.qcut(df['CreditScore'],
                                             q=3,
                                             labels=['Low', 'Medium', 'High'],
                                             duplicates='drop')
    
    return df_segments

# Update the create_churn_analysis function to handle the warning
def create_churn_analysis(df):
    """Calculate churn rates for different segments"""
    
    analysis = {}
    
    # Add observed=True to suppress the warning
    analysis['age_churn'] = df.groupby('AgeGroup', observed=True)['Exited'].mean()
    analysis['balance_churn'] = df.groupby('BalanceGroup', observed=True)['Exited'].mean()
    analysis['credit_churn'] = df.groupby('CreditScoreGroup', observed=True)['Exited'].mean()
    analysis['products_churn'] = df.groupby('NumOfProducts', observed=True)['Exited'].mean()
    
    return analysis

def calculate_customer_value(df):
    """Calculate customer value metrics"""
    
    df['CustomerValue'] = df['Balance'] * df['Tenure'] / 12  # Annual value
    df['ChurnRisk'] = (df['Balance'] == 0).astype(int) + \
                      (df['NumOfProducts'] == 1).astype(int) + \
                      (df['IsActiveMember'] == 0).astype(int)
    
    return df

def prepare_data(df_processed):
    """Prepare data for XGBoost model"""
    # Remove non-feature columns
    feature_df = df_processed.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Prepare features and target
    X = feature_df
    y = df_processed['Exited']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_df.columns

def train_xgboost_model(X_train, y_train):
    """Train XGBoost model with hyperparameter tuning"""
    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Initialize base model
    base_model = xgb.XGBClassifier(
      objective='binary:logistic',
      random_state=42,
      eval_metric='logloss'
)
    
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    print("Best ROC-AUC score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and print ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.3f}")
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
    print(f"Average CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return y_pred, y_pred_proba

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance in XGBoost Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Main execution
def run_xgboost_analysis(df_processed):
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df_processed)
    
    # Train model
    print("Training XGBoost model with hyperparameter tuning...")
    model = train_xgboost_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    y_pred, y_pred_proba = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_feature_importance(model, feature_names)
    plot_confusion_matrix(y_test, y_pred)
    
    return model, y_pred, y_pred_proba

# Run the analysis
model, y_pred, y_pred_proba = run_xgboost_analysis(df_processed)



