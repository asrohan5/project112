import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_random_forest(df):
    print('::: Training Random Forest (advanced) :::')
    features = df.drop(columns=['is_churned', 'spend_last_30d', 'spend_prior_30d'])
    target = df['is_churned']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    param_grid = {'n_estimators':[50,100,200],
                  'max_depth': [5,10,20],
                  'min_samples_split': [2,10]}

    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)

    print('Tuning Hyperparameters....')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f'Best Parameters: {grid_search.best_params_}')

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    print('\n Random Forest Report: ')
    print(classification_report(y_test, y_pred))
    print(f'New ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')

    return best_model, scaler, features.columns

def plot_feature_importance_rf(model, feature_names):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importance (Random FOrest)")
    plt.bar(range(len(importances)), importances[indices], align="center", color='forestgreen')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/6_feature_importance.png')

