import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def train_baseline_model(df):
    print('+++ baseline model +++')

    drop_cols = ['spend_last_30d', 'spend_prior_30d']
    features = df.drop(columns=['is_churned'] + drop_cols, errors='ignore')
    target=df['is_churned']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]


    print("\n Classification Report :::::")
    print(classification_report(y_test, y_pred))

    auc= roc_auc_score(y_test, y_prob)
    print(f'ROC-AUC score: {auc:.4f}')

    importance = pd.DataFrame({'Feature': features.columns, 'Coefficient': model.coef_[0]}).sort_values(by='Coefficient', ascending=False)

    print(f'\nTop predictors of Rention (neg coeff) v Churn (pos coeff):')
    print(importance.head(3))
    print(importance.tail(3))

    return model, y_test, y_pred, importance

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label (1=Churn)')
    plt.ylabel('True Label (1=Churn)')
    plt.title('Confusion Matrix: Baseline Model')
    plt.savefig('plots/4_confusion_matrix.png')
