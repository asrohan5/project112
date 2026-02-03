import pandas as pd 
import logging
from src.features import engineer_features

def create_churn_dataset(df_clean, df_raw, cutoff_date=None):

    if cutoff_date is None:
        cutoff_date=df_clean['InvoiceDate'].max() - pd.TimeDelta(days=90)
    
    print(f'===TimeSplit Config===')
    print(f'Cutoff Date: {cutoff_date}')


    df_past = df_clean[df_clean['InvoiceDate']<= cutoff_date]
    df_raw_past = df_raw[df_raw['InvoiceDate']<=cutoff_date]

    df_future = df_clean[df_clean['InvoiceDate']>cutoff_date]

    print('Engineering features on past data (training set):::')
    X=engineer_features(df_past, df_raw_past)

    active_customers = df_future['Customer ID'].unique()

    X['is_churned'] = ~X.index.isin(active_customers)
    X['is_churned'] = X['is_churned'].astype(int)

    churn_rate = X['is_churned'].mean()
    print(f'Dataset generate: {X.shape[0]} customers')
    print(f'Churn Rate: {churn_rate}')

    return X