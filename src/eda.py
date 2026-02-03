import seaborn as sns
import matplotlib.pyplot as plt

def plot_churn_separation(df_labeled):
    
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df_labeled, x='is_churned', y='spend_velocity', showfliers=False)
    
    plt.title("Impact of Spend Velocity on Churn (Outliers Hidden)")
    plt.xlabel("Churn Status (0=Stayed, 1=Left)")
    plt.ylabel("Spend Velocity (Ratio)")
    plt.show()

    plt.figure(figsize=(12, 10))
    corr = df_labeled.corr()
    sns.heatmap(corr[['is_churned']].sort_values(by='is_churned', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation with Churn Target")
    plt.show()