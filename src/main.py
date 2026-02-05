import logging
import pandas as pd
from src.ingestion import load_and_optimize_data
from src.metrics import clean_data_for_metrics, calculate_rfm_values, assign_rfm_scores, define_customer_segments, calculate_cohort_index, get_retention_matrix, build_executive_summary
from src import vizualization as viz
from src import features
from src import eda
from src.data_factory import create_churn_dataset
from src import model as ml_engine
from src import model_advanced
from src import serialization


RUN_VIZ_AND_REPORTING = False
RUN_FEATURE_ENGG = False
RUN_TARGET_GEN = False
RUN_BASELINE_MODEL = False
RUN_ADVANCED_MODEL = True


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s %(message)s]")

def run_pipeline():

    if RUN_VIZ_AND_REPORTING or RUN_FEATURE_ENGG or RUN_TARGET_GEN:
        df_raw = load_and_optimize_data()
        if df_raw is None:
            logging.error('Pipeline Failed: Could not load data')
            return
        df_clean = clean_data_for_metrics(df_raw)
    else:
        logging.info("Skipping Raw Data Load")

    
    if RUN_VIZ_AND_REPORTING:
        logging.info('::: RUNNING REPORTING LAYER :::')
        rfm = calculate_rfm_values(df_clean)
        rfm = assign_rfm_scores(rfm)
        rfm = define_customer_segments(rfm)

        df_cohort = calculate_cohort_index(df_clean)
        retention = get_retention_matrix(df_cohort)
        exec_summary = build_executive_summary(df_clean)

        viz.plot_retention_heatmap(retention)
        viz.plot_segement_distribution(rfm)
        viz.plot_revenue_growth(exec_summary)

    
    if RUN_FEATURE_ENGG:
        logging.info(" ::: RUNNING FE ::: ")
        feature_matrix = features.engineer_features(df_clean, df_raw)

        print(feature_matrix[['spend_velocity', 'return_rate', 'whale_score']].head())
        feature_matrix.to_csv('outputs/feature_matrix.csv')

    if RUN_TARGET_GEN:
        logging.info(" ::: Creating Training Set :::")
        training_cutoff = pd.TimeStamp('2011-09-01')
        train_df = create_churn_dataset(df_clean, df_raw, cutoff_date=training_cutoff)

        train_df.to_csv('outputs/modeling_data.csv')
        eda.plot_churn_separation(train_df)
    
    if RUN_BASELINE_MODEL or RUN_ADVANCED_MODEL:
        try: 
            logging.info("Loading Pre-Processed modeling data...")
            df_model = pd.read_csv('outputs/modeling_data.csv', index_col=0)

        except FileNotFoundError:
            logging.error('CRITICAL: modeling_data.csv not found. Set RUN_TARGET_GEN = True')
            return
    
        if RUN_BASELINE_MODEL:
            logging.info('Training Baseline Logistic Regression...')
            trained_model, y_test, y_pred, feat_imp = ml_engine.train_baseline_model(df_model)
            ml_engine.plot_confusion_matrix(y_test, y_pred)

    
        if RUN_ADVANCED_MODEL:
            logging.info('Tarining Advanced Random Forest (Grid Search)...')
            rf_model, scaler, feature_names = model_advanced.train_random_forest(df_model)
            model_advanced.plot_feature_importance_rf(rf_model, feature_names)

            logging.info("serialization model for production:::")
            serialization.save_production_artifacts(rf_model, scaler, feature_names)
    
    logging.info('+++++ Pipeline Execution Completed Successfully +++++')


if __name__ == "__main__":
    run_pipeline()
