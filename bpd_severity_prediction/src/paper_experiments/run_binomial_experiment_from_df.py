import os
import argparse
import random
import pandas as pd
from scipy.stats import uniform, loguniform, randint
import main_cross_validation as main_cross_val
import config_cross_validation as config_cv

# Parser for running from command line
run_with_parser = True
run_with_partition = True
if run_with_parser:
    parser = argparse.ArgumentParser(
        description="Nested cross validation experiments. Read input from df",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c', '--configFileName',
        help='Name of csv file with experiments config',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '-o', '--outputVariable',
        help='Name of variable for regression',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '-s', '--saveDir',
        help='Name of folder to save variable',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '-p', '--partition',
        help='Partition i of N of the config df, i/N',
        type=str,
        default=None,
        required=True,
    )

    args = parser.parse_args()
    name_config_file = args.configFileName
    name_output_var = args.outputVariable
    name_save_dir = args.saveDir
    part_of_config_df = args.partition
else:
    name_config_file = 'df_experiment_binomial.csv'
    name_output_var = 'bpd_no_vs_all'  # bpd_no_vs_all,  bpd_no-mild_vs_mod-sev
    name_save_dir = 'results_binomial_no_vs_all'
    part_of_config_df = '0/10'

# Choose a random seed
random.seed(8)

# User Configurations ----------------------------------------------------------------------------------
EXPERIMENT_CONFIGS_PATH = config_cv.EXPERIMENT_CONFIG_ROOT + name_config_file

if run_with_partition:
    df_experiment_full = pd.read_csv(EXPERIMENT_CONFIGS_PATH, sep=';')
    split_in_N_parts = int(part_of_config_df.split('/')[1])
    ith_part = int(part_of_config_df.split('/')[0])
    n_samples = int(len(df_experiment_full)/split_in_N_parts)
    i_start = ith_part * n_samples
    i_end = i_start + n_samples
    if ith_part == split_in_N_parts-1:
        df_experiment = df_experiment_full.iloc[i_start:]
    else:
        df_experiment = df_experiment_full.iloc[i_start:i_end]
else:
    df_experiment = pd.read_csv(EXPERIMENT_CONFIGS_PATH, sep=';')

PATH_DATA = config_cv.PATH_DATA
SAVE_DIR = '../results/'+name_save_dir+'/'

OUTPUT_VARIABLE = name_output_var
SCALER = main_cross_val.get_scaler('standard_scaler')

N_REPETITIONS = 10
random_repetitions_array = random.sample(range(1, 1000), N_REPETITIONS)

NESTED_CROSS_VALIDATION_CONFIG = config_cv.NESTED_CROSS_VALIDATION_CONFIG
NESTED_CROSS_VALIDATION_CONFIG['random_repetition_states'] = random_repetitions_array
BINS_GA_ANALYSIS = config_cv.BINS_GA_ANALYSIS


# Load configurations from df  --------------------------------------------------------------------------
for index, row in df_experiment.iterrows():

    NAME_EXPERIMENT = row['experiment_name']
    str_experiment = NAME_EXPERIMENT + '.csv'

    # if experiment not already in results folder
    if str_experiment not in os.listdir(SAVE_DIR):

        INPUT_FEATURES = row['variables'].split('/')

        # Load Model Configurations
        if row['model_type'] == 'logistic_classification':
            MODEL_CONFIG = {
                'model_type': row['model_type'],
                'penalty': row['penalty'],
                'max_iter': int(row['max_iter']),
            }

            # Read configs for parameter distribution
            temp_c = row['C_loguniform'].split('/')
            temp_c = [float(x) for x in temp_c]

            PARAMETER_DISTRIBUTIONS = {
                'C': loguniform(temp_c[0], temp_c[1]),
            }

            if row['l1_ratio'] != 'None':
                temp_l1ratio = row['l1_ratio'].split('/')
                temp_l1ratio = [float(x) for x in temp_l1ratio]
                PARAMETER_DISTRIBUTIONS['l1_ratio'] = uniform(loc=temp_l1ratio[0], scale=temp_l1ratio[1])

        elif row['model_type'] == 'random_forest_classification':

            MODEL_CONFIG = {
                'model_type': row['model_type'],
            }

            temp_max_depth = row['rf_max_depth'].split('/')
            temp_max_depth = [int(x) for x in temp_max_depth]
            temp_n_estimators = row['rf_n_estimators'].split('/')
            temp_n_estimators = [int(x) for x in temp_n_estimators]

            PARAMETER_DISTRIBUTIONS = {
                'max_depth': randint(temp_max_depth[0], temp_max_depth[1]),
                'n_estimators': randint(temp_n_estimators[0], temp_n_estimators[1]),
            }
        else:
            MODEL_CONFIG = None

        # Load Hyperparameter Search Configurations
        HYPERPARAMETER_CONFIG = {
            'method': row['hyper_search_method'],
            'param_distributions': PARAMETER_DISTRIBUTIONS,
            'score_metric': row['hyper_score_metric'],
            'n_min_iter_random_search': int(row['n_min_iter_random_search']),
        }

        # Load Feature Selection Configurations
        if row['feature_selection_method'] == 'univariate':
            FEATURE_SELECTION_CONFIG = {
                'feature_selection_method': 'univariate',
                'parameters': {
                    'select_k_features': int(row['select_k_features']),
                    'score_feature_selection': row['score_feature_selection'],
                },
            }
        else:
            FEATURE_SELECTION_CONFIG = None

        # Load PCA Configurations
        if row['PCA']:
            PCA_CONFIG = {
                'threshold': float(row['pca_threshold']),
            }
        else:
            PCA_CONFIG = None

        # Configurations for Gestational Age Analysis
        GEST_AGE_ANALYSIS = {
            'bins': BINS_GA_ANALYSIS
        }
        gest_age_data = main_cross_val.get_df_column(
            csv_path=PATH_DATA,
            col_name='gest_age',
        )
        GEST_AGE_ANALYSIS['data'] = gest_age_data

        NESTED_CROSS_VALIDATION_CONFIG['experiment_name'] = NAME_EXPERIMENT

        # --------------------------------------------------------------------------------------------
        # Load Data for Experiment
        X_in, y_in = main_cross_val.get_data(
            csv_path=PATH_DATA,
            outcome_variable=OUTPUT_VARIABLE,
            selected_features_group_or_vars=INPUT_FEATURES
        )

        # Run Nested Cross Validation for Multinomial Classification
        results_all_splits = main_cross_val.nested_cross_validation(
            X=X_in,
            y=y_in,
            cross_val_config=NESTED_CROSS_VALIDATION_CONFIG,
            model_config=MODEL_CONFIG,
            hyperparameters_config=HYPERPARAMETER_CONFIG,
            feature_selection_config=FEATURE_SELECTION_CONFIG,
            pca_config=PCA_CONFIG,
            scaler=SCALER,
            gest_age_analysis=GEST_AGE_ANALYSIS,
        )

        # Save
        print(NAME_EXPERIMENT)
        results_all_splits.to_csv(SAVE_DIR + NAME_EXPERIMENT + '.csv', index=False)
