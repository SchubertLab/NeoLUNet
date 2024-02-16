EXPERIMENT_CONFIG_ROOT = '../data/experiment_config/'
PATH_DATA = '../data/regression_features_2023_03_01.csv'
N_REPETITIONS = 10
NESTED_CROSS_VALIDATION_CONFIG = {
    'n_splits_outer': 5,
    'n_splits_inner': 5,
    'experiment_type': 'regression',
    'k_fold_type': 'k_fold',
}
BINS_GA_ANALYSIS = [
    [23, 26],
    [26, 29],
    [29, 32],
    [23, 29],
    [25.4, 28.6],
]
