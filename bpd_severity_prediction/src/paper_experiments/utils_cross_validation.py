import copy
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocess_features import pre_process_features


# Auxiliary Functions ---------------------------------------------------------------------------------------
def get_data(csv_path, outcome_variable, selected_features_group_or_vars):
    df_raw = pd.read_csv(csv_path)
    df_regression = pre_process_features(
        df_in=df_raw,
    )
    col_array = df_regression.columns

    all_lung_columns = [x for x in col_array if 'lung' in x]
    lung_per_weight_columns = [x for x in all_lung_columns if 'birth_weight_voxels' in x]
    lung_only = [x for x in all_lung_columns if x not in lung_per_weight_columns]
    lung_shapr = [x for x in lung_only if 'shapr' in x]
    lung_pyrad = [x for x in lung_only if 'pyrad' in x]

    feature_names = {
        'PATIENT_VARS': ['gender', 'gest_age', 'birth_weight_g', 'body_size_cm'],
        'CLINICAL_VARS': ['apgar_5min', 'early_onset_infection', 'steroids'],
        'LUNG_SHAPR': lung_shapr,
        'LUNG_PYRAD': lung_pyrad,
        'LUNG_by_WEIGHT': lung_per_weight_columns,
    }

    selected = []
    for x in selected_features_group_or_vars:
        if x in ['PATIENT_VARS', 'CLINICAL_VARS', 'LUNG_SHAPR', 'LUNG_PYRAD', 'LUNG_by_WEIGHT']:
            temp_features = feature_names[x]
            selected = np.concatenate((selected, temp_features), axis=None)
        else:
            selected = np.concatenate((selected, x), axis=None)

    y = df_regression[outcome_variable]
    X = df_regression[selected]
    return X, y


def get_df_column(csv_path, col_name):
    df_temp = pd.read_csv(csv_path)
    df_columns = df_temp[col_name]
    return df_columns


def get_scaler(input_scaler, quantile_range=(25.0, 75.0), unit_variance=True):
    scaler = None
    if input_scaler == 'robust_scaler':
        scaler = RobustScaler(
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        )
    elif input_scaler == 'standard_scaler':
        scaler = StandardScaler()
    return scaler


def get_k_folds_object(fold_type, n_splits, random_state, shuffle_option=True):
    fold_object = None
    if fold_type == 'stratified_k_fold':
        fold_object = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle_option,
            random_state=random_state,
        )
    elif fold_type == 'k_fold':
        fold_object = KFold(
            n_splits=n_splits,
            shuffle=shuffle_option,
            random_state=random_state,
        )
    return fold_object


def get_performance_metric(input_metric):
    metric_return = None
    if input_metric == 'f_classif':
        metric_return = f_classif
    elif input_metric == 'mutual_info_classif':
        metric_return = mutual_info_classif
    return metric_return


def initialize_results_dict(experiment_type):
    return_dict = {}
    if experiment_type == 'multinomial':
        save_result_cols = [
            'model_type', 'experiment_name',
            'repetition_n', 'outer_fold',
            'selected_features_names',
            'roc_auc_all_classes', 'f1_all_classes',
            'fpr_all_classes', 'tpr_all_classes',
            'auc_no_bpd', 'auc_mild_bpd', 'auc_moderate_bpd', 'auc_severe_bpd',
            'penalty', 'model_coefficients', 'log_reg_c', 'l1_ratio',
            'max_depth', 'n_estimators', 'feature_importances',
            'apply_pca_lung_features', 'pca_threshold',
            'y_test', 'y_pred', 'y_pred_proba',
        ]
        for i in save_result_cols:
            return_dict[i] = []
    elif experiment_type == 'binomial':
        save_result_cols = [
            'model_type', 'experiment_name',
            'repetition_n', 'outer_fold',
            'selected_features_names',
            'roc_auc', 'f1',
            'fpr', 'tpr',
            'penalty', 'model_coefficients', 'log_reg_c', 'l1_ratio',
            'max_depth', 'n_estimators', 'feature_importances',
            'apply_pca_lung_features', 'pca_threshold',
            'y_test', 'y_pred', 'y_pred_proba',
        ]
        for i in save_result_cols:
            return_dict[i] = []
    elif experiment_type == 'regression':
        save_result_cols = [
            'model_type', 'experiment_name',
            'repetition_n', 'outer_fold',
            'selected_features_names',
            'mean_squared_error', 'mean_absolute_error',
            'poisson_model_weights', 'poisson_model_intercept',
            'rf_feature_importances', 'n_estimators', 'max_depth',
            'apply_pca_lung_features', 'pca_threshold',
            'y_test', 'y_pred',
        ]
        for i in save_result_cols:
            return_dict[i] = []
    return return_dict


def build_outer_fold_model(model_config, random_state_i):
    model = None
    if model_config['model_type'] == 'logistic_classification':
        solver_temp = None
        # define solver
        if model_config['penalty'] == 'elasticnet' or model_config['penalty'] == 'l1':
            solver_temp = 'saga'
        elif model_config['penalty'] == 'l2':
            solver_temp = 'lbfgs'

        model = LogisticRegression(
            random_state=random_state_i,
            multi_class='auto',  # 'multinomial',
            penalty=model_config['penalty'],
            solver=solver_temp,
            max_iter=model_config['max_iter'],
        )
    elif model_config['model_type'] == 'random_forest_classification':
        model = RandomForestClassifier(
            random_state=random_state_i,
        )
    elif model_config['model_type'] == 'poisson_reg':
        model = PoissonRegressor(
            max_iter=model_config['poisson_max_iter'],
        )
    elif model_config['model_type'] == 'random_forest_reg':
        model = RandomForestRegressor(
            random_state=random_state_i
        )
    return model


def build_inner_fold_model(model_config, random_state_i, hyp_search_best):
    model = None
    if model_config['model_type'] == 'logistic_classification':
        solver_temp = None
        temp_l1_ratio = None
        if model_config['penalty'] == 'elasticnet':
            temp_l1_ratio = hyp_search_best.l1_ratio
            solver_temp = 'saga'
        elif model_config['penalty'] == 'l1':
            solver_temp = 'saga'
        elif model_config['penalty'] == 'l2':
            solver_temp = 'lbfgs'

        model = LogisticRegression(
            random_state=random_state_i,
            C=hyp_search_best.C,
            penalty=model_config['penalty'],
            multi_class='auto',  # 'multinomial',
            solver=solver_temp,
            l1_ratio=temp_l1_ratio,
            max_iter=model_config['max_iter'],
        )
    elif model_config['model_type'] == 'random_forest_classification':
        model = RandomForestClassifier(
            random_state=random_state_i,
            n_estimators=hyp_search_best.n_estimators,
            max_depth=hyp_search_best.max_depth,
        )
    elif model_config['model_type'] == 'poisson_reg':
        model = PoissonRegressor(
            alpha=hyp_search_best.alpha,
            max_iter=model_config['poisson_max_iter'],
        )
    elif model_config['model_type'] == 'random_forest_reg':
        model = RandomForestRegressor(
            random_state=random_state_i,
            n_estimators=hyp_search_best.n_estimators,
            max_depth=hyp_search_best.max_depth,
        )

    return model


def apply_pca_lung_features(x_train, x_test, scaler, pca_config):
    # scale lung features
    lung_feature_names = [x for x in x_train.columns if 'lung' in x]
    x_train_lung = x_train[lung_feature_names]
    x_test_lung = x_test[lung_feature_names]
    scaler_lung_features = copy.deepcopy(scaler)
    scaler_lung_features.fit(x_train_lung)
    scaled_x_lung_train = scaler_lung_features.transform(x_train_lung)
    scaled_x_lung_test = scaler_lung_features.transform(x_test_lung)
    # apply PCA
    pca = PCA()
    pca.fit(scaled_x_lung_train)
    pca_lung_features_train = pca.transform(scaled_x_lung_train)
    pca_lung_features_test = pca.transform(scaled_x_lung_test)
    # select pca features above threshold
    cum_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    index_threshold_array = np.argwhere(cum_variance_ratio > pca_config['threshold'])
    index_threshold_val = index_threshold_array[0][0]
    if index_threshold_val == 0:
        pca_lung_features_train = pca_lung_features_train[:, 0]
        pca_lung_features_test = pca_lung_features_test[:, 0]
        pca_feature_names = ['pca0']
    else:
        pca_lung_features_train = pca_lung_features_train[:, 0:index_threshold_val]
        pca_lung_features_test = pca_lung_features_test[:, 0:index_threshold_val]
        pca_feature_names = ['pca' + str(i) for i in range(index_threshold_val)]

    other_feature_names = [x for x in x_train.columns if 'lung' not in x]

    if len(other_feature_names) != 0:
        # merge other features and pca
        x_train_other = x_train[other_feature_names]
        x_test_other = x_test[other_feature_names]
        scaler_other_features = copy.deepcopy(scaler)
        scaler_other_features.fit(x_train_other)
        scaled_x_other_train = scaler_other_features.transform(x_train_other)
        scaled_x_other_test = scaler_other_features.transform(x_test_other)
        feature_names_pca = np.concatenate((other_feature_names, pca_feature_names), axis=None)
        scaled_pca_x_train = np.concatenate((scaled_x_other_train, pca_lung_features_train), axis=1)
        scaled_pca_x_test = np.concatenate((scaled_x_other_test, pca_lung_features_test), axis=1)
    else:
        feature_names_pca = pca_feature_names
        scaled_pca_x_train = pca_lung_features_train
        scaled_pca_x_test = pca_lung_features_test
    return feature_names_pca, scaled_pca_x_train, scaled_pca_x_test


def save_model_parameters(save_results_dict, model_config, test_model, hyp_search):
    if model_config['model_type'] == 'logistic_classification':
        save_results_dict['penalty'].append(model_config['penalty'])
        save_results_dict['model_coefficients'].append(test_model.coef_[0])
        save_results_dict['log_reg_c'].append(hyp_search.best_estimator_.C)
        if model_config['penalty'] == 'elasticnet':
            save_results_dict['l1_ratio'].append(hyp_search.best_estimator_.l1_ratio)

    if model_config['model_type'] == 'random_forest_classification':
        save_results_dict['max_depth'].append(test_model.max_depth)
        save_results_dict['n_estimators'].append(test_model.n_estimators)
        save_results_dict['feature_importances'].append(test_model.feature_importances_)

    if model_config['model_type'] == 'poisson_reg':
        save_results_dict['poisson_model_weights'].append(test_model.coef_)
        save_results_dict['poisson_model_intercept'].append(test_model.intercept_)

    if model_config['model_type'] == 'random_forest_reg':
        save_results_dict['max_depth'].append(test_model.max_depth)
        save_results_dict['n_estimators'].append(test_model.n_estimators)
        save_results_dict['rf_feature_importances'].append(test_model.feature_importances_)

    return save_results_dict


def gest_age_stratified_analysis(
        ga_test_set, ga_bins,
        y_test, y_predictions,
        experiment_type,
        y_predict_proba=None,
        predictions_decision=None,
        n_classes=4):
    performance_dict = {}

    for temp_bin in ga_bins:
        bin_name = 'ga_bin' + '_' + str(temp_bin[0]) + '_' + str(temp_bin[1])
        bool_temp_bin = [True if temp_bin[1] > x >= temp_bin[0] else False for x in ga_test_set]
        filtered_y_test = y_test[bool_temp_bin]
        filtered_y_predictions = y_predictions[bool_temp_bin]

        if experiment_type == 'multinomial':
            filtered_y_predict_proba = y_predict_proba[bool_temp_bin]
            targets = filtered_y_test.reshape(-1)
            filtered_one_hot_y_test = np.eye(n_classes)[targets.astype(int)]
            # Calculate roc and auc per class for each GA bin
            temp_fpr = dict()
            temp_tpr = dict()
            temp_class_roc_auc = dict()
            for k in range(n_classes):
                temp_fpr[k], temp_tpr[k], _ = roc_curve(filtered_one_hot_y_test[:, k], filtered_y_predict_proba[:, k])
                temp_class_roc_auc[k] = auc(temp_fpr[k], temp_tpr[k])
            # Find overall model performance only if all classes are available
            nan_class = False
            for n in range(n_classes):
                if np.isnan(temp_class_roc_auc[n]):
                    nan_class = True
                    break
            # Weighted auc
            if nan_class:
                roc_auc_all_classes = np.nan
                f1_whole_all_classes = np.nan
            else:
                roc_auc_all_classes = roc_auc_score(
                    filtered_one_hot_y_test,
                    filtered_y_predict_proba,
                    average='macro'
                )
                f1_whole_all_classes = f1_score(filtered_y_test, filtered_y_predictions, average='macro')

            performance_dict[bin_name + '_roc_auc_all_classes'] = roc_auc_all_classes
            performance_dict[bin_name + '_f1_all_classes'] = f1_whole_all_classes
            performance_dict[bin_name + '_auc_no_bpd'] = temp_class_roc_auc[0]
            performance_dict[bin_name + '_auc_mild_bpd'] = temp_class_roc_auc[1]
            performance_dict[bin_name + '_auc_moderate_bpd'] = temp_class_roc_auc[2]
            performance_dict[bin_name + '_auc_severe_bpd'] = temp_class_roc_auc[3]

        elif experiment_type == 'binomial':
            filtered_y_predict_proba = y_predict_proba[bool_temp_bin]
            filtered_y_predictions = y_predictions[bool_temp_bin]

            if predictions_decision is not None:
                filtered_predictions_decision = predictions_decision[bool_temp_bin]
                temp_fpr, temp_tpr, _ = roc_curve(filtered_y_test, filtered_predictions_decision)
                roc_auc_temp = roc_auc_score(filtered_y_test, filtered_y_predict_proba)
                f1_temp = f1_score(filtered_y_test, filtered_y_predictions)
            else:
                temp_fpr, temp_tpr, _ = roc_curve(filtered_y_test, filtered_y_predict_proba)
                roc_auc_temp = roc_auc_score(filtered_y_test, filtered_y_predict_proba)
                f1_temp = f1_score(filtered_y_test, filtered_y_predictions)

            performance_dict[bin_name + '_roc_auc'] = roc_auc_temp
            performance_dict[bin_name + '_f1'] = f1_temp

        elif experiment_type == 'regression':
            if len(filtered_y_test) != 0:
                mse_temp = mean_squared_error(filtered_y_test, filtered_y_predictions)
                mae_temp = mean_absolute_error(filtered_y_test, filtered_y_predictions)
            else:
                mse_temp = np.nan
                mae_temp = np.nan
            performance_dict[bin_name + '_mean_squared_error'] = mse_temp
            performance_dict[bin_name + '_mean_absolute_error'] = mae_temp

    return performance_dict
