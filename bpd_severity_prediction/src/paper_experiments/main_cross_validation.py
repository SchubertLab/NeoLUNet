import os
import copy
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils_cross_validation import *


def nested_cross_validation(
        X,
        y,
        cross_val_config,
        model_config,
        hyperparameters_config,
        feature_selection_config,
        pca_config,
        scaler,
        gest_age_analysis=None,
        clip_outliers=False,
):
    feature_names = X.columns
    random_states = cross_val_config['random_repetition_states']
    n_splits_outer = cross_val_config['n_splits_outer']
    n_splits_inner = cross_val_config['n_splits_inner']
    experiment_type = cross_val_config['experiment_type']
    k_fold_type = cross_val_config['k_fold_type']

    save_results_dict = initialize_results_dict(experiment_type)

    if gest_age_analysis is not None:
        gest_age_data = gest_age_analysis['data']
        bins_gest_age = gest_age_analysis['bins']
    else:
        gest_age_data, bins_gest_age = None, None

    # Outer loop for random seeds
    for i, random_state_i in enumerate(random_states):

        # performance for a complete repetition
        y_test_by_rep = []
        y_pred_by_rep = []
        gest_age_by_rep = []
        y_pred_proba_by_rep = None
        predictions_decision_by_rep = None

        model_outer_fold = build_outer_fold_model(model_config, random_state_i)

        # Outer Fold Cross Validation
        outer_cross_validation = get_k_folds_object(
            fold_type=k_fold_type,
            n_splits=n_splits_outer,
            random_state=random_state_i,
            shuffle_option=True,
        )

        outer_cv = outer_cross_validation.split(X, y)

        for j, (train, test) in enumerate(outer_cv):
            x_train = X.loc[train]
            y_train = y.loc[train]

            x_test = X.loc[test]
            y_test = y.loc[test]

            # Scale Features
            if pca_config is None:
                scaler_features = copy.deepcopy(scaler)
                scaler_features.fit(x_train)
                scaled_x_train = scaler_features.transform(x_train)
                scaled_x_test = scaler_features.transform(x_test)
            else:
                # PCA -> only for Lung features!
                feature_names, scaled_x_train, scaled_x_test = apply_pca_lung_features(
                    x_train=x_train,
                    x_test=x_test,
                    scaler=scaler,
                    pca_config=pca_config,
                )

            # Feature Selection
            selected_features_names = feature_names
            if feature_selection_config is not None:
                if feature_selection_config['feature_selection_method'] == 'univariate':
                    metric_selected = get_performance_metric(
                        feature_selection_config['parameters']['score_feature_selection'])
                    selector = SelectKBest(
                        metric_selected,
                        k=feature_selection_config['parameters']['select_k_features']
                    )
                    selector.fit(scaled_x_train, y_train)
                    selected_features_boolean = selector.get_support()
                    selected_features_names = feature_names[selected_features_boolean]
                    scaled_x_train = selector.transform(scaled_x_train)
                    scaled_x_test = selector.transform(scaled_x_test)

            # Clip outliers in scaled features
            if clip_outliers:
                max_std_clip_outliers = 10
                scaled_x_train = np.clip(scaled_x_train, -1*max_std_clip_outliers, max_std_clip_outliers)
                scaled_x_test = np.clip(scaled_x_test, -1*max_std_clip_outliers, max_std_clip_outliers)

            # Inner Fold
            inner_cross_validation = get_k_folds_object(
                fold_type=k_fold_type,
                n_splits=n_splits_inner,
                random_state=random_state_i,
                shuffle_option=True,
            )

            # Define type of hyperparameter search method
            if hyperparameters_config['method'] == 'RandomizedSearchCV':
                hyp_search = RandomizedSearchCV(
                    estimator=model_outer_fold,
                    param_distributions=hyperparameters_config['param_distributions'],
                    cv=inner_cross_validation,
                    scoring=hyperparameters_config['score_metric'],
                    n_iter=hyperparameters_config['n_min_iter_random_search'],
                )
                hyp_search.fit(scaled_x_train, y_train)

            # Use the best parameter of the inner fold on the test
            test_model = build_inner_fold_model(model_config, random_state_i, hyp_search.best_estimator_)
            test_model.fit(scaled_x_train, y_train)

            # Calculate Performance Metrics
            y_test_array = np.array(y_test)
            if experiment_type == 'multinomial':
                n_classes = cross_val_config['n_classes']
                # one hot encoding y_test
                targets = y_test_array.reshape(-1)
                one_hot_y_test = np.eye(n_classes)[targets]

                # Perform Predictions
                predictions = test_model.predict(scaled_x_test)
                y_predict_proba = test_model.predict_proba(scaled_x_test)

                # Calculate roc and auc per class
                temp_fpr = dict()
                temp_tpr = dict()
                temp_class_roc_auc = dict()
                for k in range(n_classes):
                    temp_fpr[k], temp_tpr[k], _ = roc_curve(one_hot_y_test[:, k], y_predict_proba[:, k])
                    temp_class_roc_auc[k] = auc(temp_fpr[k], temp_tpr[k])
                # whole model weighted auc
                roc_auc_all_classes = roc_auc_score(one_hot_y_test, y_predict_proba, average='macro')
                f1_all_classes = f1_score(y_test, predictions, average='macro')

            elif experiment_type == 'binomial':
                if model_config['model_type'] == 'logistic_classification':
                    predictions = test_model.predict(scaled_x_test)
                    predictions_decision = test_model.decision_function(scaled_x_test)
                    temp_fpr, temp_tpr, _ = roc_curve(y_test, predictions_decision)
                    y_predict_proba = test_model.predict_proba(scaled_x_test)[:, 1]
                    roc_auc_temp = roc_auc_score(y_test, y_predict_proba)
                    f1_temp = f1_score(y_test, predictions)
                elif model_config['model_type'] == 'random_forest_classification':
                    predictions = test_model.predict(scaled_x_test)
                    y_predict_proba = test_model.predict_proba(scaled_x_test)[:, 1]
                    temp_fpr, temp_tpr, _ = roc_curve(y_test, y_predict_proba)
                    roc_auc_temp = roc_auc_score(y_test, y_predict_proba)
                    f1_temp = f1_score(y_test, predictions)

            elif experiment_type == 'regression':
                predictions = test_model.predict(scaled_x_test)
                mse_temp = mean_squared_error(y_test, predictions)
                mae_temp = mean_absolute_error(y_test, predictions)
                y_predict_proba = None

            # Save iteration results ----------------------------------------------
            print('repetition_n', i)
            print('outer_fold', j)

            save_results_dict['repetition_n'].append(i)
            save_results_dict['outer_fold'].append(j)

            save_results_dict['selected_features_names'].append(selected_features_names)
            save_results_dict['model_type'].append(model_config['model_type'])
            save_results_dict['experiment_name'].append(cross_val_config['experiment_name'])

            save_results_dict['y_test'].append(y_test_array)
            save_results_dict['y_pred'].append(predictions)

            if experiment_type == 'multinomial' or experiment_type == 'binomial':
                save_results_dict['y_pred_proba'].append(y_predict_proba)

            # Save performance metrics
            if experiment_type == 'multinomial':
                save_results_dict['roc_auc_all_classes'].append(roc_auc_all_classes)
                save_results_dict['f1_all_classes'].append(f1_all_classes)
                save_results_dict['fpr_all_classes'].append(temp_fpr)
                save_results_dict['tpr_all_classes'].append(temp_tpr)
                save_results_dict['auc_no_bpd'].append(temp_class_roc_auc[0])
                save_results_dict['auc_mild_bpd'].append(temp_class_roc_auc[1])
                save_results_dict['auc_moderate_bpd'].append(temp_class_roc_auc[2])
                save_results_dict['auc_severe_bpd'].append(temp_class_roc_auc[3])

            elif experiment_type == 'binomial':
                save_results_dict['roc_auc'].append(roc_auc_temp)
                save_results_dict['f1'].append(f1_temp)
                save_results_dict['fpr'].append(temp_fpr)
                save_results_dict['tpr'].append(temp_tpr)

            elif experiment_type == 'regression':
                save_results_dict['mean_squared_error'].append(mse_temp)
                save_results_dict['mean_absolute_error'].append(mae_temp)

            # Save Model Parameters
            save_model_parameters(
                save_results_dict=save_results_dict,
                model_config=model_config,
                test_model=test_model,
                hyp_search=hyp_search,
            )

            # Save PCA Results
            if pca_config is not None:
                save_results_dict['apply_pca_lung_features'].append(True)
                save_results_dict['pca_threshold'].append(pca_config['threshold'])
            else:
                save_results_dict['apply_pca_lung_features'].append(False)
                save_results_dict['pca_threshold'].append(None)

            # Save info for GA analysis
            if gest_age_analysis is not None:
                if 'gest_age_test_vals' not in save_results_dict:
                    save_results_dict['gest_age_test_vals'] = []
                ga_age_test = gest_age_data.loc[test]
                save_results_dict['gest_age_test_vals'].append(ga_age_test.to_numpy())

            # Save performance by repetition
            y_test_by_rep = np.concatenate((y_test_by_rep, y_test_array), axis=None)
            y_pred_by_rep = np.concatenate((y_pred_by_rep, predictions), axis=None)
            gest_age_by_rep = np.concatenate((gest_age_by_rep, ga_age_test.to_numpy()), axis=None)
            if experiment_type == 'multinomial':
                if y_pred_proba_by_rep is not None:
                    y_pred_proba_by_rep = np.concatenate((y_pred_proba_by_rep, y_predict_proba), axis=0)
                else:
                    y_pred_proba_by_rep = y_predict_proba

            elif experiment_type == 'binomial':
                if y_pred_proba_by_rep is not None:
                    y_pred_proba_by_rep = np.concatenate((y_pred_proba_by_rep, y_predict_proba), axis=0)
                else:
                    y_pred_proba_by_rep = y_predict_proba
                if model_config['model_type'] == 'logistic_classification':
                    if predictions_decision_by_rep is not None:
                        predictions_decision_by_rep = np.concatenate(
                            (predictions_decision_by_rep, predictions_decision), axis=0)
                    else:
                        predictions_decision_by_rep = predictions_decision

        # Perform GA analysis by repetition (considering all outer folds)
        if gest_age_analysis is not None:
            performance_metrics_ga = gest_age_stratified_analysis(
                ga_test_set=gest_age_by_rep,
                ga_bins=bins_gest_age,
                y_test=y_test_by_rep,
                y_predictions=y_pred_by_rep,
                y_predict_proba=y_pred_proba_by_rep,
                experiment_type=experiment_type,
                predictions_decision=predictions_decision_by_rep,
            )
        else:
            performance_metrics_ga = None

        # Save gest_age test values and performance by gest_age_bin
        if gest_age_analysis is not None:
            for key, value in performance_metrics_ga.items():
                if key not in save_results_dict:
                    save_results_dict[key] = []
                for i_n_reps in range(n_splits_outer):
                    save_results_dict[key].append(value)

    # Fill None for empty fields
    current_len = len(save_results_dict['model_type'])
    for key, value in save_results_dict.items():
        if not value:
            save_results_dict[key] = [None] * current_len

    df_results = pd.DataFrame.from_dict(save_results_dict)

    return df_results
