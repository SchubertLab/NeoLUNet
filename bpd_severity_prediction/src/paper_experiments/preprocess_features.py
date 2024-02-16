def pre_process_features(df_in):
    df_regression = df_in
    # Add variable of respiratory support (invasive + non-invasive)
    resp_support_days = df_regression['respiration_inv_days'].to_numpy() + df_regression[
        'respiration_non_inv_days'].to_numpy()
    df_regression.insert(5, "resp_support_days", resp_support_days, True)

    # Change gender to numerical
    gender_dict = {'f': 0, 'm': 1}
    df_regression.replace({"gender": gender_dict}, inplace=True)

    # Create binary prediction variables
    binary_no_vs_all = df_regression['bpd_severity'].apply(lambda x: 0 if x < 1 else 1)
    df_regression.insert(1, "bpd_no_vs_all", binary_no_vs_all, True)

    # Create binary [low vs high] prediction variables
    binary_low_vs_high = df_regression['bpd_severity'].apply(lambda x: 0 if x < 2 else 1)
    df_regression.insert(2, "bpd_no-mild_vs_mod-sev", binary_low_vs_high, True)

    return df_regression

