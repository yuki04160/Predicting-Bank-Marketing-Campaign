"""
<Project Title>

Copyright (c) 2021
Licensed
Written by <Kuan-Pei Lai>
"""

def upper_outlier_percentage(df, col):
    # Q3 + 1.5 * IQR
    outlier_uplimit = df[col].quantile(q = 0.75) + 1.5*(df[col].quantile(q = 0.75) - df[col].quantile(q = 0.25))
    outlier_uplimit_count = len(df[df[col] > outlier_uplimit])
    outlier_uplimit_per = outlier_uplimit_count / len(df)
    outlier_uplimit_per = "{:.2%}".format(outlier_uplimit_per)
    return print(f'{col} has {outlier_uplimit_per} outliers.')


def create_pie(df, col):
    group = df.groupby(col).size()
    plt = group.plot.pie(autopct="%.1f%%").set(ylabel=None)
    return plt


import numpy as np
def table_of_target_var(y_train):
    target = np.bincount(y_train)
    ii = np.nonzero(target)[0]
    return print(np.vstack((ii, target[ii])).T)


import pandas as pd
def cat_to_dummy(cate_col, X):
    for col in cate_col:
        X = pd.concat([X.drop(col, axis=1),
                           pd.get_dummies(X[col], prefix=col, prefix_sep='_',
                                          drop_first=True, dummy_na=False)], axis=1)
    return X


def cat_to_one_hot(cate_col, X):
    for col in cate_col:
        X = pd.concat([X.drop(col, axis=1),
                           pd.get_dummies(X[col], prefix=col, prefix_sep='_',
                                          drop_first=False, dummy_na=False)], axis=1)
    return X


def delete_upper_outliers(df, col):
    outlier_uplimit = df[col].quantile(q = 0.75) + 1.5*(df[col].quantile(q = 0.75) - df[col].quantile(q = 0.25))
    clean_df = df.loc[df[col] <= outlier_uplimit, :]
    clean_df
    return clean_df


import matplotlib.pyplot as plt
def ROC_curve(y_test, model_pred):
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_test, model_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt


def plot_top5_feature_importances(X_train, feature_importances):
    # Build feature importances dataframe
    headers = ["variables", "score"]
    values = sorted(zip(X_train.columns, feature_importances), key=lambda x: x[1] * -1)
    df_feature_importances = pd.DataFrame(values, columns = headers)
    top5_df_feature_importances = df_feature_importances[:5]
    
    # Plot top 5 feature importances
    x_pos = np.arange(0, len(top5_df_feature_importances))
    plt.bar(x_pos, top5_df_feature_importances['score'])
    plt.xticks(x_pos, top5_df_feature_importances['variables'])
    return plt