import os
import pandas as pd
import numpy as np

# Import custom functions
import env_functions as ef
import s3_functions as sf

# Determine the environment and get appropriate vars
deepnote, env_vars = ef.load_env_vars()

# Iterate through the vars and set them as global vars
for var_name, var in env_vars.items():
    globals()[var_name] = var

# If not in the DeepNote environment, create a dict for aws creds
#   that were located in the environment file.  This will be passed
#   to all aws s3 functions.
if not deepnote:
    aws_env_vars = {
        'access_key_id': aws_access_key_id,
        'secret_access_key': aws_secret_access_key,
        'bucket_name': s3_bucket_name
    }


def import_data(dn_path="/work/data/Xy_Data", s3_path="data/Xy_Data", location_name='CARB'):
    """
    This function imports the data from the specified data path and location geography
    """

    data_path = None

    if deepnote:
        data_path = dn_path
        # Import the training, validation and holdout data from DeepNote Local
        X_train_trans = pd.read_parquet(f"{data_path}/X_train_trans_{location_name}.parquet")
        X_val_trans = pd.read_parquet(f"{data_path}/X_val_trans_{location_name}.parquet")
        X_holdout_trans = pd.read_parquet(f"{data_path}/X_holdout_trans_{location_name}.parquet")

        y_train = pd.read_parquet(f"{data_path}/y_train_{location_name}.parquet")
        y_val = pd.read_parquet(f"{data_path}/y_val_{location_name}.parquet")
        y_holdout = pd.read_parquet(f"{data_path}/y_holdout_{location_name}.parquet")
    else:
        data_path = s3_path
        # Import the training, validation and holdout data from S3
        X_train_trans = pd.read_parquet(sf.load_from_s3(f"{data_path}/X_train_trans_{location_name}.parquet", **aws_env_vars))
        X_val_trans = pd.read_parquet(sf.load_from_s3(f"{data_path}/X_val_trans_{location_name}.parquet", **aws_env_vars))
        X_holdout_trans = pd.read_parquet(sf.load_from_s3(f"{data_path}/X_holdout_trans_{location_name}.parquet", **aws_env_vars))

        y_train = pd.read_parquet(sf.load_from_s3(f"{data_path}/y_train_{location_name}.parquet", **aws_env_vars))
        y_val = pd.read_parquet(sf.load_from_s3(f"{data_path}/y_val_{location_name}.parquet", **aws_env_vars))
        y_holdout = pd.read_parquet(sf.load_from_s3(f"{data_path}/y_holdout_{location_name}.parquet", **aws_env_vars))

    # y is not transformed
    y_train_trans = y_train
    y_val_trans = y_val
    y_holdout_trans = y_holdout

    return X_train_trans, X_val_trans, X_holdout_trans, y_train_trans, y_val_trans, y_holdout_trans

# Import some data temporarily, as some of the other functions use the Xy data in argument defaults
X_train_trans, X_val_trans, X_holdout_trans, y_train_trans, y_val_trans, y_holdout_trans = import_data(location_name='CARB')


def plot_feat_importance(model):
    """
    Plot the feature importance
    model: The model object itself
    """
    model_type_name = type(model).__module__ + "." + type(model).__name__

    #if isinstance(model, lightgbm.sklearn.LGBMRegressor):
    if model_type_name == "lightgbm.sklearn.LGBMRegressor":
        fi_df = pd.DataFrame(
            {'Feature': model.booster_.feature_name(), 'Importance': model.booster_.feature_importance()}).sort_values(
            by='Importance', ascending=True)

    #elif isinstance(model, sklearn.linear_model._coordinate_descent.ElasticNet):
    elif model_type_name == "sklearn.linear_model._coordinate_descent.ElasticNet":
        fi_df = pd.DataFrame(
            {'Feature': model.feature_names_in_, 'Importance': abs(model.coef_)}).sort_values(
            by='Importance', ascending=True)

    #elif isinstance(model, sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor):
    elif model_type_name == "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor":
        raise ValueError("HistGradientBoostingRegressor does not support Feature Importance")

    else:
        fi_df = pd.DataFrame(
            {'Feature':model.feature_names_in_,'Importance':model.feature_importances_}).sort_values(
            by='Importance', ascending=True)


    # Plot the feature importance
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', color_discrete_sequence=['darkorange'])
    fig.update_xaxes(categoryorder='total ascending')
    fig.update_layout(
        title={
            'text': "Feature Importance For The Model",
            'x': 0.5, 'xanchor': 'center',
            'y': 0.90, 'yanchor': 'top'},
        xaxis=dict(title="Feature Importance", title_standoff=2),
        yaxis=dict(title="Feature", title_standoff=0),
        height=400, width=650)
    fig.add_annotation(
        dict(text=f"Data Sources: Global Coral Beaching Database, World Bank WDI\nMarine Ecoregions of the World",
             x=0.5, y=-0.25, showarrow=False,
             font=dict(
                 size=10,
                 color="grey"),
             xref="paper", yref="paper", align="center"
             )
    )

    fig.show()


def write_out(model, trials, params, feat_cols=[], dn_path="/work/models", dns_path="/datasets/s3/models",
              s3_path="/models", model_family="xgboost_reg", location_name="CARB"):
    """
    Write out the model artifacts to disk

    model: The model object itself
    trial: The Hyperopt trials object
    params: The best model hyperparameters from the trials object
    common_path: The common path for model artifacts e.g. '/work/models'
    model_family: The model family, e.g. 'xgboost_reg', 'lightgbm_reg'
    location_name: The geography that the model is trained for, e.g. 'SEAA', 'CARB', 'GLOB'
    """
    date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dn_model_path = os.path.join(dn_path, model_family, location_name)
    dns_model_path = os.path.join(dns_path, model_family, location_name)
    s3_model_path = os.path.join(s3_path, model_family, location_name)

    if deepnote:
        # make sure the path exists
        os.makedirs(dn_model_path, exist_ok=True)
        os.makedirs(dns_model_path, exist_ok=True)

        # Write out the HyperOpt Trials object
        with open(dn_model_path + '/' + date_time_str + '_trials.pkl', 'wb') as f:
            pickle.dump(trials, f)
        with open(dns_model_path + '/' + date_time_str + '_trials.pkl', 'wb') as f:
            pickle.dump(trials, f)

        # Write out the XGBoost Model Object
        with open(dn_model_path + '/' + date_time_str + '_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(dns_model_path + '/' + date_time_str + '_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Write out the XGBoost Best Params
        with open(dn_model_path + '/' + date_time_str + '_params.pkl', 'wb') as f:
            pickle.dump(params, f)
        with open(dns_model_path + '/' + date_time_str + '_params.pkl', 'wb') as f:
            pickle.dump(params, f)

        # Write out the feature columns if they exist
        if len(feat_cols) > 0:
            with open(dn_model_path + '/' + date_time_str + '_feat_cols.pkl', 'wb') as f:
                pickle.dump(feat_cols, f)
            with open(dns_model_path + '/' + date_time_str + '_feat_cols.pkl', 'wb') as f:
                pickle.dump(feat_cols, f)
    else:
        # Write out the HyperOpt Trials object
        write_to_s3(file_path=f"{s3_model_path}/{date_time_str}_trials.pkl", data=trials, pickle_file=True, **aws_env_vars)

        # Write out the XGBoost Model Object
        write_to_s3(file_path=f"{s3_model_path}/{date_time_str}_model.pkl", data=model, pickle_file=True, **aws_env_vars)

        # Write out the XGBoost Best Params
        write_to_s3(file_path=f"{s3_model_path}/{date_time_str}_params.pkl", data=params, pickle_file=True, **aws_env_vars)

        # Write out the feature columns if they exist
        if len(feat_cols) > 0:
            write_to_s3(file_path=f"{s3_model_path}/{date_time_str}_feat_cols.pkl", data=feat_cols, pickle_file=True, **aws_env_vars)
