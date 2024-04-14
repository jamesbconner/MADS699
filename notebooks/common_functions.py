# Import Standard Libraries
import os
import datetime
import pickle
import itertools
import pandas as pd
import numpy as np

# Import custom functions
import env_functions as ef
import s3_functions as sf

# Import model libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Import evaluation libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import neptune

# Import Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap

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


def import_data(dn_path="/work/data/Xy_Data", s3_path="data/Xy_Data", location_name=None):
    """
    This function imports the data from the specified data path and location geography
    """

    if location_name == None:
        raise ValueError("'location_name' must be set")
    elif location_name not in ['GLOB','CARB','SEAA']:
        raise ValueError("'location_name' must be either GLOB, CARB or SEAA")
    else:
        pass

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
              s3_path="models", model_family=None, location_name=None):
    """
    Write out the model artifacts to disk

    model: The model object itself
    trial: The Hyperopt trials object
    params: The best model hyperparameters from the trials object
    common_path: The common path for model artifacts e.g. '/work/models'
    model_family: The model family, e.g. 'xgboost_reg', 'lightgbm_reg'
    location_name: The geography that the model is trained for, e.g. 'SEAA', 'CARB', 'GLOB'
    """

    if location_name == None:
        raise ValueError("'location_name' must be set")
    elif location_name not in ['GLOB','CARB','SEAA']:
        raise ValueError("'location_name' must be either GLOB, CARB or SEAA")
    else:
        pass

    if model_family == None:
        raise ValueError("'model_family' must be set")
    elif model_family not in ['elasticnet_reg', 'histgradboost_reg', 'lightgbm_reg', 'randomforest_reg', 'xgboost_reg']:
        raise ValueError("'model_family' must be either elasticnet_reg, histgradboost_reg, lightgbm_reg, randomforest_reg, or xgboost_reg")
    else:
        pass

    date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dn_model_path = os.path.join(dn_path, model_family, location_name)
    dns_model_path = os.path.join(dns_path, model_family, location_name)
    #s3_model_path = os.path.join(s3_path, model_family, location_name)

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
        sf.write_to_s3(file_path=f"{s3_path}/{model_family}/{location_name}/{date_time_str}_trials.pkl", data=trials, pickle_file=True, **aws_env_vars)

        # Write out the XGBoost Model Object
        sf.write_to_s3(file_path=f"{s3_path}/{model_family}/{location_name}/{date_time_str}_model.pkl", data=model, pickle_file=True, **aws_env_vars)

        # Write out the XGBoost Best Params
        sf.write_to_s3(file_path=f"{s3_path}/{model_family}/{location_name}/{date_time_str}_params.pkl", data=params, pickle_file=True, **aws_env_vars)

        # Write out the feature columns if they exist
        if len(feat_cols) > 0:
            sf.write_to_s3(file_path=f"{s3_path}/{model_family}/{location_name}/{date_time_str}_feat_cols.pkl", data=feat_cols, pickle_file=True, **aws_env_vars)


def model_score(hps, val=True, train=False, holdout=False, Xtt=None, ytt=None, Xvt=None, yvt=None, Xht=None, yht=None, model_type=None):
    """
    This function rebuilds the model with the desired hyperparameters
    val: True will evaluate the model on the validation data
    holdout: True will evaluate the model on the holdout data
    train: True will evaluate the model on the train data
    Xtt: X_train_trans
    ytt: y_train_trans
    Xvt: X_val_trans
    yvt: y_val_trans
    Xht: X_holdout_trans
    yht: y_holdout_trans
    model_type: The type of model to fit and score
    """

    # Identify if there are missing data objects
    if not all(x is not None for x in [Xtt, ytt, Xvt, yvt, Xht, yht]):
        raise ValueError("Data is missing (all parameters for X and y data must be provided)")

    # Create the model object
    if model_type == None:
        raise ValueError("'model_type' argument must be set")

    elif model_type == 'xgb':
        model = xgb.XGBRegressor(**hps, random_state=42)

    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(**hps, verbose=-1, random_state=42, n_jobs=-1)

    elif model_type == 'hgbm':
        model = HistGradientBoostingRegressor(**hps, random_state=42)

    elif model_type == 'rf':
        model = RandomForestRegressor(**hps, random_state=42)

    elif model_type == 'enet':
        model = ElasticNet(**hps, random_state=42)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Fit the model against the training data
    model.fit(Xtt, ytt)

    # Evaluate model and print results
    if train:
        # Evaluation on train data
        train_pred = model.predict(Xtt)
        train_pred = np.clip(train_pred, 0, 100)
        train_mae = mean_absolute_error(ytt, train_pred)
        train_mse = mean_squared_error(ytt, train_pred)
        train_rmse = mean_squared_error(ytt, train_pred, squared=False)
        train_rsq = r2_score(ytt, train_pred)
        print(" ")
        print(f"Train Mean Absolute Error: {train_mae:.4f}")
        print(f"Train Mean Squared Error: {train_mse:.4f}")
        print(f"Train Root Mean Squared Error: {train_rmse:.4f}")
        print(f"Train R^2 Score: {train_rsq:.4f}")

    if val:
        # Evaluation on validation data
        val_pred = model.predict(Xvt)
        val_pred = np.clip(val_pred, 0, 100)
        val_mae = mean_absolute_error(yvt, val_pred)
        val_mse = mean_squared_error(yvt, val_pred)
        val_rmse = mean_squared_error(yvt, val_pred, squared=False)
        val_rsq = r2_score(yvt, val_pred)
        print(" ")
        print(f"Validation Mean Absolute Error: {val_mae:.4f}")
        print(f"Validation Mean Squared Error: {val_mse:.4f}")
        print(f"Validation Root Mean Squared Error: {val_rmse:.4f}")
        print(f"Validation R^2 Score: {val_rsq:.4f}")

    if holdout:
        # Evaluation on holdout data
        holdout_pred = model.predict(Xht)
        holdout_pred = np.clip(holdout_pred, 0, 100)
        holdout_mae = mean_absolute_error(yht, holdout_pred)
        holdout_mse = mean_squared_error(yht, holdout_pred)
        holdout_rmse = mean_squared_error(yht, holdout_pred, squared=False)
        holdout_rsq = r2_score(yht, holdout_pred)
        print(" ")
        print(f"Holdout Mean Absolute Error: {holdout_mae:.4f}")
        print(f"Holdout Mean Squared Error: {holdout_mse:.4f}")
        print(f"Holdout Root Mean Squared Error: {holdout_rmse:.4f}")
        print(f"Holdout R^2 Score: {holdout_rsq:.4f}")

    return model


def feat_ablation(model, hps, Xtt, ytt, Xvt, yvt, Xht, yht, abl_list_to_combo=[], model_type=None):
    """
    Perform feature ablation analysis
    model: The model object itself
    Xtt: X_train_trans
    ytt: y_train_trans
    Xvt: X_val_trans
    yvt: y_val_trans
    Xht: X_holdout_trans
    yht: y_holdout_trans
    """

    # Create the base model object with hyperparameters
    # Note: this model will be refit during the ablation loop
    abl_model = model_score(hps, Xtt=Xtt, ytt=ytt, Xvt=Xvt, yvt=yvt, Xht=Xht, yht=yht, train=False, val=False,
                            holdout=False, model_type=model_type)

    # Evaluation on train data
    train_pred = abl_model.predict(Xtt)
    train_pred = np.clip(train_pred, 0, 100)
    baseline_mae_train = mean_absolute_error(ytt, train_pred)

    # Evaluation on validation data
    val_pred = abl_model.predict(Xvt)
    val_pred = np.clip(val_pred, 0, 100)
    baseline_mae_val = mean_absolute_error(yvt, val_pred)

    print(f"Baseline Mean MAE: {baseline_mae_train:.4f}, Validation MAE: {baseline_mae_val:.4f}")

    # Features for ablation
    # Start with all features
    abl_list = [[x] for x in Xtt.columns]

    # Now create combos of features passed in as abl_list_to_combo
    # Create combos of all items in abl_list_to_combo
    # Range stars at 2 to skip single columns
    abl_combo_list = [combo for r in range(2, len(abl_list_to_combo) + 1)
                      for combo in itertools.combinations(abl_list_to_combo, r)]

    # Itertools combinations() creates tuples.
    #   Convert each combination from a tuple to a list for ablation
    abl_combo_list = [list(combo) for combo in abl_combo_list]

    # Add the ablation combos to the ablation list
    abl_list = abl_list + abl_combo_list

    # Create the ablation loop
    ablation_results_list = []

    # Feat ablation loop
    for feature in abl_list:
        # drop ablated cols
        modified_X_train_trans = Xtt.drop(columns=feature)
        modified_X_val_trans = Xvt.drop(columns=feature)

        # Fit the model with ablated features
        abl_model.fit(modified_X_train_trans, ytt)

        # Evaluation on train data
        modified_train_predictions = abl_model.predict(modified_X_train_trans)
        modified_train_predictions = np.clip(modified_train_predictions, 0, 100)
        modified_mae_train = mean_absolute_error(ytt, modified_train_predictions)

        # Evaluation on validation data
        modified_val_predictions = abl_model.predict(modified_X_val_trans)
        modified_val_predictions = np.clip(modified_val_predictions, 0, 100)
        modified_mae_val = mean_absolute_error(yvt, modified_val_predictions)

        # Calculate MAE changes
        mae_change_train = baseline_mae_train - modified_mae_train
        mae_change_val = baseline_mae_val - modified_mae_val

        ablation_result_dict = {
            'Removed_Feature': ", ".join(feature),
            'Train_MAE': modified_mae_train,
            'Train_MAE_Change': mae_change_train,
            'Train_MAE_Pct_Change': 100 * (1 - (modified_mae_train / baseline_mae_train)),
            'Val_MAE': modified_mae_val,
            'Val_MAE_Change': mae_change_val,
            'Val_MAE_Pct_Change': 100 * (1 - (modified_mae_val / baseline_mae_val))
        }

        ablation_results_list.append(ablation_result_dict)

    feature_ablation_df = pd.DataFrame(ablation_results_list)

    return feature_ablation_df, baseline_mae_val, baseline_mae_train