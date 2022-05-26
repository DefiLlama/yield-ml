import json

import boto3
import pandas as pd

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.pipeline.column_mapping import ColumnMapping

if __name__ == "__main__":

    s3 = boto3.client("s3")

    # read datasets
    df_current = s3.get_object(
        Bucket="llama-apy-prod-data", Key="enriched/dataEnriched.json"
    )
    df_current = pd.DataFrame(json.loads(df_current["Body"].read().decode()))

    df_reference = s3.get_object(
        Bucket="llama-apy-prediction-prod",
        Key="mlmodelartefacts/reference_data_2022_05_20.json",
    )
    df_reference = pd.DataFrame(json.loads(df_reference["Body"].read().decode()))

    features = [
        "apy",
        "tvlUsd",
        "apyMeanExpanding",
        "apyStdExpanding",
        "chain_factorized",
        "project_factorized",
    ]

    # remove outliers based on reference data max values
    for f in ["apy", "apyMeanExpanding", "apyStdExpanding"]:
        df_current = df_current[df_current[f] < df_reference[f].max()]

    # replace NaN in categorial features and remove -1 from categorical features
    df_reference = df_reference.fillna(-1)
    cat_features = ["chain_factorized", "project_factorized"]
    df_reference[cat_features] = df_reference[cat_features].astype(int)

    for f in cat_features:
        df_current = df_current[df_current[f] != -1]
        df_reference = df_reference[df_reference[f] != -1]

    cm = ColumnMapping()
    cm.numerical_features = features[:-2]
    cm.categorical_features = cat_features

    # run checks and create notebook
    drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    drift_dashboard.calculate(
        df_reference[features], df_current[features], column_mapping=cm
    )

    # store the output
    p = "/home/ec2-user/workspace/yield-ml-feature-drift-dashboard/ml_dashboard_output"
    drift_dashboard.save(f"{p}/drift_check.html")
