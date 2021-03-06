import json
import tempfile
from typing import List, Tuple, Dict

import boto3
import joblib
import numpy as np
from tabulate import tabulate
from discord_webhook import DiscordWebhook
from scipy.stats import tstd, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def handler(event, context):
    # load datasets
    X_reference, X_current, features = read_datasets(
        bucket_reference="llama-apy-prediction-prod",
        bucket_current="llama-apy-prod-data",
        prefix_reference="mlmodelartefacts/reference_data_2022_05_20.json",
        prefix_current="enriched/dataEnriched.json",
        prefix_features="mlmodelartefacts/feature_list.joblib",
    )
    # run different stats for numerical and categorical features
    num = calculate_wasserstein_distance(
        X_reference[:, :-2], X_current[:, :-2], features[:-2], 0.1
    )
    cat = calculate_jensenhannon_distance(
        X_reference[:, -2:], X_current[:, -2:], features[-2:], 0.1
    )
    stats = num + cat
    # format the stats for discord
    stats_table = tabulate(stats, headers="keys", showindex=True, tablefmt="github")
    print(f"\ntest results:\n{stats_table}\n")

    # send discord alarm if >= 1 feature drift
    if len([i for i in stats if i["drift"] == True]) > 0:
        ssm = boto3.client("ssm")
        ssm_path_webhook = (
            "/llama-apy/serverless/sls-authenticate/yield-ml-discord-webhook"
        )
        url = ssm.get_parameter(Name=ssm_path_webhook, WithDecryption=True)
        msg_string = f"Feature Drift Detected`​`​`​\n{stats_table}\n`​`​`​"

        # send msg
        webhook = DiscordWebhook(url=url["Parameter"]["Value"], content=msg_string)
        webhook.execute()


def read_datasets(
    bucket_reference: str,
    bucket_current: str,
    prefix_reference: str,
    prefix_current: str,
    prefix_features: str,
) -> Tuple[np.array, np.array, List[str]]:

    s3 = boto3.client("s3")

    # features
    with tempfile.TemporaryFile() as fp:
        s3.download_fileobj(Fileobj=fp, Bucket=bucket_reference, Key=prefix_features)
        fp.seek(0)
        features = joblib.load(fp)

    # training data
    data_reference = json.loads(
        s3.get_object(Bucket=bucket_reference, Key=prefix_reference)["Body"]
        .read()
        .decode()
    )
    # latest data
    data_current = json.loads(
        s3.get_object(Bucket=bucket_current, Key=prefix_current)["Body"].read().decode()
    )
    # keep feature fields only
    f = lambda data: [{k: v for k, v in p.items() if k in features} for p in data]
    data_current = f(data_current)
    data_reference = f(data_reference)

    # sort features
    index_map = {v: i for i, v in enumerate(features)}
    f = lambda data: [sorted(x.items(), key=lambda i: index_map[i[0]]) for x in data]
    data_current = f(data_current)
    data_reference = f(data_reference)

    # cast to numpy array
    f = lambda data: np.array([[i[1] for i in x] for x in data])
    X_current = f(data_current)
    X_reference = f(data_reference)

    # replace potential None values with -1 (should only occur in the categorical features)
    # and only in X_reference, but will do it for X_current as well just in case
    X_current[X_current == None] = -1
    X_reference[X_reference == None] = -1

    # remove outliers on current data (even a small amount of large outliers in small pools would cause
    # the tests to fail which results in lots of false positives for data drift check)
    print("max values btw reference and current:")
    for f in ["apy", "apyMeanExpanding", "apyStdExpanding"]:
        f_idx = features.index(f)
        f_reference_max = X_reference[:, f_idx].max()
        print(
            f"""
            feature: {f} 
            {X_current[:, f_idx].max()} [current data] 
            {f_reference_max} [reference data]
            {sum(X_current[:, f_idx] >= f_reference_max)} [nb above max]
            """
        )
        X_current = X_current[X_current[:, f_idx] < f_reference_max, :]

    return X_reference, X_current, features


def calculate_wasserstein_distance(
    X_reference: np.array,
    X_current: np.array,
    features: List[str],
    thr: float,
) -> List[dict]:

    d = []
    for i, f in enumerate(features):
        # normalise wasserstein distance by std
        stat = wasserstein_distance(X_reference[:, i], X_current[:, i]) / max(
            tstd(X_reference[:, i]), 0.001
        )
        d.append(
            {
                "feature": f,
                "distance": "wasserstein_distance",
                "value": stat,
                "drift": bool(stat >= thr),
            }
        )

    return d


def calculate_jensenhannon_distance(
    X_reference: np.array,
    X_current: np.array,
    features: List[str],
    thr: float,
) -> List[dict]:

    # projects and chains which haven't been seen by the algorithm during model training
    # will be encoded as -1 before inputting to the model. For the drift check of these 2
    # categorical features i will remove all samples with -1 in both `chain_factorized`
    # and `project_factorized` so that the drift check doesn't throw a false alarm
    # note: i deliberately apply this filter only inside this function because i want to catch
    # feature drifts in numerical features in case its driven by a new chain/project
    exclude = -1
    print("remove -1 values from categorical features:")
    for i, f in enumerate(features):
        print(f"\tNb {f} == -1: {sum(X_current[:, i] == -1)}")
        X_current = X_current[X_current[:, i].astype(int) != exclude, :]
        X_reference = X_reference[X_reference[:, i].astype(int) != exclude, :]

    d = []
    for i, f in enumerate(features):
        reference_percents, current_percents = get_binned_data(
            X_reference[:, i], X_current[:, i]
        )
        stat = jensenshannon(reference_percents, current_percents)
        d.append(
            {
                "feature": f,
                "distance": "jensenhannon_distance",
                "value": stat,
                "drift": bool(stat >= thr),
            }
        )

    return d


# from evidentlyai (modified)
# https://github.com/evidentlyai/evidently/blob/main/src/evidently/analyzers/stattests/utils.py
def get_binned_data(
    reference: np.array,
    current: np.array,
) -> Tuple[np.array, np.array]:
    """Split variable into n buckets based on reference quantiles
    Args:
        reference: reference data
        current: current data
    Returns:
        reference_percents: % of records in each bucket for reference
        current_percents: % of records in each bucket for reference
    """

    def get_value_counts(X: np.array) -> Dict:
        # value counts numpy style
        values, counts = np.unique(X, return_counts=True)
        # get index of sorted counts in descending order
        s = np.argsort(counts)[::-1]
        return {k: v for k, v in zip(values[s], counts[s])}

    dict_value_counts_reference = get_value_counts(reference)
    dict_value_counts_current = get_value_counts(current)

    keys = list((set(reference) | set(current)) - {np.nan})

    ref_feature_dict = {**dict.fromkeys(keys, 0), **dict_value_counts_reference}
    current_feature_dict = {
        **dict.fromkeys(keys, 0),
        **dict_value_counts_current,
    }

    reference_percents = np.array(
        [ref_feature_dict[key] / len(reference) for key in keys]
    )
    current_percents = np.array(
        [current_feature_dict[key] / len(current) for key in keys]
    )

    return reference_percents, current_percents
