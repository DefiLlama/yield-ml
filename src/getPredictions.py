import tempfile
import json
from typing import Dict, List

import boto3
import joblib
import numpy as np


def read_artefacts(
    bucket: str,
    prefix: str,
    files: List[str],
) -> Dict[str, object]:
    """read trained model and feature_list artefacts from s3"""

    s3 = boto3.client("s3")

    artefacts = {}
    for i in files:
        with tempfile.TemporaryFile() as fp:
            s3.download_fileobj(Fileobj=fp, Bucket=bucket, Key=f"{prefix}/{i}")
            fp.seek(0)
            f = joblib.load(fp)

            artefacts[i.replace(".joblib", "")] = f

    return artefacts


# call this once outside the handler to cache the trained instance
artefacts = read_artefacts(
    bucket="llama-apy-prediction-prod",
    prefix="mlmodelartefacts",
    files=["clf.joblib", "feature_list.joblib"],
)
clf = artefacts["clf"]
feature_list = artefacts["feature_list"]


def handler(event, context) -> np.array:
    """handler function for inference"""

    # parse body
    X = json.loads(event["body"])
    # will look like:
    # [
    #     {
    #         "chain_factorized": 0,
    #         "project_factorized": 1,
    #         "apy": 10,
    #         "tvlUsd": 100000,
    #         "apyMeanExpanding": 20,
    #         "apyStdExpanding": 12,
    #     },
    #     {
    #         "apy": 100,
    #         "tvlUsd": 1000000,
    #         "apyMeanExpanding": 20,
    #         "apyStdExpanding": 1,
    #         "chain_factorized": 0,
    #         "project_factorized": 1,
    #     },
    # ]

    # we make sure the order of feature fields matches exactly the order in feature_list
    # reason: sklearn models will take the feature order as is, so need to make sure
    # that everything is aligned
    index_map = {v: i for i, v in enumerate(feature_list)}
    X = [sorted(x.items(), key=lambda i: index_map[i[0]]) for x in X]
    # will look like:
    # [
    #     [
    #         ("apy", 10),
    #         ("tvlUsd", 100000),
    #         ("apyMeanExpanding", 20),
    #         ("apyStdExpanding", 12),
    #         ("chain_factorized", 0),
    #         ("project_factorized", 1),
    #     ],
    #     [
    #         ("apy", 100),
    #         ("tvlUsd", 1000000),
    #         ("apyMeanExpanding", 20),
    #         ("apyStdExpanding", 1),
    #         ("chain_factorized", 0),
    #         ("project_factorized", 1),
    #     ],
    # ]

    # cast X into required numpy array
    # we extract feature values only (index 1), and cast the list of lists into a numpy array
    # of shape (nbSamples, nbFeatures)
    X = np.array([[i[1] for i in x] for x in X])
    print("np array data shape", X.shape)

    # call the model
    y_pred = clf.predict_proba(X)

    return {"predictions": y_pred.tolist()}
