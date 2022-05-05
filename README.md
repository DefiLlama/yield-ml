### ML for apy dashboard

Objective:

- Predict pool "apy runway"

Machine learning aspects:

- target framing: Instead of trying to estimate the nb of days a pool can keep up current apy (a regression problem)
  think going with binary classification makes more sense. Reason: less noise, therefore likely better results and we get a probability distribution
  on top. So this is out target:
  _what is the probability a pool can keep up its apy (within a defined range) for the next 4weeks?_

- inference frequency: we call the model every hour

- metric: roc auc

- data: use full historical snapshot but reduce to daily granularity, either use only a specific data point (eg midnight) or aggregate daily ones

- X: everything i got so far and more, backward looking stuff such as rolling/expanding stats probably very useful

- y: important to calculate target on a sorted pool level. training though will be iid on full batch. some first ideas how to design the target in more detail:

```
2. encode target:

version A) [very simple]
y = {
    0: if ((apyFuture / apy) - 1) < 0
    1: else
}

version B) [not as strict and imo more useful]
y = {
    0: if ((apyFuture / apy) - 1) < -20
    1: else
}

where apyFuture can be multiple things, a few which come to mind:
- a) simply the apy 30day in the future (could be a baseline)
- b) the avg of 30day apy values
- c) the median of 30day apy values to correct for potential outliers on apy series
```

- use baseline, logistic regression and random forest oob with minimal settings, don't waste much time on tuning until we happy

### notes regarding lambda layer/serverless

includes a simple lambda handler function for ML inference.
the steps:

- loads the saved model artefact from s3
- casts the incoming data into a numpy array
- calls the `predict` method on the model
- returns the required prediction arrays

### notes for installation process

python runtime lambdas are a bit of a pain when using scientific computing libraries as dependencies (numpy, scikit-learn, scipy etc). multiple difficulties:

- the dependencies are very large (~100mb together) and can only be compressed to ~80mb
- must be built on linux or via docker container otherwise the python lambda runtime environment can't import the necessary libs
- size is so large that testing and changing the handler function gets very slow

hence i chose the following process instead:

- building a lambda layer with all required dependencies using the `createlayer.sh` script
- uploading the created zip file to s3
- creating a layer manually via aws console
- adding that layer to the handler function inside serverless.yml
- and only then calling `sls deploy`

The result will be a very small zip file for the lambda as the layer is already build and completly isolated

Layer consits of sklearn only (which will install numpy, scipy, joblib and threadpoolctl). All versions are the latest except scipy, had to downgrade from 1.8.0 to 1.4.1 to get rid of

```
[ERROR] Runtime.ImportModuleError: Unable to import module 'handler': /opt/python/lib/python3.8/site-packages/scipy/linalg/_fblas.cpython-38-x86_64-linux-gnu.so: ELF load command address/offset not properly aligned
```

upload layer via `aws s3 cp layer.zip bucket`
