from __future__ import print_function
import sys

sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator as H2OPCA

def pca_max_k():
    data = h2o.upload_file(pyunit_utils.locate("bigdata/laptop/jira/rotterdam.csv.zip"))
    y = set(["relapse"])
    x = list(set(data.names) - y)

    pcaRandomized = H2OPCA(k=-1, transform="STANDARDIZE", pca_method="Randomized",
                           impute_missing=True, max_iterations=100, seed=12345)
    pcaRandomized.train(x, training_frame=data)

    pcaPower = H2OPCA(k=-1, transform="STANDARDIZE", pca_method="Power",
                      impute_missing=True, max_iterations=100, seed=12345)
    pcaPower.train(x, training_frame=data)
    # eigenvalues between the PCA and Randomize should be close, I hope...
    print("@@@@@@  Comparing eigenvalues between Randomized and Power PCA...\n")
    pyunit_utils.assert_H2OTwoDimTable_equal(pcaRandomized._model_json["output"]["importance"],
                                             pcaPower._model_json["output"]["importance"],
                                             ["Standard deviation", "Cumulative Proportion", "Cumulative Proportion"])

if __name__ == "__main__":
    pyunit_utils.standalone_test(pca_max_k)
else:
    pca_max_k()
