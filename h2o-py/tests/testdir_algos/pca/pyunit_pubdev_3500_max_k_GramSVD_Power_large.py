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

    pcaGramSVD = H2OPCA(k=-1, transform="STANDARDIZE", pca_method="GramSVD", impute_missing=True, max_iterations=100)
    pcaGramSVD.train(x, training_frame=data)
    pcaPower = H2OPCA(k=-1, transform="STANDARDIZE", pca_method="Power", impute_missing=True, max_iterations=100,
                  seed=12345)
    pcaPower.train(x, training_frame=data)

    # compare singular values and stuff with GramSVD
    print("@@@@@@  Comparing eigenvalues between GramSVD and Power...\n")
    pyunit_utils.assert_H2OTwoDimTable_equal(pcaGramSVD._model_json["output"]["importance"],
                                         pcaPower._model_json["output"]["importance"],
                                         ["Standard deviation", "Cumulative Proportion", "Cumulative Proportion"],
                                         tolerance=1)

    correctEigNum = pcaPower.full_parameters["k"]["actual_value"]
    gramSVDNum = len(pcaGramSVD._model_json["output"]["importance"].cell_values[0]) - 1
    powerNum = len(pcaPower._model_json["output"]["importance"].cell_values[0]) - 1
    assert correctEigNum == gramSVDNum, "PCA GramSVD FAIL: expected number of eigenvalues: " + correctEigNum + \
                                    ", actual: " + gramSVDNum + "."
    assert correctEigNum == powerNum, "PCA Power FAIL: expected number of eigenvalues: " + correctEigNum + \
                                  ", actual: " + powerNum + "."

if __name__ == "__main__":
    pyunit_utils.standalone_test(pca_max_k)
else:
    pca_max_k()
