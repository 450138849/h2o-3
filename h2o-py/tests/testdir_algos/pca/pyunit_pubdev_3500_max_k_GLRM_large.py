from __future__ import print_function
import sys

sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator as H2OPCA

def pca_max_k():
    """
    In this test, no comparisons are made with other PCA methods.  This is due to the fact that the dataset
    contains enum columns.  The metric used for GLRM when dealing with enum columns differ from the other
    PCA methods.  Hence, the results returned will be very different.  
    """
    data = h2o.upload_file(pyunit_utils.locate("bigdata/laptop/jira/rotterdam.csv.zip"))
    y = set(["relapse"])
    x = list(set(data.names) - y)

    pcaGLRM = H2OPCA(k=-1, transform="STANDARDIZE", pca_method="GLRM", use_all_factor_levels=True,
                     max_iterations=100, seed=12345)
    pcaGLRM.train(x, training_frame=data)
    correctEigNum = pcaGLRM.full_parameters["k"]["actual_value"]
    glrmNum = len(pcaGLRM._model_json["output"]["importance"].cell_values[0]) - 1
    assert correctEigNum == glrmNum, "PCA GLRM FAIL: expected number of eigenvalues: " + correctEigNum + \
                                     ", actual: " + glrmNum + "."


if __name__ == "__main__":
    pyunit_utils.standalone_test(pca_max_k)
else:
    pca_max_k()
