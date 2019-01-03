Reproducibility
^^^^^^^^^^^^^^^

- **Given the same training set and the same GBM parameters, will GBM produce a different model with two different validation data sets, or the same model?**

 The same model will be generated, unless early stopping is turned on (by default it is turned off), which could lead to slightly different models. Using different validation sets would only affect the model being built if early stopping is turned on - early stopping uses the validation set to determine when to stop building more trees. 

- **How deterministic is GBM?**

 As long as you set the seed, GBM is deterministic up to floating point rounding errors (out-of-order atomic addition of multiple threads during histogram building). This means that if you set a seed, your results will be reproducible (even if, for example, you change the number of nodes in your cluster, change the way you ingest data, or change the number of files your data lives in, among many other examples).
 
- **How to guarantee reproducibility in single node cluster?**

 The following criteria must be met to guarantee reproducibility in a single node cluster:
 
 - same training data
 
   - **Note**: If you have H2O import a whole directory with multiple files instead of a single file, we do not guarantee reproducibility because the data may be shuffled during import.
   
 - same parameters used to train the model
 - same seed set (this is required when any sampling is done)
 - no early stopping or early stopping with ``score_tree_interval`` set and same validation data

- **How to guarantee reproducibility in multinode clusters?**

 The following criteria must be met to guarantee reproducibility in a multinode cluster:

 - the cluster configuration is the same
 - the model training is triggered from the leader node of the cluster
 - reproducible requirements for single node cluster are met (see above)

 For two clusters to be considered the same, they need to have the same number of nodes and the nodes need to have the same number of CPU cores available (or the same restriction on number of threads). When H2O is running on Hadoop, the leader node is automatically returned by the h2odriver as the node that the user should connect to. In multinode deployments of Standalone H2O, the leader node has to be manually identified by the user. Flow users can easily check whether they are connected to the leader node by opening Cluster Status (from the Admin menu) and checking that the first node has the same IP address as they see in their browser's address bar.
 
 .. figure:: ../../images/GBMReproducibility_LeaderNode.png
    :alt: Leader node in cluster status

 **Note**: In H2O versions prior to 3.18.0.1, the node automatically returned by the h2odriver while running on Hadoop was not guaranteed to be the leader node.  If you are using a version prior to 3.18.0.1 and you are running H2O on Hadoop, the leader node has to be manually identified by the user for reproducibility.

.. Floating point problems - precision vs. accuracy, roundoff error, statistical error
.. Role in map-reduce computation - operation order,  
.. Role in GBM - scoring intervals and early stopping, creating histograms
