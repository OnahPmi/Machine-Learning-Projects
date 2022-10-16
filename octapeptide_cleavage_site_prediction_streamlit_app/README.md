## Implimented Algorithms for Generating Calculated Hybrid Descriptors for Machine/Deep Learning Applications on HIV-1 Protease Cleavage Site Prediction on octapeptide sequences and the deployment of the Logistic Regression Model using the Hybrid Descriptors for the same task

##### This work has been accepted at _BMC Bioinformatics_ for publication, and a link shall be available soon. Meanwhile, a preprint is available at [ResearchSquare](https://doi.org/10.21203/rs.3.rs-1688464/v1)
***
**In most parts of the world, especially in underdeveloped countries, _Acquired Immunodeficiency Syndrome (AIDS)_ still remains a major cause of death, disability and unfavorable economic outcomes. This has necessitated intensive research to develop effective therapeutic agents for the treatment of _Human Immunodeficiency Virus (HIV)_ infection, which is responsible for AIDS.  Peptide cleavage by `HIV-1 protease` is an essential step in the replication of HIV-1. Thus, correct and timely prediction ofthe cleavage site of HIV-1 protease can significantly speed up and optimize the drug discovery process of novel HIV-1 protease inhibitors.**
***
**In this work, we present a `Logistic Regression Model` for predicting the substrate specificity and cleavage site of HIV-1 protease. First, we built and compared the performance of selected machine learning models for the prediction of HIV-1 protease cleavage site utilizing a hybrid of octapeptide sequence information comprising 
`bond composition`, `amino acid binary profile (AABP)`, and `physicochemical properties` as numerical descriptors serving as input variables for some selected machine learning algorithms. Our work differs from antecedent studies exploring the same subject in the combination of octapeptide descriptors and method used. Instead of using various subsets of the dataset for training and testing the models, we combined the dataset, applied a 3-way data split, and then used a "stratified" 10-fold cross-validation technique alongside the testing set to evaluate the models.**
***
**This procedure showed that the `logistic regression model` and the `multi-layer perceptron classifier` achieved superior performance comparable to that of the state-of-the-art model, `linear support vector machine`. Our feature selection algorithm implemented via the `Decision tree model` showed that**: 

* `AABP → Amino Acid Binary Profile` **and two of the physicochemical properties, mamely**: 
* `PCP_BS → Composition of basic residues`, **and** 
* `PCP_HL → Composition of hydrophilic residues` 

**were top features selected by during model training. This supports previous findings that water accessibility served as a discriminative factor to predict cleavage sites [Warut et. al, 2022]( https://doi.org/10.1155/2022/8513719).**
