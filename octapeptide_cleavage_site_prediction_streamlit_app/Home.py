#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(page_title="Home", page_icon="project_data/favicon.jpg", layout="wide", initial_sidebar_state="expanded")

with st.container():
    st.markdown("***")
    st.markdown("""
    <div style='text-align:justify'><h3>Implimented Algorithms for Generating Calculated Hybrid Descriptors for Machine/Deep
    Learning Applications on HIV-1 Protease Cleavage Site Prediction on octapeptide sequences and the deployment of the Logistic
    Regression Model using the Hybrid Descriptors for the same task</h3></div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **This server calculates octapeptide features such as _amino acid binary profile_, which is a 
    `one-hot-encoding` of a peptide/protein and some physicochemical properties of peptides/proteins needed
    for bulding rubost models for the task of predicting the cleavage status of an octapeptide. It further avails a `logistic 
    regression model` for the same task. The sidebar shows the implimentation of the various tools this webserver has to offer**
    """)
    col1, col2 = st.columns(2)
    with col1: 
        st.markdown("""
        * The **`Make Prediction`** tool predicts the cleavage status of octapeptide sequence(s). The generated results 
          can be applied to making improved decision on various drug design/bioinformatics tasks.
        """)
    with col2:
        st.markdown("""
        * The **`Calculate Descriptor`** tool calculates and returns a CSV file of the calculated descriptors which can
          be downloaded for various machine/deep learning applications in the design of potent HIV drugs.
        """)
    
    st.markdown("""
    **This work has been accepted at _BMC Bioinformatics_ for publication, and a link shall be available soon. Meanwhile,
    a preprint is available at [ResearchSquare](https://doi.org/10.21203/rs.3.rs-1688464/v1)**""")

with st.expander("See Details Related to the Model's Training Procedure", expanded=True):
    st.markdown("""
     In most parts of the world, especially in underdeveloped countries, _Acquired Immunodeficiency Syndrome (AIDS)_ 
     still remains a major cause of death, disability and unfavorable economic outcomes. This has necessitated 
     intensive research to develop effective therapeutic agents for the treatment of _Human Immunodeficiency Virus 
     (HIV)_ infection, which is responsible for AIDS.  Peptide cleavage by `HIV-1 protease` is an essential step in 
     the replication of HIV-1. Thus, correct and timely prediction of the cleavage site of HIV-1 protease can 
     significantly speed up and optimize the drug discovery process of novel HIV-1 protease inhibitors.
     ***
     In this work, we present a `Logistic Regression Model` for predicting the substrate specificity and cleavage 
     site of HIV-1 protease. First, we built and compared the performance of selected machine learning models for 
     the prediction of HIV-1 protease cleavage site utilizing a hybrid of octapeptide sequence information comprising 
     `bond composition`, `amino acid binary profile (AABP)`, and `physicochemical properties` as numerical descriptors 
     serving as input variables for some selected machine learning algorithms. Our work differs from antecedent 
     studies exploring the same subject in the combination of octapeptide descriptors and method used. Instead of 
     using various subsets of the dataset for training and testing the models, we combined the dataset, applied a 
     3-way data split, and then used a "stratified" 10-fold cross-validation technique alongside the testing set 
     to evaluate the models.
     ***
     This procedure showed that the `logistic regression model` and the `multi-layer perceptron classifier` achieved 
     superior performance comparable to that of the state-of-the-art model, `linear support vector machine`. Our feature
     selection algorithm implemented via the `Decision tree model` showed that: 
     * `AABP → Amino Acid Binary Profile` and two of the physicochemical properties, mamely: 
     * `PCP_BS → Composition of basic residues`, and 
     * `PCP_HL → Composition of hydrophilic residues` 
     were top features selected by during model training. This supports previous findings that water accessibility served 
     as a discriminative factor to predict cleavage sites [Warut et. al, 2022]( https://doi.org/10.1155/2022/8513719).
     ***
     Some of the performance metrics employed during model training and their values are shown below:""")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Balanced Accuracy", "90.9 %")
    col2.metric("AUC", "0.97")
    col3.metric("F-Index", "0.91")
    col4.metric("Jaccard Index", "0.83")
    col5.metric("Specificity", "0.90")
    col6.metric("Sensitivity", "0.57")
    
with st.expander("Charts Related to the Model's Training Procedure", expanded=False):    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sequence Logos of Cleaved and Uncleaved Octapeptides", 
                                            "Model Training Algorithm", 
                                            "Feature Importance Plot by GBC", 
                                            "ROC Curves", 
                                            "10-Fold CV Metrics Distribution"])

    with tab1:
        st.markdown("""
        **The sequence logos of A) cleaved octapeptide; B) un-cleaved octapeptide sequences generated
        with the online Seq2Logo webserver using Heuristic clustering algorithm, pseudo count with a
        weight of 200 and logotype as Kullback–Leibler. Enriched amino acids are shown on the positive
        y-axis and depleted amino acids on the negative y-axis**""")
        st.image("project_data/seq_logo_of_cleaved_&_uncleaved_octapeptides.png")

    with tab2:
        st.markdown("""
        **Model Training Algorithm**""")
        st.image("project_data/Model_Training_Flow_Chart.jpg")

    with tab3:
        st.markdown("""
        **Gini importance chart of the best 20 features selected by the Gradient Boosting Classifier**""")
        st.image("project_data/Plot_of_feature_importance_for_GB_classifier.png")

    with tab4:
        st.markdown("""
        **ROC Curves of the Models on the Testing Set**""")
        st.image("project_data/ROC_Curves_of_the_Models_on_the_Testing_Set.png")

    with tab5:
        st.markdown("""
        **Distribution of the performance metrics scores of the models in the 10-fold CV experiment for
        each of the 6 standard tests conducted. a) Balanced Accuracy Scores; b) Sensitivity Scores; c)
        Specificity Scores; d) F-score; e) AUC; and f) Jaccard Index Scores**""")
        st.image("project_data/Distribution_of_the_Performance_Metrics_Across_the_Models.png")
    
with st.container():
    st.markdown("""
    #### References:    
     * Warut P, Kwanluck TA, Kasidit S, Parthana P, Jirachai B. "Hyperparameter Tuning of Machine Learning Algorithms
       Using Response Surface Methodology: A Case Study of ANN, SVM, and DBN", Mathematical Problems in Engineering. 2022,
       vol. 2022, Article ID 8513719, 17 pages [https://doi.org/10.1155/2022/8513719](https://doi.org/10.1155/2022/8513719).
       
    #### Further information:  
     * A lot of inspirations was drawn from the works of the Raghava group, headed by Professor Raghava, the Head of the 
       Department of Computational Biology, Indraprastha Institute of Information Technology (IIIT-Delhi), India. Particularly, 
       their work on the [pfeature software](https://github.com/raghavagps/Pfeature) gave us insight on developing the various 
       algorithms for generating the Amino Acid Binary Profile, Composition of Basic and Hydrophilic Residues present in 
       Octapeptides. Worthy of note also is the work of [Singh & Su](https://doi.org/10.1186/s12859-016-1337-6). 
    
     * The authors have published a number of works in the _in silico_ drug design/bioinformatics domain spanning across
       various disease conditions like `Cancer`, `Infectious Diseases`, and `Neurodegenerative Disorders`. Notable examples
       include:  
        * Onah E, Uzor F.P., Ugwoke I.C., et al. Prediction of HIV-1 Protease Cleavage Site from Octapeptide Sequence 
          Information using Selected Classifiers and Hybrid Descriptors. Accepted at BMC Bioinformatics for publication.
          A Priprint (Version 1) is available at Research Square
          [https://doi.org/10.21203/rs.3.rs-1688464/v1](https://doi.org/10.21203/rs.3.rs-1688464/v1). 
        * Ibezim A., Onah E., Dim E.N., and Ntie-Kang F. (2021). A Computational Multi-targeting Approach for Psoriasis 
          Treatment. BMC Complementary Medicine and Therapies, 21(1), 193.
          [https://doi.org/10.1186/s12906-021-03359-2](https://doi.org/10.1186/s12906-021-03359-2).
        * Onah, E., Ugwoke, I., Eze, U., Eze, H., Musa, S., Ndiana-Abasi, S., Okoli, O., Ekeh, I., & Edet, A. (2021). 
          Search for Structural Scaffolds Against SARS-COV-2 Mpro: An In Silico Study. Journal of Fundamental and Applied 
          Sciences, 13(2), 740-769. 
          [https://jfas.info/index.php/JFAS/article/view/987](https://jfas.info/index.php/JFAS/article/view/987).
    """)

