#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import pathlib
import time
import math
from sklearn.linear_model import LogisticRegression

###############################################################################################################################
#                                          Page configuration and Sidebar Creation                                            #
###############################################################################################################################


st.set_page_config(page_title="Make Prediction", page_icon="project_data/favicon.jpg", layout="wide", 
                   initial_sidebar_state="expanded")


with st.container():
    st.markdown("***")
    st.markdown("""
    <div style='text-align:center'><h3>HIV-1 Protease Cleavage Status Prediction Model Employing the Logistic Regression
    Algorithm and Calculated Hybrid Descriptors as Input Features</h2></div>
    """, unsafe_allow_html=True)
    st.markdown("***")

with st.sidebar:
    user_input = st.text_area("Paste Octapeptide Sequences", 
                       placeholder="E.g.\nVDGFLVGG\nWDNLLAVI\nAECFRIFD\nHLVEALYL", 
                       max_chars=None,
                       height=130,
                       label_visibility="visible").strip(" ,:;.")
    octapeptides = user_input.upper().split("\n")

    action = st.button("Make Prediction")
    
###############################################################################################################################
#                                             AABP → Amino Acid Binary Profile                                                #
###############################################################################################################################
    
def AABP_calculator():
    A=(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    C=(0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    D=(0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    E=(0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    F=(0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    G=(0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    H=(0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)
    I=(0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)
    K=(0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0)
    L=(0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0)
    M=(0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0)
    N=(0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0)
    P=(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0)
    Q=(0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)
    R=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)
    S=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0)
    T=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)
    V=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0)
    W=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)
    Y=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)

    one_hot_encoded_peptide = []
    for octapeptide in octapeptides:
        for i in octapeptide:
            if i == "A":
                one_hot_encoded_peptide.append(A)
            elif i == "C":
                one_hot_encoded_peptide.append(C)
            elif i == "D":
                one_hot_encoded_peptide.append(D)
            elif i == "E":
                one_hot_encoded_peptide.append(E)
            elif i == "F":
                one_hot_encoded_peptide.append(F)
            elif i == "G":
                one_hot_encoded_peptide.append(G)
            elif i == "H":
                one_hot_encoded_peptide.append(H)
            elif i == "I":
                one_hot_encoded_peptide.append(I)
            elif i == "K":
                one_hot_encoded_peptide.append(K)
            elif i == "L":
                one_hot_encoded_peptide.append(L)
            elif i == "M":
                one_hot_encoded_peptide.append(M)
            elif i == "N":
                one_hot_encoded_peptide.append(N)
            elif i == "P":
                one_hot_encoded_peptide.append(P)
            elif i == "Q":
                one_hot_encoded_peptide.append(Q)
            elif i == "R":
                one_hot_encoded_peptide.append(R)
            elif i == "S":
                one_hot_encoded_peptide.append(S)
            elif i == "T":
                one_hot_encoded_peptide.append(T)
            elif i == "V":
                one_hot_encoded_peptide.append(V)
            elif i == "W":
                one_hot_encoded_peptide.append(W)
            elif i == "Y":
                one_hot_encoded_peptide.append(Y)

        AABPs_arr = np.asarray(one_hot_encoded_peptide).flatten()
        AABPs_array = AABPs_arr.reshape(-1, 160)
    return AABPs_array

    
###############################################################################################################################
#                                                PCP_BS → Composition of basic residues                                       #
###############################################################################################################################

try:
    path=pathlib.Path("project_data/PhysicoChemicalProperties(labelled).csv")
    PCPs_df= pd.read_csv(path, index_col=0)
except:
    st.markdown("### An error occured! The Physicochemical Properties could not be loaded")


def PCP_BS_calculator(PCPBS = "PCP_BS"):
    PCP_BSs = []
    for octatapeptide in octapeptides:
        l=len(octatapeptide)
        PCP_BS = np.zeros(l)
        for i in range(l):
            PCP_BS[i] = PCPs_df[octatapeptide[i]][PCPBS]
        PCP_BSs.append(sum(PCP_BS)/l)
        PCP_BSs_arr = np.asarray(PCP_BSs).reshape(-1, 1)
    return PCP_BSs_arr


###############################################################################################################################
#                                             PCP_HL → Composition of hydrophilic residues                                    #
###############################################################################################################################   

def octapeptides_to_array():
    octapeptides_arr = np.asarray(octapeptides).reshape(-1, 1)
    return octapeptides_arr

    
def PCP_HL_calculator(PCPHL = "PCP_HL"):
    PCP_HLs = []
    for octatapeptide in octapeptides:
        l=len(octatapeptide)
        PCP_HL = np.zeros(l)
        for i in range(l):
            PCP_HL[i] = PCPs_df[octatapeptide[i]][PCPHL]
        PCP_HLs.append(sum(PCP_HL)/l)
        PCP_HLs_arr = np.asarray(PCP_HLs).reshape(-1, 1)
    return PCP_HLs_arr

   
###############################################################################################################################
#                                                 Model Unpickling and deployment                                             #
###############################################################################################################################
    
def my_model():
    path=pathlib.Path("project_data/logistic_model.pkl")
    with open(path, mode="rb") as infile:
        model = pickle.load(infile)
        return model

with st.container():
    if action:
        initial_time=time.time()
        try:
            if octapeptides:
                with st.spinner("Wait for your results. They will be available shortly..."):
                    AABPs_arr = AABP_calculator()
                    PCP_BSs_arr = PCP_BS_calculator()
                    PCP_HLs_arr = PCP_HL_calculator()
                    octapeptides_arr = octapeptides_to_array()

                    AABPs_arr_df = pd.DataFrame(AABPs_arr, columns=("col %d" % i for i in range(1, 161, 1)))
                    PCP_BS_df = pd.DataFrame(PCP_BSs_arr, columns=["Composition of Basic Residue"])
                    PCP_HL_df = pd.DataFrame(PCP_HLs_arr, columns=["Composition of Hydrophilic Residue"])
                    octapeptides_arr_df = pd.DataFrame(octapeptides_arr, columns=["Octapeptide"])

                    descriptors_for_prediction = pd.concat([AABPs_arr_df, PCP_BS_df, PCP_HL_df], axis=1)
                    descriptors_with_octapeptides = pd.concat([octapeptides_arr_df, AABPs_arr_df, PCP_BS_df, PCP_HL_df], axis=1)


#                     @st.cache # IMPORTANT: Cache the conversion to prevent computation on every rerun, but not necessary here
                    def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    def map_output():
                        model = my_model()
                        predictions = model.predict(descriptors_for_prediction.to_numpy())
                        y = []
                        for i in predictions:
                            if i == 1:
                                y.append("Cleaved")
                            else:
                                y.append("Un-cleaved")
                        return np.asarray(y).reshape(-1, 1)
                    labels = map_output()

                prediction_df = pd.DataFrame(labels, columns=["Predicted Class"])
                final_df = pd.concat([octapeptides_arr_df, PCP_HL_df, PCP_BS_df, prediction_df], axis=1)
                
                st.markdown("""
                **Predicted classes**, together with the calculated `Compositions of Basic` and `Hydrophilic Residues`""", 
                            unsafe_allow_html=False)
                st.dataframe(data=final_df, use_container_width=True)
                
                st.download_button(
                    label="Download Predictions",
                    data=convert_df(final_df),
                    on_click=st.write("Click the download button below to save the CSV file"),
                    file_name='predicted_classes_of_the_octapeptides.csv',
                    mime='text/csv',
                    )
                
                final_time=time.time()
                elapesd_time=final_time-initial_time
                st.markdown(f"This took **_{round(elapesd_time, 2)}_** seconds to complete.")
                st.markdown("***")
            else:
                st.markdown("### No octapeptide sequences pasted!", unsafe_allow_html=False)
                st.stop()
        except:
            st.markdown("""
            **Invalid entry detected! Your pasted sequences may contain `empty lines` or `an invalid letter representation 
            of an amino acid residue` or `atleast one sequence length is not 8!` Please paste valid sequences!**""", 
                        unsafe_allow_html=False)
            final_time=time.time()
            elapesd_time=final_time-initial_time
            st.markdown(f"This took **_{round(elapesd_time, 2)}_** seconds to complete.")
            st.markdown("***")   
    else:
        st.markdown("""
        **Your predictions will appear below once you paste your octapeptide sequences and Click the 
        `Make Prediction` button in the Sidebar**""", unsafe_allow_html=False)
        st.markdown("***")

