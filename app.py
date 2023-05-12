import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("EColiModel.pickle", "rb"))
encoder = pickle.load(open("EColiEncoder.pickle", "rb"))

st.header("DNA E.Coli Classification")
data = st.text_input("Gen")

if st.button("Analyze"):
    data = list(data)
    data_df = pd.DataFrame(data).transpose()
    test = encoder.transform(data_df).toarray()

    result = model.predict(test).item()
    if result == 0:
        st.warning("POSITIVE")
    else:
        st.success("NEGATIVE")
