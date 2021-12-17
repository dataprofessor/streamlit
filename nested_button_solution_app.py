import streamlit as st
import pandas as pd
import numpy as np

if "submitted" not in st.session_state:
    st.session_state.submitted = False

text1 = st.text_input("Username")
text2 = st.text_input("Password", type="password")

if (st.button("Login") or st.session_state.submitted) and text1=="admin" and text2=="password":
    st.session_state.submitted = True
    st.success("Welcome!")

    if st.button("Plot Graph"):
        st.session_state.submitted = False
        chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c'])
        st.area_chart(chart_data)
