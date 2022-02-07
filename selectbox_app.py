import streamlit as st
import numpy as np

st.title("Selectbox")

contact_options = ["Email", "Phone", "Text"]

st.header("Selectbox from a list")

contact_selected = st.selectbox("How would you like to be contacted?",
                                options= contact_options)

st.write("Selectbox returns:", contact_selected,
         "of type", type(contact_selected))

if contact_selected == "Email":
    st.write("**Comfirm your email address by clicking the link sent to you**")
else:
    st.write("**Thank you, we will be in touch soon**")

st.header("Selectbox from a NumPy array")

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

col1, mid, col2 = st.columns([1, 0.1, 3])
with col1:
    st.write("My Array:")
    array

array_selection = st.selectbox("Choose an option", options = array, index=1)

st.write("Array selection returns:", array_selection,
         "of type", type(array_selection))
