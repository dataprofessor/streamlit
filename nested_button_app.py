import streamlit as st

if st.button('Level 1'):
    st.write('Level 1 passed!')
    if st.button('Level 2'):
        st.write('Level 2 passed')
