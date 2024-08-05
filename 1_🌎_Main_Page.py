import streamlit as st

st.set_page_config(
    page_title='Earthquake Predictions',
    page_icon='🌎'
)

st.title('Homepage')
st.sidebar.success('Select a page!')

st.divider()

st.title('Why this app?')
st.markdown("""
            - Provide an application platform for people to experience both the backend and frontend design of machine learning prediction systems
            - While this application does not offer a thorough walkthrough of the code used, we explain the problem solving process that leads to making accurate predictions 
            - This application may mirror the focus of programming websites and articles like GeeksforGeeks or Medium 
            - For this application, we will showcase an environmental monitoring algorithm trained at predicting the magnitude of earthquakes around the world""")

st.divider()

st.title('Table of Contents')
st.markdown("""
            1. Regression: Predicting the Severity of an Earthquake
            2. Classification: Predicting the Probability of an Earthquake 
            3. Geostationary Satellite Mapping: Earthquake Location and Concentration
            4. ChatBot: Earthquakes in Your Area
            """)

st.markdown('[Go to Regression Page](Regression_Systems)')
st.markdown('[Go to Classification Page](Classification_Systems)')
st.markdown('[Go to Geo-Mapping Page](Geomap_Widget)')
st.markdown('[Go to ChatBot](ChatBot)')