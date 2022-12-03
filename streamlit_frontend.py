"""
Purpose: Builds a frontend for the API constructed in main.py.
Author(s): Bobby Lumpkin
"""


import json
import requests
import streamlit as st


# Define functions.
@st.cache
def send_post_payload(payload: dict):
    r = requests.post(
        "https://udacity-cicd-project.herokuapp.com/scoring/",
        data=json.dumps(payload)
    )
    return r


# Title and header.
st.title("Categorical Salary Prediction")

st.subheader(
    "Enter demographic data for an individual below:"
)

# Model feature forms.
age = st.number_input(
    label="Age",
    value=21,
    help="Age of an individual."
)
workclass = st.selectbox(
    label="Workclass",
    help="Represents the employment status of an individual.",
    options=[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
        "Never-worked",
        "?"]
)
fnlgt = 189778
education = st.selectbox(
    label="Education",
    help="Max education level.",
    options=[
        "Preschool",
        "1st-4th",
        "5th-6th", 
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad",
        "Some-college",
        "Assoc-acdm",
        "Assoc-voc",
        "Bachelors",
        "Prof-school",
        "Masters",
        "Doctorate"
    ]
)
education_num = st.number_input(
    label="Education Number",
    min_value=0,
    help="Number of years spent in school."
)
marital_status = st.selectbox(
    label="Marital Status",
    help="Marital status of an individual.",
    options=[
        "Never-married",
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Separated",
        "Divorced",
        "Widowed"
    ]
)
occupation = st.selectbox(
    label="Occupation",
    help="An individual's general type of occupation.",
    options=[
        "Adm-clerical",
        "Armed-Forces",
        "Craft-repair",
        "Exec-managerial",
        "Farming-fishing",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Other-service",
        "Priv-house-serv",
        "Prof-specialty",
        "Protective-serv",
        "Sales",
        "Tech-support",
        "Transport-moving",
        "?"
    ]
)
relationship = st.selectbox(
    label="Relationship",
    help="Represents what this individual is, relative to others.",
    options=[
        "Husband",
        "Wife"
        "Own-child",
        "Unmarried",
        "Not-in-family",
        "Other-relative" 
    ]
)
race = st.selectbox(
    label="Race",
    help="Description of an individual's race.",
    options=[
        "Amer-Indian-Eskimo",
        "Asian-Pac-Islander",
        "Black",
        "White",
        "Other"
    ]
)
sex = st.selectbox(
    label="Sex",
    help="The biological sex of the individual.",
    options=[
        "Female",
        "Male"
    ]
)
capital_gain = st.number_input(
    label="Capital Gain",
    help="Capital gains for an individual.",
    min_value=0
)
capital_loss = st.number_input(
    label="Capital Loss",
    help="Capital loss for an individual.",
    min_value=0
)
hours_per_week = st.number_input(
    label="Hours per Week",
    help="The hours an individual reports to work per week.",
    min_value=0.00
)
native_country = st.selectbox(
    label="Native Country",
    help="An individual's country of origin.",
    index=38,
    options=[
        "Cambodia",
        "Canada",
        "China",
        "Columbia",
        "Cuba",
        "Dominican-Republic",
        "Ecuador",
        "El-Salvador",
        "England",
        "France",
        "Germany",
        "Greece",
        "Guatemala",
        "Haiti",
        "Holand-Netherlands",
        "Honduras",
        "Hong",
        "Hungary",
        "India",
        "Iran",
        "Ireland",
        "Italy",
        "Jamaica",
        "Japan",
        "Laos",
        "Mexico",
        "Nicaragua",
        "Outlying-US(Guam-USVI-etc)",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Puerto-Rico",
        "Scotland",
        "South",
        "Taiwan",
        "Thailand",
        "Trinadad&Tobago",
        "United-States",
        "Vietnam",
        "Yugoslavia",
        "?"
    ]
)

# Submit button and API calls.
st.button(
    label="Submit",
    key="submit_button"
)

if st.session_state.submit_button:
    st.text("Submitting data for prediction...")
    payload = {
        "age": age,
        "workclass": workclass,
        "fnlgt": fnlgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country
    }
    r = send_post_payload(payload=payload)
    
    # Return results.
    return_dict = r.json()
    return_str_body = "This individual makes less than or equal to 50K!"
    if return_dict["preds"][0] == 1:
        return_str_body = "This individual makes more than 50K!"
    st.markdown("---")
    st.text(
        return_str_body
    )