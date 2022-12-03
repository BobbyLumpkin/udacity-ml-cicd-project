"""
Purpose: Test the live API post method.
Author(s): Bobby Lumpkin
"""


import json
import requests


payload = {
    'age': 31,
    'workclass': 'Private',
    'fnlgt': 45781,
    'education': 'Masters',
    'education_num': 14,
    'marital_status': 'Never-married',
    'occupation': 'Prof-specialty',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Female',
    'capital_gain': 14084,
    'capital_loss': 0,
    'hours_per_week': 50,
    'native_country': 'United-States'
}


if __name__ == "__main__":
    r = requests.post(
        "https://udacity-cicd-project.herokuapp.com/scoring/",
        data=json.dumps(payload)
    )
    print(r)
    print(r.json())