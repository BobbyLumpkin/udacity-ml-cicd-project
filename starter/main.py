# Put the code for your API here.
from fastapi import FastAPI
import json
import pandas as pd
from pydantic import BaseModel
from typing import List, Union


app = FastAPI()


@app.get("/")
async def get_greeting():
    greetings_str = (
        "Greetings!,\n\n"
        "We hope you enjoy this API!\n\n"
        "Best,\n"
        "The Creators"
    )
    return {"greeting": greetings_str}


class Observation(BaseModel):
    age: Union[int, List[int]]
    workclass: Union[str, List[str]]
    fnlgt: Union[int, List[int]]
    education: Union[str, List[str]]
    education_num: Union[int, List[int]]


    # marital_status: Union[str, List[str]]
    # occupation: Union[str, List[str]]
    # relationship: Union[str, List[str]]
    # race: Union[str, List[str]]
    # sex: Union[str, List[str]]
    # capital_gain: Union[int, List[int]]
    # capital_loss: Union[int, List[int]]



@app.post("/scoring/")
async def score_observations(observation: Observation):
    observation_dict = json.loads(observation.json())
    df = pd.DataFrame(observation_dict)
    df.age = df.age + 1
    return df.to_dict()


