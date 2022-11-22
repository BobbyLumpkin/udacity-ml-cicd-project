# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel


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
    age: int
    workclass: str


    # fnlgt: int
    # education: str
    # education-num: int



@app.post("/scoring/")
async def score_observations(observation: Observation):
    num_obs = (
        len(observation.age) if isinstance(observation.age, list)
        else 1
    )
    df = pd.DataFrame(
        {
            "age" : observation.age,
            "workclass" : observation.workclass
        },
        index=list(range(num_obs))
    )
    df.age = df.age + 1
    return df.to_json()


