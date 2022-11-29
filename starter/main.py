# Put the code for your API here.
from fastapi import FastAPI
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
import sys
from typing import List, Union


sys.path.append(str(Path(__file__).parent / "starter"))


from ml.model import inference


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
    age: Union[int, List[int]] = Field(example=28)
    workclass: Union[str, List[str]] = Field(example="Private")
    fnlgt: Union[int, List[int]] = Field(example=338409)
    education: Union[str, List[str]] = Field(example="Masters")
    education_num: Union[int, List[int]] = Field(example=17)
    marital_status: Union[str, List[str]] = Field(example="Never-married")
    occupation: Union[str, List[str]] = Field(example="Prof-specialty")
    relationship: Union[str, List[str]] = Field(example="Not-in-family")
    race: Union[str, List[str]] = Field(example="Black")
    sex: Union[str, List[str]] = Field(example="Male")
    capital_gain: Union[int, List[int]] = Field(example=0)
    capital_loss: Union[int, List[int]] = Field(example=0)
    hours_per_week: Union[int, List[int]] = Field(example=40)
    native_country: Union[str, List[str]] = Field(example="United-States")


@app.post("/scoring/")
async def score_observations(observation: Observation):
    # Load model obj.
    model_path = "model/model_obj.pkl"
    observation_dict = json.loads(observation.json())
    observation_dict = {
        k.replace("_", "-"):v for k, v in observation_dict.items()
    }
    df = pd.DataFrame(observation_dict)
    preds = inference(model_obj=model_path, X=df)
    preds_dict = {"preds" : preds.tolist()}
    return preds_dict#df.to_dict()


