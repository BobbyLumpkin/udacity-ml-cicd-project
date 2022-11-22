# Put the code for your API here.
from fastapi import FastAPI
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