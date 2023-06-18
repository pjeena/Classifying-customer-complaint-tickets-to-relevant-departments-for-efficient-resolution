from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_pipeline
import pickle
from io import StringIO
from deta import Deta
import json
import numpy as np


# Initialize and connect to drive.
deta = Deta('a0gqhlckd6c_d2VdFso9YJ23omhGk36dAojFT8sBwuhC')
drive = deta.Drive("artifacts")


# load preprocessor , model, and mappings
preprocessor = drive.get('preprocessor.pkl').read()
preprocessor = pickle.loads(preprocessor)


model = drive.get('model.pkl').read()
model = pickle.loads(model)


mapping = drive.get('labels_mapping.json').read()
labels_mapping = json.loads(mapping)



app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    ticket_type: str



@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict")
def predict(payload: TextIn):
    ticket_type, predict_proba = predict_pipeline(
        payload.text, preprocessor=preprocessor, model=model, dict_classes=labels_mapping
    )
    return {"complaint ticket type": ticket_type, "class probabilities" : predict_proba}
