import os 
import sys
import pymongo
import certifi
import pandas as pd
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

ca = certifi.where()
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

templates = Jinja2Templates(directory="./templates")

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training Successful..")

    except Exception as e:
        raise NetworkSecurityException(e,sys)

   
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Read uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(file.file)

        # Load pre-trained preprocessor and model from disk
        preprocessor = load_object("final_models/preprocessor.pkl")
        model = load_object("final_models/model.pkl")

        # Initialize your custom NetworkModel class with preprocessor and model
        network_model = NetworkModel(preprocessor=preprocessor, model=model)

        # Debug: Print the first row of the DataFrame
        print(df.iloc[0])

        # Make predictions on the input DataFrame
        y_pred = network_model.predict(df)

        # Debug: Print raw predictions
        print(y_pred)

        # Add predictions as a new column to the DataFrame
        df["predicted_column"] = y_pred

        # Debug: Print the new column to verify it's added correctly
        print(df["predicted_column"])

        # Save the updated DataFrame to a CSV file
        df.to_csv("predicted_output/output.csv")

        # Convert DataFrame to an HTML table for rendering in the frontend
        table_html = df.to_html(classes="table table-striped")

        # Render the HTML page with the table displayed
        return templates.TemplateResponse("table.html", {
            "request": request,
            "table": table_html
        })

    except Exception as e: 
        raise NetworkSecurityException(e, sys)

    

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
