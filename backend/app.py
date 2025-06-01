from fastapi import FastAPI, Query
from predict import predict_stat

app = FastAPI()

@app.get("/")
def home():
    return {"message": "StatScope AI is live"}

@app.get("/predict")
def predict(player: str = Query(...), stat: str = Query(...), threshold: float = Query(...)):
    result = predict_stat(player, stat, threshold)
    return result
