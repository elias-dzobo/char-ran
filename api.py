import fastapi 
from fastapi import FastAPI, HTTPException
from rnn import predict

app = FastAPI()


@app.post('/predict')
def predict_name_category(name: str):
    try:
        category = predict(name)
        return {"prediction": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")