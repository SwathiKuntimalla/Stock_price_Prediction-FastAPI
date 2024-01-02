import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import uvicorn


app = FastAPI()
# training the model
model = LinearRegression()
raw_data = pd.read_csv(r'C:\Users\kunti\Desktop\VS Code\Machine Learning\Day1\NFLX.csv')
x = raw_data[['High','Low','Close','Adj Close']]
y = raw_data['Volume']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model.fit(x_train, y_train)


class StockPricePrediction(BaseModel):
    High: float
    Low: float
    Close: float
    Adj_Close: float
    
   
@app.get("/")
async def root():
    return {"message": "Netflix Stock Price Prediction"}

@app.post("/predict")
def predict_stock_volume(data: StockPricePrediction):
   
    try:
        x = np.array([[data.High, data.Low, data.Close, data.Adj_Close]])
        prediction = model.predict(x)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



