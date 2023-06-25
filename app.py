from flask import Flask
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

data = pd.read_csv('/Users/sathishkrishnan/csk/python/house_prediction/HousePriceIndia.csv')
data.drop(['id','Date'], axis=1, inplace=True)
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
forest = RandomForestRegressor()
forest.fit(X_train, y_train)

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/predict')
def predict():
    predictions = forest.predict([[4,2.50,2021,18972,2.0,0,0,4,8,2010,0,1989,0,122011,52.8082,-114.208,2100,8511,1,74]])
    return 'Prediction' + str(predictions[0])

if __name__ == '__main__':
    app.run()