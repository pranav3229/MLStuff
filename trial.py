import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
import joblib


    

df = pd.read_csv("c.csv")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area.to_numpy(), reg.predict(df[['area']].values), color='blue')
predicted_price = reg.predict([[3300]])
print("Predicted price for an area of 3300 sqft:", predicted_price)

print(reg.coef_)
print(reg.intercept_)


plt.show()

with open('model_pickle','wb') as f:
    pickle.dump(reg,f)

with open('model_pickle','rb') as f:
    mp= pickle.load(f)

print("yeet : ",mp.predict([[5000]]))

joblib.dump(reg,'model_joblib')

mj =joblib.load('model_joblib')

print("yeet again: ",mj.predict([[6000]]))