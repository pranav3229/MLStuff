import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("insurance_data.csv")
# print(df.head())

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
X_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,train_size=0.9)

model = LogisticRegression()
model.fit(X_train,y_train)
print(x_test)

print(model.predict(x_test))
print(model.score(x_test,y_test))

print(model.predict_proba(x_test))

plt.show()