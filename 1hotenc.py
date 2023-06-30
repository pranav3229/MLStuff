import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


df = pd.read_csv("homeprices.csv")

dummies = pd.get_dummies(df.town)
merged = pd.concat([df, dummies], axis='columns')
final = merged.drop(['town', 'west windsor'], axis='columns')

model = LinearRegression()
x = final.drop(['price'], axis='columns')
y = final.price
model.fit(x, y)

ans = model.predict([[2800, 0, 1]])
print(model.score(x, y))

le = LabelEncoder()
z = df
z.town = le.fit_transform(z.town)
print(z)

X = df[['town', 'area']].values

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

X=X[:,1:]
model.fit(X,y)


print(model.predict([[1,0,2800]]))

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train)
print(len(X_train))
print(len(x_test))

clf=LinearRegression()
clf.fit(X_train,y_train)

clf.predict(x_test)
print(y_test)
print(clf.score(x_test,y_test))