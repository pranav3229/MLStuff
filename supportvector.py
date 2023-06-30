import pandas as pd 
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



iris=load_iris()
print(iris.feature_names)

df=pd.DataFrame(iris.data,columns=iris.feature_names)


df['target']=iris.target
print(df.head())
print(iris.target_names)



df['flower_name'] =df.target.apply(lambda x:iris.target_names[x])

print(df[df.target==1].head())

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')
# plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red',marker='+')
# plt.show()

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(df0['petal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['petal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')
# plt.show()

X =df.drop(['target','flower_name'],axis='columns')
print(X.head())

y=df.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model =SVC(C=5,gamma=4,kernel='linear')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

