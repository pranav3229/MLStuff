from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 

df = pd.read_csv("income.csv")
print(df.head())

scalar_income = MinMaxScaler()
scalar_age = MinMaxScaler()

df[['Income($)']] = scalar_income.fit_transform(df[['Income($)']])
df[['Age']] = scalar_age.fit_transform(df[['Age']])
print(df.head())

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = y_predicted
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 1')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 2')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()

k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
print(sse)

plt.xlabel('X')
plt.ylabel('Sim of squared error')
plt.plot(k_rng,sse)
plt.show()

