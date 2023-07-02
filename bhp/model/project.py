import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json
pd.set_option('display.max_columns', None)

#performing data cleaning
matplotlib.rcParams["figure.figsize"]=(20,10)
df1=pd.read_csv("project.csv")
print(df1.head())

print(df1.groupby('area_type')['area_type'].agg('count'))

df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
print(df2.head())

print(df2.isnull().sum())

df3=df2.dropna()
print(df3.isnull().sum())

print(df3['size'].unique())
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.head())

print(df3.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(df3[-df3['total_sqft'].apply(is_float)].head())

def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head())

df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
print(df5.head())

df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

print(len(df5.location.unique()))
df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique())) #reducing data that might be irrelevant

print(df5[df5.total_sqft/df5.bhk<300].head())

df6= df5[-(df5.total_sqft/df5.bhk<300)]  #outlier removal

print(df6.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out =pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 =remove_pps_outliers(df6)
print(df7.shape)

def plot_scatter_chart(df,location):
    bhk2 =df[(df.location==location)&(df.bhk==2)]
    bhk3 =df[(df.location==location)&(df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK',s=50)
    plt.xlabel('Total square feet area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    plt.show()

def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'stf':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats =bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)
print(df8.shape)

# plot_scatter_chart(df8,"Hebbal")

matplotlib.rcParams['figure.figsize']=(20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price")
plt.ylabel("Count")
# plt.show()

df9=df8[df8.bath<df8.bhk+2]
print(df9.shape)

df10=df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head())

dummies=pd.get_dummies(df10.location)
df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')

df12=df11.drop('location',axis='columns')
print(df12.shape)

X=df12.drop('price',axis='columns')

y=df12.price

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=10)
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
print(lr_clf.score(X_test,y_test))

cv =ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
print(cross_val_score(LinearRegression(),X,y,cv=cv))

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                # 'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(X,y))

def predict_price(location,sqft,bath,bhk):
    loc_index =np.where(X.columns==location)[0][0]
    x =np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1
    return lr_clf.predict([x])[0]

print(predict_price('Hebbal',1000,2,2))
with open('price_predictor_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))