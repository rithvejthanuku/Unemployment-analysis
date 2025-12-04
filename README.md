# Unemployment-analysis
The project describes the unemployement analysis around the world using clustreing techniques.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
%matplotlib inline
import seaborn as sns
from pandas import DataFrame
warnings.filterwarnings("ignore")
df = pd.read_csv("unemployment analysis.csv")
print(df)

df.head()



df.tail()



df.describe()

df.isna()

df.dropna()

df.drop_duplicates()

df.mean()

df.mode()

df.duplicated().sum()

unemployment=df.loc[df["Country Name"]=='Zimbabwe']
unemployment



ZMM=unemployment.melt(id_vars=['Country Name','Country Code'], var_name='Years',value_name='rate%')
ZMM

import pandas as pd

  

index_names = df[ (df['2016'] >= 21) & (df['2016'] <= 23)].index
  
# drop these given row
# indexes from dataFrame
df.drop(index_names, inplace = True)
  
df

df

df.info()

mv=ZMM.loc[ZMM['rate%']==ZMM['rate%'].max(),['Years','rate%']]
mvv=ZMM.loc[ZMM['rate%']==ZMM['rate%'].min(),['Years','rate%']]
print(f'max unemployment rate:{mv.iloc[0,1]} Year:{mv.iloc[0,0]}')
print(f'min unemployment rate:{mvv.iloc[0,1]} Year:{mvv.iloc[0,0]}')

sns.lineplot(data=ZMM,x='Years',y='rate%');
plt.xticks(rotation=90);
plt.title("Unemployment in zimbabwe between 1991 and 2021")

rates = [f'{rate}%' for rate in list(ZMM.loc[:, 'rate%'])]
sns.barplot(data=ZMM, x='Years', y='rate%', label=rates,);
plt.xticks(rotation=90);
plt.legend(bbox_to_anchor=(1,1.05));
plt.figure(figsize=(30,10))
plt.title('Unemployment rate in Zimbabwe between 1991 and 2021');

df.shape

world_unemployment_rate_2021 = df.loc[:, ['Country Name', 'Country Code', '2021']]
highest_unemployment_rate = df.loc[df['2021'] == df['2021'].max()].melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%')
lowest_unemployment_rate = df.loc[df['2021'] == df['2021'].min()].melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%')
sns.lineplot(data=highest_unemployment_rate, x='Years', y='rate%', label='South Africa');
sns.lineplot(data=lowest_unemployment_rate, x='Years', y='rate%', label='Qatar');
plt.xticks(rotation=90);
plt.title(label='Difference between countries which has the lowest and highest unemployment rate in the world in 2021.')
plt.legend();


world_unemployment_rate_2021 = df.loc[:, ['Country Name', 'Country Code', '2008']]
highest_unemployment_rate = df.loc[df['2008'] == df['2008'].max()].melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%')
lowest_unemployment_rate = df.loc[df['2008'] == df['2008'].min()].melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%')
sns.lineplot(data=highest_unemployment_rate, x='Years', y='rate%', label='South Africa');
sns.lineplot(data=lowest_unemployment_rate, x='Years', y='rate%', label='Qatar');
plt.xticks(rotation=90);
plt.title(label='Difference between countries which has the lowest and highest unemployment rate in the world in 2008.')
plt.legend();


world_unemployment_rate_2021 = df.loc[:, ['Country Name', 'Country Code', '2000']]
highest_unemployment_rate = df.loc[df['2000'] == df['2000'].max()].melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%')
lowest_unemployment_rate = df.loc[df['2000'] == df['2000'].min()].melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%')
sns.lineplot(data=highest_unemployment_rate, x='Years', y='rate%', label='South Africa');
sns.lineplot(data=lowest_unemployment_rate, x='Years', y='rate%', label='Qatar');
plt.xticks(rotation=90);
plt.title(label='Difference between countries which has the lowest and highest unemployment rate in the world in 2000.')
plt.legend();


rates = [f'{rate}%' for rate in list(lowest_unemployment_rate.loc[:, 'rate%'])]
sns.barplot(data=lowest_unemployment_rate, x='Years', y='rate%', label=rates);
plt.xticks(rotation=90);
plt.figure(figsize=(30,10));

plt.title('Unemployment rate in Qatar');
sns.lineplot(data=lowest_unemployment_rate, x='Years', y='rate%', label='Qatar');

rates = [f'{rate}%' for rate in list(highest_unemployment_rate.loc[:, 'rate%'])]
sns.barplot(data=highest_unemployment_rate, x='Years', y='rate%', label=rates);
plt.xticks(rotation=90);

plt.figure(figsize=(30,10))
plt.title('Unemployment rate in South Africa');
sns.lineplot(data=highest_unemployment_rate, x='Years', y='rate%', label='South Africa');

world_unemployment_rate = df.melt(id_vars=['Country Name', 'Country Code'], var_name='Years', value_name='rate%').groupby('Years')['rate%'].agg('mean')
world_unemployment_rate = world_unemployment_rate.reset_index()

sns.lineplot(data=world_unemployment_rate, x='Years', y='rate%');
plt.xticks(rotation=90);
plt.title('Change in the average unemployment rate of all countries in the world (1991-2021)');

df

x=df.iloc[0:25]

plt.figure(figsize=(10,10))
sns.scatterplot(data=x, x='2008', y='Country Name', color='blue', hue='2008'
                )
plt.grid(axis='y')

plt.figure(figsize=(10,10))
sns.scatterplot(data=x, x='2009', y='Country Name', color='blue', hue='2009'
                )
plt.grid(axis='y')

df['2008'].value_counts

plt.figure(figsize=(10,10))
sns.scatterplot(data=x, x='2007', y='Country Name', color='blue', hue='2007'
                )
plt.grid(axis='y')

x

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.figure(figsize=(10,80))
sns.scatterplot(data=df, x='2008', y='Country Name', color='blue', hue='2008',
                size='2008', sizes=(10,100))
plt.grid(axis='y')

plt.figure(figsize=(10,60))
sns.scatterplot(data=df, x='2016', y='Country Name', color='blue', hue='2016',
                size='2016', sizes=(10,100))
plt.grid(axis='y')

## unemployment rate was same in the year 1991 and 1992 which is correlated to 2013 to 2019

df.corr()

plt.figure(figsize=(30,10))
sns.barplot(data=df)

plt.figure(figsize=(10,60))
sns.scatterplot(data=df, x='2017', y='Country Name', color='blue', hue='2017',
                size='2017', sizes=(10,100))
plt.grid(axis='y')

plt.figure(figsize=(10,60))
sns.scatterplot(data=df, x='2018', y='Country Name', color='blue', hue='2018',
                size='2018', sizes=(10,100))
plt.grid(axis='y')

plt.figure(figsize=(10,60))
sns.scatterplot(data=df, x='2019', y='Country Name', color='blue', hue='2019',
                size='2019', sizes=(10,100))
plt.grid(axis='y')

plt.figure(figsize=(10,60))
sns.scatterplot(data=df, x='2020', y='Country Name', color='blue', hue='2020',
                size='2020', sizes=(10,100))
plt.grid(axis='y')

plt.figure(figsize=(10,60))
sns.scatterplot(data=df, x='2021', y='Country Name', color='blue', hue='2021',
                size='2021', sizes=(10,100))
plt.grid(axis='y')

df.shape



df.isnull().sum()

for column in df:
    unique_vals = np.unique(df["1991"])
    nr_values = len(unique_vals)
    if nr_values < 1:
        print("The number of values for feature {} :{} -- {}".format("1991",nr_values,unique_vals))
    else:
        print("The number of values for feature {} :{} -- {}".format("1991",nr_values,unique_vals))
    

df.columns

import seaborn as sns
d = sns.pairplot(df[['1991','1992','1993','Country Name']],hue = 'Country Name',height = 5)

import seaborn as sns
d = sns.pairplot(df[['1991','1992','1993','Country Code']],hue = 'Country Code',height = 5)

d = sns.lmplot(x = "1991", y ="1992",data = df)

np.random.seed(20)
k=3
centroids ={
    i+1: [np.random.randint(0,20),np.random.randint(0,20)]
    for i in range(k)
}
fig = plt.figure(figsize=(5,5))
plt.scatter(df['1991'],df['1992'],color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

def assignment(df, centroids):
    for i in centroids.keys():
        #sqrt((x1 - x2)^2)-(y1 -y2)^2)
        df['distance_from_{}'.format(i)] = (
        np.sqrt(
            (df['1991'] - centroids[i][0]) ** 2
            + (df['1992'] -centroids[i][1]) ** 2
           )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closet'] = df.loc[:,centroid_distance_cols].idxmin(axis=1)
    df['closet'] = df['closet'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closet'].map(lambda x: colmap[x])
    return df
df = assignment(df,centroids)
print(df.head())

fig = plt.figure(figsize = (5,5))
plt.scatter(df['1991'],df['1992'], color = df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

import copy
old_centroids = copy.deepcopy(centroids)
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closet'] == i]['1991'])
        centroids[i][0] = np.mean(df[df['closet'] == i]['1992'])
    return k
centroids = update(centroids)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['1991'], df['1992'], color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][0] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()
    
        

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['1991'], df['1992'], color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

while True:
    closet_centroids = df['closet'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closet_centroids.equals(df['closet']):
        break
        
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['1991'], df['1992'], color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['1991'], df['1992'], color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

np.random.seed(20)
k=3
centroids ={
    i+1: [np.random.randint(0,20),np.random.randint(0,20)]
    for i in range(k)
}
fig = plt.figure(figsize=(5,5))
plt.scatter(df['1998'],df['2000'],color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

def assignment(df, centroids):
    for i in centroids.keys():
        #sqrt((x1 - x2)^2)-(y1 -y2)^2)
        df['distance_from_{}'.format(i)] = (
        np.sqrt(
            (df['1998'] - centroids[i][0]) ** 2
            + (df['2000'] -centroids[i][1]) ** 2
           )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closet'] = df.loc[:,centroid_distance_cols].idxmin(axis=1)
    df['closet'] = df['closet'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closet'].map(lambda x: colmap[x])
    return df
df = assignment(df,centroids)
print(df.head())

fig = plt.figure(figsize = (5,5))
plt.scatter(df['1991'],df['2000'], color = df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
X_actual = ["1991"]
Y_predic = ["1991"]
results = confusion_matrix(X_actual, Y_predic)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(X_actual, Y_predic))
print ('Classification Report : ')
print (classification_report(X_actual, Y_predic))


import copy
old_centroids = copy.deepcopy(centroids)
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closet'] == i]['1998'])
        centroids[i][0] = np.mean(df[df['closet'] == i]['2000'])
    return k
centroids = update(centroids)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['1998'], df['2000'], color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,20)
plt.ylim(0,20)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][0] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
X_actual = ["1991"]
Y_predic = ["1991"]
MSE = mean_squared_error(X_actual, Y_predic)
print ('MAE =',mean_absolute_error(X_actual, Y_predic))
print ('MSE =',mean_squared_error(X_actual, Y_predic))
print ('RMSE =', np.sqrt(MSE))

x = df.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as shc  
dendro = shc.dendrogram(shc.linkage(x, method="ward"))  
plt.title("Hireachial")  
plt.ylabel("y axis")  
plt.xlabel("x axis")
plt.figure(figsize=(10,60))
plt.show()  

from sklearn.cluster import AgglomerativeClustering  
hc= AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
y_pred= hc.fit_predict(x)  

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')  
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'green', label = 'Cluster 2')  
plt.scatter(x[y_pred== 2, 0], x[y_pred == 2, 1], s = 100, c = 'red', label = 'Cluster 3')  
plt.xlabel('years')  
plt.ylabel('percentage')  
plt.legend()  
plt.show()  

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
X_actual = ["1991"]
Y_predic = ["1991"]
results = confusion_matrix(X_actual, Y_predic)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(X_actual, Y_predic))
print ('Classification Report : ')
print (classification_report(X_actual, Y_predic))


df

x1=df1.drop(["Country Name","Country Code"],axis=1)


wcss=[]

from sklearn.cluster import KMeans  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(x1)  
    wcss.append(kmeans.inertia_) 

plt.plot(range(1,11),wcss)

kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
y_predict= kmeans.fit_predict(x1)  

df1 

y_predict

kmeans.cluster_centers_



k=pd.DataFrame(k)

plt.scatter(k[y_predict == 0], k[y_predict == 0] ,s = 100, c = 'blue', label = 'Cluster 1') 
plt.scatter(k[y_predict == 1], k[y_predict == 1], s = 100, c = 'green', label = 'Cluster 2') 
plt.scatter(k[y_predict== 2], k[y_predict == 2],s = 100, c = 'red', label = 'Cluster 3') 

plt.scatter(x1[y_predict == 0], x1[y_predict == 0] ,s = 100, c = 'blue', label = 'Cluster 1') 
plt.scatter(x1[y_predict == 1], x1[y_predict == 1], s = 100, c = 'green', label = 'Cluster 2') 
plt.scatter(x1[y_predict== 2], x1[y_predict == 2],s = 100, c = 'red', label = 'Cluster 3')   
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1 ],kmeans.cluster_centers_[:,2 ],c = 'yellow', label = 'Centroid')   

plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.show()  

