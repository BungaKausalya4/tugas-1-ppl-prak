#%% import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import missingno as msno
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
import warnings
warnings.filterwarnings("ignore")


#import data
df_full = pd.read_csv("Iris.csv", delimiter=';')
print(df_full.head())

df_full = df_full.drop(columns=['Id'])
print(df_full.shape)

# lihat data
df_full

# cek null data
df_full.isnull().sum()

#cek outlier
Q1 = df_full.select_dtypes(include=np.number).quantile(0.25)
Q3 = df_full.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

outliers = ((df_full.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) | (df_full.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)
print(df_full[outliers])

plt.figure(figsize=(10,6))
sns.boxplot(data=df_full)
plt.title('Boxplot untuk Mendeteksi Outlier')
plt.show()

#Menghilangkan outlier
lower_bound = Q1[df_full.select_dtypes(include=np.number).columns] - 1.5 * IQR
upper_bound = Q3[df_full.select_dtypes(include=np.number).columns] + 1.5 * IQR
outlier_filter = ((df_full.select_dtypes(include=np.number) < lower_bound) | (df_full.select_dtypes(include=np.number) > upper_bound)).any(axis=1)

df_full = df_full[~outlier_filter]

plt.figure(figsize=(10,6))
sns.boxplot(data= df_full)
plt.title('Boxplot untuk Mendeteksi Outlier')
plt.show()

#Amati bentuk visual masing-masing fitur
sns.pairplot(df_full)
plt.show()

#Membangun FCM
nmpy = df_full.drop(columns=['Species']).values
model = FCM(n_clusters=3)
model.fit(nmpy)
centers = model.centers
labels = model.predict(nmpy)
plt.scatter(nmpy[labels == 0, 2], nmpy[labels == 0, 3], s=10, c='r')
plt.scatter(nmpy[labels == 1, 2], nmpy[labels == 1, 3], s=10, c='b')
plt.scatter(nmpy[labels == 2, 2], nmpy[labels == 2, 3], s=10, c='g')
plt.scatter(centers[:, 2], centers[:, 3], s=300, c='black', marker='+')
plt.title('Clustering')
plt.show()

#Melihat nilai silhouette score
score2 = silhouette_score(nmpy, labels)
print("Silhouette Score: ", score2)