import pandas as pd
data=pd.read_csv('OnlineRetail.csv',encoding='ISO-8859/1')
print(data.info())
print(data.columns)
print(data.head())
print(data.shape)
print(data.describe)
print(data.isnull().sum())
data_null=round(100*(data.isnull().sum())/len(data),2)
print(data_null)
data=data.dropna()
print(data.shape)
data['CustomerID']=data['CustomerID'].astype(str)
print(data.info())
data['Amount']=data['Quantity']*data['UnitPrice']
print(data.info())
datamonetary=data.groupby('CustomerID')['Amount'].sum()
print(datamonetary.head())

#Most selling product
product=data.groupby('Description')['Quantity'].sum()
print (product.head())

#region that's selling most products
region=data.groupby('Country')['Amount'].sum()
print(region.head())

#reset the index
df=data.reset_index()
print(df.head())

#most frequently bought product
product=data.groupby('Description')['Quantity'].count()
print(product.head())


data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'],format='%m/%d/%Y %H:%M')
print(data.InvoiceDate)

#last Transaction Date
max_date=max(data['InvoiceDate'])
print(max_date)
print('Next Line')             
min_date=min(data['InvoiceDate'])
print(min_date)

#Total Sales for the last month
from datetime import timedelta
new_min_date=max_date-timedelta(days=30)
last_30_days_sales=data[(data['InvoiceDate']>=new_min_date)&(data['InvoiceDate']<=max_date)]['Amount'].count()
print(last_30_days_sales)


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
features = ['Quantity', 'UnitPrice', 'CustomerID']
data = data.dropna(subset=features)
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

kmeans = KMeans(n_clusters=4, max_iter=50, random_state=42)
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=50, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_k = 5  # Based on the analysis
kmeans = KMeans(n_clusters=optimal_k, max_iter=50, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
#print("For n_clusters={0}, the silhouette score is {1}".format(n_clusters, silhouette_avg))
data.to_csv('data_with_clusters.csv', index=False)


