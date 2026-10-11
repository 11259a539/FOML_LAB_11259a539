
import pandas as pd                                   
from sklearn.cluster import AgglomerativeClustering    

data = pd.read_csv("Mall_Customers.csv")                 
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  

model = AgglomerativeClustering(n_clusters=5)          
data["Group"] = model.fit_predict(X)                   

print(data.groupby("Group")[["Annual Income (k$)","Spending Score (1-100)"]].mean())  

// predicted data

new_customer = [80, 15]                                
averages = data.groupby("Group")[["Annual Income (k$)","Spending Score (1-100)"]].mean()  
distances = ((averages - new_customer) ** 2).sum(axis=1)  
print("Closest group:", distances.idxmin()) 
