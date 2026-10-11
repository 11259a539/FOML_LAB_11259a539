
import pandas as pd                                
from sklearn.cluster import KMeans                     

data = pd.read_csv("Mall_Customers.csv")              
X = data[["Annual Income (k$)","Spending Score (1-100)"]]  

model = KMeans(n_clusters=5, n_init=10, random_state=1) 
model.fit(X)                                             
data["Group"] = model.labels_                           

print(data["Group"].value_counts())               

// predicted data

new_customer = [[75, 85]]                              
group = model.predict(new_customer)[0]                   
print("This customer belongs to group:", group)   

// graph code

import matplotlib.pyplot as plt         

x = data["Annual Income (k$)"]                
y = data["Spending Score (1-100)"]             
plt.scatter(x, y, c=model.labels_, cmap="tab10", s=30)
cen = model.cluster_centers_                   
plt.scatter(cen[:,0], cen[:,1], marker="X", s=200, c="black")  
plt.xlabel("income (k$)"); plt.ylabel("spending score")
plt.title("5 customer groups"); plt.show()
