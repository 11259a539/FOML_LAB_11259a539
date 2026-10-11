import pandas as pd                                  
from sklearn.model_selection import train_test_split     
from sklearn.preprocessing import StandardScaler       
from sklearn.neighbors import KNeighborsClassifier    
from sklearn.svm import SVC                          

data = pd.read_csv("Social_Network_Ads.csv")          
X = data[["Age","EstimatedSalary"]]                
y = data["Purchased"]                                  

X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.25, random_state=1)

scaler = StandardScaler()                           
X_train = scaler.fit_transform(X_train)                 
X_test  = scaler.transform(X_test)                    

knn = KNeighborsClassifier(n_neighbors=5)             
knn.fit(X_train, y_train)                               
print("KNN accuracy:", round(knn.score(X_test, y_test), 2)) 

svm = SVC()                                              
svm.fit(X_train, y_train)                            
print("SVM accuracy:", round(svm.score(X_test, y_test), 2))

// predicted output

new_person = scaler.transform([[40, 90000]])           
answer = knn.predict(new_person)[0]                   
print("Will they buy? (1=yes, 0=no):", answer)

