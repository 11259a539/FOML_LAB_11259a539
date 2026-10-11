from sklearn.datasets import load_iris                  
from sklearn.model_selection import train_test_split   
from sklearn.neighbors import KNeighborsClassifier      

X, y = load_iris(return_X_y=True)                        

X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.2, random_state=1)                 
model = KNeighborsClassifier(n_neighbors=5)            
model.fit(X_train, y_train)                              

print("Accuracy:", model.score(X_test, y_test)) 
