from sklearn.datasets import load_iris                  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier      

X, y = load_iris(return_X_y=True)                      
X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.2, random_state=1)                

model = KNeighborsClassifier(n_neighbors=5)              

print("Accuracy:", model.score(X_test, y_test))      
What this code does — in 3 steps
species = ["setosa", "versicolor", "virginica"]     
new_flower = [[5.1, 3.5, 1.4, 0.2]]                      
answer = model.predict(new_flower)[0]          
print("This flower is:", species[answer])      
