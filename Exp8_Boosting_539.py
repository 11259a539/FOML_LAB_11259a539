import pandas as pd                                  
from sklearn.model_selection import train_test_split   
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("heart.csv")                      
X = data.drop("target", axis=1)                     
y = data["target"]                                 

X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=1)

model = GradientBoostingClassifier()                 
model.fit(X_train, y_train)                          
print("Accuracy:", round(model.score(X_test, y_test), 3)) 

// predicted data

new_patient = [X_test.iloc[0]]                        
result = model.predict(new_patient)[0]                 
print("Heart disease? (1=yes, 0=no):", result)   
