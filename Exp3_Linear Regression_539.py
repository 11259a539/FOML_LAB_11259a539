import seaborn as sns                                    
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression       

data = sns.load_dataset("mpg").dropna()                  
X = data[["cylinders","displacement","horsepower","weight","acceleration","model_year"]]  
y = data["mpg"]                                         

X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.2, random_state=1)

model = LinearRegression()                              
model.fit(X_train, y_train)                            

print("R2 score:", round(model.score(X_test, y_test), 3)) 

//predicted data

new_car = [[4, 120, 90, 2500, 16, 80]]                   
mpg = model.predict(new_car)[0]                       
print("Predicted mileage:", round(mpg, 1), "mpg")

//graph code

import matplotlib.pyplot as plt      
import numpy as np

w = data["weight"]; m = data["mpg"]           
plt.scatter(w, m, alpha=0.5, color="#2F49D1", label="cars")  
a, b = np.polyfit(w, m, 1)                     
plt.plot(w, a*w + b, color="red", linewidth=2, label="trend line")
plt.xlabel("weight (lbs)"); plt.ylabel("mpg")  
plt.title("Heavier cars get fewer mpg"); plt.legend()
plt.show()       
