import pandas as pd                                     
import seaborn as sns                                  

data = sns.load_dataset("titanic")                      
data = data[["survived","pclass","sex","age","fare","embarked"]]  

print(data.isnull().sum())                              

data["age"] = data["age"].fillna(data["age"].median())  
data["embarked"] = data["embarked"].fillna("S")         

data = pd.get_dummies(data, columns=["sex","embarked"], dtype=int)  

print(data.head())                                      
print("Any blanks left?", data.isnull().values.any())   
