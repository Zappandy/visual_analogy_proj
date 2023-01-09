import pandas as pd

from sklearn.model_selection import train_test_split

data = pd.read_csv("Custom_Recipes_NGL.csv")

print(data.columns)
data = data.drop(data.columns[[0]],axis=1)
print(data.columns)

train, test = train_test_split(data, test_size=.2)
val, test = train_test_split(test, test_size=.5)
print(train.shape)
print(test.shape)
print(val.shape)
print(data.shape)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
val.to_csv("validation.csv", index=False)
