import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('iphone.csv')
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[15]]))

plt.scatter(data['version'], data['price']) # for graph
# plt.bar(data['version'], data['price']) # for bar chart
plt.show()
# print(data.head())