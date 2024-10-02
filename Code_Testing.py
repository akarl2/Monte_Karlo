import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")

data = pd.read_csv('framingham.csv')
data = data.dropna()
print(data.shape)

y = data.TenYearCHD.values
x = data.drop(['TenYearCHD'], axis = 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print("Test Accuracy {:.2f}%".format(lr.score(x_test, y_test)*100))

#print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr.predict(x_test))
sns.heatmap(cm, annot=True)
print(cm)

print(lr.coef_)
print(lr.intercept_)


