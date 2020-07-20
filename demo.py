import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.random_forest.rf_selector import RFSelector

# we will use Wine dataset as an example
from src.sbs.sbs import SBS

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header=None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
              'OD280/OD315 of diluted wines', 'Proline']

# let's extract class labels and samples and split data into train and test sets
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, 5)
sbs.fit(X_train, y_train)

# RFSelector returns indices of the most important features...
# rf_selector = RFSelector({'n_estimators': 10000, 'random_state': 0, 'n_jobs': -1}, 5, X_train, y_train)
# features = list(rf_selector.select())
#
# # ...but you can easily get access to their names
# print(df.columns[1:][features])

# fig = rf_selector.plot(df.columns[1:][features], (6, 8))
# fig.show()