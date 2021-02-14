import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_wine():
    # sometimes UCI machine learning repository has problems with certificate and you will not be able to load dataset.
    # In this case, to run demo you need to download it somewhere else.
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

    df = pd.read_csv(url, header=None)
    df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                  'OD280/OD315 of diluted wines', 'Proline']

    x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    return x_train_std, x_test_std, y_train, y_test
