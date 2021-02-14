from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from examples.load_wine import load_wine
from selexor.sbs import SBS

# we will use Wine dataset for demonstration
x_train_std, x_test_std, y_train, y_test = load_wine()

# let's create a classifier and calculate accuracy score
knn = KNeighborsClassifier(n_jobs=-1)

knn.fit(x_train_std, y_train)
y_pred = knn.predict(x_test_std)
print(f'Accuracy score before SBS: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# now, let's create SBS instance and use fit_transform and transform methods to fit the dataset and transform
# samples
sbs = SBS(knn, 5)
x_train_sbs = sbs.fit_transform(x_train_std, y_train)
x_test_sbs = sbs.transform(x_test_std)

# let's fit the classifier on new data and calculate accuracy score again
knn.fit(x_train_sbs, y_train)
y_pred = knn.predict(x_test_sbs)
print(f'Accuracy score after SBS: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# you can get access to the feature sets
print(f'Feature sets: {sbs.feature_sets}')
