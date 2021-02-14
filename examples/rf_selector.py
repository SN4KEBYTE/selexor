from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from examples.load_wine import load_wine
from selexor.random_forest import RFSelector

# we will use Wine dataset for demonstration
x_train_std, x_test_std, y_train, y_test = load_wine()

# let's create a classifier and calculate accuracy score
knn = KNeighborsClassifier(n_jobs=-1)

knn.fit(x_train_std, y_train)
y_pred = knn.predict(x_test_std)
print(f'Accuracy score before RFSelector: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# now, let's create RFSelector instance and use fit_transform and transform methods to fit the dataset and transform
# samples
rf = RFSelector(n_components=2, estimator_params={'max_depth': 3, 'n_jobs': -1})
x_train_rf = rf.fit_transform(x_train_std, y_train)
x_test_rf = rf.transform(x_test_std)

# let's fit the classifier on new data and calculate accuracy score again
knn.fit(x_train_rf, y_train)
y_pred = knn.predict(x_test_rf)
print(f'Accuracy score after RFSelector: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# you can get access to the feature importances
print(f'Features importances: {rf.feature_importances}')
