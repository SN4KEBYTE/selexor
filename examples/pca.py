from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from examples.load_wine import load_wine
from selexor.pca.pca import PCA

# we will use Wine dataset for demonstration
x_train_std, x_test_std, y_train, y_test = load_wine()

# let's create a classifier and calculate accuracy score
knn = KNeighborsClassifier(n_jobs=-1)

knn.fit(x_train_std, y_train)
y_pred = knn.predict(x_test_std)
print(f'Accuracy score before PCA: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# now, let's create PCA instance and use fit_transform and transform methods to fit the dataset and apply dimensionality
# reduction
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

# let's fit the classifier on new data and calculate accuracy score again
knn.fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)
print(f'Accuracy score after PCA: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# you can get access to the projection matrix and explained variance
print(f'Projection matrix: {pca.projection_matrix}')
print(f'Explained variance: {pca.variance_explained}')
