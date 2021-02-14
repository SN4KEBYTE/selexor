from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from examples.load_wine import load_wine
from selexor.pca import KernelPCA

# we will use Wine dataset for demonstration
x_train_std, x_test_std, y_train, y_test = load_wine()

# let's create a classifier and calculate accuracy score
knn = KNeighborsClassifier(n_jobs=-1)

knn.fit(x_train_std, y_train)
y_pred = knn.predict(x_test_std)
print(f'Accuracy score before KernelPCA: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# now, let's create KernelPCA instance and use fit_transform and transform methods to fit the dataset and apply
# dimensionality reduction
k_pca = KernelPCA(n_components=2, gamma=1.0)
x_train_k_pca = k_pca.fit_transform(x_train_std)
x_test_k_pca = k_pca.transform(x_test_std)

# let's fit the classifier on new data and calculate accuracy score again
knn.fit(x_train_k_pca, y_train)
y_pred = knn.predict(x_test_k_pca)
print(f'Accuracy score after KernelPCA: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

# you can get access to the eigen values and eigen vectors
print(f'Eigen values: {k_pca.lambdas}')
print(f'Eigen vectors: {k_pca.alphas}')
