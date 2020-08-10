from sklearn.metrics import accuracy_score

from examples.load_wine import load_wine
from selexor.lda.lda import LDA
from sklearn.neighbors import KNeighborsClassifier

x_train_std, x_test_std, y_train, y_test = load_wine()

knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(x_train_std, y_train)
y_pred = knn.predict(x_test_std)

print(f'Accuracy score before LDA: {accuracy_score(y_pred=y_pred, y_true=y_test)}')

lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train_std, y_train)
x_test_lda = lda.transform(x_test_std)

knn.fit(x_train_lda, y_train)
y_pred = knn.predict(x_test_lda)

print(f'Accuracy score after LDA: {accuracy_score(y_pred=y_pred, y_true=y_test)}')