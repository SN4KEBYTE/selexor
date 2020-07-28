import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class LDA:
    def __init__(self, k: int) -> None:
        self.__k = k
        self.__w: np.ndarray = None
        self.__scaler = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, standardize: bool = False,
            encode_labels: bool = False) -> 'LDA':
        if standardize:
            self.__scaler = StandardScaler()
            x_train_std = self.__scaler.fit_transform(x_train)
        else:
            x_train_std = x_train

        if encode_labels:
            le = LabelEncoder()
            le.fit(y_train)

            y_train_encoded = le.transform(y_train)
        else:
            y_train_encoded = y_train

        labels = np.sort(np.unique(y_train_encoded))

        mean_vecs = [np.mean(x_train_std[y_train_encoded == label], axis=0) for label in labels]

        dim = x_train_std.shape[1]
        s_w = np.zeros((dim, dim))

        for label, mean_vec in zip(labels, mean_vecs):
            class_scatter = np.cov(x_train_std[y_train_encoded == label].T)
            s_w += class_scatter

        mean_overall = np.mean(x_train_std, axis=0)
        s_b = np.zeros((dim, dim))

        for i, mean_vec in enumerate(mean_vecs):
            n = x_train_std[y_train_encoded == i, :].shape[0]
            mean_vec = mean_vec.reshape(dim, 1)
            mean_overall = mean_overall.reshape(dim, 1)
            s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        eigen_pairs = sorted([(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))], reverse=True)

        new_features = [eigen_pairs[i][1][:, np.newaxis].real for i in range(self.__k)]
        self.__w = np.hstack(new_features)

        return self

    def transform(self, x_train: np.ndarray) -> np.ndarray:
        if self.__w is None:
            raise RuntimeError('Projection matrix is not calculated. Please use fit method first, or fit_transform.')

        return x_train.dot(self.__w)

    def fit_transform(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        self.fit(x_train, y_train)

        return x_train.dot(self.__w)

    @property
    def w(self):
        return self.__w
