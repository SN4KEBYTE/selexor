import numpy as np

from src.extracion.extractor import Extractor


class LDA(Extractor):
    def __init__(self, k: int) -> None:
        super(LDA, self).__init__(k)
        self.__scaler = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> 'LDA':
        labels = np.sort(np.unique(y_train))

        mean_vecs = [np.mean(x_train[y_train == label], axis=0) for label in labels]

        dim = x_train.shape[1]
        s_w = np.zeros((dim, dim))

        for label, mean_vec in zip(labels, mean_vecs):
            class_scatter = np.cov(x_train[y_train == label].T)
            s_w += class_scatter

        mean_overall = np.mean(x_train, axis=0)
        s_b = np.zeros((dim, dim))

        for i, mean_vec in enumerate(mean_vecs):
            n = x_train[y_train == i, :].shape[0]
            mean_vec = mean_vec.reshape(dim, 1)
            mean_overall = mean_overall.reshape(dim, 1)
            s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        eigen_pairs = sorted([(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))],
                             key=lambda k: k[0], reverse=True)

        new_features = [eigen_pairs[i][1][:, np.newaxis].real for i in range(self._k)]
        self._w = np.hstack(new_features)

        return self

    def fit_transform(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        self.fit(x_train, y_train)

        return self.transform(x_train)
