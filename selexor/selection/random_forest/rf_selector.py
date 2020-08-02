import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RFSelector:
    def __init__(self, estimator_params: dict, k_features: int, x_train, y_train, *args,
                 scoring: callable = accuracy_score, **kwargs) -> None:
        self.__k_features = k_features
        self.__x_train = x_train
        self.__y_train = y_train
        self.__scoring = scoring
        self.__forest = RandomForestClassifier(**estimator_params)
        self.__importances = None
        self.__indices = None

    def plot(self, feat_labels: np.ndarray, fig_size, *args, title: str = 'Importances',
             color: str = 'green', align: str = 'center', **kwargs) -> plt.figure:
        # if self.__importances is None:
        #     self.select()
        #
        # fig, ax = plt.subplots(figsize=fig_size)
        # fig.tight_layout()
        #
        # ax.set_title(title)
        # ax.set_xticks(range(self.__x_train.shape[1]), feat_labels[self.__indices])
        # ax.set_xlim([-1, self.__x_train.shape[1]])
        # ax.bar(range(self.__x_train.shape[1]), self.__importances[self.__indices], color=color, align=align)

        # return fig
        pass

    def select(self) -> np.ndarray:
        if self.__importances is None:
            self.__forest.fit(self.__x_train, self.__y_train)
            self.__importances = self.__forest.feature_importances_
            self.__indices = np.argsort(self.__importances)[::-1]

        return self.__indices[:self.__k_features]
