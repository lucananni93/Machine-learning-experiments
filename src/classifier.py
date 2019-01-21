from .base import Estimator
import numpy as np
from .utils import check_array
from scipy.stats import mode


class KNNClassifier(Estimator):
    def __init__(self, k=1, metric='euclidean'):
        super(Estimator, self).__init__()

        self.K = k
        self.__metric = metric

    def _fit(self, X, y=None):
        self.__X = check_array(X, 2)
        self.__Y = check_array(y, 1)

    def _predict(self, X):
        X = check_array(X, 2)
        y_pred = np.zeros(X.shape[0], dtype=self.__Y.dtype)
        for i in range(X.shape[0]):
            x = X[i, :]
            if self.__metric == 'euclidean':
                d_x = np.sqrt(np.sum((self.__X - x)**2, axis=1))
            else:
                raise NotImplementedError("metric not implemented")
            sorted_idxs = np.argsort(d_x)
            top_k_idxs = sorted_idxs[:self.K]
            top_k_ys = self.__Y[top_k_idxs].flatten()
            y = mode(top_k_ys)
            y_pred[i] = y
        return y_pred
