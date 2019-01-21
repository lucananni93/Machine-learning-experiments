from abc import ABC, abstractmethod


class Estimator(ABC):

    def __init__(self):
        self.__is_fit = False

    def fit(self, X, y=None):
        self._fit(X, y)
        self.__is_fit = True

    @abstractmethod
    def _fit(self, X, y=None):
        pass

    def predict(self, X):
        if self.__is_fitted:
            return self._predict(X)
        else:
            raise ValueError("Not fitted estimator")

    @abstractmethod
    def _predict(self, X):
        pass

    @property
    def __is_fitted(self):
        return self.__is_fit
