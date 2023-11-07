import numpy as np


class AbstractDistance(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, p, q):
        raise NotImplementedError("All distance methods should implement this method")

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name


class EuclideanDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self, "EuclideanDistance")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p - q), 2)))
