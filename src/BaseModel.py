import numpy as np
from fisherfaces import fisherFace
from distance import EuclideanDistance
from dataLoad import process_data


class BaseModel(object):
    def __init__(
        self,
        train_data=None,
        train_label=None,
        dist_metric=EuclideanDistance(),
        num_components=0,
    ):
        self.dist_metric = dist_metric
        self.num_components = 0
        self.projections = []
        self.W = []
        self.mu = []
        if (train_data is not None) and (train_label is not None):
            self.compute(train_data, train_label)

    def compute(self, train_data, train_label):
        raise NotImplementedError("Every BaseModel must implement the compute method")

    def predict(self, test_data):
        minDist = np.finfo("float").max
        minClass = -1
        Q = fisherFace.project(self.W, test_data.reshape(1, -1), self.mu)

        for i in range(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.train_label[i]

        return minClass


class FisherfaceModel(BaseModel):
    def __init__(
        self,
        train_data=None,
        train_label=None,
        dist_metric=EuclideanDistance(),
        num_components=0,
    ):
        super(FisherfaceModel, self).__init__(
            train_data=train_data,
            train_label=train_label,
            dist_metric=dist_metric,
            num_components=num_components,
        )

    def compute(self, X, y):
        a = fisherFace()
        [D, self.W, self.mu] = a.fisherfaces(a.asRowMatrix(X), y, self.num_components)

        self.train_label = y

        for xi in X:
            self.projections.append(a.project(self.W, xi.reshape(1, -1), self.mu))

    def predict(self, test_data):
        a = fisherFace()
        minDist = np.finfo("float").max
        minClass = -1
        Q = a.project(self.W, test_data.reshape(1, -1), self.mu)

        for i in range(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.train_label[i]

        return minClass


train_data, train_label, test_data, test_label = process_data()

# print(test_data.shape, test_label)
print(len(train_data[:1]), len(train_data[1:]))
model = FisherfaceModel(train_data, train_label)

print(test_label[19])
for index, test in enumerate(test_data):
    print(
        "Expected =",
        test_label[index],
        " / Predicted =",
        model.predict(test_data[index]),
    )
