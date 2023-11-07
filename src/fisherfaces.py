import numpy as np
import sys
from dataLoad import process_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class fisherFace:
    def project(self, W, train_data, mu=None):
        if mu is None:
            return np.dot(train_data, W)
        return np.dot(train_data - mu, W)

    def reconstruct(self, W, Y, mu=None):
        if mu is None:
            return np.dot(Y, W.T)
        return np.dot(Y, W.T) + mu

    def pca(self, train_data, train_label, num_components=0):
        [n, d] = train_data.shape

        if (num_components <= 0) or (num_components > n):
            num_components = n

        mu = train_data.mean(axis=0)
        train_data = train_data - mu

        if n > d:
            C = np.dot(train_data.T, train_data)
            [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        else:
            C = np.dot(train_data, train_data.T)
            [eigenvalues, eigenvectors] = np.linalg.eigh(C)
            eigenvectors = np.dot(train_data.T, eigenvectors)

            for i in range(n):
                eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(
                    eigenvectors[:, i]
                )
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = eigenvalues[0:num_components].copy()
        eigenvectors = eigenvectors[:, 0:num_components].copy()

        return [eigenvalues, eigenvectors, mu]

    def lda(self, train_data, train_lables, num_components=0):
        [n, d] = train_data.shape

        c = np.unique(train_lables)

        if (num_components <= 0) or (num_components > (len(c) - 1)):
            num_components = len(c) - 1

        meanTotal = train_data.mean(axis=0)

        Sw = np.zeros((d, d), dtype=np.float32)
        Sb = np.zeros((d, d), dtype=np.float32)

        for i in c:
            train_datai = train_data[np.where(train_lables == i)[0], :]
            meanClass = train_datai.mean(axis=0)
            Sw = Sw + np.dot((train_datai - meanClass).T, (train_datai - meanClass))
            Sb = Sb + n * np.dot((meanClass - meanTotal), (meanClass - meanTotal).T)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
        idx = np.argsort(-eigenvalues.real)

        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        eigenvalues = np.array(
            eigenvalues[0:num_components].real, dtype=np.float32, copy=True
        )
        eigenvectors = np.array(
            eigenvectors[0:, 0:num_components].real, dtype=np.float32, copy=True
        )

        return [eigenvalues, eigenvectors]

    def fisherfaces(self, train_data, train_labels, num_components=0):
        # print(train_data.reshape(150, 150))
        # print(train_data.shape[0])
        n = train_data.shape[0]

        c = len(np.unique(train_labels))
        [eigenvalues_pca, eigenvectors_pca, mu_pca] = self.pca(
            train_data, train_labels, n - c
        )

        [eigenvalues_lda, eigenvectors_lda] = self.lda(
            self.project(eigenvectors_pca, train_data, mu_pca),
            train_labels,
            num_components,
        )

        eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)

        return [eigenvalues_lda, eigenvectors, mu_pca]

    def create_font(self, fontname="Arial", fontsize=10):
        return {"fontname": fontname, "fontsize": fontsize}

    def normalize(self, x, low, high, dtype=None):
        # x = np.asarray(x)
        minx, marx = np.min(x), np.max(x)
        x = x - float(minx)
        x = x / float((marx - minx))
        x = x * (high - low)
        x = x + low

        if dtype is None:
            return np.asarray(x)
        return np.asarray(x, dtype=None)

    def asRowMatrix(self, X):
        if len(X) == 0:
            return np.array([])
        mat = np.empty((0, X[0].size), dtype=X[0].dtype)
        for row in X:
            mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
        return mat

    def subplot(
        self,
        title,
        images,
        rows,
        cols,
        sptitle="subplot",
        sptitles=[],
        colormap=cm.gray,
        ticks_visible=True,
        filename=None,
    ):
        fig = plt.figure()
        # main title
        fig.text(0.5, 0.95, title, horizontalalignment="center")
        for i in range(len(images)):
            ax0 = fig.add_subplot(rows, cols, (i + 1))
            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax0.get_yticklabels(), visible=False)

            if len(sptitles) == len(images):
                plt.title(
                    "%s #%s" % (sptitle, str(sptitles[i])),
                    self.create_font("Tahoma", 10),
                )
            else:
                plt.title("%s #%d" % (sptitle, (i + 1)), self.create_font("Tahoma", 10))
            plt.imshow(np.asarray(images[i]), cmap=colormap)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)


a = fisherFace()
train_data, train_label, test_data, test_label = process_data()
# print(a.fisherfaces(train_data, train_label))
# [D, W, mu] = a.fisherfaces(train_data, train_label)
# print(len(W[:, 1]))
# e = W[:, 1].reshape(-1, 1)
# print(e)

# print(a.project(e, train_data[0].reshape(1, -1), mu))
# p = a.project(e, train_data[0].reshape(1, -1), mu)
# R = a.reconstruct(e, p, mu)

# R = R.reshape(train_data[0].shape)


# print(a.reconstruct(e, p, mu))
# print(R)
[D, W, mu] = a.fisherfaces(a.asRowMatrix(train_data), train_label)

E = []
for i in range(min(W.shape[1], 16)):
    e = W[:, i].reshape(-1, 1)
    p = a.project(e, train_data[0].reshape(1, -1), mu)
    R = a.reconstruct(e, p, mu)
    R = R.reshape(train_data[0].shape)
    E.append(a.normalize(R, 0, 255))


a.subplot(
    title="Fisherfaces reconstruction yale database",
    images=E,
    rows=4,
    cols=4,
    sptitle="Fisherface",
    colormap=cm.gray,
    filename="images_new.png",
)


# class EuclideanDistance()


# class FisherfacesMode(BaseModel):

#     def __init__(self, )
