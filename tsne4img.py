import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
#from sklearn.datasets import load_iris

N = 750 # random sample
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, help="Path to the data file")
args = vars(ap.parse_args())
dataset = np.loadtxt(args["data"], delimiter=" ")
print(dataset.shape)

np.random.shuffle(dataset)
X = dataset[:N,1:4]
y = dataset[:N,0]

X_tsne = TSNE(learning_rate=100).fit_transform(X)
X_iso = Isomap().fit_transform(X)
X_mds = MDS().fit_transform(X)
X_pca = PCA().fit_transform(X)

the_labels=['background', 'object']
colo = ['chartreuse', 'mediumslateblue']
plt.figure(figsize=(12,8))

plt.subplot(221)
for o in set(y):
    print(o)
    X_pca_o = X_pca[y == o]
    # print(X_pca_o[:, 0], X_pca_o[:, 1])
    plt.scatter(X_pca_o[:, 0], X_pca_o[:, 1],
                c = colo[int(o)],
                alpha = 0.6,
                label = the_labels[int(o)])
plt.title("PCA")
plt.legend(loc=0, fontsize=8, framealpha=0.3)
plt.figtext(0.99, 0.01, "N = "+str(N), horizontalalignment="right")

plt.subplot(222)
for o in set(y):
    X_iso_o = X_iso[y == o]
    plt.scatter(X_iso_o[:, 0], X_iso_o[:, 1],
                c = colo[int(o)],
                alpha = 0.6,
                label = the_labels[int(o)])
plt.title("Isomap")
#plt.legend(loc=0, fontsize=8, framealpha=0.3)

plt.subplot(223)
for o in set(y):
    X_tsne_o = X_tsne[y == o]
    plt.scatter(X_tsne_o[:, 0], X_tsne_o[:, 1],
                c = colo[int(o)],
                alpha = 0.6,
                label = the_labels[int(o)])
plt.title("t-SNE")
#plt.legend(loc=0, fontsize=8, framealpha=0.3)

plt.subplot(224)
for o in set(y):
    X_mds_o = X_mds[y == o]
    plt.scatter(X_mds_o[:, 0], X_mds_o[:, 1],
                c = colo[int(o)],
                alpha = 0.6,
                label = the_labels[int(o)])
plt.title("MDS")
#plt.legend(loc=0, fontsize=8, framealpha=0.3)

fig1 = plt.gcf()
plt.show()

fig1.savefig("foo.pdf", orientation='landscape')
