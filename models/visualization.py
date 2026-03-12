import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


def show_t_sen(colors,data=None, label=None, embedding_size=2,  show=False,save_path=None,save_name=None):
    X = np.array(data)
    y = label


    tsne = manifold.TSNE(n_components=embedding_size, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(save_name)
    if show:
        plt.show()
    if save_path:
        plt.savefig((save_path+save_name))