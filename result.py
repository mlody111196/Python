import numpy as np
from scipy.stats import kde
import mdshare
import argparse
import sklearn
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import MeanShift, estimate_bandwidth

parser=argparse.ArgumentParser(description='Zmiana wymiarowosci danych')
parser.add_argument('-s', '--samp', type=int, metavar='', default=100, help='co ktora probka ma byc brana')
parser.add_argument('-d', '--dim', type=int, metavar='', default=2, help='ilosc wymiarow rzutowania')
parser.add_argument('-l', '--file_in', type=str, metavar='', required=True, help='wejsciowy plik z rozszerzeniem .npz')
parser.add_argument('-o', '--file_out', type=str, metavar='', default='wyjscie.png', 
                    help='wyjsciowy plik - obraz z rozszerzeniem .png')
args=parser.parse_args()

def funkcja(samp, dim, file_in, file_out):
    dataset = mdshare.fetch(file_in)
    with np.load(dataset) as f:
        X = np.vstack([f[key] for key in sorted(f.keys())])

    samples=np.arange(0, X.shape[0], samp)

    X_sample=X[samples,:]
    mds = manifold.TSNE(n_components=dim, init='pca')
    Y=mds.fit_transform(X_sample)
    k = kde.gaussian_kde(Y.T)
    xi, yi = np.mgrid[Y[:,0].min():Y[:,0].max():100*1j, Y[:,1].min():Y[:,1].max():100*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues)
    plt.contour(xi, yi, zi.reshape(xi.shape))
    plt.savefig(file_out, transparent=True, bbox_inches='tight')

if __name__ == '__main__':
    print (funkcja(args.samp, args.dim, args.file_in, args.file_out))
