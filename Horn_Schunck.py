#!/usr/bin/env python
"""
python Horn_Schunck.py data/box box.*
python Horn_Schunck.py data/office office.*
python Horn_Schunck.py data/rubic rubic.*
python Horn_Schunck.py data/sphere sphere.*
"""
import time

from scipy.ndimage.filters import gaussian_filter
import imageio
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.pyplot import show
from argparse import ArgumentParser

from src.pyoptflow import HornSchunck, getimgfiles
from src.pyoptflow.plots import compareGraphs, plotderiv

FILTER = 7


def main():
    p = ArgumentParser(description="Pure Python Horn Schunck Optical Flow")
    p.add_argument("stem", help="path/stem of files to analyze")
    p.add_argument("pat", help="glob pattern of files", default="*.bmp")
    p.add_argument("-p", "--plot", help="show plots", action="store_true")
    p.add_argument(
        "-a", "--alpha", help="regularization parameter", type=float, default=0.001
    )
    p.add_argument("-N", help="number of iterations", type=int, default=8)
    p = p.parse_args()

    U, V = horn_schunck(p.stem, p.pat, alpha=p.alpha, Niter=p.N, verbose=p.plot)

    show()


def horn_schunck(stem: Path, pat: str, alpha: float, Niter: int, verbose: bool, filterSize=None):
    flist = getimgfiles(stem, pat)

    execTime = 0

    for i in range(len(flist) - 1):
        fn1 = flist[i]
        im1 = imageio.imread(fn1, as_gray=True)

        fn2 = flist[i + 1]
        im2 = imageio.imread(fn2, as_gray=True)

        # if filterSize is not None:
        #     im1f = gaussian_filter(im1, filterSize)
        #     im2f = gaussian_filter(im2, filterSize)
        # else:
        #     im1f = im1
        #     im2f = im2

        t1 = time.time()
        U, V = HornSchunck(im1, im2, alpha=alpha, Niter=Niter, verbose=False, filterSize=filterSize)
        dt = time.time() - t1
        execTime += dt
        print(f'Execution time: {dt}')

        if verbose:
            # compareGraphs(U, V, im2, fn=fn2.name)
            plotderiv(U, V, im1 - im2)

    print(f'Average execution time: {execTime / (len(flist) - 1)}')

    return U, V, im1, im2


if __name__ == "__main__":
    # main()

    plt.ion()

    """
    python Horn_Schunck.py data/box box.*
    python Horn_Schunck.py data/office office.*
    python Horn_Schunck.py data/rubic rubic.*
    python Horn_Schunck.py data/sphere sphere.*
    """
    alpha=1
    N=8
    filterSize = 2
    plot=False
    stem='src/pyoptflow/tests/data/office/'
    pat='office.*'
    stem=r'src/pyoptflow/tests/data/office'
    pat='office.[0,2].bmp'
    U, V, im1, im2 = horn_schunck(stem, pat, alpha=alpha, Niter=N, verbose=plot, filterSize=filterSize)

    plotderiv(U, V, im1 - im2)
    plt.show()

    print('Done')