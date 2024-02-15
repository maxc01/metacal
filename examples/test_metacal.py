import numpy as np
import argparse
from numpy.random import default_rng
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from metacal.metacal import MetaCalCoverageAcc
from metacal.metacal import MetaCalMisCoverage
from metacal.evaluation import ECE


def acc(logits, labels):
    return (np.argmax(logits, axis=1) == labels).mean()


def make_data():
    train = np.load("train-logits.npz")
    X1, Y1 = train["logits"], train["targets"]
    test = np.load("test-logits.npz")
    X2, Y2 = test["logits"], test["targets"]
    return X1, X2, Y1, Y2


def test_mis_cov(target_alpha=0.05):
    X1, X2, Y1, Y2 = make_data()
    model = MetaCalMisCoverage(alpha=target_alpha)
    model.fit(X1, Y1)
    ece = ECE(model.predict(X2), Y2, 15)
    emp_mis = model.empirical_miscoverage(X2, Y2)[0]
    return ece, emp_mis


def test_cov_acc(target_acc=0.9):
    X1, X2, Y1, Y2 = make_data()
    model = MetaCalCoverageAcc(acc=target_acc)
    model.fit(X1, Y1)
    proba_cal, neg_ind = model.predict(X2, return_ind=True)
    ece = ECE(proba_cal, Y2, 15)
    emp_acc = model.empirical_coverage_acc(X2, Y2)
    return ece, emp_acc, neg_ind


def plot_tsne_fitting(X, Y, ax):
    trans_data = (
        TSNE(n_components=2, learning_rate="auto", init="random")
        .fit_transform(X)
        .T
    )
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labeltop=False,
        labelbottom=False,
        labelleft=False,
        labelright=False,
    )
    ax.scatter(trans_data[0], trans_data[1], c=Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_alpha", type=float, default=0.05)
    parser.add_argument("--target_acc", type=float, default=0.999)
    args = parser.parse_args()
    target_alpha = args.target_alpha
    target_acc = args.target_acc

    X1, X2, Y1, Y2 = make_data()
    train_acc = acc(X1, Y1)
    test_acc = acc(X2, Y2)
    assert (
        target_acc > test_acc
    ), f"target accuracy: {target_acc} must be higher than original test accuracy: {test_acc}"

    print(f"Test MisCoverage (target={target_alpha})")
    print(
        "ECE: {}, empirical miscoverage: {}".format(
            *test_mis_cov(target_alpha=target_alpha)
        )
    )
    print(f"Original test accuracy: {test_acc}")
    print(f"Test CoverageAcc (target={target_acc})")
    ece, emp_acc, neg_ind = test_cov_acc(target_acc=target_acc)
    print("ECE: {}, empirical coverageacc: {}".format(ece, emp_acc))

    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 12))
    plot_tsne_fitting(X2, Y2, ax0)
    ax0.set_title("TSNE using tets logits")

    plot_tsne_fitting(X2[neg_ind], Y2[neg_ind], ax1)
    ax1.set_title(
        "TSNE with MetaCal using test logits (unseen during MetaCal training)"
    )
    plt.savefig("TSNE-metacal.jpg")
    plt.show()


if __name__ == "__main__":
    main()
