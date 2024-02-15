import numpy as np
from numpy.random import default_rng

from sklearn.model_selection import train_test_split

from metacal.metacal import MetaCalCoverageAcc
from metacal.metacal import MetaCalMisCoverage
from metacal.evaluation import ECE


def make_data():
    rng = default_rng()
    d = 3
    n = 5000
    X = rng.uniform(-5, 5, size=(n, d))
    Y = rng.choice(d, size=n)
    X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.5)
    return X1, X2, Y1, Y2


def test_mis_cov():
    X1, X2, Y1, Y2 = make_data()
    __import__('ipdb').set_trace()
    model = MetaCalMisCoverage(alpha=0.05)
    model.fit(X1, Y1)
    ece = ECE(model.predict(X2), Y2, 15)
    emp_mis = model.empirical_miscoverage(X2, Y2)[0]
    return ece, emp_mis


def test_cov_acc():
    X1, X2, Y1, Y2 = make_data()
    model = MetaCalCoverageAcc(acc=0.34)
    model.fit(X1, Y1)
    ece = ECE(model.predict(X2), Y2, 15)
    emp_acc = model.empirical_coverage_acc(X2, Y2)
    return ece, emp_acc


def main():
    print("Test MisCoverage (target=0.05)")
    print("ECE: {}, empirical miscoverage: {}".format(*test_mis_cov()))
    print("Test CoverageAcc (target=0.34)")
    should_cont = True
    while should_cont:
        try:
            print("ECE: {}, empirical coverageacc: {}".format(*test_cov_acc()))
        except ValueError:
            pass
        else:
            should_cont = False


if __name__ == "__main__":
    main()
