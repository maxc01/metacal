import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

from metacal.ts import TemperatureScaling
from metacal.utils import errors


class MetaCalMisCoverage:
    """under miscoverage rate constraint
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, xs, ys):
        """
        xs: logits, (N,K)
        ys: labels, (N,)
        """
        # 1. divide data into two parts
        neg_ind = np.argmax(xs, axis=1) == ys
        xs_neg, ys_neg = xs[neg_ind], ys[neg_ind]
        xs_pos, ys_pos = xs[~neg_ind], ys[~neg_ind]
        n1 = int(len(xs_neg) / 10)  # 1/10 of negative, compute a threshold
        n1 = min(n1, 500)
        x1, x2, _, y2 = train_test_split(xs_neg, ys_neg, train_size=n1)
        x2 = np.r_[x2, xs_pos]
        y2 = np.r_[y2, ys_pos]
        # 2. compute threshold on x1
        scores_x1 = entropy(softmax(x1, axis=1), axis=1)
        threshold = np.quantile(scores_x1, 1 - self.alpha, method="higher")

        # 3. fit a base calibrator on (x2,y2) | h(X) < threshold
        scores_x2 = entropy(softmax(x2, axis=1), axis=1)
        cond_ind = scores_x2 < threshold
        ts_model = TemperatureScaling()
        ts_x, ts_y = x2[cond_ind], y2[cond_ind]
        ts_model.fit(ts_x, ts_y)

        # 4. return the binary classifier and fitted base calibrator
        self.threshold = threshold
        self.base_model = ts_model

    def predict(self, X):
        """
        X: logits, (N,K)
        """
        if not hasattr(self, "threshold"):
            raise AttributeError("run fit on training set first")

        scores_X = entropy(softmax(X, axis=1), axis=1)
        neg_ind = scores_X < self.threshold
        proba_cal = np.empty_like(X)
        proba_cal[neg_ind] = self.base_model.predict(X[neg_ind])
        proba_cal[~neg_ind] = 1 / X.shape[1]

        return proba_cal

    def empirical_miscoverage(self, X, Y):
        """ empirical type-i, type-ii
        X: logits, (N,K)
        Y: labels, (N,)
        """
        scores_X = entropy(softmax(X, axis=1), axis=1)
        bin_pred = scores_X > self.threshold
        bin_target = np.argmax(X, axis=1) != Y
        R0, R1, _, _, _ = errors(bin_pred, bin_target)
        return R0, R1


class MetaCalCoverageAcc:
    """under coverage accuracy constraint
    """

    def __init__(self, acc):
        self.acc = acc

    def fit(self, xs, ys):
        """
        xs: logits, (N,K)
        ys: labels, (N,)
        """
        bins = 20  # number of bins used to estimate l
        n1 = int(len(xs) / 10)
        n1 = min(n1, 500)
        x1, x2, y1, y2 = train_test_split(xs, ys, train_size=n1)
        x1_pred = np.argmax(x1, axis=1)
        scores_x1 = entropy(softmax(x1, axis=1), axis=1)

        accs = []
        ents = []
        cut_points = np.quantile(scores_x1, np.linspace(0, 1, bins + 1))
        for (a, b) in zip(cut_points, cut_points[1:]):
            indices = np.where(np.logical_and(scores_x1 > a, scores_x1 <= b))[0]
            if len(indices) > 0:
                accs.append(np.mean(y1[indices] == x1_pred[indices]))
                ents.append(np.mean(scores_x1[indices]))
            else:
                accs.append(0)
                ents.append(0)
        accs_avg = np.add.accumulate(accs) / (np.arange(len(accs)) + 1)
        model_l = IsotonicRegression(increasing=False).fit(accs_avg, ents)

        threshold = model_l.predict([self.acc])[0]
        if np.isnan(threshold):
            raise ValueError("coverage accuracy should be increased")
        scores_x2 = entropy(softmax(x2, axis=1), axis=1)
        cond_ind = scores_x2 < threshold
        ts_model = TemperatureScaling()
        ts_x, ts_y = x2[cond_ind], y2[cond_ind]
        ts_model.fit(ts_x, ts_y)

        self.threshold = threshold
        self.base_model = ts_model

    def predict(self, X, return_ind=False):
        """
        X: logits, (N,K)
        return_ind: if True, return indices whose scores less than threshold
        """
        if not hasattr(self, "threshold"):
            raise AttributeError("run fit on training set first")

        scores_X = entropy(softmax(X, axis=1), axis=1)
        neg_ind = scores_X < self.threshold
        proba_cal = np.empty_like(X)
        proba_cal[neg_ind] = self.base_model.predict(X[neg_ind])
        proba_cal[~neg_ind] = 1 / X.shape[1]

        if return_ind:
            return proba_cal, neg_ind

        return proba_cal

    def empirical_coverage_acc(self, X, Y):
        """ empirical coverage accuracy
        X: logits, (N,K)
        Y: labels, (N,)
        """
        proba_cal, neg_ind = self.predict(X, return_ind=True)
        cov_acc = np.mean(proba_cal[neg_ind].argmax(axis=1) == Y[neg_ind])
        np.argmax(X, axis=1) == Y
        return cov_acc
