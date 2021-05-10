from scipy.optimize import minimize
from sklearn.metrics import log_loss
from scipy.special import softmax
import numpy as np


class TemperatureScaling:
    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        K = probs.shape[1]
        loss = log_loss(y_true=true, y_pred=scaled_probs, labels=np.arange(K))
        return loss

    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        true = true.flatten()  # Flatten y_val
        opt = minimize(
            self._loss_fun,
            x0=1,
            args=(logits, true),
            options={"maxiter": self.maxiter},
            method=self.solver,
        )
        self.temp = opt.x[0]

        return opt

    def predict(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp, axis=1)
        else:
            return softmax(logits / temp, axis=1)


def run_ts(conf_name):
    from metacal.utils import load_data

    X_train, Y_train, X_test, Y_test = load_data(conf_name)
    ts_model = TemperatureScaling()
    ts_model.fit(X_train, Y_train)
    proba_test = ts_model.predict(X_test)
    return proba_test, Y_test
