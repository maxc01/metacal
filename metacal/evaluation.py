import numpy as np


def reliability_diag(prob, t, n_bins=10, scheme="eqdist_prob", agg="mean"):
    """ Reliability Diagrams from Guo2017

    prob: (N, K), row sum=1
    t: (N,), an integer array
    """
    assert scheme in ["eqdist_prob", "eqdist_quantile"]
    assert agg in ["mean", "median"]
    pred = np.argmax(prob, axis=1)
    conf = np.max(prob, axis=1)
    if scheme == "eqdist_prob":
        cut_points = np.linspace(0, 1, n_bins + 1)
    else:
        cut_points = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    accs = []
    confs = []
    ns = []
    mids = []
    for (a, b) in zip(cut_points, cut_points[1:]):
        mids.append((a + b) / 2.0)
        indices = np.where(np.logical_and(conf > a, conf <= b))[0]
        if len(indices) > 0:
            if agg == "mean":
                accs.append((t[indices] == pred[indices]).mean())
                confs.append(conf[indices].mean())
            elif agg == "median":
                accs.append((t[indices] == pred[indices]).mean())
                confs.append(np.median(conf[indices]))
        else:
            accs.append(0)
            confs.append(0)
        ns.append(len(indices))
    gaps = np.abs(np.asarray(confs) - np.asarray(accs))
    # better names for "confs" and "accs" are "forecast probablity" and
    # "observed frequency" resp.
    return accs, confs, ns, mids, gaps


def ECE(prob, t, n_bins=10, scheme="eqdist_prob", agg="mean"):
    accs, confs, ns, _, _ = reliability_diag(prob, t, n_bins, scheme, agg)
    N = np.sum(ns)
    return np.sum([abs(acc - conf) * n / N for acc, conf, n in zip(accs, confs, ns)])
