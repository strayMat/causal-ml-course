# %%
import logging
import numpy as np


class IHDP(object):
    def __init__(self, path_data="datasets/IHDP", replications=1000):
        self.path_data = path_data
        self.path_data_train = path_data + "/ihdp_npci_1-1000.train.npz"
        self.path_data_test = path_data + "/ihdp_npci_1-1000.test.npz"
        self.arr_train = np.load(self.path_data_train)
        self.arr_test = np.load(self.path_data_test)
        self.replications = replications
        # which features are binary
        self.binfeats = [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
        ]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]
        self.logger = logging.getLogger("models.data.ihdp")

    def split_ihdp_dataset(self, arr, i_rep):
        t, y, y_cf = arr["t"], arr["yf"], arr["ycf"]
        mu_0, mu_1, Xs = arr["mu0"], arr["mu1"], arr["x"]

        t, y, y_cf = (
            t[:, i_rep][:, np.newaxis],
            y[:, i_rep][:, np.newaxis],
            y_cf[:, i_rep][:, np.newaxis],
        )
        mu_0, mu_1, Xs = (
            mu_0[:, i_rep][:, np.newaxis],
            mu_1[:, i_rep][:, np.newaxis],
            Xs[:, :, i_rep],
        )
        Xs[:, 13] -= 1  # this binary feature is in {1, 2}
        return (Xs, t, y), (y_cf, mu_0, mu_1)

    def _get_train_test(self, i):
        train = self.split_ihdp_dataset(self.arr_train, i)
        test = self.split_ihdp_dataset(self.arr_test, i)
        return train, test

    def get_train_xt(self, i):
        (x, t, y), _ = self.split_ihdp_dataset(self.arr_train, i)
        return x, t

    def get_xty(self, data):
        (x, t, y), _ = data
        return x, t, y

    def get_eval(self, data):
        (x, t, y), (y_cf, mu_0, mu_1) = data
        return Evaluator(mu_0, mu_1)

    def get_eval_idx(self, data, idx):
        (x, t, y), (y_cf, mu_0, mu_1) = data
        return Evaluator(mu_0[idx], mu_1[idx])


class Evaluator(object):
    def __init__(self, mu0, mu1):
        self.mu0 = mu0.reshape(-1, 1) if mu0.ndim == 1 else mu0
        self.mu1 = mu1.reshape(-1, 1) if mu1.ndim == 1 else mu1
        self.cate_true = self.mu1 - self.mu0
        self.metrics = ["ate", "pehe"]

    def get_metrics(self, cate_hat):
        ate_val = abs_ate(self.cate_true, cate_hat)
        pehe_val = pehe(self.cate_true, cate_hat)
        return [ate_val, pehe_val]


# %%
ihdp = IHDP("../data/IHDP")

all_ates = []
all_oracle_ates = []
for i in range(10):
    data = ihdp.split_ihdp_dataset(ihdp.arr_train, i)
    (x, t, y), (y_cf, mu_0, mu_1) = data
    oracle_ate = np.mean(mu_1 - mu_0)
    all_oracle_ates.append(oracle_ate)
    ate = np.mean(y - y_cf)
    all_ates.append(ate)
    print(f"oracle ATE {i}: {oracle_ate}/ ATE {ate}")
# %%
distrib = np.percentile(all_ates, [2.5, 50, 97.5])
mean_ate = np.mean(all_ates)
sd_ate = np.std(all_ates)
print(f"(Mean ATE / std ATE): ({mean_ate:.2f}, {sd_ate:.2f})")

distrib = np.percentile(all_oracle_ates, [2.5, 50, 97.5])
mean_oracle_ate = np.mean(all_oracle_ates)
sd_oracle_ate = np.std(all_oracle_ates)
print(
    f"(Mean oracle ATE / std oracle ATE): ({mean_oracle_ate:.2f}, {sd_oracle_ate:.2f}:.2f)"
)
