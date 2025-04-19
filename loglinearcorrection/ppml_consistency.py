######################################################################################
#This script is used to test for the E[\eta_i|X_i] = 1 condition for PPML
# Currently only available on statsmodels GLM with Poisson family
#######################################################################################

from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
import statsmodels.api as sm
import numpy as np
from resample.bootstrap import bootstrap
from scipy.stats import norm



class AssumptionTest:

    def __init__(self, model_results:GLMResultsWrapper):
        self.model_results = model_results
        self.data = np.concatenate([model_results.model.endog.reshape(-1,1), model_results.model.exog], axis=1)
        self.len_data = len(self.data)
        yhat = model_results.mu
        self.resids = model_results.model.endog / yhat
        self.stat_numerator = np.mean(self.resids)

    def test_bootstrap(self, size=1000):
        def stat(data):
            y, x = data[:,0], data[:,1:]
            model = sm.GLM(y, x, family=sm.families.Poisson())
            results = model.fit()
            yhat = results.mu
            resids = y/yhat
            return np.mean(resids)

        # bootstrap the test statistic
        bs_samp = bootstrap(stat, self.data, size=size)
        bs_diff = bs_samp - self.stat_numerator
        bs_std = np.std(bs_diff)
        p_value_bs_samp = bs_diff/bs_std

        test_stat = (self.stat_numerator - 1)/bs_std
        p_value = np.mean(np.abs(p_value_bs_samp) > np.abs(test_stat))
        reject = p_value < 0.05
        return AssumptionTestResults(test_stat, p_value, reject)


    def test_direct(self):
        denom_1 = np.var(self.resids)
        mean_x = np.mean(self.data[:,1:], axis=0).reshape(-1,1)
        denom_2 = mean_x.transpose() @ self.model_results.cov_params() @ mean_x
        variance_stat = denom_1 + denom_2[0][0]
        stat = np.sqrt(self.len_data) * (self.stat_numerator - 1)/np.sqrt(variance_stat)
        p_value = 2 * (1 - norm.cdf(np.abs(stat)))
        reject = p_value < 0.05
        return AssumptionTestResults(stat, p_value, reject)


class AssumptionTestResults:

    def __init__(self, test_stat, p_value, reject):
        self.test_stat = test_stat
        self.p_value = p_value
        self.reject = reject

    def __str__(self):
        return f"Test Statistic: {self.test_stat}, P-value: {self.p_value}, Reject Null: {self.reject}"



