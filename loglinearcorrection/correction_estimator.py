# This script allows for using the log-linear correction estimator for OLS models, in one command.


from scripts.ppml_consistency import AssumptionTest
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CorrectedEstimator:

    def __init__(self, y, X, correction_model_type='nn', interest = 0, log_x=False):
        self.X = X
        self.y = y
        self.correction_model_type = correction_model_type.lower()
        if self.correction_model_type not in ['ols', 'nn']:
            raise ValueError("correction_model must be either 'ols' or 'nn'")
        self.x_index = interest
        self.log_x = log_x

    def fit(self, params_dict = None, **kwargs):

        if params_dict is None:
            print('Empty params_dict, using default values')
            params_dict = {}

        X_ols = self.X.copy()
        if self.log_x:
            X_ols[:, self.x_index] = np.log(X_ols[:, self.x_index])

        ols_results = sm.OLS(np.log(self.y), X_ols).fit(**kwargs)
        print(ols_results.summary())
        print('Estimating correction term...please wait')

        if self.correction_model_type == 'ols':

            correction_model = self._fit_ols_correction(ols_results, params_dict)

        elif self.correction_model_type == 'nn':
            correction_model = self._fit_nn_correction(ols_results, params_dict)

        print('Correction term estimated')
        if not self.log_x:
            return CorrectedEstimatorResultsLogLinear(self, ols_results, correction_model)
        else:
            return CorrectedEstimatorResultsLogLog(self, ols_results, correction_model)

    def _fit_ols_correction(self, ols_results, params_dict):
        poly = PolynomialFeatures(degree=params_dict.get('degree', 3), include_bias=False)
        X_poly = poly.fit_transform(self.X)
        eu = np.exp(ols_results.resid)
        correction_model = sm.OLS(eu, X_poly).fit()
        return OLSCorrectionModel(correction_model, poly)

    def _fit_nn_correction(self, ols_results, params_dict):
        nn_model = self.make_nn(params_dict)
        validation_split = params_dict.get('validation_split', 0.2)
        batch_size = params_dict.get('batch_size', 64)
        epochs = params_dict.get('epochs', 100)
        patience = params_dict.get('patience', 10)

        eu = tf.convert_to_tensor(np.exp(ols_results.resid))
        X_tensor = tf.convert_to_tensor(self.X)

        nn_model.fit(x=X_tensor, y=eu, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                     callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], verbose=1)
        return NNCorrectionModel(nn_model)



    def make_nn(self, params_dict):
        """
        :param params_dict: dictionary with parameters for the neural network
        :return: a neural network model
        """

        num_layers = params_dict.get('num_layers', 3)
        activation = params_dict.get('activation', 'relu')
        num_units = params_dict.get('num_units', 64)
        optimizer = params_dict.get('optimizer', 'adam')
        loss = params_dict.get('loss', 'mean_squared_error')


        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        elif len(num_units) != num_layers:
            raise ValueError("num_units must be an int or a list of length num_layers")


        # Create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.X.shape[1],)))
        for i in range(num_layers):
            model.add(tf.keras.layers.Dense(num_units[i], activation=activation))
        model.add(tf.keras.layers.Dense(1))

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss)

        return model

class CorrectedEstimatorResults:

    pass

class CorrectedEstimatorResultsLogLinear(CorrectedEstimatorResults):

    def __init__(self, model, ols_results, correction_model):
        self.model = model
        self.correction_model = correction_model
        self.index = model.x_index
        self.betahat = ols_results.params[self.index]
        self.se = self.correction_model.semi_elasticity(model.X, self.index)

    def average_semi_elasticity(self):
        return self.betahat + np.mean(self.se)

    def plot_dist_semi_elasticity(self):
        fig, ax = plt.subplots()
        sns.kdeplot(self.betahat + self.se, ax=ax)
        ax.vlines(self.betahat,ymin=0, ymax=1, color='red', linestyle='--', label='OLS Estimate', transform=ax.get_xaxis_transform())
        ax.vlines(self.average_semi_elasticity(), ymin=0, ymax=1, color='blue', linestyle='--', label='Corrected Estimate', transform=ax.get_xaxis_transform())
        ax.set_title('Distribution of Semi-Elasticity')
        ax.set_xlabel('Semi-Elasticity')
        ax.set_ylabel('Density')
        ax.legend()
        return fig, ax

    def semi_elasticiy_at_average(self):
        X_mean = np.mean(self.model.X, axis=0)
        return self.betahat + self.correction_model.semi_elasticity(X_mean.reshape(1, -1), self.index)[0]

    def test_ppml(self):
        ppml_mod = sm.GLM(self.model.y, self.model.X, family=sm.families.Poisson()).fit(cov_type='HC3')
        return AssumptionTest(ppml_mod).test_direct()

class CorrectedEstimatorResultsLogLog(CorrectedEstimatorResults):

    def __init__(self, model, ols_results, correction_model):
        self.model = model
        self.correction_model = correction_model
        self.index = model.x_index
        self.betahat = ols_results.params[self.index]
        self.e = self.correction_model.elasticity(model.X, self.index)

    def average_elasticity(self):
        return self.betahat + np.mean(self.e)

    def plot_dist_elasticity(self):
        fig, ax = plt.subplots()
        sns.kdeplot(self.betahat + self.e, ax=ax)
        ax.vlines(self.betahat,ymin=0, ymax=1, color='red', linestyle='--', label='OLS Estimate', transform=ax.get_xaxis_transform())
        ax.vlines(self.average_elasticity(), ymin=0, ymax=1, color='blue', linestyle='--', label='Corrected Estimate', transform=ax.get_xaxis_transform())
        ax.set_title('Distribution of Elasticity')
        ax.set_xlabel('Elasticity')
        ax.set_ylabel('Density')
        ax.legend()
        return fig, ax

    def semi_elasticiy_at_average(self):
        X_mean = np.mean(self.model.X, axis=0)
        return self.betahat + self.correction_model.semi_elasticity(X_mean.reshape(1, -1), self.index)[0]

    def test_ppml(self):
        ppml_mod = sm.GLM(self.model.y, self.model.X, family=sm.families.Poisson()).fit(cov_type='HC3')
        return AssumptionTest(ppml_mod).test_direct()


class CorrectionModel:

    def predict(self, X):
        pass

    def marginal_effects(self, X, index):
        pass

    def semi_elasticity(self, X, index):
        pass

    def elasticity(self, X, index):
        pass

class OLSCorrectionModel(CorrectionModel):

    def __init__(self, model, poly):
        self.model = model
        self.poly = poly

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def marginal_effects(self, X, index):
        delta = np.zeros(X.shape)
        delta[:, index] = 1e-6
        eu_new = self.predict(X + delta)
        eu_old = self.predict(X)
        return (eu_new - eu_old) / 1e-6

    def semi_elasticity(self, X, index):
        delta = np.zeros(X.shape)
        delta[:, index] = 1e-6
        eu_new = self.predict(X + delta)
        eu_old = self.predict(X)
        return ((eu_new - eu_old) / 1e-6)/ eu_old

    def elasticity(self, X, index):
        return self.semi_elasticity(X, index) * X[:, index].reshape(-1,)

class NNCorrectionModel(CorrectionModel):

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model(X)

    def marginal_effects(self, X, index):
        X_used = tf.convert_to_tensor(X, dtype=tf.float32)
        X_used = tf.Variable(X_used)
        with tf.GradientTape() as tape:
            tape.watch(X_used)
            euhat = self.predict(X_used)
        grads = tape.gradient(euhat, X_used)
        return (grads[:, index]).numpy()

    def semi_elasticity(self, X, index):
        X_used = tf.convert_to_tensor(X, dtype=tf.float32)
        X_used = tf.Variable(X_used)
        with tf.GradientTape() as tape:
            tape.watch(X_used)
            euhat = self.predict(X_used)
        grads = tape.gradient(euhat, X_used)
        return grads[:, index].numpy() / euhat.numpy().reshape(-1,)

    def elasticity(self, X, index):
        X_used = tf.convert_to_tensor(X, dtype=tf.float32)
        X_used = tf.Variable(X_used)
        with tf.GradientTape() as tape:
            tape.watch(X_used)
            euhat = self.predict(X_used)
        grads = tape.gradient(euhat, X_used)
        return grads[:, index].numpy() / euhat.numpy().reshape(-1,) * X_used.numpy()[:, index].reshape(-1,)
    