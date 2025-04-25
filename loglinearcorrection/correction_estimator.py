# This script allows for using the log-linear correction estimator for OLS models, in one command.


from .ppml_consistency import AssumptionTest
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CorrectedEstimator:
    """Corrected estimator for log-linear and log-log regression models.
    
    This class implements a correction method for the bias that occurs in log-transformed OLS models.
    It provides methods to estimate the correction term using either OLS with polynomial features
    or neural networks.
    """

    def __init__(self, y, X, correction_model_type='nn', interest=0, log_x=False, fe=None):
        """Initialize the corrected estimator.
        
        :param y: Response variable (untransformed)
        :type y: numpy.ndarray
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param correction_model_type: Type of model to use for estimating the correction term, either 'nn' (neural network) or 'ols' (polynomial regression), defaults to 'nn'
        :type correction_model_type: str, optional
        :param interest: Index of the regressor of interest in X, defaults to 0
        :type interest: int, optional
        :param log_x: Whether the regressor of interest is log-transformed, defaults to False
        :type log_x: bool, optional
        :raises ValueError: If correction_model_type is not 'ols' or 'nn'
        """

        if fe is not None:
            one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
            dummy_vars = one_hot_encoder.fit_transform(X[:,fe])
            X = np.delete(X, fe, axis=1)
            residual_maker = np.eye(len(dummy_vars)) - dummy_vars @ np.linalg.pinv(np.transpose(dummy_vars) @ dummy_vars) @ np.transpose(dummy_vars)
            demeaned = residual_maker @ np.concatenate([X, y.reshape(-1,1)], axis=1)
            X= demeaned[:,:-1]
            y = demeaned[:,-1]
            print('Care fe not fully implemented yet, please check the results')

        self.X = X
        self.y = y
        self.correction_model_type = correction_model_type.lower()
        if self.correction_model_type not in ['ols', 'nn']:
            raise ValueError("correction_model must be either 'ols' or 'nn'")
        self.x_index = interest
        self.log_x = log_x

    def fit(self, params_dict=None, weights=1.0, **kwargs):
        """Fit the corrected estimator model.
        
        This method first fits a standard OLS model on log-transformed data, then estimates
        the correction term using either a polynomial OLS model or a neural network.
        
        :param params_dict: Dictionary with parameters for the correction model, defaults to None
        :type params_dict: dict, optional
        :param kwargs: Additional arguments to pass to the OLS fit method
        :return: Fitted model results object (either CorrectedEstimatorResultsLogLinear or CorrectedEstimatorResultsLogLog)
        """
        if params_dict is None:
            print('Empty params_dict, using default values')
            params_dict = {}

        ols_results = sm.WLS(np.log(self.y), self.X, weights=weights).fit(**kwargs)
        print(f'OLS model fitted, estimated betahat: {ols_results.params[self.x_index]:.4f}')
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
        """Fit a polynomial OLS model for the correction term.
        
        :param ols_results: Results from the initial OLS model
        :type ols_results: statsmodels.regression.linear_model.RegressionResults
        :param params_dict: Dictionary with parameters, including 'degree' for polynomial order
        :type params_dict: dict
        :return: Fitted OLS correction model
        :rtype: OLSCorrectionModel
        """
        x_ols_correction = self.X.copy()
        if self.log_x:
            x_ols_correction[:, self.x_index] = np.exp(x_ols_correction[:, self.x_index])
        poly = PolynomialFeatures(degree=params_dict.get('degree', 3), include_bias=False)
        X_poly = poly.fit_transform(x_ols_correction)
        eu = np.exp(ols_results.resid)
        correction_model = sm.OLS(eu, X_poly).fit()
        return OLSCorrectionModel(correction_model, poly)

    def _fit_nn_correction(self, ols_results, params_dict):
        """Fit a neural network model for the correction term.
        
        :param ols_results: Results from the initial OLS model
        :type ols_results: statsmodels.regression.linear_model.RegressionResults
        :param params_dict: Dictionary with parameters for the neural network
        :type params_dict: dict
        :return: Fitted neural network correction model
        :rtype: NNCorrectionModel
        """
        import tensorflow as tf
        nn_model = self.make_nn(params_dict)
        validation_split = params_dict.get('validation_split', 0.2)
        batch_size = params_dict.get('batch_size', 64)
        epochs = params_dict.get('epochs', 100)
        patience = params_dict.get('patience', 10)
        verbose = params_dict.get('verbose', 1)

        residuals_normalized = (ols_results.resid - np.mean(ols_results.resid)) / np.std(ols_results.resid)

        residuals_clipped = np.clip(residuals_normalized, -60, 60)

        eu = tf.convert_to_tensor(np.exp(residuals_clipped))

        X_nn = self.X.copy()
        if self.log_x:
            X_nn[:, self.x_index] = np.exp(X_nn[:, self.x_index])
        X_tensor = tf.convert_to_tensor(X_nn)

        nn_model.fit(x=X_tensor, y=eu, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], verbose=verbose)
        return NNCorrectionModel(nn_model, np.mean(ols_results.resid), np.std(ols_results.resid))

    def make_nn(self, params_dict):
        """Create a neural network model for estimating the correction term.
        
        :param params_dict: Dictionary with parameters for the neural network
        :type params_dict: dict
        :return: A neural network model
        :rtype: tf.keras.Sequential
        :raises ValueError: If num_units is neither an integer nor a list of proper length
        """
        import tensorflow as tf

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
    """Base class for results of corrected estimator models.
    
    This class stores results from both the initial OLS model and the correction model.
    It provides methods to access and display these results.
    """

    def __init__(self, model, ols_results, correction_model):
        """Initialize the results object.
        
        :param model: The CorrectedEstimator model
        :type model: CorrectedEstimator
        :param ols_results: Results from the initial OLS model
        :type ols_results: statsmodels.regression.linear_model.RegressionResults
        :param correction_model: The fitted correction model
        :type correction_model: CorrectionModel
        """
        self.model = model
        self.correction_model = correction_model
        self.index = model.x_index
        self.ols_results = ols_results
        self.betahat = ols_results.params[self.index]

    def print_ols_results(self):
        """Print a summary of the OLS results.
        
        :return: None
        """
        print(self.ols_results.summary())

    def get_ols_results(self):
        """Get the OLS results object.
        
        :return: OLS results object
        :rtype: statsmodels.regression.linear_model.RegressionResults
        """
        return self.ols_results




class CorrectedEstimatorResultsLogLinear(CorrectedEstimatorResults):
    """Results for log-linear model with correction.
    
    This class extends CorrectedEstimatorResults with methods specific to
    log-linear models, focusing on semi-elasticity estimation.
    """

    def __init__(self, model, ols_results, correction_model):
        """Initialize the log-linear results object.
        
        :param model: The CorrectedEstimator model
        :type model: CorrectedEstimator
        :param ols_results: Results from the initial OLS model
        :type ols_results: statsmodels.regression.linear_model.RegressionResults
        :param correction_model: The fitted correction model
        :type correction_model: CorrectionModel
        """
        super().__init__(model, ols_results, correction_model)
        self.se = self.correction_model.semi_elasticity(model.X, self.index)

    def average_semi_elasticity(self):
        """Calculate the average semi-elasticity with correction.
        
        :return: Average corrected semi-elasticity
        :rtype: float
        """
        return self.betahat + np.mean(self.se)

    def plot_dist_semi_elasticity(self):
        """Plot the distribution of semi-elasticities.
        
        :return: Figure and axis objects for the plot
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = plt.subplots()
        sns.kdeplot(self.betahat + self.se, ax=ax)
        ax.vlines(self.betahat, ymin=0, ymax=1, color='red', linestyle='--', label='OLS Estimate', transform=ax.get_xaxis_transform())
        ax.vlines(self.average_semi_elasticity(), ymin=0, ymax=1, color='blue', linestyle='--', label='Corrected Estimate', transform=ax.get_xaxis_transform())
        ax.set_title('Distribution of Semi-Elasticity')
        ax.set_xlabel('Semi-Elasticity')
        ax.set_ylabel('Density')
        ax.legend()
        return fig, ax

    def semi_elasticity_at_average(self):
        """Calculate the semi-elasticity at the average values of regressors.
        
        :return: Semi-elasticity at average values
        :rtype: float
        """
        X_mean = np.mean(self.model.X, axis=0)
        return self.betahat + self.correction_model.semi_elasticity(X_mean.reshape(1, -1), self.index)[0]

    def plot_eu(self):
        """Plot the distribution of the correction term at average regressor values.

        :return: Figure and axis objects for the plot
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = plt.subplots()
        x_indexed = self.model.X[:, self.index].copy()
        xmin = np.quantile(x_indexed, 0.1)
        xmax = np.quantile(x_indexed, 0.9)
        x_to_use = np.linspace(xmin, xmax, 1000)
        others_to_use = self.model.X.mean(axis=0)
        X_plot = np.tile(others_to_use, (len(x_to_use), 1))
        X_plot[:, self.index] = x_to_use

        eu = self.correction_model.predict(X_plot)
        sns.scatterplot(x=x_to_use, y=eu, ax=ax)
        ax.set_title(r'$E[e^u|X]$')
        ax.set_xlabel('Regressor of Interest')
        return fig, ax

    def plot_eu_grad(self):
        """Plot the gradient of the correction term with respect to the regressor of interest.
        :return: Figure and axis objects for the plot
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = plt.subplots()
        x_indexed = self.model.X[:, self.index].copy()
        xmin = np.quantile(x_indexed, 0.1)
        xmax = np.quantile(x_indexed, 0.9)
        x_to_use = np.linspace(xmin, xmax, 1000)
        others_to_use = self.model.X.mean(axis=0)
        X_plot = np.tile(others_to_use, (len(x_to_use), 1))
        X_plot[:, self.index] = x_to_use

        eu_grad = self.correction_model.marginal_effects(X_plot, self.index)
        sns.scatterplot(x=x_to_use, y=eu_grad, ax=ax)
        ax.set_title(r'$\dfrac{dE[e^u|X]}{dx}$')
        ax.set_xlabel('Regressor of Interest')
        ax.set_ylabel('Gradient')
        return fig, ax

    def test_ppml(self):
        """Test the consistency of PPML estimation.
        
        :return: Results of the PPML consistency test
        :rtype: dict
        """
        ppml_mod = sm.GLM(self.model.y, self.model.X, family=sm.families.Poisson()).fit(cov_type='HC3')
        return AssumptionTest(ppml_mod).test_direct()


class CorrectedEstimatorResultsLogLog(CorrectedEstimatorResults):
    """Results for log-log model with correction.
    
    This class extends CorrectedEstimatorResults with methods specific to
    log-log models, focusing on elasticity estimation.
    """

    def __init__(self, model, ols_results, correction_model):
        """Initialize the log-log results object.
        
        :param model: The CorrectedEstimator model
        :type model: CorrectedEstimator
        :param ols_results: Results from the initial OLS model
        :type ols_results: statsmodels.regression.linear_model.RegressionResults
        :param correction_model: The fitted correction model
        :type correction_model: CorrectionModel
        """
        super().__init__(model, ols_results, correction_model)
        self.e = self.correction_model.elasticity(model.X, self.index)

    def average_elasticity(self):
        """Calculate the average elasticity with correction.
        
        :return: Average corrected elasticity
        :rtype: float
        """
        return self.betahat + np.mean(self.e)

    def plot_dist_elasticity(self):
        """Plot the distribution of elasticities.
        
        :return: Figure and axis objects for the plot
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = plt.subplots()
        sns.kdeplot(self.betahat + self.e, ax=ax)
        ax.vlines(self.betahat, ymin=0, ymax=1, color='red', linestyle='--', label='OLS Estimate', transform=ax.get_xaxis_transform())
        ax.vlines(self.average_elasticity(), ymin=0, ymax=1, color='blue', linestyle='--', label='Corrected Estimate', transform=ax.get_xaxis_transform())
        ax.set_title('Distribution of Elasticity')
        ax.set_xlabel('Elasticity')
        ax.set_ylabel('Density')
        ax.legend()
        return fig, ax


    def elasticity_at_average(self):
        """Calculate the elasticity at the average values of regressors.
        
        :return: Elasticity at average values
        :rtype: float
        """
        X_mean = np.mean(self.model.X, axis=0)
        return self.betahat + self.correction_model.elasticity(X_mean.reshape(1, -1), self.index)[0]

    def plot_eu(self):
        """Plot the distribution of the correction term at average regressor values.

        :return: Figure and axis objects for the plot
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = plt.subplots()
        x_indexed = self.model.X[:, self.index].copy()
        xmin = np.exp(np.quantile(x_indexed, 0.1))
        xmax = np.exp(np.quantile(x_indexed, 0.9))
        x_to_use = np.linspace(xmin, xmax, 1000)
        others_to_use = self.model.X.mean(axis=0)
        X_plot = np.tile(others_to_use, (len(x_to_use), 1))
        X_plot[:, self.index] = x_to_use

        eu = self.correction_model.predict(X_plot)
        sns.scatterplot(x=x_to_use, y=eu, ax=ax)
        ax.set_title(r'$E[e^u|X]$')
        ax.set_xlabel('Regressor of Interest')
        return fig, ax

    def plot_eu_grad(self):
        """Plot the gradient of the correction term with respect to the regressor of interest.
        :return: Figure and axis objects for the plot
        :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = plt.subplots()
        x_indexed = self.model.X[:, self.index].copy()
        xmin = np.exp(np.quantile(x_indexed, 0.1))
        xmax = np.exp(np.quantile(x_indexed, 0.9))
        x_to_use = np.linspace(xmin, xmax, 1000)
        others_to_use = self.model.X.mean(axis=0)
        X_plot = np.tile(others_to_use, (len(x_to_use), 1))
        X_plot[:, self.index] = x_to_use

        eu_grad = self.correction_model.marginal_effects(X_plot, self.index)
        sns.scatterplot(x=x_to_use, y=eu_grad, ax=ax)
        ax.set_title(r'$\dfrac{dE[e^u|X]}{dx}$')
        ax.set_xlabel('Regressor of Interest')
        ax.set_ylabel('Gradient')
        return fig, ax

    def test_ppml(self):
        """Test the consistency of PPML estimation.
        
        :return: Results of the PPML consistency test
        :rtype: dict
        """
        ppml_mod = sm.GLM(self.model.y, self.model.X, family=sm.families.Poisson()).fit(cov_type='HC3')
        return AssumptionTest(ppml_mod).test_direct()


class CorrectionModel:
    """Base class for correction models.
    
    This abstract class defines the interface for all correction models used
    to correct the bias in log-transformed OLS models.
    """

    def predict(self, X):
        """Predict the correction term for given data.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :return: Predicted correction term
        :rtype: numpy.ndarray
        """
        pass

    def marginal_effects(self, X, index):
        """Calculate the marginal effects of a regressor.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Marginal effects
        :rtype: numpy.ndarray
        """
        pass

    def semi_elasticity(self, X, index):
        """Calculate the semi-elasticity of a regressor.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Semi-elasticities
        :rtype: numpy.ndarray
        """
        pass

    def elasticity(self, X, index):
        """Calculate the elasticity of a regressor.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Elasticities
        :rtype: numpy.ndarray
        """
        pass


class OLSCorrectionModel(CorrectionModel):
    """OLS-based correction model with polynomial features.
    
    This class implements a correction model using OLS regression
    with polynomial features for the regressors.
    """

    def __init__(self, model, poly):
        """Initialize the OLS correction model.
        
        :param model: Fitted OLS model for the correction term
        :type model: statsmodels.regression.linear_model.RegressionResults
        :param poly: Polynomial features transformer
        :type poly: sklearn.preprocessing.PolynomialFeatures
        """
        self.model = model
        self.poly = poly

    def predict(self, X):
        """Predict the correction term for given data.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :return: Predicted correction term
        :rtype: numpy.ndarray
        """
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def marginal_effects(self, X, index):
        """Calculate the marginal effects of a regressor using numerical differentiation.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Marginal effects
        :rtype: numpy.ndarray
        """
        delta = np.zeros(X.shape)
        delta[:, index] = 1e-6
        eu_new = self.predict(X + delta)
        eu_old = self.predict(X)
        return (eu_new - eu_old) / 1e-6

    def semi_elasticity(self, X, index):
        """Calculate the semi-elasticity of a regressor.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Semi-elasticities
        :rtype: numpy.ndarray
        """
        delta = np.zeros(X.shape)
        delta[:, index] = 1e-6
        eu_new = self.predict(X + delta)
        eu_old = self.predict(X)
        return ((eu_new - eu_old) / 1e-6) / eu_old

    def elasticity(self, X, index):
        """Calculate the elasticity of a regressor.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Elasticities
        :rtype: numpy.ndarray
        """
        appropriate_x = X.copy()
        appropriate_x[:, index] = np.exp(appropriate_x[:, index])
        return self.semi_elasticity(appropriate_x, index) * appropriate_x[:, index].reshape(-1,)


class NNCorrectionModel(CorrectionModel):
    """Neural network-based correction model.
    
    This class implements a correction model using a neural network
    and automatic differentiation for marginal effects.
    """
    import tensorflow as tf

    def __init__(self, model, mean, std):
        """Initialize the neural network correction model.
        
        :param model: Fitted neural network model for the correction term
        :type model: tf.keras.Model
        """
        self.model = model
        self.mean = mean
        self.std = std

    def predict(self, X):
        """Predict the correction term for given data.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray or tf.Tensor
        :return: Predicted correction term
        :rtype: tf.Tensor
        """
        return self.tf.reshape(self.model(X), (-1,))

    def marginal_effects(self, X, index):
        """Calculate the marginal effects of a regressor using automatic differentiation.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Marginal effects
        :rtype: numpy.ndarray
        """

        X_used = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        X_used = self.tf.Variable(X_used)
        with self.tf.GradientTape() as tape:
            tape.watch(X_used)
            euhat = self.predict(X_used)
        grads = tape.gradient(euhat, X_used)
        return (grads[:, index]).numpy()

    def semi_elasticity(self, X, index):
        """Calculate the semi-elasticity of a regressor using automatic differentiation.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Semi-elasticities
        :rtype: numpy.ndarray
        """

        X_used = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        X_used = self.tf.Variable(X_used)
        with self.tf.GradientTape() as tape:
            tape.watch(X_used)
            euhat = self.predict(X_used) * self.std + self.mean
        grads = tape.gradient(euhat, X_used)
        return grads[:, index].numpy() / euhat.numpy().reshape(-1,)

    def elasticity(self, X, index):
        """Calculate the elasticity of a regressor using automatic differentiation.
        
        :param X: Regressor matrix
        :type X: numpy.ndarray
        :param index: Index of the regressor of interest
        :type index: int
        :return: Elasticities
        :rtype: numpy.ndarray
        """

        X_used = X.copy()
        X_used[:, index] = np.exp(X_used[:, index])
        X_used = self.tf.convert_to_tensor(X_used, dtype=self.tf.float32)
        X_used = self.tf.Variable(X_used)
        with self.tf.GradientTape() as tape:
            tape.watch(X_used)
            euhat = self.predict(X_used)
        grads = tape.gradient(euhat, X_used)
        return grads[:, index].numpy() / euhat.numpy().reshape(-1,) * X_used.numpy()[:, index].reshape(-1,)
