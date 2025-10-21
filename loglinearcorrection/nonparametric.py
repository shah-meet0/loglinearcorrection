import numpy as np
import numpy.typing as npt


class NPModel:
    def __init__(self, variable_types: dict, params: dict):
        pass

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "NPModelResults":
        pass


class NPModelResults:

    def __init__(self, model: NPModel, x: npt.ArrayLike, y: npt.ArrayLike):
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float_]:
        pass

    def derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass

    def second_derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass



class NNModel(NPModel):
    def __init__(self, variable_types: dict, params: dict):
        super().__init__(variable_types, params)
        pass

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "NNModelResults":
        pass

    def _create_model(self, params):
        pass

    def _parse_params(self, params):
        pass



class NNModelResults(NPModelResults):
    def __init__(self, model: NNModel, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(model, x, y)
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float_]:
        pass

    def derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass

    def second_derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass


class NNFactory:

    def __init__(self, nn_config: dict):
        self.nn_config = nn_config

    def create_model(self):
        return self.create_model_torch()

    def create_model_tf(self):
        raise NotImplementedError("TensorFlow implementation not available.")
        pass

    def create_model_torch(self):
        import torch
        pass