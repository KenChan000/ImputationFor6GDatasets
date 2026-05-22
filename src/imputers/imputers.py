from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


class Imputer(ABC):
    """Common interface so all methods plug into the same evaluation loop."""

    name: str

    @abstractmethod
    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        """Impute NaN cells in X and return a fully observed array."""


class MeanImputer(Imputer):
    name = "Mean"

    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        return SimpleImputer(strategy="mean").fit_transform(X)


class KNNImputerWrapper(Imputer):
    name = "kNN"

    def __init__(self, k: int = 5):
        self.k = k

    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        return KNNImputer(n_neighbors=self.k).fit_transform(X)


class MICEImputer(Imputer):
    name = "MICE"

    def __init__(self, max_iter: int = 10):
        self.max_iter = max_iter

    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            imp = IterativeImputer(
                max_iter=self.max_iter,
                random_state=seed,
                estimator=None,
            )
            return imp.fit_transform(X)


class SoftImputeWrapper(Imputer):
    """SoftImpute via fancyimpute, when compatible with installed sklearn."""

    name = "SoftImpute"

    def __init__(self, max_iters: int = 100):
        self.max_iters = max_iters

    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        from fancyimpute import SoftImpute as FancySoftImpute

        # fancyimpute 0.7 expects sklearn's older `force_all_finite` keyword.
        # Newer sklearn versions renamed that parameter to `ensure_all_finite`.
        # Patch the imported helper locally so we can keep using fancyimpute.
        import fancyimpute.solver as fancy_solver
        import fancyimpute.soft_impute as fancy_soft_impute
        from sklearn.utils.validation import check_array as sklearn_check_array

        def _compat_check_array(*args, force_all_finite=True, **kwargs):
            if "ensure_all_finite" not in kwargs:
                kwargs["ensure_all_finite"] = force_all_finite
            return sklearn_check_array(*args, **kwargs)

        fancy_solver.check_array = _compat_check_array
        fancy_soft_impute.check_array = _compat_check_array

        model = FancySoftImpute(max_iters=self.max_iters, verbose=False)
        return np.asarray(model.fit_transform(X.copy()), dtype=float)
    
class HyperImputeImputer(Imputer):
    name = "HyperImpute"

    def __init__(self):
        self._loaded_name = None

    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        from hyperimpute.plugins.imputers import Imputers
        plugins = Imputers()
        model = plugins.get("hyperimpute")  
        self._loaded_name = "hyperimpute"

        X_imp = model.fit_transform(X.copy())
        if hasattr(X_imp, "to_numpy"):
            X_imp = X_imp.to_numpy()
        return np.asarray(X_imp, dtype=float)

def get_default_imputers(
) -> list[Imputer]:
    imputers: list[Imputer] = [
        MeanImputer(),
        KNNImputerWrapper(k=5),
        MICEImputer(max_iter=10),
        SoftImputeWrapper(),
        HyperImputeImputer(),
    ]
    return imputers
