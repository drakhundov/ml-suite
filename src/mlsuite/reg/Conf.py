from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class GDConfig:
    """Hyperparameters for Linear Regression with Iterative Optimization (Gradient Descent)"""

    lr: float
    diminishing_lr: bool = False  # lr=1/(k+1)
    l2_coef: float = 0
    use_bias: bool = False
    niters: int = 1000
    stochastic: bool = False
    epsilon: float | None = None
    degree: int = 1
    normalize_data: bool = True

    def __post_init__(self):
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if self.niters <= 0:
            raise ValueError("niters should be a positive integer")
        if self.l2_coef < 0.0:
            raise ValueError("l2_coef must be zero or positive")
        if self.degree < 1:
            raise ValueError("degree must be greater than or equal to 1")


@dataclass(frozen=True)
class LSConfig:
    """Hyperparameters for Linear Regression with Direct Optimization (Least Squares)"""

    l2_coef: float = 0
    use_bias: bool = False
    degree: int = 1
    normalize_data: bool = True

    def __post_init__(self):
        if self.l2_coef < 0.0:
            raise ValueError("l2_coef must be zero or positive")


class OptimizationMethod(Enum):
    GRAD = 1  # Gradient Descent
    LSQR = 2  # Least Squares
