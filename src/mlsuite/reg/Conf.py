from dataclasses import dataclass


@dataclass(frozen=True)
class GDConfig:
    """Hyperparameters for Linear Regression with Iterative Optimization (Gradient Descent)"""

    lr: float
    diminishing_lr: bool = False  # lr=1/(k+1)
    l2_coef: float = 0
    niters: int = 1000
    stochastic: bool = False
    epsilon: float | None = None

    def __post_init__(self):
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if self.niters <= 0:
            raise ValueError("niters should be a positive integer")
        if self.l2_coef < 0.0:
            raise ValueError("l2_coef must be zero or positive")


@dataclass(frozen=True)
class LSConfig:
    """Hyperparameters for Linear Regression with Direct Optimization (Least Squares)"""

    l2_coef: float = 0
    bias: bool

    def __post_init__(self):
        if self.l2_coef < 0.0:
            raise ValueError("l2_coef must be zero or positive")
