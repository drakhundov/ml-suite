from dataclasses import dataclass

from mlsuite.reg import GDConfig

@dataclass(frozen=True)
class LogisticClfConfig(GDConfig):
    num_classes: int = 2
    use_binary_clf: bool = False
