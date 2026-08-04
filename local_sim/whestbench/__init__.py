"""
LOCAL SHIM for the `whestbench` package — cloud/offline development only.

Mirrors the participant-facing API surface documented in the starter kit:
MLP (width, depth, weights, seed), BaseEstimator, SetupContext.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MLP:
    width: int
    depth: int
    weights: List[Any]  # list of (width, width) arrays; forward: y = ReLU(W.T @ x)
    seed: int = 0

    @property
    def n_layers(self):
        return len(self.weights)


@dataclass
class SetupContext:
    width: int = 256
    depth: int = 32
    flop_budget: int = 272_000_000_000
    api_version: str = "1.0"
    scratch_dir: Optional[str] = None
    submission_dir: Optional[str] = None
    seed: int = 0


class BaseEstimator:
    def setup(self, context: SetupContext) -> None:  # optional lifecycle hook
        pass

    def predict(self, mlp: MLP, budget: int):
        raise NotImplementedError

    def teardown(self) -> None:  # optional lifecycle hook
        pass
