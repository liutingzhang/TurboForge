from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..domain import ModelInspection


class InputAdapter(ABC):
    name: str

    @abstractmethod
    def matches(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def inspect(self, path: Path) -> ModelInspection:
        raise NotImplementedError
