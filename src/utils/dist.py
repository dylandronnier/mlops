from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass, make_dataclass
from typing import Any, TypeVar

from optuna.trial import Trial

X = TypeVar(name="X")


class Distribution[X](ABC):
    @abstractmethod
    def suggest(self, trial: Trial) -> X:
        pass


class RangeInt(Distribution[int]):
    def __init__(self, name: str, low: int, high: int) -> None:
        """RangeInt.

        Args:
        ----
            name: Name of the hyperparameter.
            low: Lower bound.
            high: Upper bound.

        """
        assert low < high
        self._name = name
        self._low = low
        self._high = high

    def suggest(self, trial: Trial) -> int:
        return trial.suggest_int(name=self._name, low=self._low, high=self._high)


class RangeFloat(Distribution[float]):
    def __init__(self, name: str, low: float, high: float) -> None:
        """RangeFloat.

        Args:
        ----
            name: Name of the hyperparameter.
            low: Lower bound.
            high: Upper bound.

        """
        assert low < high
        self._name = name
        self._low = low
        self._high = high

    def suggest(self, trial: Trial) -> float:
        return trial.suggest_float(name=self._name, low=self._low, high=self._high)


def _sugg(d, trial: Trial) -> dict[str, Any]:
    assert is_dataclass(d)
    args = dict()
    for k, v in asdict(d).items():
        if isinstance(v, Distribution):
            args[k] = v.suggest(trial)
        else:
            args[k] = v
    print(args)
    return args


def make_config_suggest(cls: type[X]) -> type[Distribution[X]]:
    assert is_dataclass(cls)
    fields_dict = dict()
    for f in fields(cls):
        fields_dict[f.name] = f.type | Distribution[f.type]
    return make_dataclass(
        cls_name="Distribution" + cls.__name__,
        bases=(Distribution,),
        fields=fields_dict,
        namespace={"suggest": lambda self, trial: cls(**_sugg(self, trial))},
    )
