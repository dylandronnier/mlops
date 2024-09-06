from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass, field, fields, make_dataclass

import omegaconf
from flax import nnx
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig(ABC):
    @abstractmethod
    def to_model(
        self, *, channels: int, num_classes: int, rngs: nnx.Rngs
    ) -> nnx.Module:
        raise NotImplementedError


def store_model_config(cs: ConfigStore, module: str):
    name = module.split(sep=".")[-1]
    mod = __import__(module, fromlist=[None])

    new_fields = list()
    for f in fields(mod.Architecture):
        if f.default == MISSING:
            new_fields.append((f.name, f.type, field(default=omegaconf.MISSING)))
        else:
            new_fields.append((f.name, f.type, f.default))
    cls = make_dataclass(
        cls_name="Config" + name,
        fields=new_fields,
        bases=(ModelConfig,),
        namespace={
            "to_model": lambda self, *, channels, num_classes, rngs: mod.NeuralNetwork(
                mod.Architecture(**asdict(self)),
                channels=channels,
                num_classes=num_classes,
                rngs=rngs,
            ),
        },
    )
    cs.store(group="model", name="base_" + name, node=cls)
