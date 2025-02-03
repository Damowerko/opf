import logging
from abc import abstractmethod
from typing import Dict, Iterable, Type

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torchcps.utils import add_model_specific_args

logger = logging.getLogger(__name__)


class ModelSpecificArgsMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, "add_model_specific_args"):
            cls.add_model_specific_args = classmethod(add_model_specific_args)


class OPFModel(nn.Module, metaclass=ModelSpecificArgsMeta):
    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        return self._forward(data)

    @abstractmethod
    def _forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        raise NotImplementedError()


class ModelRegistry:
    registry: Dict[str, tuple[Type[OPFModel], bool]] = {}

    @classmethod
    def register(cls, name: str, is_dual: bool):
        """
        Register a model in the registry.

        Args:
            name: The name of the model.
            is_dual: Whether the model uses a regular (branches are edges) or dual graph (branches are nodes).
        """

        def decorator(model_class: Type[OPFModel]):
            if name in cls.registry:
                logger.warning(
                    f"OPFModel {name} already registered. It will be replaced."
                )
            cls.registry[name] = (model_class, is_dual)
            return model_class

        return decorator

    @classmethod
    def items(cls) -> Iterable[tuple[str, Type[OPFModel]]]:
        return ((name, cls) for name, (cls, _) in cls.registry.items())

    @classmethod
    def get_class(cls, name: str) -> Type[OPFModel]:
        if name not in cls.registry:
            raise ValueError(f"Model {name} not found in registry.")
        return cls.registry[name][0]

    @classmethod
    def is_dual(cls, name: str) -> bool:
        return cls.registry[name][1]
