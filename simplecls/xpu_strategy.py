# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _DEVICE
from .xpu_accelerator import is_xpu_available
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class SingleXPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single XPU device."""

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: _DEVICE = "xpu:0",
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):

        if not is_xpu_available():
            raise MisconfigurationException("`SingleXPUStrategy` requires XPU devices to run")

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)
        model, optimizer = torch.xpu.optimize(trainer.model, optimizer=trainer.optimizers[0])
        trainer.optimizers = [optimizer]
        trainer.model = model

    def model_to_device(self) -> None:
        self.model.to(self.root_device)  # type: ignore

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )


StrategyRegistry.register(SingleXPUStrategy.strategy_name, SingleXPUStrategy, description="Strategy that enables training on single XPU")
