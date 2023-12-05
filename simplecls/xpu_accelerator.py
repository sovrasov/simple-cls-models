# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Union

import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators import AcceleratorRegistry


XPU_AVAILABLE = None
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    XPU_AVAILABLE = False
    ipex = None


def is_xpu_available() -> bool:
    """Checks if XPU device is available."""
    global XPU_AVAILABLE  # noqa: PLW0603
    if XPU_AVAILABLE is None:
        XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    return XPU_AVAILABLE


class XPUAccelerator(Accelerator):
    """Support for a hypothetical XPU, optimized for large-scale machine learning."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        #device = torch.device("xpu", 0)
        if device.type != "xpu":
            raise RuntimeError(f"Device should be xpu, got {device} instead")

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    def teardown(self) -> None:
        pass

    @staticmethod
    def is_available() -> bool:
        return is_xpu_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return {}

    @classmethod
    def register_accelerators(cls, accelerator_registry):
        accelerator_registry.register(
            "xpu",
            cls,
            description=f"XPU accelerator supports Intel ARC and Max GPUs.",
        )

AcceleratorRegistry.register("xpu", XPUAccelerator, description="XPU accelerator supports Intel ARC and Max GPUs")
