import torch
import unittest
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union
import itertools
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from .gating import GATING_TO_MODEL_MAPPING

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb
from moelora import *

class TestMoELoRA(unittest.TestCase):
    def setUp(self):
        self.model = MoELoRA()  # Instantiate your MoELoRA model here

    def test_forward_no_adapters(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

    def test_forward_with_adapters(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        self.model.active_adapter = 'adapter1'  # Set the active adapter
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

    def test_forward_with_global_user_embeds(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        self.model.active_adapter = 'adapter1'  # Set the active adapter
        self.model.global_user_embeds = [torch.randn(10, 30)]  # Set the global_user_embeds
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

if __name__ == '__main__':
    unittest.main()import torch
import unittest

class TestMoELoRA(unittest.TestCase):
    def setUp(self):
        self.model = MoELoRA()  # Instantiate your MoELoRA model here

    def test_forward_no_adapters(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

    def test_forward_with_adapters(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        self.model.active_adapter = 'adapter1'  # Set the active adapter
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

    def test_forward_with_global_user_embeds(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        self.model.active_adapter = 'adapter1'  # Set the active adapter
        self.model.global_user_embeds = [torch.randn(10, 30)]  # Set the global_user_embeds
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

    def test_forward_with_global_user_embeds_exception(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        self.model.active_adapter = 'adapter1'  # Set the active adapter
        self.model.global_user_embeds = [torch.randn(5, 30)]  # Set the global_user_embeds with incompatible shape
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

    def test_forward_no_global_user_embeds(self):
        x = torch.randn(10, 20, 30)  # Create a random input tensor
        self.model.active_adapter = 'adapter1'  # Set the active adapter
        self.model.global_user_embeds = []  # Set an empty global_user_embeds
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10, 20, 30))  # Assert the output shape is correct

if __name__ == '__main__':
    unittest.main()