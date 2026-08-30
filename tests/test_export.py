import json
import os
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Replace this with your actual import
# from adjaxt.export import export_arbitrary_model_to_hf, dataclass_to_dict
from adjaxt.sharding import ModelWeightMap  #
from adjaxt.export import dataclass_to_dict, export_arbitrary_model_to_hf

# --- Dummy Data Structures for Testing ---

@dataclass
class DummyNestedConfig:
    dim: int
    eps: float

@dataclass
class DummyModelConfig:
    vocab_size: int
    nested: DummyNestedConfig

@pytest.fixture
def dummy_config():
    return DummyModelConfig(vocab_size=1000, nested=DummyNestedConfig(dim=128, eps=1e-5))

@pytest.fixture
def dummy_weights():
    return {"fake_layer": {"weight": [1, 2, 3]}}

# --- Test Suite ---

def test_dataclass_to_dict(dummy_config):
    """Ensures nested dataclasses serialize correctly without retaining Python objects."""
    # Assuming dataclass_to_dict is in the same file
    result = dataclass_to_dict(dummy_config)
    assert isinstance(result, dict)
    assert result["vocab_size"] == 1000
    assert result["nested"]["dim"] == 128
    assert result["nested"]["eps"] == 1e-05


@patch("adjaxt.export.os.path.exists") # Mock exists to pretend all files are there
@patch("adjaxt.export.adjaxt")         # Mock the imported adjaxt module
@patch("shutil.copy")
@patch("adjaxt.export.save_checkpoint")
def test_export_copies_framework_files_reliably(
    mock_save, mock_copy, mock_adjaxt, mock_exists, tmp_path, dummy_config, dummy_weights
):
    # 1. Setup our fake absolute path for adjaxt
    mock_adjaxt.__file__ = "/fake/absolute/path/adjaxt/__init__.py"
    mock_exists.return_value = True
    
    mock_weight_map = MagicMock()
    
    export_arbitrary_model_to_hf(
        model_type="test_moe",
        jax_fn_name="test_moe_model",
        config_cls_name="TestMoEModelConfig",
        cfg=dummy_config,
        weights=dummy_weights,
        weight_map=mock_weight_map,
        dim_sizes={"i": 2},
        save_directory=str(tmp_path)
    )

    # 2. Verify it used the absolute path correctly
    # Let os.path.dirname resolve the base just like the actual code does
    expected_source_dir = os.path.dirname("/fake/absolute/path/adjaxt/__init__.py")
    
    assert mock_copy.call_count == 6
    
    # Check the first call's arguments to ensure the source path was built correctly
    first_call_args = mock_copy.call_args_list[0][0]
    
    # Build the expected path using os.path.join so it matches the OS separator
    expected_path = os.path.join(expected_source_dir, "layers.py")
    assert first_call_args[0] == expected_path