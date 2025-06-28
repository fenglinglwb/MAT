"""Shared pytest fixtures and configuration."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import numpy as np
import torch
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "test_model",
            "input_size": 256,
            "output_size": 256,
            "channels": 3,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 10,
        },
        "dataset": {
            "root": "/tmp/test_data",
            "train_split": 0.8,
            "val_split": 0.2,
        },
    }


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a sample binary mask for testing."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255  # Square hole in the center
    return mask


@pytest.fixture
def sample_tensor_image() -> torch.Tensor:
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_tensor_mask() -> torch.Tensor:
    """Create a sample mask tensor for testing."""
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, 64:192, 64:192] = 1.0
    return mask


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x, mask=None):
            return self.conv(x)
    
    return MockModel()


@pytest.fixture
def mock_dataset_path(temp_dir: Path) -> Path:
    """Create a mock dataset directory structure."""
    dataset_path = temp_dir / "dataset"
    images_path = dataset_path / "images"
    masks_path = dataset_path / "masks"
    
    images_path.mkdir(parents=True)
    masks_path.mkdir(parents=True)
    
    # Create sample images and masks
    for i in range(5):
        img = Image.new('RGB', (256, 256), color=(i*50, i*50, i*50))
        img.save(images_path / f"image_{i}.png")
        
        mask = Image.new('L', (256, 256), color=0)
        mask.save(masks_path / f"mask_{i}.png")
    
    return dataset_path


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output


# Markers for conditional test execution
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU tests."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)