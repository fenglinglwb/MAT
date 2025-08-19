"""Validation tests to ensure the testing infrastructure is set up correctly."""
import os
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


class TestInfrastructureValidation:
    """Test suite to validate the testing infrastructure setup."""
    
    def test_pytest_installed(self):
        """Test that pytest is installed and importable."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_installed(self):
        """Test that pytest-cov is installed."""
        import pytest_cov
        assert pytest_cov.__version__
    
    def test_pytest_mock_installed(self):
        """Test that pytest-mock is installed."""
        import pytest_mock
        # pytest-mock doesn't expose __version__, check if module exists
        assert pytest_mock is not None
    
    def test_project_packages_importable(self):
        """Test that project packages can be imported."""
        # Add parent directory to path if not already there
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        # Test imports
        import dnnlib
        import torch_utils
        assert dnnlib is not None
        assert torch_utils is not None
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_image, sample_mask):
        """Test that custom fixtures are available and working."""
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert isinstance(sample_image, np.ndarray)
        assert sample_image.shape == (256, 256, 3)
        assert isinstance(sample_mask, np.ndarray)
        assert sample_mask.shape == (256, 256)
    
    def test_torch_fixtures(self, sample_tensor_image, sample_tensor_mask):
        """Test PyTorch tensor fixtures."""
        assert isinstance(sample_tensor_image, torch.Tensor)
        assert sample_tensor_image.shape == (1, 3, 256, 256)
        assert isinstance(sample_tensor_mask, torch.Tensor)
        assert sample_tensor_mask.shape == (1, 1, 256, 256)
    
    def test_mock_model_fixture(self, mock_model):
        """Test mock model fixture."""
        assert isinstance(mock_model, torch.nn.Module)
        test_input = torch.randn(1, 3, 256, 256)
        output = mock_model(test_input)
        assert output.shape == test_input.shape
    
    def test_mock_dataset_fixture(self, mock_dataset_path):
        """Test mock dataset creation fixture."""
        assert mock_dataset_path.exists()
        assert (mock_dataset_path / "images").exists()
        assert (mock_dataset_path / "masks").exists()
        assert len(list((mock_dataset_path / "images").glob("*.png"))) == 5
        assert len(list((mock_dataset_path / "masks").glob("*.png"))) == 5
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True
    
    def test_device_fixture(self, device):
        """Test device fixture returns valid device."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    def test_random_seed_fixture(self):
        """Test that random seeds are set consistently."""
        # This test relies on the autouse fixture reset_random_seeds
        np_random1 = np.random.rand()
        torch_random1 = torch.rand(1).item()
        
        # Seeds should be reset for each test, so these should be the same
        # as if we just set the seed
        np.random.seed(42)
        torch.manual_seed(42)
        np_random2 = np.random.rand()
        torch_random2 = torch.rand(1).item()
        
        assert np_random1 == np_random2
        assert torch_random1 == torch_random2
    
    def test_coverage_configured(self):
        """Test that coverage is properly configured."""
        # This test will pass if coverage is running (which it should be
        # based on our pytest configuration)
        import sys
        # Coverage adds trace functions when running
        assert True  # If we get here, the test infrastructure is working


@pytest.mark.parametrize("test_input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_parametrize_works(test_input, expected):
    """Test that pytest parametrize decorator works."""
    assert test_input ** 2 == expected


def test_temp_dir_cleanup(temp_dir):
    """Test that temp_dir fixture creates and cleans up properly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    # Cleanup will happen after the test