import pandas as pd
import pytest
import torch

from topic_transition.loss import BCEWithWeights


@pytest.fixture
def dummy_data():
    """Fixture for dummy data frame used in tests."""
    data = pd.DataFrame(
        {
            "label": [
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
            ]
        }
    )
    return data


@pytest.fixture
def bce_with_weights_instance(dummy_data):
    """Fixture for an instance of BCEWithWeights."""
    dtype = torch.float32
    device = torch.device("cpu")
    return BCEWithWeights(data=dummy_data, dtype=dtype, device=device)


def test_bce_with_weights_initialization(dummy_data):
    """Test that BCEWithWeights is initialized correctly."""
    dtype = torch.float32
    device = torch.device("cpu")
    instance = BCEWithWeights(data=dummy_data, dtype=dtype, device=device)
    assert instance.num_splits == 3
    assert instance.pos_ratios.shape == (3,)
    assert torch.is_tensor(instance.pos_ratios)


def test_bce_with_weights_call_no_kwargs(bce_with_weights_instance):
    """Test __call__ method without any additional kwargs."""
    model_outputs = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.6], [0.8, 0.5, 0.2]])
    targets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    loss = bce_with_weights_instance(model_outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_bce_with_weights_call_with_split_idx(bce_with_weights_instance):
    """Test __call__ method with split_idx kwarg."""
    model_outputs = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.6], [0.8, 0.5, 0.2]])
    targets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    loss = bce_with_weights_instance(model_outputs, targets, split_idx=1)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_bce_with_weights_call_with_loss_mask(bce_with_weights_instance):
    """Test __call__ method with loss_mask kwarg."""
    model_outputs = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.6], [0.8, 0.5, 0.2]])
    targets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    loss_mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    loss = bce_with_weights_instance(model_outputs, targets, loss_mask=loss_mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_bce_with_weights_call_with_aggregate_false(bce_with_weights_instance):
    """Test __call__ method with aggregate kwarg set to False."""
    model_outputs = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.6], [0.8, 0.5, 0.2]])
    targets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    loss = bce_with_weights_instance(model_outputs, targets, aggregate=False)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 2
    assert loss.shape == model_outputs.shape
