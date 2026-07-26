import torch

from src.model import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN(
        num_classes=10,
        dropout=0.3,
    )

    images = torch.randn(
        8,
        3,
        32,
        32,
    )

    logits = model(images)

    assert logits.shape == (8, 10)


def test_model_supports_different_batch_sizes():
    model = SimpleCNN(
        num_classes=10,
        dropout=0.3,
    )

    images = torch.randn(
        3,
        3,
        32,
        32,
    )

    logits = model(images)

    assert logits.shape == (3, 10)


def test_model_backward_pass():
    model = SimpleCNN(
        num_classes=10,
        dropout=0.3,
    )

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    criterion = torch.nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()

    has_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert has_gradient