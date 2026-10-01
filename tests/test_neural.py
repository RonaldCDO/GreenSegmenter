import numpy as np
import pytest

from greensegmenter.neural import predict_neural


def test_neural_requires_an_explicit_model() -> None:
    with pytest.raises(ValueError, match="Carregue um modelo"):
        predict_neural(np.zeros((8, 8, 3), dtype=np.uint8), None, "CPU", 0.5)


@pytest.mark.parametrize("channels", [1, 2])
def test_torchscript_binary_logits_are_restored_and_thresholded(tmp_path, channels):
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def forward(self, image):
            foreground = image[:, 1:2, ::2, ::2] * 10 - 5
            if channels == 2:
                return torch.cat([-foreground, foreground], dim=1)
            return foreground

    path = tmp_path / "binary.pt"
    torch.jit.trace(TinyModel(), torch.zeros(1, 3, 8, 8)).save(str(path))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:, 1] = 255
    result = predict_neural(image, str(path), "CPU", 0.5)

    assert result.probability.shape == (8, 8)
    assert not result.mask[:, :3].any()
    assert result.mask[:, 5:].all()
    assert np.isfinite(result.probability).all()
