from post_training_pipeline.models.loader import get_dtype


def test_get_dtype() -> None:
    import torch

    assert get_dtype("bfloat16") == torch.bfloat16
    assert get_dtype("float16") == torch.float16
    assert get_dtype("float32") == torch.float32
    assert get_dtype("unknown") == torch.bfloat16  # default
