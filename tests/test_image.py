from vnet import inv_scale_and_pad


def test_inv_scale_and_pad() -> None:
    original = (300, 400)
    padded = (512, 512)
    scale, pad = inv_scale_and_pad(original, padded)
    assert scale == 400 / 512
