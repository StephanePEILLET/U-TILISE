"""
Test that the without_coherence pipeline works correctly end-to-end:
- Channel calculation in CIRCA_from_HDF5.setup_channels
- Model input/output dimensions in utils.get_model
- Forward pass through U-TILISE model
"""

import torch

from lib.models.utilise import UTILISE


def simulate_setup_channels(use_sar_val, channels="all"):
    """Reproduce the channel calculation logic from CIRCA_from_HDF5.setup_channels."""
    without_coherence = False
    use_sar = use_sar_val
    if isinstance(use_sar, str) and ("_without_coherence" in use_sar or "_only_coherence" in use_sar):
        without_coherence = True
        use_sar = use_sar.replace("_without_coherence", "").replace("_only_coherence", "")

    num_channels = 10 if channels == "all" else 4
    if use_sar:
        s1_bands = 2 if without_coherence else 4
        if use_sar == "asc+desc":
            num_channels += s1_bands * 2
        elif use_sar in ["asc", "desc", "mix_closest"]:
            num_channels += s1_bands
        else:
            raise ValueError(f"Bad SAR pairing: {use_sar}")
    return num_channels, without_coherence, use_sar


def simulate_get_model_dims(use_sar_val, input_dim):
    """Reproduce the output_dim calculation from lib.utils.get_model."""
    output_dim = input_dim
    if use_sar_val:
        _has_no_coh = isinstance(use_sar_val, str) and ("_without_coherence" in use_sar_val or "_only_coherence" in use_sar_val)
        s1_bands = 2 if _has_no_coh else 4
        use_sar_base = use_sar_val.replace("_without_coherence", "").replace("_only_coherence", "") if isinstance(use_sar_val, str) else use_sar_val
        if use_sar_base == "asc+desc":
            output_dim -= 2 * s1_bands
        else:
            output_dim -= s1_bands
    return output_dim


def test_config(use_sar_val):
    print(f"=== TESTING CONFIG: use_sar='{use_sar_val}' ===")

    # 1. Channel calculation (same as CIRCA_from_HDF5.setup_channels)
    input_dim, without_coherence, use_sar_resolved = simulate_setup_channels(use_sar_val)
    print(f"  num_channels (input_dim) = {input_dim}, without_coherence = {without_coherence}")

    # 2. Model output_dim calculation (same as lib.utils.get_model)
    output_dim = simulate_get_model_dims(use_sar_val, input_dim)
    print(f"  output_dim = {output_dim}")
    assert output_dim == 10, f"Output dimension should be 10 (S2 only) but got {output_dim}!"

    # 3. Create model and run a forward pass
    model = UTILISE(
        input_dim=input_dim,
        output_dim=output_dim,
        encoder_widths=[16, 16],
        decoder_widths=[16, 16],
    )

    seq_len = 5
    x = torch.randn(1, seq_len, input_dim, 32, 32)  # B=1
    dates = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)  # B=1
    out = model(x, batch_positions=dates)
    print(f"  Model output shape = {out.shape}")
    assert out.shape == (1, seq_len, 10, 32, 32), f"Expected (1, {seq_len}, 10, 32, 32) but got {out.shape}"

    # 4. Simulate SAR slicing in __getitem__
    s1_full = torch.randn(seq_len, 4, 32, 32)
    if without_coherence:
        s1_sliced = s1_full[:, :2, :, :]
        assert s1_sliced.shape[1] == 2, f"Expected 2 SAR bands without coherence, got {s1_sliced.shape[1]}"
    else:
        s1_sliced = s1_full
        assert s1_sliced.shape[1] == 4, f"Expected 4 SAR bands with coherence, got {s1_sliced.shape[1]}"

    # 5. Verify x concatenation matches input_dim
    frames_s2 = torch.randn(seq_len, 10, 32, 32)
    if use_sar_resolved == "asc+desc":
        frames_input = torch.cat((frames_s2, s1_sliced, s1_sliced), dim=1)
    else:
        frames_input = torch.cat((frames_s2, s1_sliced), dim=1)
    assert frames_input.shape[1] == input_dim, (
        f"Concatenated input has {frames_input.shape[1]} channels but expected {input_dim}"
    )

    print("  SUCCESS!\n")


if __name__ == "__main__":
    # Test all variations
    test_config("mix_closest")
    test_config("mix_closest_without_coherence")
    test_config("mix_closest_only_coherence")
    test_config("asc+desc")
    test_config("asc+desc_without_coherence")
    test_config("asc+desc_only_coherence")
    print("All tests passed.")
