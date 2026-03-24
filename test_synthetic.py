
import torch

from dataloader_CIRCA.datasets.UTILISE_adapter import CIRCA_ADAPTED2UTILISE_Dataset
from lib.models.utilise import UTILISE


class MockDataset:
    def __init__(self, phase, hdf5_file, shuffle, use_sar, channels, image_size, load_transforms):
        self.use_sar = use_sar
        self.channels = channels
        self.without_coherence = False
        if isinstance(self.use_sar, str) and "_without_coherence" in self.use_sar:
            self.without_coherence = True
            self.use_sar = self.use_sar.replace("_without_coherence", "")

        # Calculate num_channels identical to how setup_channels does it
        self.num_channels = 10
        if self.use_sar:
            s1_bands = 2 if self.without_coherence else 4
            if self.use_sar == "asc+desc":
                self.num_channels += s1_bands * 2
            elif self.use_sar in ["asc", "desc", "mix_closest"]:
                self.num_channels += s1_bands
            else:
                raise ValueError("Bad SAR pairing")

        self.c_index_rgb = [0, 1, 2]
        self.c_index_nir = 3
        self.s2_channels = list(range(10))

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        seq_len = 10
        t_seq = list(range(seq_len))
        s1_asc = torch.randn(seq_len, 4, 32, 32)
        s1_desc = torch.randn(seq_len, 4, 32, 32)
        s1 = torch.randn(seq_len, 4, 32, 32)

        return {
            "S2": torch.randn(seq_len, 10, 32, 32),
            "S2_dates": [torch.tensor(d, dtype=torch.float32) for d in t_seq],
            "cloud_mask": torch.randint(0, 2, (seq_len, 1, 32, 32)),
            "cloud_prob": torch.rand(seq_len, 1, 32, 32),
            "S1": {
                "S1_asc": s1_asc,
                "S1_desc": s1_desc,
                "S1_dates_asc": [torch.tensor(d, dtype=torch.float32) for d in t_seq],
                "S1_dates_desc": [torch.tensor(d, dtype=torch.float32) for d in t_seq],
                "S1": s1,
                "S1_dates": [torch.tensor(d, dtype=torch.float32) for d in t_seq]
            }
        }


# Monkey patch CIRCA_from_HDF5
import dataloader_CIRCA.datasets.UTILISE_adapter as uta

uta.CIRCA_from_HDF5 = MockDataset


def test_config(use_sar_val):
    print(f"=== TESTING CONFIG: use_sar='{use_sar_val}' ===")

    # 1. Adapter creation
    adapter = CIRCA_ADAPTED2UTILISE_Dataset(
        method="utilise",
        mask_settings={
            "mask_type": "random_fully_masked",
            "ratio_masked_frames": 0.5,
            "ratio_fully_masked_frames": 0.0,
            "fixed_masking_ratio": False,
            "non_masked_frames": [0],
            "intersect_real_cloud_masks": False,
            "fill_value": 0,
            "p_filter": 0.1
        },
        filter_settings={
            "type": "cloud-free",
            "min_length": 3,
            "return_valid_obs_only": True
        },
        phase="train",
        use_sar=use_sar_val,
        channels="all",
        image_size=(32, 32)
    )

    # Check dataset initialized channel correctly
    input_dim = adapter.dataset.num_channels
    print(f"Dataset num_channels = {input_dim}")

    # 2. Get item
    batch = adapter[0]
    x = batch["x"]
    print(f"Batch 'x' tensor shape = {x.shape}")
    assert x.shape[1] == input_dim, f"Error: Dataset claims {input_dim} channels but returned {x.shape[1]}!"

    # 3. Model setup logic
    output_dim = input_dim
    s1_bands = 2 if "_without_coherence" in use_sar_val else 4
    base_use_sar = use_sar_val.replace("_without_coherence", "")
    if base_use_sar == "asc+desc":
        output_dim -= 2 * s1_bands
    else:
        output_dim -= s1_bands
    print(f"Model config output_dim resolved to = {output_dim}")
    assert output_dim == 10, "Output dimension didn't correctly subtract back to 10!"

    model = UTILISE(
        input_dim=input_dim,
        output_dim=output_dim,
        encoder_widths=[16, 16],
        decoder_widths=[16, 16],
    )

    # 4. Forward pass
    x_input = x.unsqueeze(0)  # B=1
    dates_input = batch["dates"].unsqueeze(0)  # B=1
    out = model(x_input, batch_positions=dates_input)
    print(f"Model output shape = {out.shape}")
    assert out.shape[2] == 10, f"Expected 10 channels but got {out.shape[2]}"
    print("SUCCESS!\n")


if __name__ == "__main__":
    # Test all variations
    test_config("mix_closest")
    test_config("mix_closest_without_coherence")
    test_config("asc+desc")
    test_config("asc+desc_without_coherence")
