import torch
import torch.nn as nn


class MIDIFuzzDisturb(nn.Module):
    """Applies fuzzing perturbations to MIDI latent representations.

    The raw MIDI teacher model output preserves good prosody but causes
    pronunciation interference. This module mitigates that by applying
    blur, temporal dropout, and noise to the melody latent.
    """

    def __init__(
        self, dim=128, drop_prob=0.3, noise_scale=0.1, blur_kernel=3, drop_type="random"
    ):
        super().__init__()
        self.blur = None
        self.drop_prob = None
        self.noise_scale = None
        self.dim = dim
        self.drop_type = drop_type

        assert drop_prob is not None
        assert drop_type is not None
        if drop_type == "random":
            # drop_prob is a float
            if drop_prob != 0:
                self.drop_prob = drop_prob
        elif drop_type == "equal_space":
            # drop_prob is a [drop, keep] list, e.g., [1, 1] means 1 frame drop, 1 frame keep
            self.drop_prob = drop_prob
        else:
            raise ValueError(f"Unknown drop_type: {drop_type}")

        if noise_scale != 0:
            self.noise_scale = noise_scale
        if blur_kernel != 0:
            assert blur_kernel % 2 == 1, f"blur_kernel {blur_kernel} must be odd"
            self.blur = nn.AvgPool1d(
                kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2
            )

    def _create_equal_space_mask(self, batch_size, seq_len, device):
        """Create an equally-spaced mask cycling [drop, keep] frames."""
        drop_frames, keep_frames = self.drop_prob
        cycle_len = drop_frames + keep_frames

        # Pattern: first drop_frames are 0 (drop), next keep_frames are 1 (keep)
        pattern = torch.cat(
            [
                torch.zeros(drop_frames, device=device),
                torch.ones(keep_frames, device=device),
            ]
        )

        # Repeat pattern to cover the full sequence length
        num_repeats = (seq_len + cycle_len - 1) // cycle_len
        mask = pattern.repeat(num_repeats)[:seq_len]  # [T]

        # Expand to [B, T, 1]
        mask = mask.view(1, seq_len, 1).expand(batch_size, -1, -1)

        return mask

    def forward(self, x):
        # x: [B, T, D=128], pre-sigmoid logits
        x = torch.sigmoid(x)

        assert x.shape[-1] == self.dim, (
            f"MIDIFuzzDisturb: expected dim={self.dim}, got {x.shape[-1]}"
        )

        if self.blur:
            x = self.blur(x.transpose(1, 2)).transpose(1, 2)

        if self.drop_prob:
            if self.drop_type == "random":
                time_mask = (
                    torch.rand(x.shape[0], x.shape[1], 1, device=x.device)
                    > self.drop_prob
                )
                x = x * time_mask.float()
            elif self.drop_type == "equal_space":
                time_mask = self._create_equal_space_mask(
                    x.shape[0], x.shape[1], x.device
                )
                x = x * time_mask.float()
            else:
                raise ValueError(f"Unknown drop_type: {self.drop_type}")

        if self.noise_scale:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        return x


class MIDIDigitalEmbedding(nn.Module):
    """Embeds continuous MIDI values into discrete token embeddings.

    Continuous MIDI values in [0, 127] are quantized at a configurable
    resolution (mark_distinguish_scale) and mapped to learned embeddings.
    """

    def __init__(self, embed_dim=128, num_classes=128, mark_distinguish_scale=2):
        super().__init__()

        # num_classes covers the input range [0, 127] plus 2 special tokens
        self.num_classes = num_classes + 2
        self.mark_distinguish_scale = mark_distinguish_scale
        self.embedding_input_num_class = self.num_classes * self.mark_distinguish_scale
        self.embedding = nn.Embedding(self.embedding_input_num_class, embed_dim)

    def midi_to_class(self, midi_values):
        """Map continuous MIDI values to discrete class indices.

        Args:
            midi_values: [B, T] continuous MIDI values, roughly in [0, 127]

        Returns:
            class_indices: [B, T] discrete class indices
        """
        # Round to nearest quantization step
        # e.g., with scale=2: 0->0, 0.3->1, 0.5->1, 0.8->2, 1.0->2, ...
        class_indices = torch.round(midi_values * self.mark_distinguish_scale).long()

        # Clamp to valid range
        class_indices = torch.clamp(
            class_indices, 0, self.embedding_input_num_class - 1
        )

        return class_indices

    def forward(self, midi_values):
        """
        Args:
            midi_values: [B, T] continuous MIDI values

        Returns:
            embeddings: [B, T, embed_dim] embedding vectors
        """
        class_indices = self.midi_to_class(midi_values)
        embeddings = self.embedding(class_indices)
        return embeddings
