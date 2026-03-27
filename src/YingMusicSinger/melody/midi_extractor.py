import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.YingMusicSinger.melody.Gconform import Gmidi_conform

# midi decoding utils


def decode_gaussian_blurred_probs(probs, vmin, vmax, deviation, threshold):
    num_bins = int(probs.shape[-1])
    interval = (vmax - vmin) / (num_bins - 1)
    width = int(3 * deviation / interval)  # 3 * sigma
    idx = torch.arange(num_bins, device=probs.device)[None, None, :]  # [1, 1, N]
    idx_values = idx * interval + vmin
    center = torch.argmax(probs, dim=-1, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - width, min=0)  # [B, T, 1]
    end = torch.clip(center + width + 1, max=num_bins)  # [B, T, 1]
    idx_masks = (idx >= start) & (idx < end)  # [B, T, N]
    weights = probs * idx_masks  # [B, T, N]
    product_sum = torch.sum(weights * idx_values, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    values = product_sum / (
        weight_sum + (weight_sum == 0)
    )  # avoid dividing by zero, [B, T]
    rest = probs.max(dim=-1)[0] < threshold  # [B, T]
    return values, rest


def decode_bounds_to_alignment(bounds, use_diff=True):
    bounds_step = bounds.cumsum(dim=1).round().long()
    if use_diff:
        bounds_inc = (
            torch.diff(
                bounds_step,
                dim=1,
                prepend=torch.full(
                    (bounds.shape[0], 1),
                    fill_value=-1,
                    dtype=bounds_step.dtype,
                    device=bounds_step.device,
                ),
            )
            > 0
        )
    else:
        bounds_inc = F.pad(
            (bounds_step[:, 1:] > bounds_step[:, :-1]), [1, 0], value=True
        )
    frame2item = bounds_inc.long().cumsum(dim=1)
    return frame2item


def decode_note_sequence(frame2item, values, masks, threshold=0.5):
    """

    :param frame2item: [1, 1, 1, 1, 2, 2, 3, 3, 3]
    :param values:
    :param masks:
    :param threshold: minimum ratio of unmasked frames required to be regarded as an unmasked item
    :return: item_values, item_dur, item_masks
    """
    b = frame2item.shape[0]
    space = frame2item.max() + 1

    item_dur = frame2item.new_zeros(b, space, dtype=frame2item.dtype).scatter_add(
        1, frame2item, torch.ones_like(frame2item)
    )[:, 1:]
    item_unmasked_dur = frame2item.new_zeros(
        b, space, dtype=frame2item.dtype
    ).scatter_add(1, frame2item, masks.long())[:, 1:]
    item_masks = item_unmasked_dur / item_dur >= threshold

    values_quant = values.round().long()
    histogram = (
        frame2item.new_zeros(b, space * 128, dtype=frame2item.dtype)
        .scatter_add(
            1, frame2item * 128 + values_quant, torch.ones_like(frame2item) * masks
        )
        .unflatten(1, [space, 128])[:, 1:, :]
    )
    item_values_center = histogram.float().argmax(dim=2).to(dtype=values.dtype)
    values_center = torch.gather(F.pad(item_values_center, [1, 0]), 1, frame2item)
    values_near_center = (
        masks & (values >= values_center - 0.5) & (values <= values_center + 0.5)
    )
    item_valid_dur = frame2item.new_zeros(b, space, dtype=frame2item.dtype).scatter_add(
        1, frame2item, values_near_center.long()
    )[:, 1:]
    item_values = values.new_zeros(b, space, dtype=values.dtype).scatter_add(
        1, frame2item, values * values_near_center
    )[:, 1:] / (item_valid_dur + (item_valid_dur == 0))

    return item_values, item_dur, item_masks


def expand_batch_padded(feature_tensor, counts_tensor, padding_value=0.0):
    assert feature_tensor.dim() == 2 and counts_tensor.dim() == 2

    lengths = torch.sum(counts_tensor, dim=1)

    feature_tensor = feature_tensor.reshape(-1)
    counts_tensor = counts_tensor.reshape(-1)
    expanded_flat = torch.repeat_interleave(feature_tensor, counts_tensor)

    ragged_list = torch.split(expanded_flat, lengths.tolist())

    padded_tensor = pad_sequence(
        ragged_list, batch_first=True, padding_value=padding_value
    )

    return padded_tensor, lengths


class midi_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, x, target):
        midiout, cutp = x
        midi_target, cutp_target = target

        cutploss = self.loss(cutp, cutp_target)
        midiloss = self.loss(midiout, midi_target)
        return midiloss, cutploss


class MIDIExtractor(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        super().__init__()

        cfg = {
            "attention_drop": 0.1,
            "attention_heads": 8,
            "attention_heads_dim": 64,
            "conv_drop": 0.1,
            "dim": 512,
            "ffn_latent_drop": 0.1,
            "ffn_out_drop": 0.1,
            "kernel_size": 31,
            "lay": 8,
            "use_lay_skip": True,
            "indim": 80,
            "outdim": 128,
        }
        if in_dim is not None:
            cfg["indim"] = in_dim
        if out_dim is not None:
            cfg["outdim"] = out_dim

        self.midi_conform = Gmidi_conform(**cfg)

        self.midi_min = 0
        self.midi_max = 127
        self.midi_deviation = 1.0
        self.rest_threshold = 0.1

    def _load_form_ckpt(self, ckpt_path, device="cpu"):
        from collections import OrderedDict

        if ckpt_path is None:
            raise ValueError("midi_extractor_path is required")

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        prefix_in_ckpt = "model.model"
        state_dict = OrderedDict(
            {
                k.replace(f"{prefix_in_ckpt}.", "midi_conform."): v
                for k, v in state_dict.items()
                if k.startswith(f"{prefix_in_ckpt}.")
            }
        )
        self.load_state_dict(state_dict, strict=True)
        # self.to(device)

    def forward(self, x, mask=None):
        midi, bound = self.midi_conform(x, mask)

        return midi, bound

    def postprocess(self, midi, bounds, with_expand=False):
        probs = torch.sigmoid(midi)

        bound_probs = torch.sigmoid(bounds)
        bound_probs = torch.squeeze(bound_probs, -1)

        masks = torch.ones_like(bound_probs).bool()
        # Avoid in-place ops on tensors needed for autograd (outputs of SigmoidBackward)
        probs = probs * masks[..., None]
        bound_probs = bound_probs * masks
        unit2note_pred = decode_bounds_to_alignment(bound_probs) * masks
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs,
            vmin=self.midi_min,
            vmax=self.midi_max,
            deviation=self.midi_deviation,
            threshold=self.rest_threshold,
        )
        note_midi_pred, note_dur_pred, note_mask_pred = decode_note_sequence(
            unit2note_pred, midi_pred, ~rest_pred & masks
        )
        if not with_expand:
            return note_midi_pred, note_dur_pred

        note_midi_expand, _ = expand_batch_padded(note_midi_pred, note_dur_pred)
        return note_midi_expand, None
