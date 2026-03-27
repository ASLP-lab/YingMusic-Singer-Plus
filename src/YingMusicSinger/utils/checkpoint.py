import torch

# from vocos import Vocos
from singer.model import Singer


def load_model(model_cls, model_cfg, ckpt_path, vocab_char_map, device="cuda"):
    model_arc = model_cfg.model.arch
    mel_spec_kwargs = model_cfg.model.mel_spec
    vocab_size = len(vocab_char_map)

    backbone = model_cls(
        **model_arc, text_num_embeds=vocab_size, mel_dim=mel_spec_kwargs.n_mel_channels
    )

    model = Singer(
        transformer=backbone,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle module prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def load_vocoder(vocoder_name, is_local, local_path, device="cuda"):
    if vocoder_name == "vocos":
        if is_local:
            vocoder = Vocos.from_hparams(local_path).to(device)
        else:
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    elif vocoder_name == "bigvgan":
        # Placeholder for bigvgan
        # You might need to import bigvgan here
        raise NotImplementedError("BigVGAN loading not implemented yet")
    else:
        # Fallback or error
        print(
            f"Warning: Unknown vocoder {vocoder_name}, trying to load from local path if provided"
        )
        if is_local:
            # Try loading as vocos or similar if generic
            vocoder = Vocos.from_hparams(local_path).to(device)
        else:
            raise ValueError(f"Unknown vocoder: {vocoder_name}")
    return vocoder
