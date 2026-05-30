import os
import yaml
import torch

# ── Key-format helpers ───────────────────────────────────────────────────────

# Root-level names that appear in a clean TimeSenCLIPEncoder state_dict
_ENCODER_ROOTS = {
    "time_pos_encoding", "time_token", "spectral_embedding",
    "transformer", "to_latent", "mlp_head",
}

# Prefixes used when the encoder is wrapped inside a Lightning module
_WRAPPER_PREFIXES = ("learner.ts_encoder.", "ts_encoder.", "TS_ViT.")


def _is_clean_encoder_state(state_dict: dict) -> bool:
    """Return True if keys are already bare encoder keys (no wrapper prefix)."""
    return any(k.split(".")[0] in _ENCODER_ROOTS for k in state_dict)


# ── Config loading ───────────────────────────────────────────────────────────

def replace_placeholders(config_section: dict, replacements: dict):
    """Recursively replace ``{placeholder}`` strings in a config dict."""
    for key, value in config_section.items():
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            config_section[key] = replacements.get(value[1:-1], value)
        elif isinstance(value, dict):
            replace_placeholders(value, replacements)


def model_config_load(args, dropout_type: str = "None") -> dict:
    """Load model architecture config from YAML, substituting CLI arg values."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    replacements = {
        "input_resolution": args.input_resolution,
        "time_frames":      args.time_frames,
        "num_channels":     len(args.channels),
        "device":           args.device,
        "dropout_type":     dropout_type,
    }
    model_config = config[args.ts_arch]
    replace_placeholders(model_config, replacements)
    print(f"Model config: {model_config}")
    return model_config


# ── Weight utilities ─────────────────────────────────────────────────────────

def pretrained_weights_ts(checkpoint: dict) -> dict:
    """Extract TimeSenCLIPEncoder weights from a Lightning state_dict.

    Handles:
    - Clean encoder keys (final saved format) — returned as-is.
    - Keys wrapped under a prefix (``ts_encoder.*``, ``TS_ViT.*``,
      ``learner.ts_encoder.*``) — prefix is stripped.
    """
    if _is_clean_encoder_state(checkpoint):
        return checkpoint

    out = {}
    for prefix in _WRAPPER_PREFIXES:
        out.update({
            k[len(prefix):]: v
            for k, v in checkpoint.items()
            if k.startswith(prefix)
        })
    return out if out else checkpoint


def before_load_weights(checkpoint_path: str):
    """Strip validation-only sub-models from a Lightning checkpoint in-place.

    Removes keys starting with ``ts_encoder.`` or ``clip_model`` so that
    training can be resumed without carrying unused weights.
    """
    ckpt = torch.load(checkpoint_path)
    ckpt["state_dict"] = {
        k: v for k, v in ckpt["state_dict"].items()
        if not (k.startswith("ts_encoder.") or k.startswith("clip_model"))
    }
    torch.save(ckpt, checkpoint_path)


def pretrained_weights_val(checkpoint: dict) -> dict:
    """Extract encoder weights using the ``learner.ts_encoder.*`` prefix."""
    filtered = {
        k: v for k, v in checkpoint.items()
        if not (k.startswith("ts_encoder.") or k.startswith("clip_model"))
    }
    return {
        k.replace("learner.ts_encoder.", ""): v
        for k, v in filtered.items()
        if k.startswith("learner.ts_encoder.")
    }
