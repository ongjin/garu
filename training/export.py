"""Export a trained PyTorch BiLSTM+CRF model to GMDL binary format.

The GMDL format is read by the Rust Model::from_bytes() in garu-core.

Binary layout:
    [b"GMDL"][version: u32-LE]
    [section...]*

Each section:
    [type: u8][length: u32-LE][data: bytes]

Section types:
    0 = embedding
    1 = bilstm
    2 = output_weights
    3 = output_bias
    4 = crf (transition matrix)
    5 = tagset (BIO label definitions)
    6 = dict (optional dictionary, not exported here)

INT8 quantization: per-tensor symmetric quantization.
    scale = max(|tensor|) / 127
    quantized[i] = clip(round(tensor[i] / scale), -127, 127)
"""

import json
import struct
import sys
from typing import List, Tuple

import numpy as np
import torch

from model import BiLstmCrf
from preprocess import POS_TAGS

# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------


def quantize_int8(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize a float tensor to INT8 with per-tensor symmetric scaling.

    Returns:
        (quantized i8 array, scale factor)
    """
    abs_max = np.abs(tensor).max()
    if abs_max < 1e-10:
        return np.zeros_like(tensor, dtype=np.int8), 1e-10
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _pack_u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _pack_f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _pack_section(section_type: int, data: bytes) -> bytes:
    """Pack a section: [type: u8][length: u32-LE][data]."""
    return struct.pack("<B", section_type) + _pack_u32(len(data)) + data


def _pack_quantized_matrix(weight: np.ndarray) -> bytes:
    """Pack a quantized matrix: [rows: u32][cols: u32][scale: f32][i8 data]."""
    rows, cols = weight.shape
    q_data, scale = quantize_int8(weight)
    buf = _pack_u32(rows) + _pack_u32(cols) + _pack_f32(scale)
    buf += q_data.tobytes()
    return buf


def build_embedding_section(model: BiLstmCrf) -> bytes:
    """Section 0: embedding weights as f32.

    Layout: [vocab_size: u32][embed_dim: u32][f32 weights...]
    """
    weights = model.embedding.weight.detach().cpu().numpy()  # [vocab, embed]
    vocab_size, embed_dim = weights.shape
    buf = _pack_u32(vocab_size) + _pack_u32(embed_dim)
    buf += weights.astype(np.float32).tobytes()
    return _pack_section(0, buf)


def build_bilstm_section(model: BiLstmCrf) -> bytes:
    """Section 1: BiLSTM weights.

    PyTorch LSTM packs gates as [i, f, g, o] in weight_ih and weight_hh.
    The Rust model expects gates in order [i, f, g, o], with each gate
    stored as separate quantized matrices.

    Layout:
        [num_layers: u32][hidden_size: u32]
        For each layer:
            For direction in [forward, backward]:
                For gate in [i, f, g, o]:
                    [quantized w_i matrix]
                    [quantized w_h matrix]
                    [f32 bias (hidden_size floats)]

    Note: PyTorch's bias is split across bias_ih and bias_hh, combined
    as bias_ih + bias_hh for the effective bias.
    """
    lstm = model.lstm
    num_layers = lstm.num_layers
    hidden_size = lstm.hidden_size

    buf = _pack_u32(num_layers) + _pack_u32(hidden_size)

    for layer_idx in range(num_layers):
        for direction in range(2):
            suffix = f"_l{layer_idx}" + ("_reverse" if direction == 1 else "")

            # weight_ih: [4*hidden, input_size], weight_hh: [4*hidden, hidden]
            w_ih = getattr(lstm, f"weight_ih{suffix}").detach().cpu().numpy()
            w_hh = getattr(lstm, f"weight_hh{suffix}").detach().cpu().numpy()
            b_ih = getattr(lstm, f"bias_ih{suffix}").detach().cpu().numpy()
            b_hh = getattr(lstm, f"bias_hh{suffix}").detach().cpu().numpy()

            # Combined bias
            bias = b_ih + b_hh

            # Split into 4 gates: each gate has hidden_size rows
            # PyTorch gate order: [i, f, g, o]
            for gate_idx in range(4):
                start = gate_idx * hidden_size
                end = (gate_idx + 1) * hidden_size

                gate_w_ih = w_ih[start:end, :]  # [hidden, input_size]
                gate_w_hh = w_hh[start:end, :]  # [hidden, hidden]
                gate_bias = bias[start:end]      # [hidden]

                buf += _pack_quantized_matrix(gate_w_ih)
                buf += _pack_quantized_matrix(gate_w_hh)
                buf += gate_bias.astype(np.float32).tobytes()

    return _pack_section(1, buf)


def build_output_weights_section(model: BiLstmCrf) -> bytes:
    """Section 2: output projection weights as quantized matrix.

    Linear layer weight shape: [num_tags, hidden_size*2]
    """
    weight = model.output_proj.weight.detach().cpu().numpy()
    return _pack_section(2, _pack_quantized_matrix(weight))


def build_output_bias_section(model: BiLstmCrf) -> bytes:
    """Section 3: output projection bias as f32 array."""
    bias = model.output_proj.bias.detach().cpu().numpy().astype(np.float32)
    return _pack_section(3, bias.tobytes())


def build_crf_section(model: BiLstmCrf) -> bytes:
    """Section 4: CRF transition matrix.

    Layout: [num_tags: u32][f32 transitions flattened row-major]
    """
    trans = model.transitions.detach().cpu().numpy().astype(np.float32)
    num_tags = trans.shape[0]
    buf = _pack_u32(num_tags) + trans.tobytes()
    return _pack_section(4, buf)


def build_tagset_section(bio_labels: List[str]) -> bytes:
    """Section 5: tagset mapping.

    Layout: [num_labels: u32]
    Per label: [bio_byte: u8][pos_byte: u8]

    bio_byte: 0=B, 1=I, 2=O
    pos_byte: index into POS_TAGS (0-38), or 0 for O tag
    """
    pos_to_idx = {tag: idx for idx, tag in enumerate(POS_TAGS)}

    buf = _pack_u32(len(bio_labels))
    for label in bio_labels:
        if label == "O":
            buf += struct.pack("<BB", 2, 0)  # O tag, pos_byte unused
        elif label.startswith("B-"):
            pos_tag = label[2:]
            pos_idx = pos_to_idx.get(pos_tag, 0)
            buf += struct.pack("<BB", 0, pos_idx)
        elif label.startswith("I-"):
            pos_tag = label[2:]
            pos_idx = pos_to_idx.get(pos_tag, 0)
            buf += struct.pack("<BB", 1, pos_idx)
        else:
            # Fallback: treat as O
            buf += struct.pack("<BB", 2, 0)

    return _pack_section(5, buf)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def export_gmdl(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
) -> None:
    """Export a trained model to GMDL binary format.

    Args:
        checkpoint_path: Path to best_model.pt state dict.
        config_path: Path to config.json from training.
        output_path: Path to write the .gmdl file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    embed_dim = config["embed_dim"]
    hidden_size = config["hidden_size"]
    num_tags = config["num_tags"]
    num_layers = config["num_layers"]
    dropout = config.get("dropout", 0.3)
    bio_labels = config["bio_labels"]

    # Reconstruct model and load weights
    model = BiLstmCrf(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_tags=num_tags,
        num_layers=num_layers,
        dropout=dropout,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.train(False)

    # Build GMDL binary
    header = b"GMDL" + _pack_u32(1)  # version 1

    sections = b""
    sections += build_embedding_section(model)
    sections += build_bilstm_section(model)
    sections += build_output_weights_section(model)
    sections += build_output_bias_section(model)
    sections += build_crf_section(model)
    sections += build_tagset_section(bio_labels)

    with open(output_path, "wb") as f:
        f.write(header + sections)

    file_size = len(header) + len(sections)
    print(f"Exported GMDL model to {output_path} ({file_size:,} bytes)")
    print(f"  vocab_size={vocab_size}, embed_dim={embed_dim}, "
          f"hidden_size={hidden_size}, num_tags={num_tags}, "
          f"num_layers={num_layers}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <best_model.pt> <config.json> <output.gmdl>")
        sys.exit(1)
    export_gmdl(sys.argv[1], sys.argv[2], sys.argv[3])
