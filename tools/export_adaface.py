from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

ROOT = Path(__file__).resolve().parent.parent


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class BasicBlockIR(nn.Module):
    def __init__(self, cin, depth, stride):
        super().__init__()
        self.shortcut_layer = (nn.MaxPool2d(1, stride) if cin == depth else
                               nn.Sequential(nn.Conv2d(cin, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth)))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(cin), nn.Conv2d(cin, depth, (3, 3), (1, 1), 1, bias=False), nn.BatchNorm2d(depth),
            nn.PReLU(depth), nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


Block = namedtuple("Block", ["in_channel", "depth", "stride"])


def _stage(cin, depth, n):
    return [Block(cin, depth, 2)] + [Block(depth, depth, 1) for _ in range(n - 1)]


class IR101(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        blocks = _stage(64, 64, 3) + _stage(64, 128, 13) + _stage(128, 256, 30) + _stage(256, 512, 3)
        self.body = nn.Sequential(*[BasicBlockIR(b.in_channel, b.depth, b.stride) for b in blocks])
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.4), Flatten(),
                                          nn.Linear(512 * 7 * 7, dim), nn.BatchNorm1d(dim, affine=False))

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))


def build(weights: Path) -> nn.Module:
    sd = load_file(str(weights))
    prefix = next(p for p in ("model.net.", "net.", "") if any(k.startswith(p + "input_layer") for k in sd))
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    net = IR101()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise SystemExit(f"weights do not fit IR-101: missing {missing[:5]}, unexpected {unexpected[:5]}")
    net.eval()
    lin, bn = net.output_layer[3], net.output_layer[4]
    s = torch.sqrt(bn.running_var + bn.eps)
    folded = nn.Linear(lin.in_features, lin.out_features)
    with torch.no_grad():
        folded.weight.copy_(lin.weight / s[:, None])
        folded.bias.copy_((lin.bias - bn.running_mean) / s)
    ref = net
    fused = IR101()
    fused.load_state_dict(net.state_dict())
    fused.output_layer = nn.Sequential(*list(net.output_layer)[:3], folded)
    fused.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 112, 112) * 0.5
        a, b = ref(x), fused(x)
    err = float((a - b).abs().max() / a.abs().max())
    assert err < 1e-4, err
    return fused


def main(weights: Path, out: Path):
    net = build(weights)
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (8, 3, 112, 112)).astype(np.float32))
    with torch.no_grad():
        y = net(x).numpy()
    torch.onnx.export(net, x[:1], str(out), input_names=["input"], output_names=["embedding"], opset_version=17,
                      dynamic_axes={"input": {0: "N"}, "embedding": {0: "N"}}, dynamo=False)
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    z = sess.run(None, {"input": x.numpy()})[0]
    cos = float(np.min(np.sum(z * y, 1) / (np.linalg.norm(z, axis=1) * np.linalg.norm(y, axis=1))))
    if cos <= 0.99999:
        raise SystemExit(f"exported model does not match the network (min cosine {cos})")
    onnx.save(float16.convert_float_to_float16(onnx.load(str(out)), keep_io_types=True),
              str(out.with_name(out.stem + "_fp16.onnx")))
    print(f"exported {out.name} and {out.stem}_fp16.onnx (verified, min cosine {cos:.6f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "adaface_ir101_webface12m.onnx")
