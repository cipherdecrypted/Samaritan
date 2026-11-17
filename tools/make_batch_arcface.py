from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"
SRC = MODELS / "w600k_r50.onnx"
DST = MODELS / "w600k_r50_batch.onnx"


def convert():
    m = onnx.load(str(SRC))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    producers = {o: n for n in g.node for o in n.output}
    replaced = 0
    for idx, n in enumerate(list(g.node)):
        if n.op_type != "BatchNormalization" or producers.get(n.input[0]) is None \
                or producers[n.input[0]].op_type != "Gemm":
            continue
        gamma, beta, mean, var = (numpy_helper.to_array(inits[x]).astype(np.float64) for x in n.input[1:5])
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        a = (gamma / np.sqrt(var + eps)).astype(np.float32)
        b = (beta - mean * a).astype(np.float32)
        an, bn = n.name + "_scale", n.name + "_shift"
        g.initializer.extend([numpy_helper.from_array(a, an), numpy_helper.from_array(b, bn)])
        mul_out = n.name + "_mul"
        mul = helper.make_node("Mul", [n.input[0], an], [mul_out], name=n.name + "_Mul")
        add = helper.make_node("Add", [mul_out, bn], list(n.output), name=n.name + "_Add")
        pos = list(g.node).index(n)
        g.node.remove(n)
        g.node.insert(pos, add)
        g.node.insert(pos, mul)
        replaced += 1
    for o in g.output:
        o.type.tensor_type.shape.dim[0].dim_param = "N"
    onnx.checker.check_model(m)
    onnx.save(m, str(DST))
    return replaced


def verify():
    import onnxruntime as ort
    ort.set_default_logger_severity(4)
    x = np.random.default_rng(0).normal(0, 0.5, (8, 3, 112, 112)).astype(np.float32)
    ref = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
    r = np.concatenate([ref.run(None, {ref.get_inputs()[0].name: x[i:i + 1]})[0] for i in range(8)])
    b = new.run(None, {new.get_inputs()[0].name: x})[0]
    err = float(np.abs(r - b).max() / np.abs(r).max())
    cos = float(np.min(np.sum(r * b, 1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(b, axis=1))))
    return err, cos


if __name__ == "__main__":
    n = convert()
    err, cos = verify()
    print(f"replaced {n} BatchNormalization node(s); batch-8 vs original single runs: max rel error {err:.2e}, "
          f"min cosine {cos:.6f} -> {DST.name}")
    if cos <= 0.99999:
        sys.exit(1)
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load(str(DST)), keep_io_types=True),
              str(DST.with_name(DST.stem + "_fp16.onnx")))
    print("wrote", DST.stem + "_fp16.onnx")
