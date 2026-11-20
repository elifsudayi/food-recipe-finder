"""ONNX export + INT8 post-training quantization + ONNX Runtime inference wrappers.

VGG16 classifier : static INT8 quantization (QDQ format, per-channel weights), calibrated on
                   200 TRAINING images (never on the test set).
MiniLM encoder   : dynamic INT8 quantization of the transformer weights.
"""
import os
import random
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ helpers
def file_mb(path):
    return os.path.getsize(path) / 1e6          # MB = 10^6 bytes


def time_calls(fn, runs=100, warmup=10):
    """Latency in ms of calling fn() `runs` times (after `warmup` untimed calls)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(ts)), "std_ms": float(np.std(ts)), "runs": runs}


def _onnx_export(model, args, path, **kw):
    import torch
    try:                                         # classic TorchScript-based exporter
        torch.onnx.export(model, args, str(path), dynamo=False, **kw)
    except TypeError:                            # older torch without the `dynamo` argument
        torch.onnx.export(model, args, str(path), **kw)


def _make_reader(arrays, input_name):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{input_name: a} for a in arrays])

        def get_next(self):
            return next(self.it, None)

    return Reader()


# --------------------------------------------------------------- VGG16 (image)
def export_vgg16_onnx(model, path, img_size=224, opset=17):
    import torch
    import torch.nn as nn
    model = model.eval().cpu()
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    dummy = torch.randn(1, 3, img_size, img_size)
    _onnx_export(model, (dummy,), path, input_names=["input"], output_names=["logits"],
                 dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset)
    return Path(path)


def calib_tensors(root, per_class=20, img_size=224, seed=42):
    """`per_class` TRAIN images per class, preprocessed like validation images (no augmentation)."""
    from torchvision import datasets
    from src.image.data import build_transforms
    ds = datasets.ImageFolder(f"{root}/train", transform=build_transforms(img_size, train=False))
    by = {}
    for i, (_, y) in enumerate(ds.samples):
        by.setdefault(y, []).append(i)
    rng = random.Random(seed)
    idx = [i for y in sorted(by) for i in rng.sample(by[y], per_class)]
    return [ds[i][0].numpy()[None].astype(np.float32) for i in idx]


def quantize_vgg16_static(fp32_path, int8_path, arrays, reduce_range=False):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    src = str(fp32_path)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = str(Path(fp32_path).with_suffix(".pre.onnx"))
        quant_pre_process(src, pre)
        src = pre
    except Exception as e:                       # pre-processing is recommended, not required
        print("quant_pre_process skipped:", repr(e)[:150])
    quantize_static(src, str(int8_path), _make_reader(arrays, "input"),
                    quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=reduce_range,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    return Path(int8_path)


class OnnxClassifier:
    def __init__(self, path, threads=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def logits(self, x):
        return self.sess.run(None, {self.input_name: np.asarray(x, dtype=np.float32)})[0]

    def probs(self, x):
        z = self.logits(x)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def evaluate_onnx(clf, loader, max_batches=None):
    from tqdm import tqdm
    n = c1 = c5 = 0
    preds, targets = [], []
    for b, (x, y) in enumerate(tqdm(loader)):
        if max_batches and b >= max_batches:
            break
        lg = clf.logits(x.numpy())
        top5 = np.argsort(-lg, axis=1)[:, :5]
        yy = y.numpy()
        c1 += int((top5[:, 0] == yy).sum())
        c5 += int((top5 == yy[:, None]).any(axis=1).sum())
        n += len(yy)
        preds += top5[:, 0].tolist()
        targets += yy.tolist()
    return {"top1": c1 / n, "top5": c5 / n, "preds": preds, "targets": targets, "n": n}


# ------------------------------------------------------------ MiniLM (text)
def export_minilm_onnx(name, path, opset=17):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids, return_dict=False)[0]

    tok = AutoTokenizer.from_pretrained(name)
    model = Wrap(AutoModel.from_pretrained(name, attn_implementation="eager").eval())
    b = tok(["hello world", "a somewhat longer example sentence used for tracing"],
            padding=True, return_tensors="pt")
    args = (b["input_ids"], b["attention_mask"], b["token_type_ids"])
    dyn = {k: {0: "batch", 1: "seq"} for k in ["input_ids", "attention_mask", "token_type_ids"]}
    dyn["last_hidden_state"] = {0: "batch", 1: "seq"}
    _onnx_export(model, args, path, input_names=["input_ids", "attention_mask", "token_type_ids"],
                 output_names=["last_hidden_state"], dynamic_axes=dyn, opset_version=opset)
    return Path(path)


def quantize_minilm_dynamic(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return Path(int8_path)


class OnnxEncoder:
    """Same interface as STEncoder: tokenize -> ONNX transformer -> mean pooling -> L2 normalise."""

    def __init__(self, onnx_path, tokenizer_name, max_len=256, threads=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def encode(self, texts, batch_size=32):
        texts, out = list(texts), []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in b.items() if k in self.names}
            h = self.sess.run(None, feed)[0]
            m = b["attention_mask"][..., None].astype(np.float32)
            emb = (h * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            out.append(emb)
        return np.vstack(out).astype(np.float32)
