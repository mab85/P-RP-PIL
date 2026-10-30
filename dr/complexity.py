import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                     Conv2D, MaxPooling2D, Flatten)
import argparse


# ----------------------------------------------------------------------
# 1. Build the same DL models used in baseline.py
# ----------------------------------------------------------------------
def build_transfer(name, num_classes):
    if name == 'ResNet-50':
        base = ResNet50(weights=None, include_top=False, input_shape=(128,128,3)); hidden = 512
    elif name == 'DenseNet-121':
        base = DenseNet121(weights=None, include_top=False, input_shape=(128,128,3)); hidden = 512
    elif name == 'MobileNetV2':
        base = MobileNetV2(weights=None, include_top=False, input_shape=(128,128,3)); hidden = 256
    elif name == 'EfficientNet-B0':
        base = EfficientNetB0(weights=None, include_top=False, input_shape=(128,128,3)); hidden = 256
    else:
        raise ValueError(name)

    base.trainable = False
    inputs = tf.keras.Input(shape=(128,128,3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(hidden, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)


def build_custom_cnn(num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,128,3)),
        Conv2D(32, (3,3), activation='relu'), MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'), MaxPooling2D((2,2)),
        Conv2D(128,(3,3), activation='relu'), MaxPooling2D((2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])


# ----------------------------------------------------------------------
# 2. Recursive FLOPs counter (handles nested Keras Applications)
# ----------------------------------------------------------------------
def _accumulate_flops(layer, total):
    # Dive into nested models (ResNet50 / DenseNet121 / etc.)
    if hasattr(layer, 'layers') and len(getattr(layer, 'layers', [])) > 0:
        for sub in layer.layers:
            total = _accumulate_flops(sub, total)
        return total
    try:
        in_shape  = layer.input.shape
        out_shape = layer.output.shape
        if isinstance(in_shape,  list): in_shape  = in_shape[0]
        if isinstance(out_shape, list): out_shape = out_shape[0]
        in_shape  = [int(d) if d is not None else 1 for d in in_shape[1:]]
        out_shape = [int(d) if d is not None else 1 for d in out_shape[1:]]

        if isinstance(layer, Conv2D):
            kh, kw = layer.kernel_size
            cin    = in_shape[-1]
            cout   = out_shape[-1]
            hout, wout = out_shape[0], out_shape[1]
            total += 2 * kh * kw * cin * cout * hout * wout
        elif isinstance(layer, Dense):
            total += 2 * in_shape[-1] * out_shape[-1]
    except Exception:
        pass
    return total


def get_flops(model):
    total = 0
    for layer in model.layers:
        total = _accumulate_flops(layer, total)
    return total / 1e9


# ----------------------------------------------------------------------
# 3. Memory measurement (recursive)
# ----------------------------------------------------------------------
def _accumulate_memory(layer, w_bytes, a_bytes):
    if hasattr(layer, 'layers') and len(getattr(layer, 'layers', [])) > 0:
        for sub in layer.layers:
            w_bytes, a_bytes = _accumulate_memory(sub, w_bytes, a_bytes)
        return w_bytes, a_bytes
    try:
        for w in layer.weights:
            dt = np.dtype(w.dtype).itemsize if isinstance(w.dtype, str) else w.dtype.itemsize
            w_bytes += int(np.prod(w.shape)) * dt
    except Exception:
        pass
    try:
        out_shape = layer.output.shape
        if isinstance(out_shape, list): out_shape = out_shape[0]
        out_shape = [int(d) if d is not None else 1 for d in out_shape[1:]]
        a_bytes += int(np.prod(out_shape)) * 4
    except Exception:
        pass
    return w_bytes, a_bytes


def get_memory_mb(model):
    w, a = 0, 0
    for layer in model.layers:
        w, a = _accumulate_memory(layer, w, a)
    return (w + a) / (1024 * 1024)


# ----------------------------------------------------------------------
# 4. P-RP-PIL complexity — analytic, from Section 3 equations
# ----------------------------------------------------------------------
def compute_prp_pil_complexity(dataset):
    H = W = 128; C = 3
    if dataset == 'papilledema':
        patch_size, stride, proj_dim, n_hidden, n_classes = 8, 2, 256, 2048, 3
    else:
        patch_size, stride, proj_dim, n_hidden, n_classes = 6, 3, 256, 2048, 2

    n_patches   = ((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1)
    feature_dim = 2 * proj_dim

    params_hidden = feature_dim * n_hidden + n_hidden
    params_output = n_hidden * n_classes + n_classes
    total_params  = params_hidden + params_output

    patch_dim = patch_size * patch_size * C
    flops_proj   = n_patches * patch_dim * proj_dim * 2
    flops_pool   = n_patches * proj_dim * 4
    flops_hidden = 2 * feature_dim * n_hidden
    flops_output = 2 * n_hidden * n_classes
    total_flops  = flops_proj + flops_pool + flops_hidden + flops_output

    weight_bytes = total_params * 4
    act_bytes = (n_patches * proj_dim * 4 + feature_dim * 4
                 + n_hidden * 4 + n_classes * 4)
    total_mem_mb = (weight_bytes + act_bytes) / (1024 * 1024)

    return {'Params (M)': total_params / 1e6,
            'FLOPs (G)':  total_flops / 1e9,
            'Memory (MB)': total_mem_mb}


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main(dataset='papilledema'):
    print("=" * 70)
    print(f"Complexity measurement — dataset = {dataset}")
    print("=" * 70)

    num_classes = 3 if dataset == 'papilledema' else 2
    rows = []

    prp = compute_prp_pil_complexity(dataset)
    rows.append({
        'Model': 'P-RP-PIL',
        'Params (M)':  f"{prp['Params (M)']:.2f}",
        'FLOPs (G)':   f"{prp['FLOPs (G)']:.2f}",
        'Memory (MB)': f"{prp['Memory (MB)']:.0f}",
    })
    print(f"P-RP-PIL        : {prp['Params (M)']:.2f}M | {prp['FLOPs (G)']:.2f}G | {prp['Memory (MB)']:.0f}MB")

    for name in ['ResNet-50', 'DenseNet-121', 'MobileNetV2',
                 'EfficientNet-B0', 'Custom-CNN']:
        try:
            model = build_custom_cnn(num_classes) if name == 'Custom-CNN' \
                    else build_transfer(name, num_classes)
            _ = model(tf.zeros((1,128,128,3)))

            params_m = model.count_params() / 1e6
            flops_g  = get_flops(model)
            mem_mb   = get_memory_mb(model)

            rows.append({
                'Model': name,
                'Params (M)':  f"{params_m:.2f}",
                'FLOPs (G)':   f"{flops_g:.2f}",
                'Memory (MB)': f"{mem_mb:.0f}",
            })
            print(f"{name:<16}: {params_m:.2f}M | {flops_g:.2f}G | {mem_mb:.0f}MB")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            rows.append({'Model': name, 'Params (M)': '—',
                         'FLOPs (G)': '—', 'Memory (MB)': '—'})

    df = pd.DataFrame(rows)
    csv = f'complexity_{dataset}.csv'
    df.to_csv(csv, index=False)
    print(f"\n Saved: {csv}")
    print("\n" + df.to_string(index=False))

    print("\n" + "-" * 60)
    print("LaTeX body (paste into Table VII or Table X):")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['Model']} & {r['Params (M)']} & {r['FLOPs (G)']} & {r['Memory (MB)']} \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='papilledema', choices=['papilledema', 'dr'])
    args = parser.parse_args()
    main(args.dataset)