# densenet_latency.py — single-image inference latency for DenseNet-121
import time
import numpy as np
import data_loader
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

DATASET = 'papilledema'   # or 'dr'

if DATASET == 'papilledema':
    X, y = data_loader.load_papilledema_dataset(
        img_size=(128, 128), max_samples_per_class=None)
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.preprocess_medical_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42)
else:
    X_tr, X_va, X_te, y_tr, y_va, y_te, le = \
        data_loader.load_diabetic_retinopathy_binary(
            img_size=(128, 128), max_samples_per_class=None,
            use_retino=False)

n_classes = len(le.classes_)
base = DenseNet121(weights='imagenet', include_top=False,
                   input_shape=(128, 128, 3))
base.trainable = False
inputs = tf.keras.Input(shape=(128, 128, 3))
x = base(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(n_classes, activation='softmax')(x)
model = Model(inputs, out)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Warm up
for _ in range(3):
    _ = model.predict(X_te[:1], verbose=0)

N = 100
t0 = time.time()
for i in range(N):
    _ = model.predict(X_te[i:i + 1], verbose=0)
elapsed = time.time() - t0
print(f"{DATASET} DenseNet-121 single-image latency: {elapsed / N:.4f} s")