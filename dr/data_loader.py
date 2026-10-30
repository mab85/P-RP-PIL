
import os, glob, gc
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ======================================================================
# 1. Author-provided DR loader
# ======================================================================
def load_diabetic_retinopathy_binary(data_path='data',
                                     img_size=(128, 128),
                                     max_samples_per_class=None,
                                     use_retino=False):
    """
    Load the DR dataset using the author-provided splits.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, le

    Labels are integers: 0 = No_DR, 1 = DR.
    """
    folder = 'Retino' if use_retino else 'Diagnosis of Diabetic Retinopathy'
    base = os.path.join(data_path, folder)

    le = LabelEncoder()
    le.classes_ = np.array(['No_DR', 'DR'])

    if not os.path.isdir(base):
        print(f"Error: {base} not found.")
        return None, None, None, None, None, None, le

    print(f"Loading DR from: {folder}")
    if img_size[0] > 128:
        img_size = (128, 128)

    splits_data = {}
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: {split_dir} not found.")
            continue

        images, labels = [], []
        for class_name, label in [('DR', 1), ('No_DR', 0)]:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                for alt in ['NoDR', 'No-DR', 'No DR']:
                    alt_dir = os.path.join(split_dir, alt)
                    if os.path.isdir(alt_dir):
                        class_dir = alt_dir
                        break
                else:
                    continue

            files = [f for f in glob.glob(os.path.join(class_dir, '*'))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp',
                                            '.tif', '.tiff', '.jfif'))]
            if max_samples_per_class:
                files = files[:max_samples_per_class]

            for f in tqdm(files, desc=f"  {split}/{class_name}", leave=False):
                img = cv2.imread(f)
                if img is None:
                    continue
                img = cv2.resize(img, img_size).astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)
                if len(images) % 200 == 0:
                    gc.collect()

        if not images:
            print(f"Warning: no images found in {split_dir}")
            continue

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        splits_data[split] = (X, y)
        print(f"  {split}: {len(X)} images (DR={int((y==1).sum())}, "
              f"No_DR={int((y==0).sum())})")

    X_tr, y_tr = splits_data.get('train', (None, None))
    X_va, y_va = splits_data.get('valid', (None, None))
    X_te, y_te = splits_data.get('test', (None, None))

    if X_tr is None:
        print("Error: training split not found.")
        return None, None, None, None, None, None, le

    return X_tr, X_va, X_te, y_tr, y_va, y_te, le


# ======================================================================
# 2. Stratified 70/15/15 splitters
# ======================================================================
def preprocess_medical_data(X, y, test_size=0.15, val_size=0.15,
                            random_state=42):
    """
    Stratified 70/15/15 split for multi-class labels.
    Returns X_tr, X_va, X_te, y_tr, y_va, y_te, le.
    """
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y_enc, test_size=test_size,
        random_state=random_state, stratify=y_enc)

    val_frac = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_frac,
        random_state=random_state, stratify=y_tv)

    return X_tr, X_va, X_te, y_tr, y_va, y_te, le


def preprocess_medical_data_binary(X, y, test_size=0.15, val_size=0.15,
                                   random_state=42):
    """
    Stratified 70/15/15 split for binary labels.
    Forces le.classes_ = ['No_DR', 'DR'].
    """
    le = LabelEncoder()
    le.classes_ = np.array(['No_DR', 'DR'])

    y_enc = np.asarray(y).astype(np.int64)

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y_enc, test_size=test_size,
        random_state=random_state, stratify=y_enc)

    val_frac = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_frac,
        random_state=random_state, stratify=y_tv)

    return X_tr, X_va, X_te, y_tr, y_va, y_te, le