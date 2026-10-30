# baseline_papilledema.py
import numpy as np
import time
import os
import gc
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from data_loader import load_papilledema_dataset, preprocess_medical_data
from prp_pil import PRP_PIL

# Suppress excessive TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

def setup_dataset():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        return False
    expected = ['Normal', 'Papilledema', 'Pseudopapilledema']
    found = [c for c in expected if os.path.exists(os.path.join(data_dir, c))]
    if len(found) == 0:
        print(f"Error: No Papilledema dataset found in {data_dir}!")
        return False
    print(f"Found Papilledema dataset with classes: {found}")
    return True

# Model builders
def build_resnet50():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(128,128,3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_densenet121():
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(128,128,3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mobilenetv2():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(128,128,3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_efficientnetb0():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(128,128,3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_custom_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,128,3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_multi_class_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    metrics = {'overall_accuracy': accuracy_score(y_true, y_pred), 'per_class_metrics': {}}
    precisions, recalls, specs, f1s = [], [], [], []
    for i in range(n):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp; tn = cm.sum()-(tp+fp+fn)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        precisions.append(prec); recalls.append(rec); specs.append(spec); f1s.append(f1)
        metrics['per_class_metrics'][i] = {'precision': prec, 'recall': rec, 'specificity': spec, 'f1_score': f1, 'support': cm[i,:].sum()}
    metrics['macro_precision'] = np.mean(precisions)
    metrics['macro_recall'] = np.mean(recalls)
    metrics['macro_specificity'] = np.mean(specs)
    metrics['macro_f1'] = np.mean(f1s)
    return metrics

def train_with_fallback(model_builder, X_train, y_train_cat, X_val, y_val_cat,
                        X_test, y_test, class_names, model_name, callbacks):
    global start_time
    try:
        model = model_builder()
        start_time = time.time()
        history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                           epochs=30, batch_size=32, callbacks=callbacks, verbose=0)
        train_time = time.time() - start_time
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        metrics = calculate_multi_class_metrics(y_test, y_pred)
        metrics['training_time'] = train_time
        return {'model_name': model_name, 'model': model, 'metrics': metrics, 'predictions': y_pred, 'history': history.history}
    except Exception as e:
        if 'DNN library initialization failed' in str(e) or 'CuDNN' in str(e):
            print(f"   ⚠️ CuDNN error for {model_name}. Retrying on CPU...")
            tf.config.set_visible_devices([], 'GPU')
            model = model_builder()
            start_time = time.time()
            history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                               epochs=30, batch_size=32, callbacks=callbacks, verbose=0)
            train_time = time.time() - start_time
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            metrics = calculate_multi_class_metrics(y_test, y_pred)
            metrics['training_time'] = train_time
            return {'model_name': model_name, 'model': model, 'metrics': metrics, 'predictions': y_pred, 'history': history.history}
        else:
            raise e

def run_deep_learning_model(model_name, builder, X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    print(f"\nTraining {model_name}...")
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
                 ReduceLROnPlateau(factor=0.5, patience=3)]
    return train_with_fallback(builder, X_train, y_train_cat, X_val, y_val_cat,
                               X_test, y_test, class_names, model_name, callbacks)

def generate_comparison_table():
    print("="*90)
    print("Generating Comparison Table for Papilledema Classification")
    if not setup_dataset():
        return None
    X, y = load_papilledema_dataset(img_size=(128,128), max_samples_per_class=800)
    if X is None:
        return None
    X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_medical_data(
        X, y, test_size=0.15, val_size=0.15, random_state=42
    )
    class_names = le.classes_
    del X, y; gc.collect()
    results = []

    # P-RP-PIL
    try:
        print("\nTraining P-RP-PIL...")
        start = time.time()
        model = PRP_PIL(patch_size=8, stride=2, projection_dim=256, n_hidden=2048,
                       activation='tanh', use_advanced_features=True, use_attention=True,
                       use_gradients=True, regularization=1e-3, random_state=42, batch_size=200)
        model.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = model.predict(X_test)
        metrics = calculate_multi_class_metrics(y_test, y_pred)
        metrics['training_time'] = train_time
        results.append({'model_name': 'P-RP-PIL', 'metrics': metrics, 'predictions': y_pred})
        print(f"  ✓ P-RP-PIL: Acc={metrics['overall_accuracy']:.4f}, Time={train_time:.1f}s")
    except Exception as e:
        print(f"  ✗ P-RP-PIL failed: {e}")

    # DL baselines
    builders = [
        ('ResNet-50', build_resnet50),
        ('DenseNet-121', build_densenet121),
        ('MobileNetV2', build_mobilenetv2),
        ('EfficientNet-B0', build_efficientnetb0),
        ('Custom-CNN', build_custom_cnn)
    ]
    for name, builder in builders:
        try:
            res = run_deep_learning_model(name, builder, X_train, X_val, X_test,
                                         y_train, y_val, y_test, class_names)
            results.append(res)
            print(f"  ✓ {name}: Acc={res['metrics']['overall_accuracy']:.4f}, Time={res['metrics']['training_time']:.1f}s")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    # Generate tables
    print("\n" + "="*90)
    print("📊 COMPARISON TABLE (Tables I–IV, VII)")
    table_data = []
    for r in results:
        m = r['metrics']
        table_data.append({
            'Model': r['model_name'],
            'Accuracy (%)': f"{m['overall_accuracy']*100:.2f}",
            'Training Time (s)': f"{m['training_time']:.1f}",
            'Macro Precision': f"{m['macro_precision']:.4f}",
            'Macro Recall (Sensitivity)': f"{m['macro_recall']:.4f}",
            'Macro Specificity': f"{m['macro_specificity']:.4f}",
            'Macro F1-Score': f"{m['macro_f1']:.4f}"
        })
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    df.to_csv('papilledema_model_comparison_results.csv', index=False)

    # Per-class (Table II, IV)
    print("\n PER-CLASS DETAILS")
    detailed = []
    for r in results:
        for cls_idx, cls_metrics in r['metrics']['per_class_metrics'].items():
            cls_name = le.inverse_transform([cls_idx])[0]
            detailed.append({
                'Model': r['model_name'],
                'Class': cls_name,
                'Precision': cls_metrics['precision'],
                'Recall (Sensitivity)': cls_metrics['recall'],
                'Specificity': cls_metrics['specificity'],
                'F1-Score': cls_metrics['f1_score'],
                'Support': cls_metrics['support']
            })
    detailed_df = pd.DataFrame(detailed)
    print(detailed_df.to_string(index=False))
    detailed_df.to_csv('papilledema_detailed_metrics.csv', index=False)
    return results, df

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    generate_comparison_table()