import os
import numpy as np
import pandas as pd
import polars as pl
import kaggle_evaluation.cmi_inference_server

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from joblib import dump, load

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = 'models'

# --------------------- Feature Engineering ---------------------

def extract_features(df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.Series:
    """Create extended statistical features from sensor data."""
    features = {}
    sensor_cols = [c for c in df.columns if any(s in c for s in ['acc_', 'rot_', 'thm_', 'tof_'])]

    for col in sensor_cols:
        data = df[col].dropna().values
        if len(data) == 0:
            data = np.array([0])
        diff = np.diff(data) if len(data) > 1 else np.array([0])
        features.update({
            f'{col}_mean': data.mean(),
            f'{col}_std': data.std(),
            f'{col}_max': data.max(),
            f'{col}_min': data.min(),
            f'{col}_median': np.median(data),
            f'{col}_skew': skew(data),
            f'{col}_kurtosis': kurtosis(data),
            f'{col}_diff_mean': diff.mean(),
            f'{col}_diff_std': diff.std(),
            f'{col}_abs_mean': np.abs(data).mean(),
            f'{col}_num_peaks': len(find_peaks(data)[0]),
            f'{col}_zero_crossings': np.sum(np.diff(np.sign(data)) != 0),
        })

        if 'acc_' in col or 'rot_' in col:
            fft_vals = np.abs(np.fft.rfft(data))
            fft_freq = np.fft.rfftfreq(len(data))
            features[f'{col}_fft_peak_freq'] = fft_freq[np.argmax(fft_vals)]
            features[f'{col}_fft_mean'] = fft_vals.mean()

    demo = demo_df.iloc[0]
    features.update({
        'age': demo['age'],
        'sex': demo['sex'],
        'height_cm': demo['height_cm'],
        'shoulder_to_wrist_cm': demo['shoulder_to_wrist_cm'],
        'elbow_to_wrist_cm': demo['elbow_to_wrist_cm'],
        'handedness': demo['handedness'],
    })
    return pd.Series(features)

# --------------------- Sequence Preprocessing ---------------------

def prepare_sequence(df: pd.DataFrame, seq_length: int = 128) -> np.ndarray:
    """Convert a sequence dataframe into a fixed length 2D array."""
    sensor_cols = [c for c in df.columns if any(s in c for s in ['acc_', 'rot_', 'thm_', 'tof_'])]
    data = df[sensor_cols].fillna(0).to_numpy(dtype=np.float32)
    if len(data) >= seq_length:
        data = data[:seq_length]
    else:
        pad = np.zeros((seq_length - len(data), len(sensor_cols)), dtype=np.float32)
        data = np.vstack([data, pad])
    return data

# --------------------- Model Builders ---------------------

def build_lstm_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------- Training ---------------------

def train_models(k_folds: int = 5, seq_length: int = 128):
    """Train LightGBM, RandomForest and LSTM+CNN using k-fold CV."""
    train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
    demo_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')

    features, sequences, labels = [], [], []
    for seq_id, seq in train_df.groupby('sequence_id'):
        subject = seq['subject'].iloc[0]
        demo = demo_df[demo_df.subject == subject]
        features.append(extract_features(seq, demo))
        sequences.append(prepare_sequence(seq, seq_length))
        labels.append(seq['gesture'].iloc[0])

    X_feat = pd.DataFrame(features).fillna(-1)
    X_seq = np.stack(sequences)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_feat, y)):
        X_tr_feat, X_va_feat = X_feat.iloc[tr_idx], X_feat.iloc[va_idx]
        X_tr_seq, X_va_seq = X_seq[tr_idx], X_seq[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # LightGBM with early stopping to avoid long training when there is no
        # improvement.  The verbose level is also lowered to suppress repeated
        # "No further splits" warnings from LightGBM.
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=500,
            random_state=42,
            verbosity=-1,
        )
        lgb_clf.fit(
            X_tr_feat,
            y_tr,
            eval_set=[(X_va_feat, y_va)],
            early_stopping_rounds=50,
            verbose=False,
        )

        # RandomForest
        rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_clf.fit(X_tr_feat, y_tr)

        # LSTM+CNN
        lstm_model = build_lstm_cnn((seq_length, X_seq.shape[2]), len(le.classes_))
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lstm_model.fit(
            X_tr_seq, y_tr,
            validation_data=(X_va_seq, y_va),
            epochs=10,
            batch_size=32,
            callbacks=[es],
            verbose=0,
        )

        # Save models per fold
        dump(lgb_clf, os.path.join(MODEL_DIR, f'lgb_fold{fold}.joblib'))
        dump(rf_clf, os.path.join(MODEL_DIR, f'rf_fold{fold}.joblib'))
        lstm_model.save(os.path.join(MODEL_DIR, f'lstm_fold{fold}.h5'))

# --------------------- Prediction ---------------------

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame, seq_length: int = 128) -> str:
    """Predict gesture for a single sequence using ensemble of models.

    When running inside the official Kaggle rerun environment, the prediction is
    also sent to the competition's inference server for evaluation."""
    if not os.path.exists(MODEL_DIR):
        train_models()

    le = load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    features = extract_features(sequence.to_pandas(), demographics.to_pandas()).fillna(-1).to_frame().T
    seq_array = prepare_sequence(sequence.to_pandas(), seq_length)[None, :, :]

    fold = 0
    predictions = np.zeros(len(le.classes_))
    while True:
        lgb_path = os.path.join(MODEL_DIR, f'lgb_fold{fold}.joblib')
        rf_path = os.path.join(MODEL_DIR, f'rf_fold{fold}.joblib')
        lstm_path = os.path.join(MODEL_DIR, f'lstm_fold{fold}.h5')
        if not os.path.exists(lgb_path):
            break
        lgb_clf = load(lgb_path)
        rf_clf = load(rf_path)
        lstm_model = load_model(lstm_path, compile=False)
        pred_lgb = lgb_clf.predict_proba(features)[0]
        pred_rf = rf_clf.predict_proba(features)[0]
        pred_lstm = lstm_model.predict(seq_array, verbose=0)[0]
        predictions += (pred_lgb + pred_rf + pred_lstm) / 3
        fold += 1

    predictions /= max(fold, 1)
    pred_label = predictions.argmax()
    gesture = le.inverse_transform([pred_label])[0]

    # During competition reruns, send the result to Kaggle's inference server
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        try:
            inference_server.send_prediction(gesture)
        except AttributeError:
            pass

    return gesture

# --------------------- Kaggle Inference Server ---------------------

# Wrap predict with Kaggle's evaluation server so that predictions are
# automatically sent to the official scoring endpoint during competition
# reruns.  When running locally it will load the public test data and
# provide a gateway for manual testing.
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if __name__ == '__main__':
    # Train models if they do not already exist.
    if not os.path.exists(MODEL_DIR):
        train_models()

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        test_csv = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
        demo_csv = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
        if os.path.exists(test_csv) and os.path.exists(demo_csv):
            inference_server.run_local_gateway(data_paths=(test_csv, demo_csv))
        else:
            print('Local gateway skipped: test data not found.')


