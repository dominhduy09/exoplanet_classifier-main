import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from functools import reduce

# Load and preprocess datasets
def load_kepler_koi(file_path):
    data = pd.read_csv(file_path, comment='#')
    features = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad']
    target = 'koi_disposition'
    data = data[features + [target]].dropna()
    data['mission'] = 'kepler'
    return data, features

def load_tess_toi(file_path):
    data = pd.read_csv(file_path, comment='#')
    feature_map = {
        'pl_orbper': 'koi_period', 'pl_tranmid': 'koi_time0bk', 'pl_trandurh': 'koi_duration',
        'pl_trandep': 'koi_depth', 'pl_rade': 'koi_prad', 'pl_eqt': 'koi_teq', 'pl_insol': 'koi_insol',
        'st_teff': 'koi_steff', 'st_rad': 'koi_srad'
    }
    data = data.rename(columns=feature_map)
    features = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad']
    target = 'tfopwg_disp'
    data = data[features + [target]].dropna()
    data['mission'] = 'tess'
    return data, features

def load_k2_candidates(file_path):
    data = pd.read_csv(file_path, comment='#')
    feature_map = {
        'pl_orbper': 'koi_period', 'pl_rade': 'koi_prad', 'pl_eqt': 'koi_teq', 'pl_insol': 'koi_insol',
        'st_teff': 'koi_steff', 'st_rad': 'koi_srad'
    }
    data = data.rename(columns=feature_map)
    features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad']
    target = 'disposition'
    data = data.rename(columns={'disposition': 'k2_disposition'})
    data['k2_disposition'] = data['k2_disposition'].fillna('FALSE').astype(str).str.upper()
    data = data[features + ['k2_disposition']].dropna()
    data['mission'] = 'k2'
    return data, features

# Unified preprocessing
def preprocess_data(data_list, mission_features):
    all_features = reduce(set.intersection, [set(f) for f in mission_features])
    combined = pd.concat(data_list, ignore_index=True)
    target_map = {
        'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0, 'FALSE': 0,
        'PC': 1, 'FP': 0, 'APC': 1, 'KP': 2, 'CP': 1, 'FA': 0, 'REFUTED': 0
    }
    combined['disposition'] = np.nan
    for data in data_list:
        if 'koi_disposition' in data.columns:
            mask = combined['mission'] == 'kepler'
            combined.loc[mask, 'disposition'] = combined.loc[mask, 'koi_disposition'].astype(str).str.upper().map(target_map)
        elif 'tfopwg_disp' in data.columns:
            mask = combined['mission'] == 'tess'
            combined.loc[mask, 'disposition'] = combined.loc[mask, 'tfopwg_disp'].astype(str).str.upper().map(target_map)
        elif 'k2_disposition' in data.columns:
            mask = combined['mission'] == 'k2'
            combined.loc[mask, 'disposition'] = combined.loc[mask, 'k2_disposition'].map(target_map)
    combined['disposition'] = combined['disposition'].fillna(0)
    combined = combined[list(all_features) + ['mission', 'disposition']]
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].mean())
    combined = pd.get_dummies(combined, columns=['mission'], drop_first=True)
    X = combined.drop('disposition', axis=1)
    y = combined['disposition'].astype(int)
    return X, y, X.columns.tolist()

# Train model
def train_model(X, y, n_estimators=50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    report = classification_report(y_test, y_pred, target_names=['FALSE', 'CANDIDATE', 'CONFIRMED'])
    joblib.dump(model, 'exoplanet_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model, scaler, accuracy, auc, report, X.columns.tolist()

# Main pipeline
def main():
    koi_data, koi_features = load_kepler_koi('kepler_koi.csv')
    toi_data, toi_features = load_tess_toi('tess_toi.csv')
    k2_data, k2_features = load_k2_candidates('k2_candidates.csv')
    X, y, _ = preprocess_data([koi_data, toi_data, k2_data], [koi_features, toi_features, k2_features])
    model, scaler, accuracy, auc, report, features = train_model(X, y, n_estimators=50)
    joblib.dump(features, 'features.pkl')  # Save features for backend
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print("Classification Report:\n", report)
    print(f"Trained Features: {features}")

if __name__ == "__main__":
    main()