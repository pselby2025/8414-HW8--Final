# train_model.py (fixed)
import os
import numpy as np
import pandas as pd
from pycaret.classification import setup as c_setup, compare_models, finalize_model, save_model as c_save
from pycaret.clustering import setup as k_setup, create_model as k_create, save_model as k_save
from matplotlib import pyplot as plt

np.random.seed(42)

def generate_synthetic_data(n_per_class=500):
    # --- helper distributions (same idea as before) ---
    def benign(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1,-1], n, p=[0.03,0.97]),
            'URL_Length':        np.random.choice([1,0,-1], n, p=[0.10,0.55,0.35]),
            'Shortining_Service':np.random.choice([1,-1], n, p=[0.08,0.92]),
            'having_At_Symbol':  np.random.choice([1,-1], n, p=[0.04,0.96]),
            'double_slash_redirecting': np.random.choice([1,-1], n, p=[0.05,0.95]),
            'Prefix_Suffix':     np.random.choice([1,-1], n, p=[0.08,0.92]),
            'having_Sub_Domain': np.random.choice([1,0,-1], n, p=[0.10,0.30,0.60]),
            'SSLfinal_State':    np.random.choice([-1,0,1], n, p=[0.05,0.10,0.85]),
            'URL_of_Anchor':     np.random.choice([-1,0,1], n, p=[0.10,0.20,0.70]),
            'Links_in_tags':     np.random.choice([-1,0,1], n, p=[0.10,0.20,0.70]),
            'SFH':               np.random.choice([-1,0,1], n, p=[0.10,0.10,0.80]),
            'Abnormal_URL':      np.random.choice([1,-1], n, p=[0.08,0.92]),
            'profile':           'benign'
        })

    def state_sponsored(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1,-1], n, p=[0.10,0.90]),
            'URL_Length':        np.random.choice([1,0,-1], n, p=[0.30,0.60,0.10]),
            'Shortining_Service':np.random.choice([1,-1], n, p=[0.15,0.85]),
            'having_At_Symbol':  np.random.choice([1,-1], n, p=[0.20,0.80]),
            'double_slash_redirecting': np.random.choice([1,-1], n, p=[0.20,0.80]),
            'Prefix_Suffix':     np.random.choice([1,-1], n, p=[0.75,0.25]),
            'having_Sub_Domain': np.random.choice([1,0,-1], n, p=[0.50,0.40,0.10]),
            'SSLfinal_State':    np.random.choice([-1,0,1], n, p=[0.05,0.15,0.80]),
            'URL_of_Anchor':     np.random.choice([-1,0,1], n, p=[0.30,0.40,0.30]),
            'Links_in_tags':     np.random.choice([-1,0,1], n, p=[0.30,0.40,0.30]),
            'SFH':               np.random.choice([-1,0,1], n, p=[0.50,0.30,0.20]),
            'Abnormal_URL':      np.random.choice([1,-1], n, p=[0.55,0.45]),
            'profile':           'state'
        })

    def organized_crime(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1,-1], n, p=[0.65,0.35]),
            'URL_Length':        np.random.choice([1,0,-1], n, p=[0.70,0.25,0.05]),
            'Shortining_Service':np.random.choice([1,-1], n, p=[0.70,0.30]),
            'having_At_Symbol':  np.random.choice([1,-1], n, p=[0.55,0.45]),
            'double_slash_redirecting': np.random.choice([1,-1], n, p=[0.45,0.55]),
            'Prefix_Suffix':     np.random.choice([1,-1], n, p=[0.65,0.35]),
            'having_Sub_Domain': np.random.choice([1,0,-1], n, p=[0.65,0.25,0.10]),
            'SSLfinal_State':    np.random.choice([-1,0,1], n, p=[0.60,0.30,0.10]),
            'URL_of_Anchor':     np.random.choice([-1,0,1], n, p=[0.60,0.30,0.10]),
            'Links_in_tags':     np.random.choice([-1,0,1], n, p=[0.55,0.30,0.15]),
            'SFH':               np.random.choice([-1,0,1], n, p=[0.70,0.20,0.10]),
            'Abnormal_URL':      np.random.choice([1,-1], n, p=[0.75,0.25]),
            'profile':           'crime'
        })

    def hacktivist(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1,-1], n, p=[0.20,0.80]),
            'URL_Length':        np.random.choice([1,0,-1], n, p=[0.45,0.40,0.15]),
            'Shortining_Service':np.random.choice([1,-1], n, p=[0.40,0.60]),
            'having_At_Symbol':  np.random.choice([1,-1], n, p=[0.25,0.75]),
            'double_slash_redirecting': np.random.choice([1,-1], n, p=[0.25,0.75]),
            'Prefix_Suffix':     np.random.choice([1,-1], n, p=[0.55,0.45]),
            'having_Sub_Domain': np.random.choice([1,0,-1], n, p=[0.45,0.40,0.15]),
            'SSLfinal_State':    np.random.choice([-1,0,1], n, p=[0.35,0.35,0.30]),
            'URL_of_Anchor':     np.random.choice([-1,0,1], n, p=[0.40,0.30,0.30]),
            'Links_in_tags':     np.random.choice([-1,0,1], n, p=[0.35,0.35,0.30]),
            'SFH':               np.random.choice([-1,0,1], n, p=[0.55,0.25,0.20]),
            'Abnormal_URL':      np.random.choice([1,-1], n, p=[0.60,0.40]),
            'profile':           'hacktivist'
        })

    b = benign(n_per_class); s = state_sponsored(n_per_class)
    c = organized_crime(n_per_class); h = hacktivist(n_per_class)

    b['label'] = 0
    for df in (s, c, h):
        df['label'] = 1

    full = pd.concat([b, s, c, h], ignore_index=True)
    return full.sample(frac=1, random_state=42).reset_index(drop=True)

def train():
    os.makedirs('models', exist_ok=True)
    raw = generate_synthetic_data(n_per_class=500)
    raw.to_csv('data_phishing_synth.csv', index=False)

    # --- Classification: drop 'profile' so model won't expect it at inference
    cls_data = raw.drop(columns=['profile'])
    c_setup(data=cls_data, target='label', session_id=42, verbose=False)
    best = compare_models(n_select=1)
    final_clf = finalize_model(best)
    c_save(final_clf, 'models/phishing_url_detector')

    # --- Clustering: also drop 'profile' (keep only features)
    X = raw.drop(columns=['label', 'profile'])
    k_setup(data=X, session_id=42, verbose=False, normalize=True)
    kmeans = k_create('kmeans', num_clusters=3)
    k_save(kmeans, 'models/threat_actor_profiler')

    # (elbow plot code optional; leave as you had it)
    print("Saved: models/phishing_url_detector.pkl and models/threat_actor_profiler.pkl")

if __name__ == "__main__":
    train()
