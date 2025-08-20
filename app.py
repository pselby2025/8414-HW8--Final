# app.py  (safe clustering + attribution + existing functionality)
import os
import re
import time
import pandas as pd
import streamlit as st

from pycaret.classification import load_model as c_load_model, predict_model as c_predict
from pycaret.clustering import load_model as k_load_model, predict_model as k_predict

from genai_prescriptions import generate_prescription

st.set_page_config(page_title="GenAI-Powered Phishing SOAR", layout="wide")


# ---------------------------
# Utilities
# ---------------------------

def _safe_cluster_id(val):
    """
    Convert PyCaret clustering output to an integer cluster id.
    Handles ints and strings like 'Cluster 2' or '2'.
    """
    if isinstance(val, (int, float)):
        try:
            return int(val)
        except Exception:
            return 0
    if isinstance(val, str):
        m = re.search(r"\d+", val)
        if m:
            return int(m.group(0))
        # sometimes it's already '2' as a string
        try:
            return int(val)
        except Exception:
            return 0
    return 0


@st.cache_resource
def load_assets():
    """
    Load the classification and clustering models once.
    Also optionally expose a feature-importance plot if you saved one.
    """
    clf = c_load_model("models/phishing_url_detector") if os.path.exists("models/phishing_url_detector.pkl") else None
    clu = k_load_model("models/threat_actor_profiler") if os.path.exists("models/threat_actor_profiler.pkl") else None
    plot = "models/feature_importance.png" if os.path.exists("models/feature_importance.png") else None
    return clf, clu, plot


@st.cache_resource
def infer_cluster_mapping(_clu_model, csv_path: str = "data_phishing_synth.csv"):
    """
    Infer a mapping from numeric cluster id -> threat actor label
    by running the clustering model on your training data and taking
    the majority 'profile' within each cluster.

    NOTE: _clu_model is prefixed with '_' so Streamlit won't try to hash it.
    """
    if _clu_model is None or not os.path.exists(csv_path):
        # Reasonable fallback mapping if we can't learn it
        # You can edit these labels if desired.
        return {
            0: "Organized Cybercrime",
            1: "State-Sponsored",
            2: "Hacktivist",
        }

    df = pd.read_csv(csv_path)
    # Your synthetic data had columns including 'profile' and 'label'
    # Keep only features the clusterer expects (drop label/profile if present)
    features = df.drop(columns=[c for c in ["label", "profile"] if c in df.columns])

    preds = k_predict(_clu_model, data=features).copy()
    # PyCaret typically adds a column named "Cluster"
    # Normalize to int ids
    if "Cluster" not in preds.columns:
        # Some versions may call it "Label" – handle gracefully
        cluster_col = "Label" if "Label" in preds.columns else None
    else:
        cluster_col = "Cluster"

    if not cluster_col:
        # Can't infer, return default
        return {0: "Organized Cybercrime", 1: "State-Sponsored", 2: "Hacktivist"}

    # Attach back to profiles to majority-vote
    preds["cluster_id"] = preds[cluster_col].apply(_safe_cluster_id)
    if "profile" in df.columns:
        preds["profile"] = df["profile"].values
    else:
        # If profile is missing, default fallback
        return {0: "Organized Cybercrime", 1: "State-Sponsored", 2: "Hacktivist"}

    mapping = {}
    for cid, chunk in preds.groupby("cluster_id"):
        # majority label within the cluster
        label = chunk["profile"].value_counts().idxmax()
        # normalize names a bit
        label_norm = {
            "crime": "Organized Cybercrime",
            "state": "State-Sponsored",
            "hacktivist": "Hacktivist",
            "benign": "Benign"
        }.get(str(label).lower(), str(label))
        mapping[int(cid)] = label_norm

    # Ensure we have some mapping for 0..2
    for cid in range(0, 3):
        mapping.setdefault(cid, ["Organized Cybercrime", "State-Sponsored", "Hacktivist"][cid % 3])

    return mapping


ACTOR_DESCRIPTIONS = {
    "Organized Cybercrime": "Financially motivated groups that rapidly iterate on credential harvest, phishing kits, and monetization (fraud, ransomware).",
    "State-Sponsored": "Strategic, persistent operations focused on espionage and long-term access; often disciplined infra and tradecraft.",
    "Hacktivist": "Ideologically motivated actors seeking publicity and disruption; campaigns may be noisy and episodic.",
    "Benign": "Signals do not indicate malicious activity."
}


# ---------------------------
# Load models once
# ---------------------------
clf_model, cluster_model, feature_plot = load_assets()
if clf_model is None:
    st.error("Classification model not found. Re-run training or check 'models/phishing_url_detector.pkl'.")
    st.stop()

# Build mapping once (cached)
cluster_id_to_actor = infer_cluster_mapping(cluster_model)


# ---------------------------
# Sidebar: input + provider
# ---------------------------
with st.sidebar:
    st.title("URL Feature Input")

    form_values = {
        "url_length": st.select_slider("URL Length", ["Short", "Normal", "Long"], "Long"),
        "ssl_state": st.select_slider("SSL Status", ["Trusted", "Suspicious", "None"], "Suspicious"),
        "sub_domain": st.select_slider("Sub-domain", ["None", "One", "Many"], "One"),
        "prefix_suffix": st.checkbox("Has Prefix/Suffix", True),
        "has_ip": st.checkbox("Uses IP Address", False),
        "short_service": st.checkbox("Is Shortened", False),
        "at_symbol": st.checkbox("Has '@'", False),
        "abnormal_url": st.checkbox("Is Abnormal", True),
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI"])  # keep your two providers
    submitted = st.button("Analyze & Respond", use_container_width=True, type="primary")


# ---------------------------
# Main UI
# ---------------------------
st.title("GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Provide URL features in the sidebar and click 'Analyze' to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot, caption="Which features the model weighs most heavily.")
    st.stop()


# Build a single-row dataframe from the UI choices (feature engineering mirrors your training)
input_dict = {
    "having_IP_Address": 1 if form_values["has_ip"] else -1,
    "URL_Length": -1 if form_values["url_length"] == "Short" else (0 if form_values["url_length"] == "Normal" else 1),
    "Shortining_Service": 1 if form_values["short_service"] else -1,
    "having_At_Symbol": 1 if form_values["at_symbol"] else -1,
    "double_slash_redirecting": -1,
    "Prefix_Suffix": 1 if form_values["prefix_suffix"] else -1,
    "having_Sub_Domain": -1 if form_values["sub_domain"] == "None" else (0 if form_values["sub_domain"] == "One" else 1),
    "SSLfinal_State": -1 if form_values["ssl_state"] == "None" else (0 if form_values["ssl_state"] == "Suspicious" else 1),
    "Abnormal_URL": 1 if form_values["abnormal_url"] else -1,
    "URL_of_Anchor": 0,
    "Links_in_tags": 0,
    "SFH": 0,
}
input_df = pd.DataFrame([input_dict])

# For the “Visual Insights” bars later
risk_scores = {
    "Bad SSL": 25 if input_dict["SSLfinal_State"] < 1 else 0,
    "Abnormal URL": 20 if input_dict["Abnormal_URL"] == 1 else 0,
    "Prefix/Suffix": 15 if input_dict["Prefix_Suffix"] == 1 else 0,
    "Shortened URL": 15 if input_dict["Shortining_Service"] == 1 else 0,
    "Complex Sub-domain": 10 if input_dict["having_Sub_Domain"] == 1 else 0,
    "Long URL": 10 if input_dict["URL_Length"] == 1 else 0,
    "Uses IP Address": 5 if input_dict["having_IP_Address"] == 1 else 0,
}
risk_df = pd.DataFrame(list(risk_scores.items()), columns=["Feature", "Risk Contribution"]).sort_values(
    "Risk Contribution", ascending=False
)


# ---------------------------
# Execute Playbook
# ---------------------------
with st.status("Executing SOAR playbook...", expanded=True) as status:
    st.write("Step 1: Predictive Analysis...")
    time.sleep(0.5)
    pred = c_predict(clf_model, data=input_df)
    is_malicious = bool(pred["prediction_label"].iloc[0] == 1)

    st.write(f"Step 2: Verdict is {'MALICIOUS' if is_malicious else 'BENIGN'}.")

    predicted_actor = None
    predicted_cluster_id = None

    if is_malicious and cluster_model is not None:
        # Only attribute if malicious and we have a clustering model
        st.write("Step 2.5: Threat attribution...")
        try:
            c_pred = k_predict(cluster_model, data=input_df)
            # take the first row's cluster
            cluster_val = c_pred.iloc[0].get("Cluster", c_pred.iloc[0].get("Label", 0))
            predicted_cluster_id = _safe_cluster_id(cluster_val)
            predicted_actor = cluster_id_to_actor.get(predicted_cluster_id, "Unknown")
        except Exception as e:
            # Do not break the flow if attribution fails
            predicted_actor = None
            predicted_cluster_id = None
            st.warning(f"Attribution step skipped: {e}")

    # Step 3 (GenAI) – unchanged logic
    if is_malicious:
        st.write(f"Step 3: Engaging {genai_provider} for prescription...")
        try:
            prescription = generate_prescription(genai_provider, input_dict)
            status.update(label="Playbook Executed!", state="complete", expanded=False)
        except Exception as e:
            st.error("Step 3 failed.")
            st.exception(e)
            prescription = {"recommended_actions": [], "communication_draft": "", "error": str(e)}
            status.update(label="Playbook finished with errors", state="error", expanded=True)
    else:
        prescription = None
        status.update(label="Benign. Analysis Complete.", state="complete", expanded=False)


# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Analysis Summary", "Visual Insights", "Prescriptive Plan", "Threat Attribution"])

with tab1:
    st.subheader("Verdict and Key Findings")
    if is_malicious:
        st.error("**Prediction: Malicious Phishing URL**")
        st.metric("Malicious Confidence Score", f"{pred['prediction_score'].iloc[0]:.2%}")
    else:
        st.success("**Prediction: Benign URL**")
        st.metric("Benign Confidence Score", f"{1 - pred['prediction_score'].iloc[0]:.2%}")

with tab2:
    st.subheader("Visual Analysis")
    st.write("#### Risk Contribution by Feature")
    st.bar_chart(risk_df.set_index("Feature"))
    if feature_plot:
        st.write("#### Model Feature Importance (Global)")
        st.image(feature_plot)

with tab3:
    st.subheader("Actionable Response Plan")
    if prescription:
        st.success("A prescriptive response plan has been generated by the AI.")
        st.write("#### Recommended Actions")
        actions = prescription.get("recommended_actions", [])
        if actions:
            for i, action in enumerate(actions, 1):
                st.markdown(f"**{i}.** {action}")
        st.write("#### Communication Draft")
        st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        if prescription.get("error"):
            st.caption(f"Note: {prescription['error']}")
    else:
        st.info("URL was benign. No plan needed.")

with tab4:
    st.subheader("Threat Attribution")
    if not is_malicious:
        st.info("Attribution is only performed for malicious URLs.")
    elif predicted_actor:
        st.metric("Predicted Cluster", f"{predicted_cluster_id}")
        st.metric("Suspected Actor", predicted_actor)
        st.write("#### Profile")
        st.write(ACTOR_DESCRIPTIONS.get(predicted_actor, "No description available."))
    else:
        st.info("Attribution not available for this sample.")
