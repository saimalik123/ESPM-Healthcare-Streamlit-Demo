import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ESPM Demo (Healthcare)", layout="wide")

# ---------------------------
# Helpers: scoring components
# ---------------------------

def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def compute_Ba(login_hour, files_accessed, usb_used, record_access_count, unique_hosts):
    """
    Simple explainable anomaly score (0..1), NOT a full ML model.
    This is intentionally lightweight for demo + presentation.
    If you want Isolation Forest later, we can plug it in.
    """

    # Off-hours: outside 6..18 increases anomaly
    off_hours = 0.0
    if login_hour < 6 or login_hour > 18:
        off_hours = 1.0
    elif login_hour < 8 or login_hour > 17:
        off_hours = 0.5

    # Normalize behavior roughly (tuned for demo)
    files_score = clamp01(files_accessed / 200.0)
    records_score = clamp01(record_access_count / 100.0)
    hosts_score = clamp01(unique_hosts / 20.0)
    usb_score = 1.0 if usb_used else 0.0

    # Weighted anomaly (explainable)
    Ba = (
        0.25 * off_hours +
        0.25 * files_score +
        0.25 * records_score +
        0.15 * hosts_score +
        0.10 * usb_score
    )
    return clamp01(Ba)

def compute_Ic(role, asset_type, login_hour):
    """
    Identity context: role alignment with asset + off-hours penalty.
    """
    # Role-asset expectation (simple healthcare mapping)
    role_asset_matrix = {
        ("Doctor", "EHR"): 0.20,
        ("Doctor", "Lab"): 0.25,
        ("Doctor", "Billing"): 0.60,

        ("Nurse", "EHR"): 0.25,
        ("Nurse", "Lab"): 0.35,
        ("Nurse", "Billing"): 0.70,

        ("Billing", "Billing"): 0.20,
        ("Billing", "EHR"): 0.80,
        ("Billing", "Lab"): 0.75,

        ("Admin", "EHR"): 0.55,
        ("Admin", "Lab"): 0.45,
        ("Admin", "Billing"): 0.40,
    }

    base = role_asset_matrix.get((role, asset_type), 0.60)

    # Add-on: off-hours sensitive access increases identity-context risk
    off_hours_add = 0.0
    if (login_hour < 6 or login_hour > 18) and asset_type in ["EHR", "Lab"]:
        off_hours_add = 0.20

    return clamp01(base + off_hours_add)

def compute_Pl(role):
    """
    Privilege score: more privilege => higher impact
    """
    mapping = {
        "Doctor": 0.50,
        "Nurse": 0.40,
        "Billing": 0.35,
        "Admin": 0.80
    }
    return mapping.get(role, 0.50)

def compute_Ac(asset_type):
    """
    Asset criticality (healthcare context)
    """
    mapping = {
        "EHR": 0.90,
        "Lab": 0.75,
        "Billing": 0.60
    }
    return mapping.get(asset_type, 0.60)

def compute_R(Ic, Ba, Pl, Ac, w1, w2, w3, w4):
    # Normalize weights in case user changes
    s = w1 + w2 + w3 + w4
    if s == 0:
        w1, w2, w3, w4 = 0.25, 0.35, 0.15, 0.25
        s = 1.0
    w1, w2, w3, w4 = w1/s, w2/s, w3/s, w4/s
    R = w1*Ic + w2*Ba + w3*Pl + w4*Ac
    return clamp01(R), (w1, w2, w3, w4)

def priority_label(R):
    if R >= 0.80: return "P0 (Critical)"
    if R >= 0.60: return "P1 (High)"
    if R >= 0.30: return "P2 (Medium)"
    return "P3 (Low)"

# ---------------------------
# UI
# ---------------------------

st.title("Endpoint Security Priority Model (ESPM) — Interactive Demo (Healthcare)")

st.markdown(
    """
This demo shows how ESPM converts **endpoint behavior + identity context + privilege + asset sensitivity**  
into a single **risk score (R)** and SOC priority **P0–P3**.
"""
)

# Prebuilt scenarios (ready-made inputs)
SCENARIOS = {
    "A) Normal clinician access": dict(role="Doctor", login_hour=10, asset_type="EHR", files_accessed=10, usb_used=False, record_access_count=12, unique_hosts=4),
    "B) Off-hours bulk EHR + USB (insider-risk)": dict(role="Billing", login_hour=2, asset_type="EHR", files_accessed=160, usb_used=True, record_access_count=90, unique_hosts=10),
    "C) High network spread + high access": dict(role="Admin", login_hour=3, asset_type="Lab", files_accessed=80, usb_used=False, record_access_count=70, unique_hosts=18),
    "D) Late shift nurse bulk reads": dict(role="Nurse", login_hour=21, asset_type="EHR", files_accessed=35, usb_used=False, record_access_count=40, unique_hosts=7),
}

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Load a scenario (ready-made inputs)")
    scenario_name = st.selectbox("Select scenario", list(SCENARIOS.keys()))
    s = SCENARIOS[scenario_name]

    st.subheader("2) Inputs (editable)")
    role = st.selectbox("Role", ["Doctor", "Nurse", "Billing", "Admin"], index=["Doctor","Nurse","Billing","Admin"].index(s["role"]))
    asset_type = st.selectbox("Asset Type", ["EHR", "Lab", "Billing"], index=["EHR","Lab","Billing"].index(s["asset_type"]))
    login_hour = st.slider("Login hour (0–23)", 0, 23, int(s["login_hour"]))
    files_accessed = st.slider("Files accessed in session", 0, 200, int(s["files_accessed"]))
    record_access_count = st.slider("Patient records accessed", 0, 120, int(s["record_access_count"]))
    unique_hosts = st.slider("Unique hosts contacted", 0, 25, int(s["unique_hosts"]))
    usb_used = st.checkbox("USB used?", value=bool(s["usb_used"]))

with right:
    st.subheader("3) Weight settings (explainable & tunable)")
    st.caption("Weights are policy-tunable. Default emphasizes anomaly + asset criticality.")
    w1 = st.slider("w1 — Identity Context (Ic)", 0.0, 1.0, 0.25, 0.05)
    w2 = st.slider("w2 — Behavioral Anomaly (Ba)", 0.0, 1.0, 0.35, 0.05)
    w3 = st.slider("w3 — Privilege Level (Pl)", 0.0, 1.0, 0.15, 0.05)
    w4 = st.slider("w4 — Asset Criticality (Ac)", 0.0, 1.0, 0.25, 0.05)

    st.subheader("4) Threshold (for SOC triage policy)")
    st.caption("This changes how strict you are in raising high priority.")
    p0_thr = st.slider("P0 threshold", 0.70, 0.95, 0.80, 0.01)
    p1_thr = st.slider("P1 threshold", 0.50, 0.80, 0.60, 0.01)
    p2_thr = st.slider("P2 threshold", 0.20, 0.60, 0.30, 0.01)

# Compute scores
Ba = compute_Ba(login_hour, files_accessed, usb_used, record_access_count, unique_hosts)
Ic = compute_Ic(role, asset_type, login_hour)
Pl = compute_Pl(role)
Ac = compute_Ac(asset_type)
R, (w1n,w2n,w3n,w4n) = compute_R(Ic, Ba, Pl, Ac, w1, w2, w3, w4)

def priority_custom(R):
    if R >= p0_thr: return "P0 (Critical)"
    if R >= p1_thr: return "P1 (High)"
    if R >= p2_thr: return "P2 (Medium)"
    return "P3 (Low)"

prio = priority_custom(R)

st.divider()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ba (Anomaly)", f"{Ba:.2f}")
c2.metric("Ic (Identity)", f"{Ic:.2f}")
c3.metric("Pl (Privilege)", f"{Pl:.2f}")
c4.metric("Ac (Asset)", f"{Ac:.2f}")
c5.metric("Final Risk R", f"{R:.2f}", prio)

# Explainability: factor contributions
st.subheader("Explainability — factor contributions to R")
contrib = pd.DataFrame({
    "Factor": ["Ic", "Ba", "Pl", "Ac"],
    "Weight": [w1n, w2n, w3n, w4n],
    "Score": [Ic, Ba, Pl, Ac],
})
contrib["Contribution"] = contrib["Weight"] * contrib["Score"]
contrib = contrib.sort_values("Contribution", ascending=False)

cc1, cc2 = st.columns([1,1])

with cc1:
    st.dataframe(contrib, use_container_width=True)

with cc2:
    fig = plt.figure()
    plt.bar(contrib["Factor"], contrib["Contribution"])
    plt.title("Contribution to Final Risk R")
    plt.xlabel("Factor")
    plt.ylabel("Weight × Score")
    st.pyplot(fig)

# Threshold demo curve (simple)
st.subheader("Threshold analysis (visual)")
thr = np.array([0.30,0.40,0.50,0.60,0.70,0.80])
# Example curves (demo-friendly): higher threshold => higher precision, lower recall
recall = np.array([0.92,0.90,0.88,0.82,0.75,0.60])
precision = np.array([0.55,0.62,0.71,0.78,0.84,0.90])

fig2 = plt.figure()
plt.plot(thr, recall, label="Recall")
plt.plot(thr, precision, label="Precision")
plt.xlabel("Risk threshold")
plt.ylabel("Metric")
plt.title("Precision–Recall tradeoff (illustrative)")
plt.legend()
st.pyplot(fig2)

st.caption(
    "Note: Curves are illustrative for presentation. If you want, we can compute these from your synthetic dataset in Colab and paste the real plot here."
)