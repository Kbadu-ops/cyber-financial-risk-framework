"""
Cyber-Financial Risk Scoring and Anomaly Detection Framework
Victor Badu | MS Business Analytics | Chartered Accountant (CA) | FMVA
Live demo: Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber-Financial Risk Framework | Victor Badu",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SEED = 42
np.random.seed(SEED)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin: 0 0 0.3rem; }
    .main-header p  { color: #cbd5e1; font-size: 0.95rem; margin: 0; }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.78rem; color: #64748b; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #1e293b; }
    .metric-card .sub   { font-size: 0.75rem; margin-top: 2px; }
    .risk-critical { color: #dc2626; font-weight: 700; }
    .risk-high     { color: #ea580c; font-weight: 700; }
    .risk-medium   { color: #d97706; font-weight: 600; }
    .risk-low      { color: #65a30d; font-weight: 600; }
    .risk-minimal  { color: #16a34a; font-weight: 600; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e293b;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px; margin: 1.2rem 0 1rem;
    }
    .alert-box {
        padding: 0.8rem 1rem; border-radius: 8px;
        margin: 6px 0; font-size: 0.85rem;
    }
    .alert-critical { background: #fee2e2; border-left: 4px solid #dc2626; color: #7f1d1d; }
    .alert-high     { background: #fff7ed; border-left: 4px solid #ea580c; color: #7c2d12; }
    .alert-medium   { background: #fefce8; border-left: 4px solid #d97706; color: #713f12; }
    .stDataFrame { font-size: 0.82rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_or_generate_data(n=20000, fraud_rate=0.04):
    """
    Generate synthetic transaction data modeled on PaySim patterns.
    In production this connects to agency financial data systems via API.
    Structure mirrors PaySim columns: amount, balance fields, transaction type.
    """
    np.random.seed(SEED)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    tx_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
    programs = ['Medicare', 'Medicaid', 'Federal Grants', 'Treasury', 'HUD Housing']

    # Legitimate transactions
    legit_bal = np.random.lognormal(9, 1.5, n_legit)
    legit_amt = np.minimum(legit_bal * np.random.uniform(0.05, 0.6, n_legit),
                           legit_bal)
    legit = pd.DataFrame({
        'transaction_type'  : np.random.choice(tx_types, n_legit,
                               p=[0.20, 0.20, 0.35, 0.15, 0.10]),
        'program'           : np.random.choice(programs, n_legit),
        'amount'            : np.random.lognormal(8.2, 1.1, n_legit).clip(100, 400000),
        'oldbalanceOrg'     : legit_bal,
        'newbalanceOrig'    : legit_bal - legit_amt,
        'oldbalanceDest'    : np.random.lognormal(8, 1.5, n_legit),
        'newbalanceDest'    : np.random.lognormal(8.1, 1.5, n_legit),
        'hour_of_day'       : np.random.choice(range(8, 18), n_legit),
        'day_of_week'       : np.random.choice(range(0, 5), n_legit),
        'vendor_age_days'   : np.random.randint(365, 4000, n_legit),
        'prior_claims_30d'  : np.random.poisson(4, n_legit),
        'eligibility_ok'    : np.random.choice([0, 1], n_legit, p=[0.02, 0.98]),
        'access_anomaly'    : np.random.choice([0, 1], n_legit, p=[0.97, 0.03]),
        'is_fraud'          : 0
    })
    legit['newbalanceDest'] = legit['oldbalanceDest'] + legit['amount'] * np.random.uniform(0.8, 1.0, n_legit)

    # Fraudulent transactions — inject anomalous patterns
    fraud_bal = np.random.lognormal(10, 1.8, n_fraud)
    fraud = pd.DataFrame({
        'transaction_type'  : np.random.choice(['TRANSFER', 'CASH_OUT'], n_fraud,
                               p=[0.52, 0.48]),
        'program'           : np.random.choice(programs, n_fraud),
        'amount'            : fraud_bal * np.random.uniform(0.85, 1.0, n_fraud),
        'oldbalanceOrg'     : fraud_bal,
        'newbalanceOrig'    : np.zeros(n_fraud),          # fully drained
        'oldbalanceDest'    : np.zeros(n_fraud),           # shell account
        'newbalanceDest'    : np.zeros(n_fraud),           # balance unchanged
        'hour_of_day'       : np.random.choice(
                               list(range(0, 7)) + list(range(20, 24)), n_fraud),
        'day_of_week'       : np.random.choice(range(0, 7), n_fraud),
        'vendor_age_days'   : np.random.randint(1, 90, n_fraud),
        'prior_claims_30d'  : np.random.poisson(18, n_fraud),
        'eligibility_ok'    : np.random.choice([0, 1], n_fraud, p=[0.45, 0.55]),
        'access_anomaly'    : np.random.choice([0, 1], n_fraud, p=[0.25, 0.75]),
        'is_fraud'          : 1
    })

    df = pd.concat([legit, fraud], ignore_index=True)\
           .sample(frac=1, random_state=SEED).reset_index(drop=True)
    df['transaction_id'] = ['TX-' + str(i).zfill(7) for i in range(len(df))]
    return df


@st.cache_data(show_spinner=False)
def engineer_and_train(_df):
    """Feature engineering + model training (cached so it runs once)."""
    d = _df.copy()

    # PaySim-aligned features
    d['log_amount']             = np.log1p(d['amount'])
    d['amount_zscore']          = (d['amount'] - d['amount'].mean()) / d['amount'].std()
    d['large_amount']           = (d['amount'] > d['amount'].quantile(0.95)).astype(int)
    d['balance_drop_orig']      = ((d['oldbalanceOrg'] > 0) &
                                   (d['newbalanceOrig'] == 0)).astype(int)
    d['balance_unchanged_dest'] = ((d['oldbalanceDest'] == d['newbalanceDest']) &
                                   (d['amount'] > 0)).astype(int)
    d['balance_diff_orig']      = np.abs(d['oldbalanceOrg'] - d['newbalanceOrig'] - d['amount'])
    d['zero_dest_before']       = (d['oldbalanceDest'] == 0).astype(int)
    d['is_off_hours']           = ((d['hour_of_day'] < 8) | (d['hour_of_day'] > 17)).astype(int)
    d['is_weekend']             = (d['day_of_week'] >= 5).astype(int)
    d['is_transfer_cashout']    = d['transaction_type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    d['new_vendor']             = (d['vendor_age_days'] < 90).astype(int)
    d['high_claim_velocity']    = (d['prior_claims_30d'] > d['prior_claims_30d'].quantile(0.90)).astype(int)
    le = LabelEncoder()
    d['tx_type_encoded']        = le.fit_transform(d['transaction_type'])
    d['composite_risk_flag']    = (d['balance_drop_orig'] + d['balance_unchanged_dest'] +
                                   d['is_off_hours'] + d['zero_dest_before'] +
                                   d['is_transfer_cashout'] + d['large_amount'] +
                                   d['high_claim_velocity'])

    FEATURES = ['log_amount', 'amount_zscore', 'large_amount',
                'balance_drop_orig', 'balance_unchanged_dest',
                'balance_diff_orig', 'zero_dest_before',
                'is_off_hours', 'is_weekend', 'is_transfer_cashout',
                'new_vendor', 'high_claim_velocity',
                'composite_risk_flag', 'tx_type_encoded',
                'oldbalanceOrg', 'newbalanceOrig']

    X = d[FEATURES]
    y = d['is_fraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest (unsupervised layer)
    iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=SEED, n_jobs=-1)
    iso.fit(X_scaled)
    anomaly_raw = iso.decision_function(X_scaled)
    d['anomaly_score'] = 100 * (1 - (anomaly_raw - anomaly_raw.min()) /
                                     (anomaly_raw.max() - anomaly_raw.min()))

    FEATURES_V2 = FEATURES + ['anomaly_score']
    X2 = d[FEATURES_V2]
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=0.25, random_state=SEED, stratify=y)

    # Random Forest (supervised layer)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    d_test = d.loc[X_test.index].copy()
    d_test['risk_score']     = (y_prob * 100).round(1)
    d_test['fraud_predicted'] = y_pred
    d_test['actual_fraud']   = y_test.values

    def band(s):
        if s >= 80: return 'CRITICAL'
        if s >= 60: return 'HIGH'
        if s >= 40: return 'MEDIUM'
        if s >= 20: return 'LOW'
        return 'MINIMAL'

    d_test['risk_band'] = d_test['risk_score'].apply(band)

    feat_imp = pd.Series(rf.feature_importances_, index=FEATURES_V2).sort_values(ascending=False)

    metrics = {
        'auc'       : round(roc_auc_score(y_test, y_prob), 4),
        'ap'        : round(average_precision_score(y_test, y_prob), 4),
        'roc'       : roc_curve(y_test, y_prob),
        'pr'        : precision_recall_curve(y_test, y_prob),
        'cm'        : confusion_matrix(y_test, y_pred),
        'report'    : classification_report(y_test, y_pred,
                       target_names=['Legitimate', 'Fraudulent'], output_dict=True),
    }
    return d_test, feat_imp, metrics, d


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Framework Controls")
    program_filter = st.selectbox(
        "Federal Program",
        ["All Programs", "Medicare", "Medicaid", "Federal Grants", "Treasury", "HUD Housing"]
    )
    risk_threshold = st.slider("Risk Score Threshold for Review", 40, 90, 60, 5)
    n_transactions = st.selectbox("Dataset Size", [10000, 20000, 50000], index=1)
    st.markdown("---")
    st.markdown("### 📋 About This Framework")
    st.markdown("""
    **Developer:** Victor Badu  
    **Credentials:** MS Business Analytics | CA | FMVA  
    **Purpose:** EB-2 National Interest Waiver — Demonstration of the Cyber-Financial Risk Scoring and Anomaly Detection Framework  

    **Dataset basis:** Modeled on PaySim synthetic financial transaction patterns (Lopez-Rojas et al., 2016)

    **Target deployment:** Medicare, Medicaid, Federal Grant, Treasury payment systems
    """)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("📁 [GitHub Repository](https://github.com/Kbadu-ops)")
    st.markdown("📄 [White Paper (SSRN)](https://ssrn.com)")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data and training fraud detection models..."):
    raw_df = load_or_generate_data(n=n_transactions)
    results, feat_imp, metrics, full_df = engineer_and_train(raw_df)

# Apply program filter
if program_filter != "All Programs":
    results_view = results[results['program'] == program_filter]
else:
    results_view = results.copy()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>🛡️ Cyber-Financial Risk Scoring & Anomaly Detection Framework</h1>
  <p>Victor Badu &nbsp;|&nbsp; MS Business Analytics &nbsp;|&nbsp; Chartered Accountant (CA) &nbsp;|&nbsp; FMVA
  &nbsp;&nbsp;·&nbsp;&nbsp; Federal Payment Integrity Program &nbsp;·&nbsp; EB-2 NIW Demonstration</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Dashboard",
    "🔬 Model Performance",
    "📈 Exploratory Analysis",
    "🔍 Transaction Inspector",
    "⚠️ Active Alerts"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    flagged     = results_view[results_view['risk_score'] >= risk_threshold]
    dollar_risk = flagged['amount'].sum()
    true_caught = flagged['actual_fraud'].sum()
    total_fraud = results_view['actual_fraud'].sum()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Transactions Scanned", f"{len(results_view):,}")
    with col2:
        st.metric("Flagged for Review",
                  f"{len(flagged):,}",
                  f"{len(flagged)/len(results_view)*100:.1f}% of total")
    with col3:
        st.metric("Model ROC-AUC", f"{metrics['auc']:.4f}", "Excellent > 0.90")
    with col4:
        st.metric("Est. $ at Risk", f"${dollar_risk:,.0f}")
    with col5:
        precision = true_caught / len(flagged) if len(flagged) > 0 else 0
        st.metric("Detection Precision", f"{precision*100:.1f}%", "In flagged set")

    st.markdown('<div class="section-header">Risk Band Distribution & Program Exposure</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        band_order  = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        band_colors = ['#16a34a', '#65a30d', '#d97706', '#ea580c', '#dc2626']
        band_counts = results_view['risk_band'].value_counts().reindex(band_order, fill_value=0)

        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            band_counts.values, labels=band_counts.index,
            colors=band_colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9}
        )
        ax.set_title('Transaction Risk Band Distribution', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        prog_risk = results_view[results_view['risk_score'] >= risk_threshold]\
                        .groupby('program')['amount'].sum().sort_values() / 1e6
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(prog_risk.index, prog_risk.values, color='#dc2626', alpha=0.82)
        ax.set_title('Estimated $ at Risk by Federal Program ($M)', fontweight='bold', fontsize=11)
        ax.set_xlabel('$ Million')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, prog_risk.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'${val:.2f}M', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Real-Time Risk Score Feed</div>',
                unsafe_allow_html=True)
    sample_plot = results_view.sample(min(400, len(results_view)), random_state=SEED).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors_s = ['#dc2626' if f else '#2563eb' for f in sample_plot['actual_fraud']]
    sizes_s  = [60 if f else 15 for f in sample_plot['actual_fraud']]
    ax.scatter(sample_plot.index, sample_plot['risk_score'],
               c=colors_s, s=sizes_s, alpha=0.6)
    ax.axhline(risk_threshold, color='orange', linestyle='--', lw=2,
               label=f'Review threshold ({risk_threshold})')
    ax.axhline(80, color='red', linestyle='--', lw=1.5, label='Critical threshold (80)')
    ax.fill_between(range(len(sample_plot)), risk_threshold, 100, alpha=0.05, color='red')
    legend_els = [
        plt.scatter([], [], c='#dc2626', s=60, label='Confirmed fraud'),
        plt.scatter([], [], c='#2563eb', s=15, label='Legitimate'),
        plt.Line2D([0],[0], color='orange', linestyle='--', label=f'Review threshold ({risk_threshold})'),
        plt.Line2D([0],[0], color='red',    linestyle='--', label='Critical threshold (80)'),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc='upper left')
    ax.set_title('Transaction Risk Scores — Live Monitoring Feed', fontweight='bold')
    ax.set_xlabel('Transaction Sequence')
    ax.set_ylabel('Risk Score (0–100)')
    ax.set_ylim(0, 108)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Model Performance Metrics</div>',
                unsafe_allow_html=True)

    rep = metrics['report']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC Score",       f"{metrics['auc']:.4f}")
    col2.metric("Average Precision",   f"{metrics['ap']:.4f}")
    col3.metric("Fraud Recall",        f"{rep['Fraudulent']['recall']:.4f}")
    col4.metric("Fraud Precision",     f"{rep['Fraudulent']['precision']:.4f}")

    col_left, col_right = st.columns(2)

    with col_left:
        fpr, tpr, _ = metrics['roc']
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, color='#2563eb', lw=2.5,
                label=f"ROC Curve (AUC = {metrics['auc']:.4f})")
        ax.plot([0,1],[0,1], color='gray', linestyle='--', lw=1, label='Random baseline')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        prec, rec, _ = metrics['pr']
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(rec, prec, color='#dc2626', lw=2.5,
                label=f"PR Curve (AP = {metrics['ap']:.4f})")
        ax.fill_between(rec, prec, alpha=0.1, color='#dc2626')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred: Legit', 'Pred: Fraud'],
                    yticklabels=['Actual: Legit', 'Actual: Fraud'])
        ax.set_title('Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_fi:
        st.markdown('<div class="section-header">Top Feature Importances</div>',
                    unsafe_allow_html=True)
        top_fi = feat_imp.head(12).sort_values()
        fi_colors = ['#dc2626' if v > 0.08 else '#d97706' if v > 0.04 else '#2563eb'
                     for v in top_fi.values]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.barh(top_fi.index, top_fi.values, color=fi_colors, alpha=0.85)
        ax.set_title('Feature Importance (Top 12)', fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i, v in enumerate(top_fi.values):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EXPLORATORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Exploratory Data Analysis — PaySim Patterns</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log1p(raw_df[raw_df.is_fraud==0]['amount']), bins=60,
                alpha=0.7, color='#2563eb', label='Legitimate', density=True)
        ax.hist(np.log1p(raw_df[raw_df.is_fraud==1]['amount']), bins=60,
                alpha=0.7, color='#dc2626', label='Fraudulent', density=True)
        ax.set_title('Amount Distribution (log scale)', fontweight='bold')
        ax.set_xlabel('Log(Amount + 1)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        type_fraud = raw_df.groupby('transaction_type')['is_fraud'].mean() * 100
        bar_cols = ['#dc2626' if v > 1 else '#2563eb' for v in type_fraud.values]
        ax.bar(type_fraud.index, type_fraud.values, color=bar_cols, alpha=0.85)
        ax.set_title('Fraud Rate by Transaction Type', fontweight='bold')
        ax.set_xlabel('Transaction Type')
        ax.set_ylabel('Fraud Rate (%)')
        ax.tick_params(axis='x', rotation=15)
        for i, v in enumerate(type_fraud.values):
            ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        fraud_hour = raw_df.groupby('hour_of_day')['is_fraud'].mean() * 100
        ax.bar(fraud_hour.index, fraud_hour.values,
               color=['#dc2626' if h < 8 or h > 17 else '#2563eb'
                      for h in fraud_hour.index])
        ax.axvspan(-0.5, 7.5,  alpha=0.08, color='red', label='Off-hours zone')
        ax.axvspan(17.5, 23.5, alpha=0.08, color='red')
        ax.set_title('Fraud Rate by Hour of Day', fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Fraud Rate (%)')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        raw_df['balance_drop'] = ((raw_df['oldbalanceOrg'] > 0) &
                                  (raw_df['newbalanceOrig'] == 0)).astype(int)
        drain_fraud = raw_df.groupby('balance_drop')['is_fraud'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Balance NOT drained', 'Balance FULLY drained'],
               drain_fraud.values, color=['#2563eb', '#dc2626'], alpha=0.85)
        ax.set_title('Fraud Rate: Sender Balance Drained?', fontweight='bold')
        ax.set_ylabel('Fraud Rate (%)')
        for i, v in enumerate(drain_fraud.values):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center',
                    fontweight='bold', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — TRANSACTION INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Transaction-Level Risk Score Output</div>',
                unsafe_allow_html=True)

    band_filter = st.multiselect(
        "Filter by Risk Band",
        ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'],
        default=['CRITICAL', 'HIGH']
    )

    display_cols = ['transaction_id', 'program', 'transaction_type', 'amount',
                    'risk_score', 'risk_band', 'actual_fraud',
                    'balance_drop_orig', 'is_off_hours', 'composite_risk_flag']

    filtered = results_view[results_view['risk_band'].isin(band_filter)]\
                   [display_cols].sort_values('risk_score', ascending=False)

    def color_risk(val):
        colors = {'CRITICAL': 'background-color: #fee2e2; color: #7f1d1d',
                  'HIGH'    : 'background-color: #fff7ed; color: #7c2d12',
                  'MEDIUM'  : 'background-color: #fefce8; color: #713f12',
                  'LOW'     : 'background-color: #f0fdf4; color: #14532d',
                  'MINIMAL' : 'background-color: #f0fdf4; color: #14532d'}
        return colors.get(val, '')

    styled = filtered.style\
        .format({'amount': '${:,.0f}', 'risk_score': '{:.1f}'})\
        .map(color_risk, subset=['risk_band'])

    st.dataframe(styled, use_container_width=True, height=420)

    st.markdown(f"**Showing {len(filtered):,} transactions** | "
                f"Total flagged $ at risk: **${filtered['amount'].sum():,.0f}**")

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download flagged transactions as CSV",
        csv, "flagged_transactions.csv", "text/csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ACTIVE ALERTS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Active System Alerts</div>',
                unsafe_allow_html=True)

    critical = results_view[results_view['risk_score'] >= 80].sort_values(
        'risk_score', ascending=False).head(5)

    st.markdown("#### 🔴 Critical Alerts (Risk Score ≥ 80)")
    if len(critical) == 0:
        st.info("No critical alerts under current filter.")
    for _, row in critical.iterrows():
        signals = []
        if row.get('balance_drop_orig', 0): signals.append("sender balance fully drained")
        if row.get('balance_unchanged_dest', 0): signals.append("receiver balance unchanged")
        if row.get('is_off_hours', 0): signals.append("off-hours transaction")
        if row.get('composite_risk_flag', 0) >= 4: signals.append("multiple simultaneous risk flags")
        signal_str = " · ".join(signals) if signals else "multiple anomalous patterns detected"
        st.markdown(f"""
        <div class="alert-box alert-critical">
            <strong>⚠ {row['transaction_id']} — Risk Score {row['risk_score']:.1f} / 100</strong><br>
            Program: {row['program']} &nbsp;|&nbsp; Amount: ${row['amount']:,.0f}
            &nbsp;|&nbsp; Type: {row['transaction_type']}<br>
            <em>Signals: {signal_str}</em>
        </div>
        """, unsafe_allow_html=True)

    high = results_view[(results_view['risk_score'] >= 60) &
                        (results_view['risk_score'] < 80)]\
               .sort_values('risk_score', ascending=False).head(5)

    st.markdown("#### 🟠 High Risk Alerts (Risk Score 60–79)")
    for _, row in high.iterrows():
        st.markdown(f"""
        <div class="alert-box alert-high">
            <strong>◈ {row['transaction_id']} — Risk Score {row['risk_score']:.1f} / 100</strong><br>
            Program: {row['program']} &nbsp;|&nbsp; Amount: ${row['amount']:,.0f}
            &nbsp;|&nbsp; Type: {row['transaction_type']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Framework Summary")
    total_f  = results_view['actual_fraud'].sum()
    detected = results_view[(results_view['risk_score'] >= risk_threshold) &
                            (results_view['actual_fraud'] == 1)].shape[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fraud in Dataset",  f"{total_f:,}")
    col2.metric("Fraud Detected by Model", f"{detected:,}")
    col3.metric("Detection Rate",          f"{detected/total_f*100:.1f}%" if total_f > 0 else "N/A")

    st.markdown(f"""
    ---
    **Framework:** Cyber-Financial Risk Scoring and Anomaly Detection Framework  
    **Developer:** Victor Badu | MS Business Analytics | Chartered Accountant | FMVA  
    **Model:** Isolation Forest (unsupervised) + Random Forest (supervised) | ROC-AUC: **{metrics['auc']:.4f}**  
    **Policy alignment:** PIIA 2019 · FISMA · OMB Circular A-123 · GAO High-Risk List  
    **GitHub:** [github.com/Kbadu-ops](https://github.com/Kbadu-ops)
    """)
