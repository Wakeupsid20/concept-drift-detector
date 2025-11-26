import streamlit as st
import pandas as pd
import plotly.express as px
from drift_utils import analyze_drift

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Concept Drift Detector",
    page_icon="🧠",
    layout="wide",
)

# ---------- CUSTOM CSS (gradient + glass cards + buttons) ----------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f1c2c, #141E30);
        color: #f5f5f5;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
    }
    .glass-card {
        background: rgba(15, 20, 35, 0.80);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(18px);
    }
    .metric-card {
        background: rgba(15, 20, 35, 0.90);
        border-radius: 16px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(135deg,#6366F1,#EC4899);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border: 1px solid rgba(148, 163, 184, 0.8);
        color: #e5e7eb;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-green { background: rgba(22,163,74,0.12); color: #bbf7d0; }
    .status-yellow { background: rgba(234,179,8,0.12); color: #facc15; }
    .status-red { background: rgba(239,68,68,0.12); color: #fecaca; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙ Control Panel")
st.sidebar.markdown(
    """
    Steps:
    1. Upload **reference** & **current** feature data  
    2. (Optional) Upload **prediction** scores  
    3. Tune PSI thresholds  
    4. Hit **Analyze** 🚀  
    """
)

psi_stable = st.sidebar.slider("Max PSI for Stable", 0.00, 0.50, 0.10, 0.01)
psi_moderate = st.sidebar.slider("Max PSI for Moderate", 0.05, 0.80, 0.25, 0.01)

# ---------- HEADER ----------
with st.container():
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap;">
                <div>
                    <span class="badge">REAL-TIME MODEL HEALTH</span>
                    <h1 style="margin-top:0.6rem; margin-bottom:0.3rem;">Concept & Prediction Drift Monitor 🧠📉</h1>
                    <p style="max-width:650px; color:#d1d5db; font-size:0.95rem;">
                        Upload training vs live data to monitor <b>feature drift</b> and <b>prediction drift</b>
                        using <b>PSI</b>, <b>KL-divergence</b>, and the <b>KS test</b>. 
                        Built to feel like an internal FAANG model monitoring dashboard.
                    </p>
                </div>
                <div style="text-align:right; min-width:200px;">
                    <p style="font-size:0.8rem; color:#9ca3af; margin-bottom:0.25rem;">Designed for 🔍 production ML systems</p>
                    <p style="font-size:0.9rem; color:#e5e7eb; margin:0;">
                        Built by <b>Data Scientist on Duty</b> 👩‍💻
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")  # spacing

# ---------- TABS ----------
tab_features, tab_preds, tab_about = st.tabs(
    ["📊 Feature Drift", "📈 Prediction Drift", "ℹ About Concept Drift"]
)

# ==========================================================
# TAB 1: FEATURE DRIFT
# ==========================================================
with tab_features:
    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader("📂 Reference Dataset (training CSV)", type=["csv"], key="ref_features")
    with col2:
        curr_file = st.file_uploader("📂 Current Dataset (production CSV)", type=["csv"], key="curr_features")

    if ref_file and curr_file:
        df_ref = pd.read_csv(ref_file)
        df_curr = pd.read_csv(curr_file)

        # Data preview
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👀 Feature Data Preview")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.caption("Reference (training)")
            st.dataframe(df_ref.head(), use_container_width=True)
        with subcol2:
            st.caption("Current (production)")
            st.dataframe(df_curr.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Numeric columns & feature selection
        numeric_cols = df_ref.select_dtypes(include="number").columns.tolist()

        st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.subheader("🎛 Feature Selection")
        selected_cols = st.multiselect(
            "Choose which numeric features to include in drift analysis:",
            options=numeric_cols,
            default=numeric_cols,
            help="Only numeric columns can be analyzed using PSI / KL / KS.",
            key="feature_cols",
        )
        if len(selected_cols) == 0:
            st.warning("⚠ Select at least one feature.")
        st.markdown("</div>", unsafe_allow_html=True)

        analyze_features = st.button("🚀 Analyze Feature Drift")

        if analyze_features:
            if not selected_cols:
                st.warning("Pick at least one numeric feature.")
            else:
                # Run drift analysis
                results = analyze_drift(df_ref, df_curr, numeric_cols=selected_cols)

                # Recompute severity using custom PSI thresholds
                severity_list = []
                for psi in results["psi"]:
                    if psi < psi_stable:
                        severity_list.append("Stable")
                    elif psi < psi_moderate:
                        severity_list.append("Moderate")
                    else:
                        severity_list.append("Severe")
                results["severity"] = severity_list

                # ---------- Metrics strip ----------
                overall_drift = min(results["psi"].mean() * 100, 100)
                severe_count = int((results["severity"] == "Severe").sum())
                n_features = len(results)

                st.markdown(
                    """
                    <div class="glass-card" style="margin-top:1rem;">
                        <h3 style="margin-bottom:0.8rem;">📡 Feature Drift Snapshot</h3>
                    """,
                    unsafe_allow_html=True,
                )
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Overall Drift Score</div>
                            <div class="metric-value">{overall_drift:.1f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Features Analyzed</div>
                            <div class="metric-value">{n_features}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Severe Drift Features</div>
                            <div class="metric-value">{severe_count}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ---------- Overall Status Block ----------
                if severe_count == 0 and overall_drift < 10:
                    status = ("🟢 Stable", "status-green",
                              "Your feature distributions look healthy. No significant drift detected.")
                elif severe_count <= 1 and overall_drift < 25:
                    status = ("🟡 Monitor", "status-yellow",
                              "Mild feature drift detected. Not urgent, but monitor these features over time.")
                else:
                    status = ("🔴 Retrain", "status-red",
                              "Significant feature drift detected across key inputs. Model retraining is recommended.")

                title_text, pill_class, explanation = status

                st.markdown(
                    f"""
                    <div style="margin-top:0.9rem; margin-bottom:0.5rem;">
                        <span class="status-pill {pill_class}" style="font-size:1rem; margin-bottom:0.7rem;">
                            {title_text}
                        </span>
                        <p style="color:#d1d5db; margin-top:0.8rem; font-size:0.92rem;">
                            {explanation}
                        </p>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---------- Feature-wise table ----------
                st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
                st.subheader("📋 Feature-wise Drift Report")
                st.dataframe(results, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # ---------- Plots ----------
                template = "plotly_dark"
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    fig_psi = px.bar(
                        results,
                        x="feature",
                        y="psi",
                        color="severity",
                        title="PSI per Feature",
                        template=template,
                    )
                    st.plotly_chart(fig_psi, use_container_width=True)
                with pcol2:
                    fig_ks = px.bar(
                        results,
                        x="feature",
                        y="ks_stat",
                        title="KS Statistic per Feature",
                        template=template,
                    )
                    st.plotly_chart(fig_ks, use_container_width=True)

                # ---------- Download report ----------
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Feature Drift Report (CSV)",
                    data=csv_bytes,
                    file_name="feature_drift_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # ---------- Plain-English explanation ----------
                with st.expander("🧾 Plain-English Feature Drift Explanation"):
                    for _, row in results.sort_values("psi", ascending=False).iterrows():
                        st.markdown(
                            f"- **{row['feature']}** → PSI = `{row['psi']:.3f}` "
                            f"({row['severity']}). KS p-value ≈ `{row['ks_p_value']:.2e}`."
                        )
    else:
        st.info("Upload both reference and current feature CSV files to analyze feature drift.")


# ==========================================================
# TAB 2: PREDICTION DRIFT
# ==========================================================
with tab_preds:
    st.markdown(
        """
        <div class="glass-card">
            <h3>📈 Prediction Drift</h3>
            <p style="color:#d1d5db; font-size:0.92rem;">
                Monitor how your <b>model outputs</b> change over time. 
                Even if feature drift looks small, prediction drift can reveal 
                calibration issues or silent failures in your model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    colp1, colp2 = st.columns(2)
    with colp1:
        train_pred_file = st.file_uploader(
            "📂 Training Predictions (e.g. y_pred_train.csv)", type=["csv"], key="train_pred"
        )
    with colp2:
        prod_pred_file = st.file_uploader(
            "📂 Production Predictions (e.g. y_pred_prod.csv)", type=["csv"], key="prod_pred"
        )

    if train_pred_file and prod_pred_file:
        df_train_pred = pd.read_csv(train_pred_file)
        df_prod_pred = pd.read_csv(prod_pred_file)

        st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.subheader("🔍 Prediction Data Preview")
        subp1, subp2 = st.columns(2)
        with subp1:
            st.caption("Training predictions")
            st.dataframe(df_train_pred.head(), use_container_width=True)
        with subp2:
            st.caption("Production predictions")
            st.dataframe(df_prod_pred.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        pred_cols = df_train_pred.select_dtypes(include="number").columns.tolist()

        st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.subheader("🎛 Prediction Columns")
        selected_pred_cols = st.multiselect(
            "Select prediction columns to analyze (e.g. y_pred, prob_class1):",
            options=pred_cols,
            default=pred_cols,
            key="pred_cols",
        )
        if len(selected_pred_cols) == 0:
            st.warning("⚠ Select at least one prediction column.")
        st.markdown("</div>", unsafe_allow_html=True)

        analyze_preds = st.button("🚀 Analyze Prediction Drift")

        if analyze_preds:
            if not selected_pred_cols:
                st.warning("Pick at least one prediction column.")
            else:
                pred_results = analyze_drift(
                    df_train_pred, df_prod_pred, numeric_cols=selected_pred_cols
                )

                # severity based on PSI thresholds
                severity_list = []
                for psi in pred_results["psi"]:
                    if psi < psi_stable:
                        severity_list.append("Stable")
                    elif psi < psi_moderate:
                        severity_list.append("Moderate")
                    else:
                        severity_list.append("Severe")
                pred_results["severity"] = severity_list

                # metrics
                overall_pred_drift = min(pred_results["psi"].mean() * 100, 100)
                severe_pred_count = int((pred_results["severity"] == "Severe").sum())
                n_pred_cols = len(pred_results)

                st.markdown(
                    """
                    <div class="glass-card" style="margin-top:1rem;">
                        <h3 style="margin-bottom:0.8rem;">📡 Prediction Drift Snapshot</h3>
                    """,
                    unsafe_allow_html=True,
                )
                mp1, mp2, mp3 = st.columns(3)
                with mp1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Overall Pred Drift</div>
                            <div class="metric-value">{overall_pred_drift:.1f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with mp2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Pred Columns</div>
                            <div class="metric-value">{n_pred_cols}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with mp3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Severe Pred Drift</div>
                            <div class="metric-value">{severe_pred_count}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # status for prediction drift
                if severe_pred_count == 0 and overall_pred_drift < 10:
                    p_status = ("🟢 Stable", "status-green",
                                "Model outputs look stable. No major prediction drift detected.")
                elif severe_pred_count <= 1 and overall_pred_drift < 25:
                    p_status = ("🟡 Monitor", "status-yellow",
                                "Mild prediction drift detected. Monitor business metrics and calibration.")
                else:
                    p_status = ("🔴 Retrain / Recalibrate", "status-red",
                                "Significant prediction drift detected. Investigate data, labels, and model retraining.")

                p_title, p_pill_class, p_explanation = p_status

                st.markdown(
                    f"""
                    <div style="margin-top:0.9rem; margin-bottom:0.5rem;">
                        <span class="status-pill {p_pill_class}" style="font-size:1rem; margin-bottom:0.7rem;">
                            {p_title}
                        </span>
                        <p style="color:#d1d5db; margin-top:0.8rem; font-size:0.92rem;">
                            {p_explanation}
                        </p>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # table
                st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
                st.subheader("📋 Prediction Drift Report")
                st.dataframe(pred_results, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # plots
                template = "plotly_dark"
                pp1, pp2 = st.columns(2)
                with pp1:
                    fig_pred_psi = px.bar(
                        pred_results,
                        x="feature",
                        y="psi",
                        color="severity",
                        title="PSI per Prediction Column",
                        template=template,
                    )
                    st.plotly_chart(fig_pred_psi, use_container_width=True)
                with pp2:
                    fig_pred_ks = px.bar(
                        pred_results,
                        x="feature",
                        y="ks_stat",
                        title="KS Statistic per Prediction Column",
                        template=template,
                    )
                    st.plotly_chart(fig_pred_ks, use_container_width=True)

                # download
                pred_csv_bytes = pred_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Prediction Drift Report (CSV)",
                    data=pred_csv_bytes,
                    file_name="prediction_drift_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # explanation
                with st.expander("🧾 Plain-English Prediction Drift Explanation"):
                    for _, row in pred_results.sort_values("psi", ascending=False).iterrows():
                        st.markdown(
                            f"- **{row['feature']}** → PSI = `{row['psi']:.3f}` "
                            f"({row['severity']}). KS p-value ≈ `{row['ks_p_value']:.2e}`."
                        )
    else:
        st.info("Upload training & production prediction CSVs to analyze prediction drift.")


# ==========================================================
# TAB 3: ABOUT
# ==========================================================
with tab_about:
    st.markdown(
        """
        <div class="glass-card">
            <h3>What is Concept & Prediction Drift?</h3>
            <p style="color:#e5e7eb; font-size:0.94rem;">
            <b>Concept drift</b> happens when the relationship between inputs and target changes over time.
            <b>Feature drift</b> tracks how input distributions move. 
            <b>Prediction drift</b> monitors how your model outputs shift between training and production.
            </p>
            <p style="color:#e5e7eb; font-size:0.9rem;">
            This app combines both, using:
            </p>
            <ul style="color:#e5e7eb; font-size:0.9rem;">
                <li><b>PSI (Population Stability Index)</b> for distribution shift magnitude</li>
                <li><b>KL-Divergence</b> for information gain / loss between distributions</li>
                <li><b>KS Test</b> for statistical significance of shifts</li>
            </ul>
            <p style="color:#9ca3af; font-size:0.88rem; margin-top:0.4rem;">
            Use it as a blueprint for production-grade model monitoring in credit risk, ads, recommenders, and fraud systems.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

