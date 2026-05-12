import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SegmentIQ · Customer Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("deployment/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("deployment/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("deployment/pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("deployment/cluster_mapping.pkl", "rb") as f:
        cluster_mapping = pickle.load(f)
    return kmeans, scaler, pca, cluster_mapping

kmeans, scaler, pca, cluster_mapping = load_models()

# ── Design tokens ──────────────────────────────────────────────────────────────
ACCENT       = "#7C6FFF"
ACCENT_ALT   = "#4ECDC4"
DANGER       = "#FF6B6B"
BG_BASE      = "#0D0D14"
BG_SURFACE   = "#13131F"
BG_CARD      = "#1A1A2E"
BG_CARD_ALT  = "#1E1E30"
BORDER       = "rgba(124,111,255,0.18)"
TEXT_PRIMARY = "#F0F0FA"
TEXT_MUTED   = "#8888AA"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG_BASE} !important;
    color: {TEXT_PRIMARY};
    font-family: 'DM Sans', sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{
    padding: 2rem 2.5rem 3rem;
    max-width: 1280px;
}}

[data-testid="stSidebar"] {{
    background: {BG_SURFACE} !important;
    border-right: 1px solid {BORDER};
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 22px 20px 18px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 6px;
}}
.sidebar-brand .hex {{ font-size: 26px; line-height: 1; color: {ACCENT}; }}
.sidebar-brand .name {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.02em;
}}
.sidebar-brand .sub {{
    font-size: 10px;
    color: {TEXT_MUTED};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    line-height: 1;
    margin-top: 1px;
}}

h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: .25rem !important;
}}
h2 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
h3 {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: {TEXT_MUTED};
    letter-spacing: .06em;
    text-transform: uppercase;
}}
p, li {{ color: {TEXT_MUTED}; font-size: 14.5px; line-height: 1.65; }}

.kpi-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1;
    min-width: 160px;
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_ALT});
    border-radius: 14px 14px 0 0;
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
}}
.kpi-sub {{ font-size: 12px; color: {TEXT_MUTED}; margin-top: 4px; }}

.seg-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}}
.seg-card h4 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: {ACCENT};
    margin: 0 0 6px;
}}
.seg-card p {{ color: {TEXT_MUTED}; font-size: 13.5px; margin: 0; }}

.result-luxury {{
    background: linear-gradient(135deg, rgba(124,111,255,.12), rgba(78,205,196,.08));
    border: 1px solid rgba(124,111,255,.4);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
}}
.result-budget {{
    background: linear-gradient(135deg, rgba(255,107,107,.10), rgba(255,142,83,.08));
    border: 1px solid rgba(255,107,107,.35);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
}}
.result-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .10em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 8px;
}}
.result-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
.result-rec {{
    font-size: 14px;
    color: {TEXT_MUTED};
    margin-top: 12px;
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 9px !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(124,111,255,.15) !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 9px !important;
    color: {TEXT_PRIMARY} !important;
}}
label {{ color: {TEXT_MUTED} !important; font-size: 13px !important; font-weight: 500 !important; }}

.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, #5B50DD) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: .02em !important;
    transition: opacity .18s, transform .18s !important;
}}
.stButton > button:hover {{
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, {ACCENT}, #5B50DD) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
}}

[data-testid="stPopover"] button {{
    background: transparent !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_MUTED} !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}}

.fancy-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {BORDER}, transparent);
    margin: 28px 0;
}}
.section-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .10em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 8px;
}}
.insight-row {{ display: flex; gap: 14px; margin-top: 18px; flex-wrap: wrap; }}
.insight-chip {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 13px;
    color: {TEXT_MUTED};
}}
.insight-chip span {{ color: {TEXT_PRIMARY}; font-weight: 600; }}
.footer {{
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid {BORDER};
    font-size: 12px;
    color: {TEXT_MUTED};
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = None
if "predicted_segment" not in st.session_state:
    st.session_state.predicted_segment = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="hex">⬡</span>
        <div>
            <div class="name">SegmentIQ</div>
            <div class="sub">Customer Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Overview", "Predict Segment", "Insights & Trends", "Contact"],
        icons=["grid-1x2", "cpu", "bar-chart-line", "envelope"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "8px 12px"},
            "icon": {"font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "500",
                "border-radius": "8px",
                "margin-bottom": "2px",
                "color": TEXT_MUTED,
            },
            "nav-link-selected": {
                "background-color": "rgba(124,111,255,.18)",
                "color": ACCENT,
                "font-weight": "600",
            },
        },
    )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    with st.popover("ℹ️  How it works"):
        st.markdown(f"""
        <p style='font-size:13px; color:{TEXT_MUTED}; margin-bottom:10px'>
        The model uses <b style='color:{TEXT_PRIMARY}'>KMeans Clustering</b> with PCA dimensionality reduction.
        </p>
        <p style='font-size:13px; color:{TEXT_MUTED}'>
        <b style='color:{TEXT_PRIMARY}'>1. Input</b> — Enter customer features on the Predict page.<br><br>
        <b style='color:{TEXT_PRIMARY}'>2. Predict</b> — The pipeline (Scaler → PCA → KMeans) assigns a cluster label.<br><br>
        <b style='color:{TEXT_PRIMARY}'>3. Insights</b> — Visualise spending breakdowns and get tailored recommendations.
        </p>
        """, unsafe_allow_html=True)

    st.markdown(f"<p style='font-size:11px;color:{TEXT_MUTED};padding:8px 0 0 4px'>v1.0 · KMeans · PCA</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if selected == "Overview":
    st.markdown("<div class='section-label'>Platform Overview</div>", unsafe_allow_html=True)
    st.title("Customer Segmentation\nIntelligence")
    st.markdown(f"<p style='font-size:15px;color:{TEXT_MUTED};margin-top:-6px;margin-bottom:28px'>Machine-learning powered segmentation to sharpen targeting, reduce churn, and grow revenue.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Model</div>
            <div class="kpi-value">KMeans</div>
            <div class="kpi-sub">Unsupervised clustering</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Segments</div>
            <div class="kpi-value">2</div>
            <div class="kpi-sub">Luxury · Budget</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Features</div>
            <div class="kpi-value">10</div>
            <div class="kpi-sub">Demographic + spend</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Reduction</div>
            <div class="kpi-value">PCA</div>
            <div class="kpi-sub">Dimensionality reduction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Customer Segments</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(f"""
        <div class="seg-card">
            <h4>💎 Luxury Shopper</h4>
            <p>High-income customers with frequent, large-ticket purchases across premium categories. Respond well to VIP memberships, exclusive early-access offers, and personalised rewards programs.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="seg-card">
            <h4>🛒 Budget-Conscious Buyer</h4>
            <p>Value-driven customers with moderate spend and higher price sensitivity. Best engaged through flash sales, bundle discounts, and targeted promotions that emphasise value and savings.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Pipeline</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, step, icon, desc in zip(
        [c1, c2, c3, c4],
        ["Input Features", "StandardScaler", "PCA", "KMeans Predict"],
        ["📋", "⚖️", "📐", "🎯"],
        ["10 customer attributes", "Normalise distributions", "Reduce to 2 components", "Assign cluster label"],
    ):
        with col:
            st.markdown(f"""
            <div class="seg-card" style="text-align:center;padding:18px 14px;">
                <div style="font-size:26px;margin-bottom:8px">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;color:{TEXT_PRIMARY};margin-bottom:4px">{step}</div>
                <div style="font-size:12px;color:{TEXT_MUTED}">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"<div class='footer'>Built with Streamlit · KMeans Clustering · Plotly</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT SEGMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Predict Segment":
    st.markdown("<div class='section-label'>Segmentation Engine</div>", unsafe_allow_html=True)
    st.title("Predict Customer\nSegment")
    st.markdown(f"<p style='font-size:15px;color:{TEXT_MUTED};margin-top:-4px;margin-bottom:28px'>Enter customer attributes to classify them into the appropriate segment.</p>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f"<div class='section-label' style='margin-bottom:12px'>Demographic</div>", unsafe_allow_html=True)
        income   = st.number_input("Annual Income (€)", min_value=1000, step=500, value=50000)
        age      = st.slider("Age", min_value=18, max_value=100, value=40)
        kidhome  = st.selectbox("Children at Home", list(range(6)), index=0)
        teenhome = st.selectbox("Teenagers at Home", list(range(6)), index=0)

    with col_right:
        st.markdown(f"<div class='section-label' style='margin-bottom:12px'>Spending & Behaviour</div>", unsafe_allow_html=True)
        mnt_wines       = st.slider("Wine Spend (€)", min_value=0, max_value=1000, step=10, value=200)
        mnt_meat        = st.slider("Meat Spend (€)", min_value=0, max_value=1000, step=10, value=150)
        mnt_fish        = st.slider("Fish Spend (€)", min_value=0, max_value=1000, step=10, value=50)
        web_visits      = st.slider("Web Visits / Month", min_value=0, max_value=100, step=1, value=5)
        total_campaigns = st.slider("Campaigns Responded To", min_value=0, max_value=4, step=1, value=1)
        purchase_freq   = st.number_input("Purchase Frequency", min_value=0, step=1, value=10)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⬡  Run Segmentation Model", use_container_width=False)

    if predict_btn:
        recommendations = {
            "Luxury Shopper":         "Offer VIP memberships, personalised shopping experiences, and exclusive early-access discounts.",
            "Budget-Conscious Buyer": "Run flash sales, bundle deals, and promote value-driven product combinations.",
        }
        user_data   = np.array([[income, kidhome, teenhome, mnt_wines, mnt_meat, mnt_fish, web_visits, age, total_campaigns, purchase_freq]])
        user_scaled = scaler.transform(user_data)
        user_pca    = pca.transform(user_scaled)
        cluster     = kmeans.predict(user_pca)[0]
        segment     = cluster_mapping[cluster]

        is_luxury  = segment == "Luxury Shopper"
        banner_cls = "result-luxury" if is_luxury else "result-budget"
        emoji      = "💎" if is_luxury else "🛒"

        st.markdown(f"""
        <div class="{banner_cls}">
            <div class="result-label">Predicted Segment</div>
            <div class="result-title">{emoji} {segment}</div>
            <div class="result-rec">{recommendations[segment]}</div>
        </div>
        """, unsafe_allow_html=True)

        total_spend = mnt_wines + mnt_meat + mnt_fish
        st.markdown(f"""
        <div class="insight-row">
            <div class="insight-chip">Income <span>€{income:,}</span></div>
            <div class="insight-chip">Total Spend <span>€{total_spend:,}</span></div>
            <div class="insight-chip">Age <span>{age}</span></div>
            <div class="insight-chip">Web Visits <span>{web_visits}/mo</span></div>
            <div class="insight-chip">Campaigns <span>{total_campaigns}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.user_inputs = {
            "income": income, "mnt_wines": mnt_wines, "mnt_meat": mnt_meat,
            "mnt_fish": mnt_fish, "web_visits": web_visits, "age": age,
        }
        st.session_state.predicted_segment = segment


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS & TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Insights & Trends":
    st.markdown("<div class='section-label'>Analytics</div>", unsafe_allow_html=True)
    st.title("Spending Insights\n& Trends")

    if st.session_state.predicted_segment is None:
        st.markdown(f"""
        <div class="seg-card" style="text-align:center;padding:40px 24px;">
            <div style="font-size:36px;margin-bottom:12px">📊</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:6px">No prediction yet</div>
            <div style="font-size:13.5px;color:{TEXT_MUTED}">Run the segmentation model first — then return here to explore insights.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        seg = st.session_state.predicted_segment
        inp = st.session_state.user_inputs
        is_luxury = seg == "Luxury Shopper"
        seg_color = ACCENT if is_luxury else DANGER
        emoji = "💎" if is_luxury else "🛒"

        total_spend = inp["mnt_wines"] + inp["mnt_meat"] + inp["mnt_fish"]
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Segment</div>
                <div class="kpi-value" style="font-size:1.3rem;color:{seg_color}">{emoji} {seg}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Annual Income</div>
                <div class="kpi-value">€{inp['income']:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Spend</div>
                <div class="kpi-value">€{total_spend:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Age</div>
                <div class="kpi-value">{inp['age']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        plotly_layout = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color=TEXT_MUTED),
            margin=dict(l=16, r=16, t=40, b=16),
        )

        col_bar, col_pie = st.columns(2, gap="large")

        with col_bar:
            st.markdown(f"<div class='section-label'>Spend by Category</div>", unsafe_allow_html=True)
            categories = ["Wine", "Meat", "Fish", "Web Visits"]
            values     = [inp["mnt_wines"], inp["mnt_meat"], inp["mnt_fish"], inp["web_visits"]]
            bar_colors = [ACCENT, ACCENT_ALT, "#FF6B6B", "#FFD93D"]
            fig_bar = go.Figure(go.Bar(
                x=categories, y=values,
                marker_color=bar_colors, marker_line_width=0,
                hovertemplate="%{x}: €%{y}<extra></extra>",
            ))
            fig_bar.update_layout(
                **plotly_layout,
                yaxis=dict(gridcolor="rgba(255,255,255,.06)", zeroline=False),
                xaxis=dict(showgrid=False),
                bargap=0.35,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with col_pie:
            st.markdown(f"<div class='section-label'>Food Spend Breakdown</div>", unsafe_allow_html=True)
            pie_df = pd.DataFrame({
                "Category": ["Wine", "Meat", "Fish"],
                "Spending": [inp["mnt_wines"], inp["mnt_meat"], inp["mnt_fish"]],
            })
            fig_pie = px.pie(
                pie_df, names="Category", values="Spending",
                hole=0.55,
                color_discrete_sequence=[ACCENT, ACCENT_ALT, "#FF6B6B"],
            )
            fig_pie.update_traces(
                textfont_color=TEXT_PRIMARY,
                hovertemplate="%{label}: €%{value}<extra></extra>",
            )
            fig_pie.update_layout(
                **plotly_layout,
                showlegend=True,
                legend=dict(font=dict(color=TEXT_MUTED)),
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>Customer Profile Radar</div>", unsafe_allow_html=True)

        max_vals = {"mnt_wines": 1000, "mnt_meat": 1000, "mnt_fish": 1000, "web_visits": 100, "income": 200000}
        labels   = ["Wine", "Meat", "Fish", "Web Visits", "Income"]
        raw      = [inp["mnt_wines"], inp["mnt_meat"], inp["mnt_fish"], inp["web_visits"], inp["income"]]
        normed   = [v / max_vals[k] for v, k in zip(raw, ["mnt_wines", "mnt_meat", "mnt_fish", "web_visits", "income"])]
        normed  += [normed[0]]
        labels  += [labels[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=normed, theta=labels, fill="toself",
            fillcolor="rgba(124,111,255,.15)",
            line=dict(color=ACCENT, width=2),
            hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
        ))
        fig_radar.update_layout(
            **plotly_layout,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)", tickfont=dict(color=TEXT_MUTED)),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CONTACT
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Contact":
    st.markdown("<div class='section-label'>Get in Touch</div>", unsafe_allow_html=True)
    st.title("Contact")
    st.markdown(f"<p style='font-size:15px;color:{TEXT_MUTED};margin-top:-4px;margin-bottom:28px'>Questions, feedback, or collaboration enquiries — drop a message below.</p>", unsafe_allow_html=True)

    col_form, col_info = st.columns([2, 1], gap="large")

    with col_form:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        from datetime import datetime

        def save_to_gsheet(name, email, message):
            scope  = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds  = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
            client = gspread.authorize(creds)
            sheet  = client.open_by_key("1Jwyum5tV-rphQd5-YG4heh-1cnYx766oNuKC9-RRpkM").worksheet("Sheet1")
            sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, email, message])

        with st.form("contact_form"):
            name       = st.text_input("Name")
            user_email = st.text_input("Email")
            message    = st.text_area("Message", height=140)
            submitted  = st.form_submit_button("Send Message")

            if submitted:
                if name and user_email and message:
                    try:
                        save_to_gsheet(name, user_email, message)
                        st.success("Message sent — I'll get back to you soon.")
                    except Exception as e:
                        st.error("Something went wrong. Please try again.")
                        st.exception(e)
                else:
                    st.warning("Please fill in all fields.")

    with col_info:
        st.markdown(f"""
        <div class="seg-card">
            <h4>About This Project</h4>
            <p>An end-to-end customer segmentation pipeline built with KMeans clustering, PCA, and Streamlit — designed for portfolio demonstration and real-world analytics.</p>
        </div>
        <div class="seg-card" style="margin-top:14px">
            <h4>Stack</h4>
            <p>Python · Scikit-learn · Streamlit · Plotly · Pandas · Google Sheets API</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div class='footer'>SegmentIQ · Built by Hoshang Sheth</div>", unsafe_allow_html=True)
