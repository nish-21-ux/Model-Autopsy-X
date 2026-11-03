"""Pure-black theme: CSS for the page + a Plotly template + chart theme override.
Call apply_theme() once, right after st.set_page_config()."""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

ACCENT, ACCENT2, DANGER, WARN = "#00E5A0", "#7C5CFF", "#FF4D6D", "#FFB020"
COLORWAY = [ACCENT, ACCENT2, "#29B6F6", WARN, DANGER, "#F06EAA", "#B6F36B", "#FFFFFF"]

CSS = f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: #000000; }}
[data-testid="stSidebar"] {{ background: #050505; border-right: 1px solid #161616; }}
h1 {{ background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); -webkit-background-clip: text;
      -webkit-text-fill-color: transparent; font-weight: 800; letter-spacing: .5px; }}
h2 {{ border-left: 4px solid {ACCENT}; padding-left: .6rem; }}
[data-testid="stMetric"] {{ background: #0B0B0B; border: 1px solid #1C1C1C; border-radius: 12px;
      padding: 14px 16px; box-shadow: 0 0 0 1px #000, 0 0 18px rgba(0,229,160,.05); }}
[data-testid="stMetricValue"] {{ color: {ACCENT}; }}
div.stButton > button, div.stDownloadButton > button {{ background: #0B0B0B; color: #EDEDED;
      border: 1px solid {ACCENT}; border-radius: 10px; }}
div.stButton > button:hover, div.stDownloadButton > button:hover {{ background: {ACCENT}; color: #000; }}
[data-testid="stDataFrame"] {{ border: 1px solid #1C1C1C; border-radius: 10px; }}
[data-testid="stExpander"] {{ background: #070707; border: 1px solid #1C1C1C; border-radius: 10px; }}
hr {{ border-color: #161616; }}
code, pre {{ background: #0B0B0B !important; }}
</style>
"""


def _register_template():
    axis = dict(gridcolor="#1A1A1A", zerolinecolor="#2A2A2A", linecolor="#2A2A2A", tickfont=dict(color="#CFCFCF"))
    pio.templates["autopsy_black"] = go.layout.Template(layout=go.Layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000", font=dict(color="#EDEDED"),
        colorway=COLORWAY, xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)"), title=dict(font=dict(color="#FFFFFF")),
        margin=dict(l=40, r=20, t=60, b=40)))
    pio.templates.default = "autopsy_black"


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)
    _register_template()
    # Streamlit's default chart theme overrides custom templates; turn it off globally.
    if not getattr(st.plotly_chart, "_autopsy_patched", False):
        _orig = st.plotly_chart

        def _plotly_chart(fig, *a, **k):
            k.setdefault("theme", None)
            return _orig(fig, *a, **k)

        _plotly_chart._autopsy_patched = True
        st.plotly_chart = _plotly_chart