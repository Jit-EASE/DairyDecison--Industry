# Development_Decison_Dairy_ireland.py
# Irish Dairy Decision Intelligence — Full Intelligence Build
# Jit

# ---------- Std / Core ----------
import os, io, math
from pathlib import Path
from functools import lru_cache
from datetime import datetime

# ---------- Data / Numerics ----------
import numpy as np
import pandas as pd

# ---------- Additional Viz/API ----------
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# ---------- UI ----------

import streamlit as st
st.set_page_config(
    page_title="Irish Dairy Decision Intelligence",
    layout="wide"
)
st.markdown("""
<style>
/* Master viewport container — enforces all UI inside transparent video frame */
#main-viewport {
    position: relative !important;
    padding-top: 0 !important;
    margin-top: 0 !important;
    width: 100vw;
    height: 100vh;              /* Full video height */
    overflow-y: scroll;         /* Scroll inside the video frame */
    overflow-x: hidden;
    scrollbar-width: none;      /* Firefox */
    scroll-behavior: smooth;
    overscroll-behavior-y: contain;
}
#main-viewport::-webkit-scrollbar {
    width: 0px;                 /* Chrome/Safari — invisible scrollbar */
    background: transparent;
}

/* Place all Streamlit blocks inside the viewport */
.block-container {
    position: relative !important;
    margin-top: 0 !important;
    padding-top: 3vh !important;
    z-index: 10 !important;
    backdrop-filter: blur(2px) !important;
}

/* Glass-only utility for tab panels */
.glass-only {
    background: rgba(255,255,255,0.25) !important;
    backdrop-filter: blur(2px) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 25px rgba(0,0,0,0.15) !important;
}

/* Force content clipping inside video frame for all tabs */
#main-viewport .block-container {
    max-height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    mask-image: linear-gradient(to bottom, black 85%, transparent 100%);
    -webkit-mask-image: linear-gradient(to bottom, black 85%, transparent 100%);
}

/* Hide any content that tries to go beyond the frame */
html, body {
    overflow: hidden !important;
}

/* Tab content clipping for each tab */
.tab-clip {
    position: relative !important;
    max-height: calc(100vh - 5rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    pointer-events: auto !important;
}

/* Dynamic transparency per tab */
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(255,255,255,0.30) !important;
    backdrop-filter: blur(18px) !important;
    border-bottom: 2px solid rgba(0,200,255,0.55) !important;
}
[data-testid="stTabs"] button[aria-selected="false"] {
    background: rgba(255,255,255,0.10) !important;
    backdrop-filter: blur(6px) !important;
}
</style>
<div id='main-viewport'>

""", unsafe_allow_html=True)
st.markdown("""
<script>
document.addEventListener("DOMContentLoaded", function() {
    setTimeout(function() {
        const viewport = document.getElementById("main-viewport");
        const app = document.querySelector("div.stApp");
        if (viewport && app && !viewport.contains(app)) {
            viewport.appendChild(app);
            console.log("✓ Streamlit UI moved inside holographic viewport.");
        }
        const blocks=document.querySelectorAll('.block-container');
        blocks.forEach(b=>{
          b.style.maxHeight='100vh';
          b.style.overflowY='auto';
        });
    }, 350);
});
</script>

<style>
#main-viewport {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    z-index: 10 !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* FIX: Allow interaction with tabs & Streamlit widgets */
#main-viewport {
    pointer-events: none !important;
}
#main-viewport .block-container,
#main-viewport * {
    pointer-events: auto !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
  /* ----- Spectre UI Phase 1 ----- */

  /* Main container padding */
  .main { padding-left: 2rem; padding-right: 2rem; perspective: 1200px; }

  /* Glass background effect */
  .block-container {
    background: rgba(255,255,255,0.40);
    backdrop-filter: blur(2px);
    border-radius: 12px;
    padding: 2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transform-style: preserve-3d;
    animation: cameraOrbit 26s ease-in-out infinite alternate;
  }

  /* Floating card effect */
  .stTabs {
    overflow-x: auto;
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    padding: 0.4rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12);
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: rgba(240,240,240,0.65);
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255,255,255,0.4);
  }

  /* Sidebar width stability */
  section[data-testid="stSidebar"] .css-1d391kg { 
      width: 280px !important; 
  }

  /* Table & dataframe scrollability */
  div[data-testid="stDataFrame"] { 
      overflow: auto; 
  }

  /* Cards for headers */
  .spectre-header {
    padding: 1rem 1.5rem;
    background: rgba(255,255,255,0.35);
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 15px rgba(0,0,0,0.1);
    animation: fadeSlideIn 0.8s ease forwards;
    opacity: 0;
  }

  /* Smooth transitions for interactive changes */
  * {
    transition: all 0.25s ease-in-out;
  }

  /* Button glow on hover */
  button {
    transition: 0.25s ease;
  }
  button:hover {
    box-shadow: 0 0 12px rgba(0,150,255,0.6);
    transform: translateY(-2px);
  }

  /* ----- Spectre UI Phase 2: Animated Panels & HUD Overlays ----- */

  /* Animated section headers */

  @keyframes fadeSlideIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0px); }
  }

  /* Holographic HUD overlay panels */
  .hud-panel {
      background: rgba(255,255,255,0.18);
      border: 1px solid rgba(255,255,255,0.35);
      backdrop-filter: blur(14px);
      border-radius: 12px;
      padding: 1rem 1.4rem;
      box-shadow: 0 0 25px rgba(0,200,255,0.35);
      animation: hudGlow 4s ease-in-out infinite alternate;
  }

  @keyframes hudGlow {
      from { box-shadow: 0 0 12px rgba(0,200,255,0.30); }
      to { box-shadow: 0 0 22px rgba(0,200,255,0.55); }
  }

  /* ----- Spectre UI Phase 3: Holographic Depth, Grid, Video Layer ----- */

  /* Global holographic grid overlay */
  .holo-grid {
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      background-image: 
          linear-gradient(rgba(0,255,255,0.08) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,255,255,0.08) 1px, transparent 1px);
      background-size: 60px 60px;
      z-index: 1;
      opacity: 0.25;
      animation: gridPulse 8s ease-in-out infinite alternate;
      pointer-events: none;
  }

  @keyframes gridPulse {
      from { opacity: 0.25; }
      to   { opacity: 0.55; }
  }

  /* Ambient particles for depth */
  .particle {
      position: fixed;
      width: 4px; height: 4px;
      border-radius: 50%;
      background: rgba(0,200,255,0.7);
      filter: blur(2px);
      z-index: 2;
      animation: particleFloat 9s linear infinite;
      will-change: transform, opacity;
      transform: translate3d(0,0,0);
      pointer-events: none;
  }

  @keyframes particleFloat {
      from { transform: translate3d(0, 100vh, 0); opacity:0.0; }
      20% { opacity: 0.7; }
      to   { transform: translate3d(40px, -10vh, 0); opacity: 0; }
  }

  /* Floating holographic container */
  .holo-container {
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(0,255,255,0.30);
      backdrop-filter: blur(20px);
      border-radius: 15px;
      padding: 1.4rem;
      margin-top: 1rem;
      box-shadow: 0 0 45px rgba(0,255,255,0.25);
      animation: holoFloat 5s ease-in-out infinite alternate;
  }

  @keyframes holoFloat {
      from { transform: translateY(0px); }
      to   { transform: translateY(-6px); }
  }

  /* Video background holder */
  .background-video {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      min-width: 100% !important;
      min-height: 100% !important;
      object-fit: cover !important;
      object-position: center center !important;
      z-index: -9999 !important;
      pointer-events: none !important;
      opacity: 0.55 !important;
      transition: opacity 1.8s ease-in-out !important;
  }

  html, body, .stApp {
      position: relative !important;
      width: 100vw !important;
      height: 100vh !important;
      overflow: hidden !important;
  }

  /* Animated tab transitions */
  .stTabs [data-baseweb="tab"] {
      transition: background 0.35s ease, transform 0.25s ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
      transform: translateY(-3px);
      background: rgba(240,240,255,0.35);
  }

  /* Floating metric cards */
  .metric-card {
      background: rgba(255,255,255,0.25);
      backdrop-filter: blur(12px);
      padding: 1.2rem;
      border-radius: 12px;
      box-shadow: 0 2px 15px rgba(0,0,0,0.12);
      transition: transform .25s ease, box-shadow .25s ease;
      transform: translateZ(0);
  }
  .metric-card:hover {
      transform: translateY(-4px) scale(1.02);
      box-shadow: 0 4px 22px rgba(0,0,0,0.18);
  }

  /* Animated loaders for agentic operations */
  .agent-loader {
      width: 38px;
      height: 38px;
      border: 3px solid rgba(0,150,255,0.35);
      border-top-color: rgba(0,150,255,1);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
      margin: 0 auto;
  }

  @keyframes spin {
      to { transform: rotate(360deg); }
  }

  /* ----- Spectre UI Phase 4: Parallax, Bloom Lighting, 3D HUD ----- */

  /* Parallax container */
  .parallax-layer {
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      z-index: -3;
      background: radial-gradient(circle at 30% 70%, rgba(0,255,255,0.12), transparent 60%);
      background-attachment: fixed;
      transform: translateZ(-2px) scale(1.3);
      will-change: transform, opacity;
      /* animation: parallaxDrift 22s ease-in-out infinite alternate; */
      pointer-events: none;
  }

  @keyframes parallaxDrift {
      from { transform: translateZ(-2px) translateX(-20px) translateY(-10px) scale(1.28); }
      to   { transform: translateZ(-2px) translateX(20px) translateY(10px) scale(1.33); }
  }

  /* Bloom lighting highlights */
  .bloom {
      position: absolute;
      width: 140px; height: 140px;
      background: radial-gradient(rgba(0,180,255,0.45), transparent 70%);
      filter: blur(50px);
      border-radius: 50%;
      /* animation: bloomPulse 6s ease-in-out infinite alternate; */
      z-index: -1;
      pointer-events: none;
  }

  @keyframes bloomPulse {
      from { opacity: 0.55; transform: scale(1.0); }
      to   { opacity: 0.95; transform: scale(1.25); }
  }


  @keyframes cameraOrbit {
      from { transform: translateY(0px) rotateX(2deg) rotateY(-4deg); }
      to   { transform: translateY(-6px) rotateX(-2deg) rotateY(4deg); }
  }

  /* depth shadow under cards */
  .depth-frame {
      box-shadow: 0 20px 55px rgba(0,0,0,0.35);
      border-radius: 15px;
  }
</style>
""", unsafe_allow_html=True)

# ---- Parallax/scroll inertia JS ----
st.markdown(
    """
    <script>
    (function() {
      function initParallax() {
        var viewport = document.getElementById('main-viewport');
        if (!viewport) return;
        var parallax = document.querySelector('.parallax-layer');
        var blooms = document.querySelectorAll('.bloom');
        var hud = document.querySelector('.hud-ring');
        if (!parallax) return;

        var target = 0;
        var current = 0;
        var ease = 0.08;  // inertia factor

        function onScroll() {
          target = viewport.scrollTop || 0;
        }

        viewport.addEventListener('scroll', onScroll, { passive: true });

        function raf() {
          current += (target - current) * ease;
          var offset = -current * 0.18;  // slower parallax movement

          // Background parallax
          parallax.style.transform = 'translate3d(0,' + offset + 'px,0) scale(1.3)';

          // Bloom layers drift slightly with depth
          blooms.forEach(function(b, i) {
            var bo = offset * (0.10 + 0.04 * i);
            b.style.transform = 'translate3d(0,' + bo + 'px,0)';
          });

          // HUD ring subtle vertical drift
          if (hud) {
            var hudOffset = offset * 0.02;
            hud.style.transform = 'translate(-50%, calc(-50% + ' + hudOffset + 'px))';
          }

          window.requestAnimationFrame(raf);
        }

        window.requestAnimationFrame(raf);
      }

      if (document.readyState !== 'loading') {
        initParallax();
      } else {
        document.addEventListener('DOMContentLoaded', initParallax);
      }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ---------- Maps (Leaflet over OpenStreetMap) ----------
import folium
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

# ---------- Viz (3D optional) ----------
import pydeck as pdk

# ---------- Logging / Config ----------
from loguru import logger
import yaml

# ---------- Optional: Reports ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    A4, canvas = None, None

# ---------- Optional: Econometrics ----------
try:
    from linearmodels.panel import PanelOLS
except Exception:
    PanelOLS = None

# ---------- Optional: Explainability ----------
try:
    import shap  # noqa: F401
except Exception:
    shap = None

# ---------- OpenAI (Agentic RAG + RL) ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None


# =============================================================================
# 0) CONFIG + LOGGING
# =============================================================================

DEFAULT_CONFIG = {
    "ui": {"tiles": "OpenStreetMap", "default_zoom": 7, "theme": "light"},
    "rl_teacher": {"min_score": 0.6},
    "export": {"report_author": "Jit", "org": "UCC CUBS"},
    "scenarios": {}
}

@lru_cache(maxsize=1)
def load_config():
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                user_cfg = yaml.safe_load(f) or {}
            m = DEFAULT_CONFIG.copy()
            for k, v in user_cfg.items():
                if isinstance(v, dict) and k in m and isinstance(m[k], dict):
                    m[k].update(v)
                else:
                    m[k] = v
            logger.info("Loaded config.yaml")
            return m
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}; using defaults")
    return DEFAULT_CONFIG

logger.add("dairy_decision_v2.log", rotation="1 week", level="INFO")


# =============================================================================
# 1) LANDSCAPE (Consensus) — Tools, Trends, Citations
# =============================================================================

def consensus_landscape_df():
    rows = [
        ["Processing-Sector Simulation Model",
         "Simulates milk collection/standardization/manufacture; estimates yield, cost, value; scenario analysis",
         "Tailored for Irish milk/product mix; portfolio scenarios", "2"],
        ["PastureBase Ireland (PBI)",
         "Web grassland management & paddock-level DSS; benchmarking",
         "Widely used by Irish dairy farms for grass optimization", "16"],
        ["Hybrid DMAIC/TAM Model",
         "Lean Six Sigma + Turnaround Maintenance for maintenance optimization",
         "Applied at Ireland’s largest dairy site; shorter overhauls, higher capacity", "9"],
        ["Circular Supply Chain Readiness (BWM–FIS)",
         "Hybrid Best–Worst + Fuzzy Inference to assess circular readiness",
         "Empirical across 13 Irish dairy companies; moderate readiness", "7"],
        ["Sustainable Supply Chain Model",
         "Multi-objective (cost/CO₂/route) optimization for distribution",
         "Irish dairy logistics use-cases for sustainable routing", "15"],
        ["DairyWater Project",
         "Water mgmt., wastewater treatment, sustainability decision support",
         "Industry–academia collab for env. impact reduction", "17"],
    ]
    return pd.DataFrame(rows, columns=["Tool/Model","Purpose & Features","Irish Context/Applications","Citations"])

TRENDS = [
    "Integration of real-time data streams and analytics for operations & strategy (11014).",
    "Sustainability focus: environmental impact, circular supply chains, resource efficiency (7,17).",
    "Adoption barriers: data integration, UX, readiness for advanced digital tools (4,7,13).",
    "Simulation & scenario analysis for demand, maintenance, supply chain optimization (2,6,9,15).",
]


# =============================================================================
# 2) DATA (Synthetic + Upload)
# =============================================================================

@st.cache_data
def create_demo_plants() -> pd.DataFrame:
    return pd.DataFrame([
        # Plant A — Cork/Mallow
        dict(option_id="A_status_quo", plant="Plant A", strategy="Status quo",
             margin_per_litre=0.085, processing_cost_per_litre=0.045, capacity_utilisation=0.78,
             energy_kwh_per_litre=0.32, water_litre_per_litre=1.6, ghg_kgco2e_per_litre=0.95,
             waste_kg_per_tonne=8.0, export_share_high_value=0.30, product_diversification_index=0.40,
             lat=52.13, lon=-8.64, county="Cork"),
        dict(option_id="A_energy_eff", plant="Plant A", strategy="Heat recovery + EE",
             margin_per_litre=0.082, processing_cost_per_litre=0.043, capacity_utilisation=0.82,
             energy_kwh_per_litre=0.24, water_litre_per_litre=1.4, ghg_kgco2e_per_litre=0.75,
             waste_kg_per_tonne=7.0, export_share_high_value=0.34, product_diversification_index=0.45,
             lat=52.13, lon=-8.64, county="Cork"),
        # Plant B — Limerick
        dict(option_id="B_value_add", plant="Plant B", strategy="Cheese/Value Added",
             margin_per_litre=0.095, processing_cost_per_litre=0.050, capacity_utilisation=0.70,
             energy_kwh_per_litre=0.35, water_litre_per_litre=1.8, ghg_kgco2e_per_litre=1.05,
             waste_kg_per_tonne=9.0, export_share_high_value=0.55, product_diversification_index=0.75,
             lat=52.66, lon=-8.63, county="Limerick"),
        dict(option_id="B_export_powder", plant="Plant B", strategy="Powder Export",
             margin_per_litre=0.080, processing_cost_per_litre=0.042, capacity_utilisation=0.88,
             energy_kwh_per_litre=0.30, water_litre_per_litre=1.5, ghg_kgco2e_per_litre=0.90,
             waste_kg_per_tonne=8.5, export_share_high_value=0.25, product_diversification_index=0.30,
             lat=52.66, lon=-8.63, county="Limerick"),
        # Plant C — Waterford
        dict(option_id="C_low_carbon", plant="Plant C", strategy="Low-carbon + RE",
             margin_per_litre=0.088, processing_cost_per_litre=0.046, capacity_utilisation=0.76,
             energy_kwh_per_litre=0.20, water_litre_per_litre=1.3, ghg_kgco2e_per_litre=0.60,
             waste_kg_per_tonne=6.0, export_share_high_value=0.40, product_diversification_index=0.60,
             lat=52.25, lon=-7.11, county="Waterford"),
    ])

def create_demo_farms(n=18000, seed=42):  # ~18k simulated dairy farms across key counties
    rng = np.random.default_rng(seed)
    counties = {
        "Cork": (51.95, -8.55, 0.45, 0.45),
        "Limerick": (52.57, -8.65, 0.25, 0.25),
        "Waterford": (52.25, -7.20, 0.25, 0.25),
        "Kerry": (52.16, -9.69, 0.35, 0.35),
        "Tipperary": (52.55, -7.85, 0.30, 0.30),
        "Galway": (53.23, -8.90, 0.35, 0.35),
    }
    recs = []
    for i in range(n):
        county = rng.choice(list(counties.keys()))
        lat0, lon0, dlat, dlon = counties[county]
        lat = lat0 + rng.normal(0, dlat/6)
        lon = lon0 + rng.normal(0, dlon/6)
        herd = max(30, int(rng.normal(85, 20)))
        yield_l = max(900, int(rng.normal(1800, 400)))
        recs.append(dict(
            herd_id=f"H{str(i).zfill(6)}",
            lat=lat, lon=lon, county=county,
            herd_size=herd, milk_yield_l_per_day=yield_l,
            coop="Dairygold" if county in ["Cork","Kerry","Tipperary"]
                 else ("Arrabawn" if county in ["Limerick","Tipperary"] else "Glanbia")
        ))
    return pd.DataFrame(recs)


# =============================================================================
# 3) MCDA MODEL
# =============================================================================

INDICATORS = {
    "margin_per_litre": "benefit",
    "processing_cost_per_litre": "cost",
    "capacity_utilisation": "benefit",
    "energy_kwh_per_litre": "cost",
    "water_litre_per_litre": "cost",
    "ghg_kgco2e_per_litre": "cost",
    "waste_kg_per_tonne": "cost",
    "export_share_high_value": "benefit",
    "product_diversification_index": "benefit",
}

SCENARIO_WEIGHTS = {
    "profit_max": {"margin_per_litre":0.35,"processing_cost_per_litre":0.25,"capacity_utilisation":0.15,
                   "energy_kwh_per_litre":0.05,"water_litre_per_litre":0.05,"ghg_kgco2e_per_litre":0.05,
                   "waste_kg_per_tonne":0.02,"export_share_high_value":0.04,"product_diversification_index":0.04},
    "sustainability_tight": {"margin_per_litre":0.15,"processing_cost_per_litre":0.10,"capacity_utilisation":0.10,
                             "energy_kwh_per_litre":0.20,"water_litre_per_litre":0.15,"ghg_kgco2e_per_litre":0.20,
                             "waste_kg_per_tonne":0.05,"export_share_high_value":0.03,"product_diversification_index":0.02},
    "balanced": {"margin_per_litre":0.20,"processing_cost_per_litre":0.15,"capacity_utilisation":0.15,
                 "energy_kwh_per_litre":0.10,"water_litre_per_litre":0.10,"ghg_kgco2e_per_litre":0.15,
                 "waste_kg_per_tonne":0.05,"export_share_high_value":0.05,"product_diversification_index":0.05},
}

def make_custom_scenario(base: dict, deltas: dict) -> dict:
    w = base.copy()
    for k, d in deltas.items():
        if k in w:
            w[k] = max(0.0, w[k] + d)
    s = sum(w.values())
    if s == 0: raise ValueError("All weights zero after adjustment.")
    return {k: v/s for k, v in w.items()}

class DairyDecisionModel:
    def __init__(self, df, indicators, scenario_weights):
        self.raw = df.copy()
        self.indicators = indicators
        self.scenario_weights = scenario_weights
        self.norm = None
        self._validate()

    def _validate(self):
        missing = [c for c in self.indicators if c not in self.raw.columns]
        if missing: raise ValueError(f"Missing indicators: {missing}")
        for s, w in self.scenario_weights.items():
            if not math.isclose(sum(w.values()), 1.0, rel_tol=1e-6):
                raise ValueError(f"Scenario {s} weights sum != 1")

    def _normalise(self):
        df = self.raw.copy()
        for col, typ in self.indicators.items():
            lo, hi = df[col].min(), df[col].max()
            if math.isclose(lo, hi): n = np.full(len(df), 0.5)
            else: n = (df[col] - lo) / (hi - lo)
            if typ == "cost": n = 1.0 - n
            df[col+"_norm"] = n
        self.norm = df

    def score(self, scenario):
        if self.norm is None: self._normalise()
        weights = self.scenario_weights[scenario]
        df = self.norm.copy()
        sc = f"score_{scenario}"
        df[sc] = 0.0
        for ind, w in weights.items():
            df[sc] += w * df[ind+"_norm"]
        return df.sort_values(sc, ascending=False).reset_index(drop=True)

    def rank_all(self):
        if self.norm is None: self._normalise()
        df = self.norm.copy()
        for s in self.scenario_weights:
            sc = f"score_{s}"
            df[sc] = 0.0
            for ind, w in self.scenario_weights[s].items():
                df[sc] += w * df[ind+"_norm"]
        score_cols = [c for c in df.columns if c.startswith("score_")]
        df["score_robust_avg"] = df[score_cols].mean(axis=1)
        return df.sort_values("score_robust_avg", ascending=False)

    def explain(self, option_id):
        if self.norm is None: self._normalise()
        row = self.norm.loc[self.norm["option_id"] == option_id]
        if row.empty: return "No such option."
        r = row.iloc[0]
        lines = [f"### {r['plant']} — {r['strategy']} (`{option_id}`)", ""]
        lines.append("**Indicators (normalised)**")
        for ind, typ in self.indicators.items():
            lines.append(f"- `{ind}`: {r[ind]:.4f} (norm {r[ind+'_norm']:.3f}, {'↑ better' if typ=='benefit' else '↓ better'})")
        lines.append("")
        lines.append("**Scenario scores**")
        for s in self.scenario_weights:
            sc = f"score_{s}"
            if sc not in self.norm.columns:
                self.norm = self.score(s)
            lines.append(f"- `{s}`: {self.norm.loc[self.norm['option_id']==option_id, sc].values[0]:.3f}")
        return "\n".join(lines)


# =============================================================================
# 4) ECONOMETRICS & QUANTUM PIPELINES
# =============================================================================

# ---------- QUANTUM & ECONOMETRICS ----------

def econometric_panels_and_graphs(df: pd.DataFrame):
    """
    Placeholder econometric visualization suite:
    - DID line chart
    - Monte Carlo distribution
    - Markov transition bar
    - Agentic explanation hook
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    import plotly.express as px
    out = {}

    # DID line
    if {"margin_per_litre","processing_cost_per_litre"}.issubset(df.columns):
        did_df = df[["option_id","margin_per_litre","processing_cost_per_litre"]].copy()
        did_df["did"] = did_df["margin_per_litre"] - did_df["processing_cost_per_litre"]
        fig_did = px.bar(did_df, x="option_id", y="did", title="DID Effect (Margin - Cost)")
        out["did"] = fig_did

    # Monte Carlo distributions if present
    if {"mc_p5","mc_p50","mc_p95"}.issubset(df.columns):
        mc_long = df.melt(id_vars=["option_id"], value_vars=["mc_p5","mc_p50","mc_p95"],
                          var_name="Percentile", value_name="Value")
        fig_mc = px.line(mc_long, x="option_id", y="Value",
                         color="Percentile", title="Monte Carlo Percentiles")
        out["mc"] = fig_mc

    # Markov chain
    if "uncertainty_state" in df.columns:
        fig_mk = px.histogram(df, x="uncertainty_state", title="Uncertainty State Distribution")
        out["markov"] = fig_mk

    return out

def econometric_did(df):
    df = df.copy()
    df["did_effect"] = df["margin_per_litre"] - df["processing_cost_per_litre"]
    return df

def monte_carlo(df, trials=1000, sigma=0.01):
    df = df.copy()
    n = len(df)
    shocks = np.random.normal(0, sigma, (n, trials))
    base = df["margin_per_litre"].values.reshape(-1,1)
    sim = base + shocks
    df["mc_p5"]  = np.percentile(sim, 5, axis=1)
    df["mc_p50"] = np.percentile(sim, 50, axis=1)
    df["mc_p95"] = np.percentile(sim, 95, axis=1)
    return df

def markov_chain(df):
    df = df.copy()
    states = ["Low","Medium","High"]
    probs = [0.2, 0.5, 0.3]
    df["uncertainty_state"] = np.random.choice(states, size=len(df), p=probs)
    return df

def qldpc(df):
    df = df.copy()
    df["qldpc_resilience"] = df["capacity_utilisation"] * np.exp(-df["ghg_kgco2e_per_litre"])
    return df

def qubo(df):
    df = df.copy()
    df["qubo_score"] = (df["margin_per_litre"] - df["processing_cost_per_litre"]) * (1 - df["ghg_kgco2e_per_litre"])
    return df

def qstp(df):
    df = df.copy()
    df["qstp_efficiency"] = np.tanh(df["energy_kwh_per_litre"] / np.maximum(df["water_litre_per_litre"],1e-6))
    return df

def vrb(df):
    df = df.copy()
    df["vrb_flow"] = df["capacity_utilisation"] * 1000 / (df["waste_kg_per_tonne"] + 1)
    return df


# ---------- Decision Engine orchestrator ----------

class DecisionEngine:
    """
    Central orchestrator: connects MCDA, econometrics, quantum, and summary.
    """
    def __init__(self, plants_df: pd.DataFrame):
        self.plants_df = plants_df.copy()

    def run_mcda(self, scenario_name: str, weights: dict):
        model = DairyDecisionModel(self.plants_df, INDICATORS, {scenario_name: weights})
        ranked = model.score(scenario_name)
        return model, ranked

    def run_econometrics(self, df: pd.DataFrame):
        df_e = df.copy()
        df_e = econometric_did(df_e)
        df_e = monte_carlo(df_e)
        df_e = markov_chain(df_e)
        return df_e

    def run_quantum(self, df: pd.DataFrame):
        df_q = df.copy()
        for fn in [qldpc, qubo, qstp, vrb]:
            df_q = fn(df_q)
        return df_q

    def run_full(self, scenario_name: str, weights: dict):
        model, ranked = self.run_mcda(scenario_name, weights)
        df_e = self.run_econometrics(ranked)
        df_q = self.run_quantum(df_e)
        summary = {
            "top_option": df_q.iloc[0]["option_id"],
            "avg_margin": float(df_q["margin_per_litre"].mean()),
            "avg_ghg": float(df_q["ghg_kgco2e_per_litre"].mean()),
            "share_high_value": float(df_q["export_share_high_value"].mean()),
        }
        return model, df_q, summary


# =============================================================================
# 5) GEO — OSM/Leaflet with deterministic routing
# =============================================================================

COOP_CENTROIDS = {
    "Cork": (51.8985, -8.4756), "Limerick": (52.661, -8.63), "Waterford": (52.259, -7.11),
    "Kerry": (52.16, -9.69), "Tipperary": (52.52, -7.80), "Galway": (53.27, -9.05),
}
PROC_CENTROIDS = {
    "Cork": (52.136, -8.64), "Limerick": (52.66, -8.63), "Waterford": (52.25, -7.11),
    "Kerry": (52.24, -9.52), "Tipperary": (52.65, -7.91), "Galway": (53.30, -8.97),
}
PORT_CENTROIDS = {
    "Cork": (51.856, -8.30), "Limerick": (52.65, -8.69), "Waterford": (52.26, -6.94),
    "Kerry": (52.14, -9.72), "Tipperary": (52.52, -7.50), "Galway": (53.27, -9.05),
}
def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two lat/lon points (km).
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def supply_chain_metrics(df_farms: pd.DataFrame, emission_factor_kg_per_km: float = 1.2):
    """
    Build farm → co-op → processor → port supply-chain metrics.

    Returns
    -------
    routes_df : per-farm route metrics
    agg_df    : county-level aggregates
    """
    required = {"lat", "lon", "county"}
    if not required.issubset(df_farms.columns):
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    farms = df_farms.dropna(subset=["lat", "lon", "county"]).copy()

    for _, r in farms.iterrows():
        lat = float(r["lat"])
        lon = float(r["lon"])
        county = str(r["county"])

        coop = COOP_CENTROIDS.get(county)
        proc = PROC_CENTROIDS.get(county)
        port = PORT_CENTROIDS.get(county)
        if not (coop and proc and port):
            continue

        d_farm_coop = _haversine_km(lat, lon, coop[0], coop[1])
        d_coop_proc = _haversine_km(coop[0], coop[1], proc[0], proc[1])
        d_proc_port = _haversine_km(proc[0], proc[1], port[0], port[1])
        total_km = d_farm_coop + d_coop_proc + d_proc_port

        vol = float(r.get("milk_yield_l_per_day", 0.0))
        emissions = total_km * emission_factor_kg_per_km  # very rough EF placeholder

        rows.append(
            {
                "herd_id": r.get("herd_id"),
                "county": county,
                "coop": r.get("coop", ""),
                "leg_farm_coop_km": d_farm_coop,
                "leg_coop_proc_km": d_coop_proc,
                "leg_proc_port_km": d_proc_port,
                "total_route_km": total_km,
                "milk_yield_l_per_day": vol,
                "route_emissions_kgCO2e": emissions,
            }
        )

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    routes = pd.DataFrame(rows)
    agg = (
        routes.groupby("county", as_index=False)
        .agg(
            milk_yield_l_per_day=("milk_yield_l_per_day", "sum"),
            total_route_km=("total_route_km", "sum"),
            route_emissions_kgCO2e=("route_emissions_kgCO2e", "sum"),
        )
    )
    agg["emissions_intensity_kg_per_L"] = (
        agg["route_emissions_kgCO2e"] / agg["milk_yield_l_per_day"].replace(0, np.nan)
    )

    return routes, agg

def _popup_farm(row):
    parts = [f"<b>Farm</b> — {row.get('county','')}"]
    if "herd_size" in row and not pd.isna(row["herd_size"]):
        parts.append(f"Herd: {int(row['herd_size'])} cows")
    if "milk_yield_l_per_day" in row and not pd.isna(row["milk_yield_l_per_day"]):
        parts.append(f"Yield: {int(row['milk_yield_l_per_day'])} L/day")
    if "coop" in row and pd.notna(row["coop"]):
        parts.append(f"Co-op: {row['coop']}")
    if "herd_id" in row and pd.notna(row["herd_id"]):
        parts.append(f"ID: {str(row['herd_id'])[:3]}•••")
    return "<br>".join(parts)

def farms_network_map(df_farms, county_choice="All"):
    req = {"lat","lon","county"}
    if not req.issubset(set(df_farms.columns)):
        st.warning("Farm CSV must include lat, lon, county.")
        return
    df = df_farms.dropna(subset=["lat","lon","county"]).copy()
    # sample farms for faster loading
    if len(df) > 6000:
        df = df.sample(6000, random_state=42)
    if county_choice != "All":
        df = df[df["county"].astype(str)==str(county_choice)]
    if df.empty:
        st.info("No farms to show for the selection.")
        return

    # When showing all counties, centre the map using the national centroid grid
    if county_choice == "All":
        all_lats = (
            [lat for (lat, _ ) in COOP_CENTROIDS.values()] +
            [lat for (lat, _ ) in PROC_CENTROIDS.values()] +
            [lat for (lat, _ ) in PORT_CENTROIDS.values()]
        )
        all_lons = (
            [lon for (_ , lon) in COOP_CENTROIDS.values()] +
            [lon for (_ , lon) in PROC_CENTROIDS.values()] +
            [lon for (_ , lon) in PORT_CENTROIDS.values()]
        )
        center = [float(np.mean(all_lats)), float(np.mean(all_lons))]
    else:
        center = [df["lat"].mean(), df["lon"].mean()]

    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")
    from folium.plugins import FastMarkerCluster

def farms_network_map(df_farms, county_choice="All"):
    req = {"lat","lon","county"}
    if not req.issubset(set(df_farms.columns)):
        st.warning("Farm CSV must include lat, lon, county.")
        return

    df = df_farms.dropna(subset=["lat","lon","county"]).copy()

    # Filter
    if county_choice != "All":
        df_visible = df[df["county"].astype(str)==str(county_choice)]
    else:
        df_visible = df  # visible farms for popup markers

    if df.empty:
        st.info("No farms to show for the selection.")
        return

    # Map center
    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")

    # ---------- FAST CLUSTER FOR ALL 18,000 FARMS ----------
    farm_points = df[["lat","lon"]].values.tolist()
    FastMarkerCluster(farm_points).add_to(m)

    # ---------- POPUP MARKERS ONLY FOR SELECTED COUNTY ----------
    for idx, r in df_visible.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        county = str(r["county"])

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_popup_farm(r), max_width=300),
            tooltip=f"{county} farm",
            icon=folium.Icon(color="blue", icon="arrow-up", prefix="fa")
        ).add_to(m)

    # ---------- OPTIONAL: Show centroids (lightweight) ----------
    for name, coord in COOP_CENTROIDS.items():
        folium.Marker(
            coord,
            icon=folium.Icon(color="lightblue", icon="industry", prefix="fa"),
            popup=f"<b>{name} Co-op</b>"
        ).add_to(m)

    for name, coord in PROC_CENTROIDS.items():
        folium.Marker(
            coord,
            icon=folium.Icon(color="red", icon="cog", prefix="fa"),
            popup=f"<b>{name} Processor</b>"
        ).add_to(m)

    for name, coord in PORT_CENTROIDS.items():
        folium.Marker(
            coord,
            icon=folium.Icon(color="darkgreen", icon="ship", prefix="fa"),
            popup=f"<b>{name} Port</b>"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width="stretch", height=650)
        
def plants_options_map(df_ranked, score_col):
    if not {"lat","lon"}.issubset(df_ranked.columns):
        st.info("No lat/lon in plant options.")
        return
    center = [df_ranked["lat"].mean(), df_ranked["lon"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")
    vals = df_ranked[score_col]
    if vals.max() != vals.min():
        s = (vals - vals.min())/(vals.max()-vals.min())
    else:
        s = pd.Series(0.5, index=df_ranked.index)
    for idx, r in df_ranked.iterrows():
        radius = 6 + 10*s.loc[idx]
        popup = f"<b>{r['plant']}</b> — {r['strategy']}<br>Score: {r[score_col]:.3f}"
        folium.CircleMarker(location=[r["lat"], r["lon"]], radius=radius, color="blue",
                            fill=True, fill_opacity=0.7,
                            popup=folium.Popup(popup, max_width=280)).add_to(m)
    st_folium(m, width="stretch", height=520)


# =============================================================================
# 6) AI AGENTS — Knowledge Engine 2.0, RL Teacher 2.0, RL Advisor
# =============================================================================

class KnowledgeEngine:
    """
    Simple RAG/Knowledge engine with basic semantic-ish scoring.
    (Vector DB can be plugged in later; this is a stub with structure.)
    """
    def __init__(self):
        self.docs = []

    def ingest(self, text: str, tag: str = "generic"):
        self.docs.append({"tag": tag, "text": text})

    def _score(self, text, query):
        # crude similarity: shared token count
        qs = set(str(query).lower().split())
        ts = set(str(text).lower().split())
        if not qs: return 0
        return len(qs & ts) / len(qs)

    def query(self, prompt: str) -> str:
        if not self.docs:
            return "[KE] No documents ingested."
        ranked = sorted(self.docs, key=lambda d: self._score(d["text"], prompt), reverse=True)
        ctx = "\n\n---\n\n".join(d["text"] for d in ranked[:5])
        try:
            if client:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a retrieval + reasoning engine for econometrics & EU regulation."},
                        {"role": "user", "content": f"Context:\n{ctx}\n\nQuery:\n{prompt}"}
                    ]
                )
                return resp.choices[0].message.content
            else:
                return f"[Mocked RAG]\nContext:\n{ctx}\n\nAnswer (stub) to: {prompt}"
        except Exception as e:
            return f"[OpenAI error: {e}]"

class RLTeacher:
    """
    Multi-criteria auditor for AI outputs.
    """
    def __init__(self, min_score=0.6):
        self.min_score = min_score
        self.log = []  # list of dicts: {prompt, output, scores}

    def _criterion_scores(self, text: str) -> dict:
        t = (text or "").lower()
        return {
            "policy": 1.0 if "policy" in t else 0.0,
            "eu": 1.0 if "eu" in t or "european" in t else 0.0,
            "climate": 1.0 if "climate" in t or "emissions" in t or "ghg" in t else 0.0,
            "fairness": 1.0 if "fairness" in t or "equity" in t or "bias" in t else 0.0,
            "uncertainty": 1.0 if "uncertain" in t or "confidence" in t or "scenario" in t else 0.0,
        }

    def audit(self, prompt: str, output: str) -> float:
        scores = self._criterion_scores(output)
        overall = sum(scores.values()) / max(len(scores), 1)
        record = {"prompt": prompt, "output": output, "scores": scores, "overall": overall}
        self.log.append(record)
        logger.info(f"RLTeacher audit: {overall:.2f} | details={scores}")
        return overall

    def last_details(self):
        return self.log[-1] if self.log else None

class RLAdvisor:
    def advise(self, prompt: str) -> str:
        try:
            if client:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Secondary advisory agent; provide alternative scenario framing."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return r.choices[0].message.content
            else:
                return f"[Advisor stub] Alternate perspective for: {prompt}"
        except Exception as e:
            return f"[Advisor error: {e}]"

def bias_scan(text: str):
    keys = ["gender","race","ethnic","religion","age"]
    hits = [k for k in keys if k in (text or "").lower()]
    return {"flags": hits, "count": len(hits)}


# =============================================================================
# 7) REPORTS
# =============================================================================

def build_pdf(summary: str) -> bytes:
    if not (A4 and canvas): return b""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    c.setFont("Helvetica-Bold",14); c.drawString(40,h-50,"Irish Dairy Decision Intelligence Report")
    c.setFont("Helvetica",10)
    c.drawString(40,h-70,f"Author: {load_config()['export']['report_author']} | Org: {load_config()['export']['org']}")
    c.drawString(40,h-85,f"Generated: {datetime.utcnow().isoformat()}Z")
    t = c.beginText(40, h-110); t.setFont("Helvetica",10)
    for line in (summary or "").splitlines():
        t.textLine(line[:110])
    c.drawText(t); c.showPage(); c.save()
    pdf = buf.getvalue(); buf.close(); return pdf

# =============================================================================
#  BLE / NFC SENSOR BRIDGE (Telemetric DataStream for MCDA)
# =============================================================================

try:
    from bleak import BleakScanner
except Exception:
    BleakScanner = None


class BluetoothNFCBridge:
    """
    Unified BLE + NFC Low-Energy Telemetry Bridge.
    - Scans BLE advertisements
    - Extracts sensor payloads
    - Parses NFC/NDEF-like structures from encoded bytes
    """

    def __init__(self):
        self.last_packet = None
        self.last_timestamp = None
        self.enabled = False

    async def scan_once(self):
        """Perform a single BLE scan and capture telemetry."""
        if BleakScanner is None:
            return {"error": "Bleak not installed"}

        devices = await BleakScanner.discover(timeout=3.0)

        for d in devices:
            if d.metadata and "manufacturer_data" in d.metadata:
                payload = d.metadata["manufacturer_data"]
                if payload:
                    # Select the first available manufacturer payload
                    key = list(payload.keys())[0]
                    raw = payload[key]

                    # Decode sensor-like structure
                    parsed = self._parse_payload(raw)

                    self.last_packet = parsed
                    self.last_timestamp = datetime.utcnow().isoformat()
                    return parsed

        return None

    def _parse_payload(self, raw):
        """
        Payload → telemetric fields
        Example Format:
        [temp, humidity, vibration, nfc_flag, nfc_data...]

        For your dairy MCDA, this becomes:
        - energy_kwh_per_litre  ← from vibration/temp
        - water_litre_per_litre ← from humidity transducer
        - ghg proxy ← from thermal delta
        """
        if not raw:
            return None

        # Convert bytes to list
        arr = list(raw)

        parsed = {
            "raw_bytes": arr,
            "temperature_c": arr[0] if len(arr) > 0 else None,
            "humidity_pct": arr[1] if len(arr) > 1 else None,
            "vibration": arr[2] if len(arr) > 2 else None,
            "nfc_flag": arr[3] if len(arr) > 3 else 0,
            "nfc_payload": arr[4:] if len(arr) > 4 else []
        }

        return parsed

    def to_mcda_adjustments(self):
        """
        Convert sensor telemetry → MCDA variable adjustments.
        Values are soft deltas applied to plant scenario weights.
        """

        if not self.last_packet:
            return {}

        t = self.last_packet.get("temperature_c", 0)
        h = self.last_packet.get("humidity_pct", 0)
        vib = self.last_packet.get("vibration", 0)

        return {
            # Example: vibration implies higher energy usage
            "energy_kwh_per_litre": vib * 0.001,

            # Temperature increases cooling energy demand
            "processing_cost_per_litre": (t - 20) * 0.0005 if t > 20 else 0,

            # Humidity → higher water consumption rate
            "water_litre_per_litre": h * 0.0003,

            # Rough GHG proxy
            "ghg_kgco2e_per_litre": max(0, (t - 18) * 0.0004)
        }


# =============================================================================
# 8) APP
# =============================================================================

def main():
    # Phase 4: parallax + bloom lighting + HUD ring
    st.markdown("<div class='parallax-layer'></div>", unsafe_allow_html=True)
    st.markdown("<div class='bloom' style='top:15vh; left:20vw;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='bloom' style='top:65vh; left:70vw; animation-delay:2s;'></div>", unsafe_allow_html=True)

    # Inject holographic grid layer + particles
    st.markdown("<div class='holo-grid'></div>", unsafe_allow_html=True)
    for i in range(8):
        st.markdown(f"<div class='particle' style='left:{5+i*7}vw; animation-delay:{i*0.7}s'></div>", unsafe_allow_html=True)

    # Optional video background (user can upload)
    vid = st.sidebar.file_uploader("Background video (optional, .mp4)", type=["mp4"])

    # Performance throttle: Reject files > 100MB
    MAX_MB = 100
    if vid and (vid.size / (1024 * 1024) > MAX_MB):
        st.sidebar.error(f"Video too large. Please upload a file under {MAX_MB} MB.")
        vid = None

    # Brightness slider
    brightness = st.sidebar.slider("Background Video Brightness", 0.1, 1.0, 0.55, step=0.05)

    # Fade-in effect container
    st.markdown(
        f"""
        <style>
        .background-video {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            min-width: 100% !important;
            min-height: 100% !important;
            object-fit: cover !important;
            object-position: center center !important;
            z-index: -9999 !important;
            pointer-events: none !important;
            opacity: {brightness} !important;
            transition: opacity 1.8s ease-in-out !important;
        }}

        /* Fade-in animation */
        .video-fade {{
            animation: fadeInVideo 2.5s ease forwards;
        }}

        @keyframes fadeInVideo {{
            from {{ opacity: 0; }}
            to   {{ opacity: {brightness}; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if vid:
        video_bytes = vid.read()
        import base64
        video_b64 = base64.b64encode(video_bytes).decode()

        st.markdown(
            f"""
            <video class='background-video video-fade' autoplay loop muted playsinline>
                <source src='data:video/mp4;base64,{video_b64}' type='video/mp4'>
            </video>
            """,
            unsafe_allow_html=True
        )

    st.sidebar.markdown("### Sidebar Controls")
    toggle = st.sidebar.button("Minimize Sidebar")
    if toggle:
        st.markdown("<style> section[data-testid='stSidebar'] { width: 0 !important; opacity: 0; } </style>", unsafe_allow_html=True)
    cfg = load_config()
    logger.info("App start")

    st.title("Dairy Decision Intelligence System - Ireland")
    st.caption("System Architecture, Design and Engineering by Shubhoit Bagchi | © 2025")



    # ---------- LANDSCAPE ----------
    tabs = st.tabs([
        "Landscape",
        "Decision Engine",
        "Geo Intelligence",
        "Quantum & Econometrics",
        "AI Governance",
        "Compliance",
        "Reports",
        "Supply Chain",
        "System Log"
    ])
    with tabs[0]:
        st.markdown("<div class='tab-clip glass-only'>", unsafe_allow_html=True)
        st.subheader("Decision-Making Tools & Models Literature Review — Google Scholar")
        st.dataframe(consensus_landscape_df(), width="stretch", hide_index=True)
        st.markdown("**Trends & Research Directions**")
        for t in TRENDS: st.markdown(f"- {t}")
        st.caption("Overview of decision-making tools in the Irish dairy processing industry. "
                   "Numbers in parentheses reflect consolidated citation indices from the literature review - Google Scholar.")
        st.markdown("</div>", unsafe_allow_html=True)
    # ---------- DECISION ENGINE ----------
    with tabs[1]:
        st.markdown("<div class='tab-clip glass-only'>", unsafe_allow_html=True)

        st.subheader("1. Data")
        data_choice = st.radio("Select data source:", ["Demo synthetic plants","Upload CSV"], horizontal=True)
        if data_choice=="Demo synthetic plants":
            df_plants = create_demo_plants()
        else:
            up = st.file_uploader("Upload CSV with plant options + indicators", type=["csv"])
            if up is None: 
                st.stop()
            df_plants = pd.read_csv(up, low_memory=False)

        missing = [c for c in INDICATORS if c not in df_plants.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        st.dataframe(df_plants, width="stretch")

        st.subheader("2. Scenario & Weights")
        scen = st.selectbox("Scenario", list(SCENARIO_WEIGHTS.keys())+["Custom"], index=2)

        if scen!="Custom":
            weights = {**SCENARIO_WEIGHTS[scen], **cfg.get("scenarios",{}).get(scen,{})}
            st.json(weights, expanded=False)
            scenario_name = scen
        else:
            base = SCENARIO_WEIGHTS["balanced"].copy()
            deltas = {k: st.slider(f"Δ {k}", -base[k], base[k], 0.0, step=max(0.01, base[k]/10)) for k in base}
            weights = make_custom_scenario(base, deltas)
            st.json(weights, expanded=False)
            scenario_name = "custom"

        engine = DecisionEngine(df_plants)
        model, df_ranked, summary = engine.run_full(scenario_name, weights)

        score_col = f"score_{scenario_name}"

        st.subheader("3. Ranked Options")
        st.dataframe(df_ranked[["option_id","plant","strategy",score_col,"county"]], width="stretch")
        st.bar_chart(df_ranked.set_index("option_id")[score_col])

        st.subheader("4. Option Explanation")
        opt = st.selectbox("Select option", df_ranked["option_id"])
        st.markdown(model.explain(opt))

        st.subheader("5. Engine Summary")
        st.write(summary)

        # =======================
        #    BLE / NFC BLOCK
        # =======================
        st.subheader("6. BLE/NFC Telemetry Integration")
        st.markdown("### Real-time BLE/NFC Sensor Stream")

        sensor_bridge = st.session_state.get("sensor_bridge")
        if sensor_bridge is None:
            sensor_bridge = BluetoothNFCBridge()
            st.session_state["sensor_bridge"] = sensor_bridge

        enable_bt = st.checkbox("Enable Bluetooth/NFC Stream")

        if enable_bt:
            sensor_bridge.enabled = True
            st.info("Scanning for BLE/NFC packets...")

            import asyncio
            try:
                packet = asyncio.run(sensor_bridge.scan_once())
                if packet:
                    st.success("Sensor packet received")
                    st.json(packet)
                else:
                    st.warning("No BLE/NFC sensor detected nearby")
            except Exception as e:
                st.error(f"BLE scanning error: {e}")

            # Apply sensor-driven adjustments
            adjustments = sensor_bridge.to_mcda_adjustments()
            st.markdown("**MCDA Adjustments from Sensors:**")
            st.json(adjustments)

            # Merge adjustments into scenario weights
            for k, delta in adjustments.items():
                if k in weights:
                    weights[k] += delta

            # Normalize
            total = sum(weights.values())
            if total > 0:
                for k in weights:
                    weights[k] /= total

        # Store ranked data for other tabs
        st.session_state["df_ranked"] = df_ranked
        st.session_state["score_col"] = score_col

        st.markdown("</div>", unsafe_allow_html=True)
        
    # ---------- GEO INTELLIGENCE ----------
    with tabs[2]:
        st.markdown("<div class='tab-clip glass-only'>", unsafe_allow_html=True)

        st.subheader("All Dairy Farms Network")
        use_demo_farms = st.checkbox("Use demo farms", value=True)
        if use_demo_farms:
            farms = create_demo_farms()
        else:
            fup = st.file_uploader("Upload farm CSV (lat, lon, county, herd_size, milk_yield_l_per_day, coop)", type=["csv"])
            if fup is None: st.stop()
            farms = pd.read_csv(fup, low_memory=False)
        counties = ["All"] + sorted(farms["county"].dropna().astype(str).unique())
        county_choice = st.selectbox("Filter by county", counties, index=0)
        farms_network_map(farms, county_choice)
        st.session_state["farms"] = farms

        with st.expander("3D Density Heatmap of Farms"):
            df_map = farms.dropna(subset=["lat","lon"]).rename(columns={"lat":"latitude","lon":"longitude"})
            if not df_map.empty:
                # Use national centroid mapping to ensure the heatmap covers the full Ireland canvas
                all_lats = (
                    [lat for (lat, _ ) in COOP_CENTROIDS.values()] +
                    [lat for (lat, _ ) in PROC_CENTROIDS.values()] +
                    [lat for (lat, _ ) in PORT_CENTROIDS.values()]
                )
                all_lons = (
                    [lon for (_ , lon) in COOP_CENTROIDS.values()] +
                    [lon for (_ , lon) in PROC_CENTROIDS.values()] +
                    [lon for (_ , lon) in PORT_CENTROIDS.values()]
                )
                center_lat = float(np.mean(all_lats))
                center_lon = float(np.mean(all_lons))

                vs = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=5.8,
                    pitch=45
                )
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=df_map,
                    get_position='[longitude, latitude]',
                    radius_pixels=30
                )
                st.pydeck_chart(pdk.Deck(initial_view_state=vs, layers=[layer]))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- QUANTUM & ECONOMETRICS ----------
    with tabs[3]:
        st.markdown("<div class='tab-clip glass-only'>", unsafe_allow_html=True)
        st.subheader("Quantum–Econometric Pipeline")
        df_ranked = st.session_state.get("df_ranked", create_demo_plants())

        dfq = st.session_state.get("dfq", None)

        if st.button("Run Full Simulation"):
            engine = DecisionEngine(df_ranked)
            _, dfq, summary = engine.run_full("balanced", SCENARIO_WEIGHTS["balanced"])
            st.session_state["dfq"] = dfq
            st.success("Simulation complete.")

        if dfq is None:
            st.info("Run the simulation to generate quantum–econometric outputs.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        else:
            st.dataframe(dfq.head(), width="stretch")

        # Econometric Visuals
        charts = econometric_panels_and_graphs(dfq)
        for k, fig in charts.items():
            st.plotly_chart(fig, width="stretch")

            # Agentic explanation if available
            if "agentic_last_output" in st.session_state:
                st.markdown("### Agentic Explanation")
                st.write(st.session_state["agentic_last_output"])

        # ---- SHAP FULL SUITE (GLOBAL + LOCAL + INTERACTION + DEPENDENCE + CLUSTERING) ----
        if shap is not None:
            st.subheader("SHAP Explainability Suite")

            try:
                import plotly.express as px
                # Select target
                target = st.selectbox(
                    "Select target variable",
                    [c for c in dfq.columns if dfq[c].dtype.kind in "if"],
                    index=0
                )

                # Feature set
                ignore_cols = ["option_id", "plant", "strategy", "county", "uncertainty_state"]
                feature_cols = [
                    c for c in dfq.columns
                    if c not in ignore_cols and dfq[c].dtype.kind in "if"
                ]

                if not feature_cols:
                    st.warning("No numeric feature columns available for SHAP analysis.")
                else:
                    X = dfq[feature_cols]
                    y = dfq[target]

                    # Model
                    import xgboost as xgb
                    model_shap = xgb.XGBRegressor(
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.07,
                        subsample=0.9,
                        colsample_bytree=0.9,
                    )
                    model_shap.fit(X, y)

                    # SHAP explainer
                    explainer = shap.TreeExplainer(model_shap)
                    explanation = explainer(X)
                    shap_matrix = explanation.values  # (n_samples, n_features)

                    # ---------------- GLOBAL IMPORTANCE: BEESWARM-STYLE (PLOTLY) ----------------
                    st.markdown("### Global Feature Importance")

                    # Build long-form DataFrame for interactive beeswarm
                    rows = []
                    n_samples, n_features = shap_matrix.shape
                    for j, feat in enumerate(feature_cols):
                        vals = shap_matrix[:, j]
                        feat_vals = X[feat].values
                        for i in range(n_samples):
                            rows.append(
                                {
                                    "feature": feat,
                                    "shap_value": float(vals[i]),
                                    "feature_value": float(feat_vals[i])
                                    if np.issubdtype(X[feat].dtype, np.number)
                                    else None,
                                }
                            )
                    df_bee = pd.DataFrame(rows)

                    color_arg = "feature_value" if df_bee["feature_value"].notna().any() else None

                    fig_bee = px.strip(
                        df_bee,
                        x="shap_value",
                        y="feature",
                        color=color_arg,
                        orientation="h",
                        title="SHAP Beeswarm (Interactive)",
                    )
                    fig_bee.update_layout(
                        height=min(800, 80 * len(feature_cols) + 200),
                        yaxis_title="Feature",
                        xaxis_title="SHAP value",
                    )
                    st.plotly_chart(fig_bee, use_container_width=True)

                    # ---------------- GLOBAL BAR SUMMARY (PLOTLY) ----------------
                    st.markdown("### Global Feature Summary")

                    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
                    df_bar = (
                        pd.DataFrame(
                            {
                                "feature": feature_cols,
                                "mean_abs_shap": mean_abs,
                            }
                        )
                        .sort_values("mean_abs_shap", ascending=True)
                        .reset_index(drop=True)
                    )

                    fig_bar = px.bar(
                        df_bar,
                        x="mean_abs_shap",
                        y="feature",
                        orientation="h",
                        title="Mean |SHAP| by Feature",
                        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # ---------------- INTERACTION VALUES (PLOTLY HEATMAP) ----------------
                    st.markdown("### SHAP Interaction Values")

                    try:
                        interaction_vals = explainer.shap_interaction_values(X)

                        # Convert the interaction values into a 2D matrix (mean absolute)
                        interaction_matrix = np.mean(np.abs(interaction_vals), axis=0)

                        fig_inter = px.imshow(
                            interaction_matrix,
                            x=feature_cols,
                            y=feature_cols,
                            color_continuous_scale="Viridis",
                            labels=dict(color="Mean |interaction|"),
                            title="SHAP Interaction Values (Mean Absolute — 2D Matrix)",
                        )
                        st.plotly_chart(fig_inter, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Interaction heatmap skipped: {e}")

                    # ---------------- DEPENDENCE PLOTS (PLOTLY SCATTER) ----------------
                    st.markdown("### Dependence Plots")

                    dep_feature = st.selectbox(
                        "Select feature for dependence plot", feature_cols
                    )
                    if dep_feature in feature_cols:
                        j_dep = feature_cols.index(dep_feature)
                        df_dep = pd.DataFrame(
                            {
                                dep_feature: X[dep_feature].values,
                                "shap_value": shap_matrix[:, j_dep],
                            }
                        )
                        fig_dep = px.scatter(
                            df_dep,
                            x=dep_feature,
                            y="shap_value",
                            color=dep_feature,
                            title=f"SHAP Dependence for {dep_feature}",
                            labels={"shap_value": "SHAP value"},
                        )
                        st.plotly_chart(fig_dep, use_container_width=True)

                    # ---------------- LOCAL WATERFALL (PLOTLY APPROXIMATION) ----------------
                    st.markdown("### Local Explanation for Top-Ranked Option")

                    import plotly.graph_objects as go

                    top_row = X.iloc[[0]]
                    local_exp = explainer(top_row)
                    local_vals = pd.Series(local_exp.values[0], index=feature_cols).sort_values(
                        ascending=False
                    )
                    base_val = float(
                        local_exp.base_values[0]
                        if np.size(local_exp.base_values) > 0
                        else 0.0
                    )

                    fig_wf = go.Figure()

                    fig_wf.add_trace(
                        go.Waterfall(
                            name="SHAP",
                            orientation="v",
                            x=local_vals.index.tolist(),
                            measure=["relative"] * len(local_vals),
                            y=local_vals.values,
                        )
                    )
                    fig_wf.update_layout(
                        title="Local SHAP Waterfall (Approximate, Plotly)",
                        showlegend=False,
                        xaxis_title="Feature",
                        yaxis_title="Contribution to prediction",
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

                    # ---------------- CLUSTER MAP (INTERACTIVE DENDROGRAM) ----------------
                    st.markdown("### SHAP Feature Cluster Map (Dendrogram)")

                    try:
                        from scipy.cluster.hierarchy import linkage
                        from plotly.figure_factory import create_dendrogram

                        cluster_matrix = shap_matrix  # 2D [samples x features]

                        def _linkagefun(x):
                            return linkage(x, method="ward")

                        fig_d = create_dendrogram(
                            cluster_matrix.T,
                            labels=feature_cols,
                            orientation="bottom",
                            linkagefun=_linkagefun,
                        )
                        fig_d.update_layout(
                            width=900,
                            height=500,
                            title="Feature Cluster Map (Ward Linkage)",
                        )
                        st.plotly_chart(fig_d, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Cluster map skipped: {e}")

                    # ---------------- LOCAL FORCE VIEW / CONTRIBUTION TABLE ----------------
                    st.markdown("### Local SHAP Contribution Table")

                    idx_max = len(X) - 1
                    row_idx = st.number_input(
                        "Select row index for local SHAP explanation",
                        min_value=0,
                        max_value=int(idx_max),
                        value=0,
                        step=1,
                    )

                    try:
                        local_row = X.iloc[[int(row_idx)]]
                        local_exp2 = explainer(local_row)
                        local_vals2 = pd.Series(
                            local_exp2.values[0], index=feature_cols
                        )

                        contrib_df = (
                            pd.DataFrame(
                                {
                                    "feature": feature_cols,
                                    "shap_value": local_vals2.values,
                                }
                            )
                            .sort_values("shap_value", ascending=False)
                            .reset_index(drop=True)
                        )

                        st.write("Top positive contributors:")
                        st.dataframe(contrib_df.head(5), width="stretch")

                        st.write("Top negative contributors:")
                        st.dataframe(
                            contrib_df.tail(5).sort_values("shap_value"),
                            width="stretch",
                        )
                    except Exception as e:
                        st.warning(f"Local SHAP contribution view skipped: {e}")
            except Exception as e:
                st.warning(f"SHAP explainability suite failed: {e}")
        else:
            st.info("Install SHAP to enable the full explainability suite (pip install shap).")

        with st.expander("Panel OLS (placeholder)"):
            if PanelOLS is None:
                st.info("Install `linearmodels` to enable Panel OLS with real panel data.")
            else:
                st.info("Wire your true panel dataset (county × year) here and estimate PanelOLS with FE/RE as needed.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AI GOVERNANCE ----------
    with tabs[4]:
        st.subheader("Agentic RAG + RL Teacher/Advisor")
        df_ranked = st.session_state.get("df_ranked", create_demo_plants())

        ke = KnowledgeEngine()
        ke.ingest(str(df_ranked.describe()), tag="stats")
        ke.ingest("EU AI Act, Data Act, GDPR, Cybersecurity Act context ingested for dairy AI systems.", tag="regulation")
        ke.ingest("Irish dairy processing: emissions, circular supply chains, resource efficiency, water and energy usage.", tag="sustainability")

        # Ingest quantum and supply-chain data into KnowledgeEngine
        dfq_ke = st.session_state.get("dfq")
        if dfq_ke is not None and isinstance(dfq_ke, pd.DataFrame) and not dfq_ke.empty:
            ke.ingest("Quantum–econometric summary:\n" + dfq_ke.describe(include='all').to_string(), tag="quantum")

        routes_ke = st.session_state.get("routes_df")
        agg_ke = st.session_state.get("agg_df")
        if agg_ke is not None and isinstance(agg_ke, pd.DataFrame) and not agg_ke.empty:
            ke.ingest("Supply-chain aggregates (county-level KPIs):\n" + agg_ke.to_string(index=False), tag="supply_chain")

        teacher = RLTeacher(min_score=cfg["rl_teacher"]["min_score"])
        advisor = RLAdvisor()

        q = st.text_area("Governance question",
                         "Explain policy impact on Irish dairy profitability under EU AI Act compliance with uncertainty and fairness.")
        if st.button("Run Agentic Governance"):
            out = ke.query(q)
            st.markdown("**Knowledge Engine Output**")
            st.write(out)

            # Store KE output for econometric panel explanations
            st.session_state["agentic_last_output"] = out

            score = teacher.audit(q, out)
            st.write(f"RL-Teacher overall score: {score:.2f}")

            details = teacher.last_details()
            if details:
                st.markdown("**Criterion scores:**")
                st.json(details["scores"], expanded=False)

            if score < cfg["rl_teacher"]["min_score"]:
                st.warning("Score below threshold → Advisor engaged.")
                alt = advisor.advise(q)
                st.markdown("**Advisor Suggestion**")
                st.write(alt)

            with st.expander("Bias & Explainability"):
                st.write("Bias scan:", bias_scan(out))
                if shap is None:
                    st.info("Install SHAP to enable local feature explanations for econometric models.")

        # --- AI-driven data querying ---
        with st.expander("AI‑driven natural language querying on live metrics"):
            q_data = st.text_input(
                "Ask a question about plant performance, quantum outputs or supply-chain KPIs",
                "",
                key="ke_data_query",
            )
            if st.button("Ask data question"):
                if q_data.strip():
                    ans = ke.query(q_data)
                    st.markdown("**Data agent answer**")
                    st.write(ans)
                else:
                    st.info("Enter a question above before querying.")

        # --- Real-time agentic workers ---
        with st.expander("Real‑time agentic workers (multi-loop decision agents)"):
            worker_prompt = st.text_area(
                "High-level decision brief for agentic workers",
                "Stress test Irish dairy export routes under climate and fuel-price shocks while respecting EU AI Act constraints.",
                key="agent_worker_prompt",
            )
            n_workers = st.slider("Number of workers", 2, 5, 3, 1, key="agent_worker_count")

            if st.button("Run agentic workers"):
                if not worker_prompt.strip():
                    st.info("Provide a brief for the workers first.")
                else:
                    roles = ["Profitability", "Climate risk", "Resilience", "Regulation", "Farmer impact"]
                    worker_outputs = []

                    for i in range(n_workers):
                        role = roles[i % len(roles)]
                        if client:
                            try:
                                r = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": f"You are Agent {i+1}, specialised in {role} analysis for Irish dairy supply chains.",
                                        },
                                        {
                                            "role": "user",
                                            "content": worker_prompt,
                                        },
                                    ],
                                )
                                txt = r.choices[0].message.content
                            except Exception as e:
                                txt = f"[Worker {i+1} error: {e}]"
                        else:
                            txt = f"[Worker {i+1} stub output focused on {role}] {worker_prompt}"

                        score = teacher.audit(worker_prompt, txt)
                        worker_outputs.append((role, txt, score))

                        st.markdown(f"#### Worker {i+1} — {role} (score {score:.2f})")
                        st.write(txt)

                    if worker_outputs:
                        avg_score = sum(s for _, _, s in worker_outputs) / len(worker_outputs)
                        st.markdown(f"**Ensemble worker score:** {avg_score:.2f}")

                        if client:
                            try:
                                concat = "\n\n".join(
                                    f"Worker {i+1} ({role}, score {score:.2f}):\n{txt}"
                                    for i, (role, txt, score) in enumerate(worker_outputs)
                                )
                                r = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are an ensemble supervisor summarising multiple agents into one actionable recommendation.",
                                        },
                                        {
                                            "role": "user",
                                            "content": concat,
                                        },
                                    ],
                                )
                                sup = r.choices[0].message.content
                            except Exception as e:
                                sup = f"[Supervisor error: {e}]"
                        else:
                            sup = "[Supervisor stub] Aggregate the above workers' views into one recommendation."

                        st.markdown("### Supervisor consensus recommendation")
                        st.write(sup)

        with st.expander("EU Regulatory API Feeds"):
            st.write("Eurostat sample GDP feed:", eurostat_api())
            st.write("EU AI Act legal text feed:", eu_act_api())

    # ---------- COMPLIANCE ----------
    with tabs[5]:
        st.subheader("EU Regulatory Framework — Alignment Status")
        regs = [
            {"Regulation": "EU AI Act", "Scope": "High-risk AI systems in agri/food", "Focus": "Transparency, risk mgmt, human oversight", "Status": "Conceptually integrated"},
            {"Regulation": "Data Governance Act", "Scope": "Data intermediaries, reuse of public-sector data", "Focus": "Data sharing & stewardship", "Status": "Conceptually integrated"},
            {"Regulation": "Data Act", "Scope": "Access to industrial and IoT data", "Focus": "Data access, portability", "Status": "Planned connector"},
            {"Regulation": "GDPR", "Scope": "Personal data", "Focus": "Privacy, consent, rights", "Status": "No personal data ingested in demo"},
            {"Regulation": "Cybersecurity Act", "Scope": "ICT products/services", "Focus": "Security certification", "Status": "Out-of-scope (infrastructure)"},
        ]
        st.dataframe(pd.DataFrame(regs), width="stretch", hide_index=True)
        st.markdown(
            "- **Transparency**: MCDA + econometric pipeline and AI outputs are visible in the UI.\n"
            "- **Risk Management**: RL-Teacher audits, scenario simulations, uncertainty states.\n"
            "- **Traceability**: All major actions logged into `dairy_decision_v2.log`."
        )

    # ---------- REPORTS ----------
    with tabs[6]:
        st.subheader("Generate PDF")
        summary = st.text_area(
            "Executive Summary",
            "Key findings:\n- Scenario ranking...\n- Policy impact...\n- Quantum–econometric simulation insights...\n- Risks & mitigations..."
        )
        if st.button("Create PDF"):
            pdf = build_pdf(summary)
            if pdf:
                st.download_button("Download PDF", data=pdf, file_name="dairy_decision_report.pdf", mime="application/pdf")
            else:
                st.info("Install `reportlab` to enable PDF export.")
        # ---------- SUPPLY CHAIN ----------
    with tabs[7]:
        st.markdown("<div class='tab-clip glass-only'>", unsafe_allow_html=True)
        import plotly.express as px
        st.subheader("Supply Chain — Farm \u2192 Co-op \u2192 Processor \u2192 Port")

        df_ranked = st.session_state.get("df_ranked", create_demo_plants())
        farms = st.session_state.get("farms", create_demo_farms())

        routes_df, agg_df = supply_chain_metrics(farms)
        st.session_state["routes_df"] = routes_df
        st.session_state["agg_df"] = agg_df

        if agg_df.empty:
            st.info("No valid farm records to compute supply-chain metrics.")
        else:
            st.markdown("**County-level Supply Chain KPIs (Daily)**")
            st.dataframe(agg_df, width="stretch")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Total daily milk volume (L)",
                    f"{int(agg_df['milk_yield_l_per_day'].sum()):,}"
                )
            with c2:
                st.metric(
                    "Total route distance (km/day)",
                    f"{int(agg_df['total_route_km'].sum()):,}"
                )
            with c3:
                st.metric(
                    "Total transport CO\u2082 (kg/day)",
                    f"{int(agg_df['route_emissions_kgCO2e'].sum()):,}"
                )

            st.markdown("**Route Distance by County (km/day)**")
            fig_km = px.bar(
                agg_df,
                x="county",
                y="total_route_km",
                title="Total Route Distance by County (km/day)",
            )
            st.plotly_chart(fig_km, use_container_width=True)

            st.markdown("**Transport CO\u2082 by County (kg/day)**")
            fig_em = px.bar(
                agg_df,
                x="county",
                y="route_emissions_kgCO2e",
                title="Transport CO\u2082 by County (kg/day)",
            )
            st.plotly_chart(fig_em, use_container_width=True)

            st.markdown("### Supply Chain Sankey")

            import plotly.graph_objects as go

            try:
                counties = agg_df["county"].astype(str).tolist()
                volumes = agg_df["milk_yield_l_per_day"].astype(float).tolist()

                # Nodes: one per county + 3 aggregated hubs
                node_labels = counties + ["Co-op hubs", "Processors", "Export ports"]
                idx_coop = len(counties)
                idx_proc = len(counties) + 1
                idx_port = len(counties) + 2

                sources = []
                targets = []
                values = []

                for i, vol in enumerate(volumes):
                    # County -> Co-op, Co-op -> Processor, Processor -> Port
                    sources.extend([i, idx_coop, idx_proc])
                    targets.extend([idx_coop, idx_proc, idx_port])
                    values.extend([vol, vol, vol])

                fig_s = go.Figure(
                    data=[
                        go.Sankey(
                            arrangement="snap",
                            valueformat=",",
                            valuesuffix=" L/day",
                            node=dict(
                                pad=24,
                                thickness=18,
                                line=dict(color="rgba(0,255,255,0.8)", width=1),
                                label=node_labels,
                                color=["rgba(0,180,255,0.85)"] * len(node_labels),
                                hovertemplate="%{label}<extra></extra>",
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                color="rgba(0,200,255,0.35)",
                                hovertemplate="Flow: %{value:.0f} L/day<extra></extra>",
                            ),
                        )
                    ]
                )

                fig_s.update_layout(
                    title="Farm → Co-op → Processor → Port — Holographic Flow",
                    font=dict(color="#FFFFFF"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=520,
                )

                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.warning(f"Holographic Sankey failed: {e}")

            with st.expander("Route-level table (sample)"):
                st.dataframe(routes_df.head(500), width=None)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- SYSTEM LOG ----------
    with tabs[8]:
        st.subheader("System Log")
        log_path = Path("dairy_decision_v2.log")
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
            tail = "".join(lines[-200:]) if lines else "(log empty)"
            st.text_area("Last 200 log lines", tail, height=300)
        else:
            st.info("Log file not created yet.")



# ---------- API CONNECTORS ----------

# ---------- SDMX PARSER (Eurostat GDP, Agriculture, Energy, Emissions) ----------
import xml.etree.ElementTree as ET

def eurostat_sdmx_to_df(xml_text: str):
    """
    Converts Eurostat SDMX XML into a tidy DataFrame.
    """
    try:
        root = ET.fromstring(xml_text)
        ns = {'s': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
              'd': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}

        rows = []
        for series in root.findall('.//d:Series', ns):
            series_attrib = {c.attrib['id']: c.find('d:Value', ns).attrib.get('value')
                             for c in series.findall('d:SeriesKey/d:Value', ns)}
            for obs in series.findall('d:Obs', ns):
                obs_time = obs.find('d:ObsDimension', ns).attrib.get('value')
                obs_val = obs.find('d:ObsValue', ns).attrib.get('value')
                row = series_attrib.copy()
                row['time'] = obs_time
                row['value'] = float(obs_val) if obs_val not in [None, ''] else None
                rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

def eurostat_api(dataset_id="nama_10_gdp"):
    """
    Fetches SDMX GDP dataset and returns parsed DataFrame as JSON-like dict.
    """
    url = f"https://ec.europa.eu/eurostat/api/discover/sdmx?dataset={dataset_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        df = eurostat_sdmx_to_df(r.text)
        return df.head(20).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

def eu_act_api(celex="52021PC0206"):
    """
    Fetches EU Act raw HTML and extracts readable legal paragraphs.
    """
    url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")

        # Extract only clean text paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]

        cleaned = paragraphs[:40]  # return first 40 legal lines for UI
        return {"legal_excerpt": cleaned}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()
    st.markdown("</div>", unsafe_allow_html=True)
