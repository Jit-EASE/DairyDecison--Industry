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

# ---------- UI ----------
import streamlit as st

# ---------- Maps (Leaflet over OpenStreetMap) ----------
import folium
from folium.plugins import MarkerCluster
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

def create_demo_farms(n=1200, seed=42):
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
    if county_choice != "All":
        df = df[df["county"].astype(str)==str(county_choice)]
    if df.empty:
        st.info("No farms to show for the selection.")
        return

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")
    cluster = MarkerCluster(name="Farms").add_to(m)

    # Add farms with blue arrow markers + deterministic routing
    for _, r in df.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        county = str(r["county"])
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="blue", icon="arrow-up", prefix="fa"),
            popup=folium.Popup(_popup_farm(r), max_width=300),
            tooltip=f"{county} farm",
        ).add_to(cluster)

        coop = COOP_CENTROIDS.get(county)
        proc = PROC_CENTROIDS.get(county)
        port = PORT_CENTROIDS.get(county)
        if coop and proc and port:
            folium.PolyLine([[lat,lon], coop], color="cyan", weight=1.2, opacity=0.8).add_to(m)
            folium.PolyLine([coop, proc], color="orange", weight=1.6, dash_array="6,6").add_to(m)
            folium.PolyLine([proc, port], color="green", weight=2.0, opacity=0.9).add_to(m)

    # Draw centroids
    for name, coord in COOP_CENTROIDS.items():
        folium.Marker(coord, icon=folium.Icon(color="lightblue", icon="industry", prefix="fa"),
                      popup=f"<b>{name} Co-op</b>").add_to(m)
    for name, coord in PROC_CENTROIDS.items():
        folium.Marker(coord, icon=folium.Icon(color="red", icon="cog", prefix="fa"),
                      popup=f"<b>{name} Processor</b>").add_to(m)
    for name, coord in PORT_CENTROIDS.items():
        folium.Marker(coord, icon=folium.Icon(color="darkgreen", icon="ship", prefix="fa"),
                      popup=f"<b>{name} Port</b>").add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=650)

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
    st_folium(m, use_container_width=True, height=520)


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
# 8) APP
# =============================================================================

def main():
    st.set_page_config(page_title="Irish Dairy Decision Intelligence — Full Build", layout="wide")
    cfg = load_config()
    logger.info("App start")

    st.title("Irish Dairy Decision Intelligence — Full Intelligence Build")
    st.caption("MCDA • Econometrics • Quantum • Agentic Governance • EU AI Act • OSM Leaflet")

    tabs = st.tabs([
        "Landscape",
        "Decision Engine",
        "Geo Intelligence",
        "Quantum & Econometrics",
        "AI Governance",
        "Compliance",
        "Reports",
        "System Log"
    ])

    # ---------- LANDSCAPE ----------
    with tabs[0]:
        st.subheader("Decision-Making Tools & Models (Ireland) — Consensus Landscape")
        st.dataframe(consensus_landscape_df(), use_container_width=True, hide_index=True)
        st.markdown("**Trends & Research Directions**")
        for t in TRENDS: st.markdown(f"- {t}")
        st.caption("Figure 1. Overview of decision-making tools in the Irish dairy processing industry. "
                   "Numbers in parentheses reflect consolidated citation indices from the literature snapshot.")

    # ---------- DECISION ENGINE ----------
    with tabs[1]:
        st.subheader("1. Data")
        data_choice = st.radio("Select data source:", ["Demo synthetic plants","Upload CSV"], horizontal=True)
        if data_choice=="Demo synthetic plants":
            df_plants = create_demo_plants()
        else:
            up = st.file_uploader("Upload CSV with plant options + indicators", type=["csv"])
            if up is None: st.stop()
            df_plants = pd.read_csv(up)

        missing = [c for c in INDICATORS if c not in df_plants.columns]
        if missing: st.error(f"Missing columns: {missing}"); st.stop()
        st.dataframe(df_plants, use_container_width=True)

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
        st.dataframe(df_ranked[["option_id","plant","strategy",score_col,"county"]], use_container_width=True)
        st.bar_chart(df_ranked.set_index("option_id")[score_col])

        st.subheader("4. Option Explanation")
        opt = st.selectbox("Select option", df_ranked["option_id"])
        st.markdown(model.explain(opt))

        st.subheader("5. Engine Summary")
        st.write(summary)

        # stash for other tabs
        st.session_state["df_ranked"] = df_ranked
        st.session_state["score_col"] = score_col

    # ---------- GEO INTELLIGENCE ----------
    with tabs[2]:
        st.subheader("Plants / Options (Leaflet OSM)")
        df_ranked = st.session_state.get("df_ranked", create_demo_plants())
        score_col = st.session_state.get("score_col", "score_balanced")
        plants_options_map(df_ranked, score_col)

        st.subheader("All Dairy Farms Network (Upload optional; demo otherwise)")
        use_demo_farms = st.checkbox("Use demo farms", value=True)
        if use_demo_farms:
            farms = create_demo_farms()
        else:
            fup = st.file_uploader("Upload farm CSV (lat, lon, county, herd_size, milk_yield_l_per_day, coop)", type=["csv"])
            if fup is None: st.stop()
            farms = pd.read_csv(fup)
        counties = ["All"] + sorted(farms["county"].dropna().astype(str).unique())
        county_choice = st.selectbox("Filter by county", counties, index=0)
        farms_network_map(farms, county_choice)

        with st.expander("3D Density (pydeck)"):
            df_map = farms.dropna(subset=["lat","lon"]).rename(columns={"lat":"latitude","lon":"longitude"})
            if not df_map.empty:
                vs = pdk.ViewState(latitude=df_map["latitude"].mean(), longitude=df_map["longitude"].mean(), zoom=6, pitch=45)
                layer = pdk.Layer("HeatmapLayer", data=df_map, get_position='[longitude, latitude]', radius_pixels=30)
                st.pydeck_chart(pdk.Deck(initial_view_state=vs, layers=[layer]))

    # ---------- QUANTUM & ECONOMETRICS ----------
    with tabs[3]:
        st.subheader("Quantum–Econometric Pipeline")
        df_ranked = st.session_state.get("df_ranked", create_demo_plants())
        if st.button("Run Full Simulation"):
            engine = DecisionEngine(df_ranked)
            _, dfq, summary = engine.run_full("balanced", SCENARIO_WEIGHTS["balanced"])
            st.success("Simulation complete.")
            st.dataframe(dfq.head(), use_container_width=True)
            st.markdown("**Summary**")
            st.json(summary, expanded=False)

        with st.expander("Panel OLS (placeholder)"):
            if PanelOLS is None:
                st.info("Install `linearmodels` to enable Panel OLS with real panel data.")
            else:
                st.info("Wire your true panel dataset (county × year) here and estimate PanelOLS with FE/RE as needed.")

    # ---------- AI GOVERNANCE ----------
    with tabs[4]:
        st.subheader("Agentic RAG + RL Teacher/Advisor")
        df_ranked = st.session_state.get("df_ranked", create_demo_plants())

        ke = KnowledgeEngine()
        ke.ingest(str(df_ranked.describe()), tag="stats")
        ke.ingest("EU AI Act, Data Act, GDPR, Cybersecurity Act context ingested for dairy AI systems.", tag="regulation")
        ke.ingest("Irish dairy processing: emissions, circular supply chains, resource efficiency, water and energy usage.", tag="sustainability")

        teacher = RLTeacher(min_score=cfg["rl_teacher"]["min_score"])
        advisor = RLAdvisor()

        q = st.text_area("Governance question",
                         "Explain policy impact on Irish dairy profitability under EU AI Act compliance with uncertainty and fairness.")
        if st.button("Run Agentic Governance"):
            out = ke.query(q)
            st.markdown("**Knowledge Engine Output**")
            st.write(out)

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

    # ---------- COMPLIANCE ----------
    with tabs[5]:
        st.subheader("EU Regulatory Framework — Alignment View")
        regs = [
            {"Regulation": "EU AI Act", "Scope": "High-risk AI systems in agri/food", "Focus": "Transparency, risk mgmt, human oversight", "Status": "Conceptually integrated"},
            {"Regulation": "Data Governance Act", "Scope": "Data intermediaries, reuse of public-sector data", "Focus": "Data sharing & stewardship", "Status": "Conceptually integrated"},
            {"Regulation": "Data Act", "Scope": "Access to industrial and IoT data", "Focus": "Data access, portability", "Status": "Planned connector"},
            {"Regulation": "GDPR", "Scope": "Personal data", "Focus": "Privacy, consent, rights", "Status": "No personal data ingested in demo"},
            {"Regulation": "Cybersecurity Act", "Scope": "ICT products/services", "Focus": "Security certification", "Status": "Out-of-scope (infrastructure)"},
        ]
        st.dataframe(pd.DataFrame(regs), use_container_width=True, hide_index=True)
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

    # ---------- SYSTEM LOG ----------
    with tabs[7]:
        st.subheader("System Log")
        log_path = Path("dairy_decision_v2.log")
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
            tail = "".join(lines[-200:]) if lines else "(log empty)"
            st.text_area("Last 200 log lines", tail, height=300)
        else:
            st.info("Log file not created yet.")


if __name__ == "__main__":
    main()
