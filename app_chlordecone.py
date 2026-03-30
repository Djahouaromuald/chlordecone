"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Tableau de bord interactif — Contamination au Chlordécone en Martinique     ║
║  Romuald DJAHOUA | ENSAR | 2025-2026                                         ║
║                                                                              ║
║  Usage :  pip install dash dash-bootstrap-components plotly pandas numpy     ║
║           python app_chlordecone.py                                          ║
║           → http://127.0.0.1:8050                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from pathlib import Path
import re

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# 1. Charte graphique
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":          "#F8F9FA",
    "card":        "#FFFFFF",
    "border":      "#333638",
    "text":        "#212529",
    "text_sub":    "#6C757D",
    "seuil":       "#C0392B",
    "green":       "#27AE60",
    "orange":      "#F5A623",
    "red":         "#E94560",
    "blue":        "#2980B9",
    "dark":        "#1A1A2E",
    "andosol":     "#8E1B1B",
    "ferralsol":   "#D4500A",
    "nitisol":     "#F5A623",
    "vertisol":    "#27AE60",
    "alluvium":    "#2980B9",
}

SOL_COLORS = {
    "Andosol":             "#8E1B1B",
    "Ferralsol":           "#D4500A",
    "Nitisol":             "#F5A623",
    "Vertisol":            "#27AE60",
    "Alluvium, Colluvium": "#2980B9",
}
CLASSE_COLORS = {
    "Non contaminé":       "#2196F3",
    "Contaminé modéré":    "#FF9800",
    "Fortement contaminé": "#C62828",
}
PLUVIO_COLORS = {
    "Très sec":     "#FFD54F",
    "Sec":          "#FFB300",
    "Sub-humide":   "#66BB6A",
    "Humide":       "#1E88E5",
    "Très humide":  "#1565C0",
    "Hyper-humide": "#4A148C",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text"]),
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor="#FAFAFA",
        xaxis=dict(gridcolor="#EEEEEE", linecolor="#DDDDDD"),
        yaxis=dict(gridcolor="#EEEEEE", linecolor="#DDDDDD"),
        margin=dict(l=50, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#CCCCCC"),
    )
)

SEUIL = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# 2. Chargement et préparation des données
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare() -> pd.DataFrame:
    csv_path = Path("BaseCLD2026.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "BaseCLD2026.csv introuvable — placez-le dans le même dossier que app_chlordecone.py"
        )

    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8")

    # Dates
    for col in ["Date_prelevement", "Date_enregistrement", "Date_analyse"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Sol
    df["Sol_simple"] = df["Sol_simple"].replace({"No data": np.nan, "Urban area": np.nan})

    # Taux_5b
    def parse_num(val):
        if pd.isna(val): return np.nan
        try:
            f = float(str(val).replace(",", ".").strip())
            return np.nan if np.isinf(f) else f
        except (ValueError, OverflowError):
            return np.nan

    df["Taux_5b_hydro_num"] = df["Taux_5b_hydro"].apply(parse_num)

    # Censure
    df["cld_censure"] = (df["Operateur_chld"] == "<").astype(int)

    # COMMU_LAB
    df["COMMU_LAB"] = df["COMMU_LAB"].fillna(df["COMMU_LAB"].mode()[0])

    # Log
    df["log_cld"] = np.log1p(df["Taux_Chlordecone"])

    # Classe
    def classe(v):
        if v < 0.1:  return "Non contaminé"
        elif v < 1.0: return "Contaminé modéré"
        else:         return "Fortement contaminé"

    df["classe_cld"] = df["Taux_Chlordecone"].apply(classe)

    # Pente
    def cat_pente(p):
        if p <  5: return "Plat"
        elif p < 15: return "Peu pentu"
        elif p < 30: return "Pentu"
        else:        return "Très pentu"

    df["cat_pente"] = df["mnt_pente_mean"].apply(cat_pente)

    # Pluviométrie
    ref_pluvio = {"0-1250":"Très sec","1250-1500":"Sec","1500-2000":"Sub-humide",
                  "2000-3000":"Humide","3000-5000":"Très humide","5000-8000":"Hyper-humide"}
    df["pluvio_label"] = df["RAIN"].map(ref_pluvio)
    df["rain_mm"]      = df["RAIN"].map({"0-1250":625,"1250-1500":1375,"1500-2000":1750,
                                          "2000-3000":2500,"3000-5000":4000,"5000-8000":6500})

    # Historique bananier
    histo_map = {1.0:"Faible",2.0:"Modéré",3.0:"Intense"}
    df["histo_banane_cat"] = df["histoBanane_Histo_ban"].map(histo_map).fillna("Inconnu")

    return df


df = load_and_prepare()

# ─── Données agrégées pour les graphiques ─────────────────────────────────────
agg_annee = (
    df.groupby("ANNEE")["Taux_Chlordecone"]
    .agg(mediane="median",
         q25=lambda x: x.quantile(0.25),
         q75=lambda x: x.quantile(0.75),
         q10=lambda x: x.quantile(0.10),
         q90=lambda x: x.quantile(0.90),
         n="count")
    .reset_index()
)

agg_sol = (
    df.dropna(subset=["Sol_simple"])
    .groupby("Sol_simple")["Taux_Chlordecone"]
    .agg(mediane="median", n="count",
         pct_fort=lambda x: (x > 1.0).mean() * 100)
    .sort_values("mediane", ascending=False)
    .reset_index()
)

agg_commune = (
    df.groupby("COMMU_LAB")["Taux_Chlordecone"]
    .agg(mediane="median", n="count",
         pct_fort=lambda x: (x > 1.0).mean() * 100)
    .sort_values("mediane", ascending=False)
    .reset_index()
)

df_spatial = (
    df.groupby("ID")
    .agg(X=("X","mean"), Y=("Y","mean"),
         cld_med=("Taux_Chlordecone","median"),
         classe=("classe_cld", lambda x: x.mode()[0]),
         Sol_simple=("Sol_simple","first"),
         commune=("COMMU_LAB","first"),
         annee=("ANNEE", lambda x: x.mode()[0]))
    .reset_index()
)

ALL_COMMUNES = ["Toutes"] + sorted(df["COMMU_LAB"].dropna().unique().tolist())
ALL_SOLS     = ["Tous"]   + sorted(df["Sol_simple"].dropna().unique().tolist())
YEARS        = sorted(df["ANNEE"].unique().tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fonctions de graphiques
# ─────────────────────────────────────────────────────────────────────────────
def apply_template(fig):
    fig.update_layout(
        font=PLOTLY_TEMPLATE["layout"]["font"],
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor="#FAFAFA",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#CCCCCC"),
    )
    fig.update_xaxes(gridcolor="#EEEEEE", linecolor="#DDDDDD", showgrid=True)
    fig.update_yaxes(gridcolor="#EEEEEE", linecolor="#DDDDDD", showgrid=True)
    return fig


def fig_kpis(dff):
    total  = len(dff)
    nc     = (dff["classe_cld"] == "Non contaminé").sum()
    cm     = (dff["classe_cld"] == "Contaminé modéré").sum()
    fc     = (dff["classe_cld"] == "Fortement contaminé").sum()
    med    = dff["Taux_Chlordecone"].median()
    moy    = dff["Taux_Chlordecone"].mean()
    return total, nc, cm, fc, med, moy


def fig_distribution(dff):
    fig = go.Figure()

    # Histogramme
    clip = dff["Taux_Chlordecone"].quantile(0.95)
    data_c = dff["Taux_Chlordecone"].clip(upper=clip)

    fig.add_trace(go.Histogram(
        x=data_c, nbinsx=60, name="Distribution",
        marker=dict(color="#4472C4", line=dict(width=0)),
        opacity=0.8, histnorm=""
    ))

    # Seuil
    fig.add_vline(x=SEUIL, line_dash="dash", line_color=PALETTE["seuil"],
                  annotation_text="Seuil ANSES (0.1 mg/kg)",
                  annotation_position="top right",
                  annotation_font=dict(color=PALETTE["seuil"], size=11))
    fig.add_vline(x=dff["Taux_Chlordecone"].median(),
                  line_dash="solid", line_color="#E65100",
                  annotation_text=f"Médiane : {dff['Taux_Chlordecone'].median():.4f}",
                  annotation_position="top left",
                  annotation_font=dict(color="#E65100", size=10))

    fig.update_layout(title="Distribution du taux de chlordécone (95ᵉ percentile)",
                      xaxis_title="Taux Chlordécone (mg/kg)",
                      yaxis_title="Nombre de parcelles",
                      showlegend=False, height=360)
    return apply_template(fig)


def fig_temporal(dff):
    agg = (
        dff.groupby("ANNEE")["Taux_Chlordecone"]
        .agg(mediane="median",
             q25=lambda x: x.quantile(0.25),
             q75=lambda x: x.quantile(0.75),
             n="count")
        .reset_index()
    )
    if len(agg) < 2:
        return go.Figure()

    fig = go.Figure()

    # Intervalle IQR
    fig.add_trace(go.Scatter(
        x=list(agg["ANNEE"]) + list(agg["ANNEE"])[::-1],
        y=list(agg["q75"]) + list(agg["q25"])[::-1],
        fill="toself", fillcolor="rgba(41,128,185,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="P25–P75"
    ))

    # Médiane
    fig.add_trace(go.Scatter(
        x=agg["ANNEE"], y=agg["mediane"],
        mode="lines+markers",
        line=dict(color=PALETTE["blue"], width=2.5),
        marker=dict(size=9, color=PALETTE["blue"],
                    line=dict(color="white", width=2)),
        name="Médiane",
        hovertemplate="<b>%{x}</b><br>Médiane : %{y:.4f} mg/kg<br>n=%{text}",
        text=[f"{n:,}" for n in agg["n"]]
    ))

    # Ligne de seuil
    fig.add_hline(y=SEUIL, line_dash="dash", line_color=PALETTE["seuil"],
                  annotation_text="Seuil ANSES", annotation_position="top right",
                  annotation_font=dict(color=PALETTE["seuil"]))

    # Tendance linéaire
    if len(agg) >= 3:
        m, b = np.polyfit(agg["ANNEE"], agg["mediane"], 1)
        fig.add_trace(go.Scatter(
            x=agg["ANNEE"], y=m * agg["ANNEE"] + b,
            mode="lines",
            line=dict(color="#95A5A6", dash="dot", width=1.5),
            name=f"Tendance ({m:+.4f}/an)",
            hoverinfo="skip"
        ))

    fig.update_layout(
        title="Évolution temporelle de la contamination",
        xaxis_title="Année", yaxis_title="Taux Chlordécone médian (mg/kg)",
        xaxis=dict(tickmode="array", tickvals=agg["ANNEE"].tolist()),
        height=360, legend=dict(orientation="h", y=-0.2)
    )
    return apply_template(fig)


def fig_by_sol(dff):
    data = (
        dff.dropna(subset=["Sol_simple"])
        .groupby("Sol_simple")["Taux_Chlordecone"]
        .agg(mediane="median", n="count",
             pct_fort=lambda x: (x > 1.0).mean() * 100)
        .sort_values("mediane", ascending=False)
        .reset_index()
    )
    if data.empty:
        return go.Figure()

    colors = [SOL_COLORS.get(s, "#7F8C8D") for s in data["Sol_simple"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["Sol_simple"], y=data["mediane"],
        marker_color=colors,
        text=[f"{v:.4f}" for v in data["mediane"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Médiane : %{y:.4f} mg/kg<br>n=%{customdata:,}",
        customdata=data["n"],
        name="Médiane CLD"
    ))
    fig.add_hline(y=SEUIL, line_dash="dash", line_color=PALETTE["seuil"],
                  annotation_text="Seuil ANSES")
    fig.update_layout(
        title="Contamination médiane par type de sol",
        xaxis_title="Type de sol", yaxis_title="Taux Chlordécone médian (mg/kg)",
        showlegend=False, height=360
    )
    return apply_template(fig)


def fig_commune_lollipop(dff):
    data = (
        dff.groupby("COMMU_LAB")["Taux_Chlordecone"]
        .agg(mediane="median", n="count")
        .sort_values("mediane")
        .reset_index()
    )
    if data.empty:
        return go.Figure()

    def col(v):
        if v > 1.0: return PALETTE["red"]
        elif v >= SEUIL: return PALETTE["orange"]
        else: return PALETTE["green"]

    colors = [col(v) for v in data["mediane"]]

    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_shape(type="line",
                      x0=0, x1=row["mediane"],
                      y0=row["COMMU_LAB"], y1=row["COMMU_LAB"],
                      line=dict(color="#BDC3C7", width=1.5))
    fig.add_trace(go.Scatter(
        x=data["mediane"], y=data["COMMU_LAB"],
        mode="markers",
        marker=dict(color=colors, size=10, line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{y}</b><br>Médiane : %{x:.4f} mg/kg<br>n=%{customdata:,}",
        customdata=data["n"]
    ))
    fig.add_vline(x=SEUIL, line_dash="dash", line_color=PALETTE["seuil"])
    fig.add_vline(x=1.0,   line_dash="dot",  line_color=PALETTE["red"])

    fig.update_layout(
        title="Taux médian de chlordécone par commune",
        xaxis_title="Taux Chlordécone médian (mg/kg)",
        height=max(400, len(data) * 22),
        showlegend=False,
        margin=dict(l=140, r=30, t=50, b=50)
    )
    return apply_template(fig)


def fig_heatmap_commune_annee(dff):
    pivot = (
        dff.groupby(["COMMU_LAB","ANNEE"])["Taux_Chlordecone"]
        .median().unstack("ANNEE")
    )
    if pivot.empty:
        return go.Figure()

    commune_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[commune_order]
    years_l = sorted(pivot.columns.tolist())

    z     = pivot[years_l].values
    text  = [[f"{v:.3f}" if not np.isnan(v) else "—"
              for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(y) for y in years_l], y=commune_order,
        text=text, texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale=[[0,"#FFF9E6"],[0.25,"#FFD54F"],
                    [0.5,"#FF6D00"],[0.75,"#B71C1C"],[1,"#4A0000"]],
        hovertemplate="<b>%{y}</b><br>%{x}<br>Médiane : %{z:.4f} mg/kg",
        colorbar=dict(title="mg/kg", tickfont=dict(size=9))
    ))
    fig.update_layout(
        title="Heatmap — Taux médian de chlordécone (commune × année)",
        xaxis_title="Année", yaxis_title="Commune",
        height=max(400, len(commune_order)*24+100),
        margin=dict(l=150, r=50, t=60, b=50)
    )
    return apply_template(fig)


def fig_spatial(dff):
    dfs = (
        dff.groupby("ID")
        .agg(X=("X","mean"), Y=("Y","mean"),
             cld_med=("Taux_Chlordecone","median"),
             classe=("classe_cld", lambda x: x.mode()[0]),
             commune=("COMMU_LAB","first"))
        .reset_index()
    )
    if dfs.empty:
        return go.Figure()

    fig = go.Figure()
    for cls, col in CLASSE_COLORS.items():
        sub = dfs[dfs["classe"]==cls]
        sz  = 4 if cls=="Non contaminé" else (8 if cls=="Contaminé modéré" else 12)
        fig.add_trace(go.Scatter(
            x=sub["X"], y=sub["Y"],
            mode="markers",
            marker=dict(color=col, size=sz, opacity=0.6,
                        line=dict(width=0)),
            name=f"{cls} (n={len(sub):,})",
            hovertemplate="<b>%{text}</b><br>X=%{x:.0f}<br>Y=%{y:.0f}<br>CLD=%{customdata:.4f} mg/kg",
            text=sub["commune"],
            customdata=sub["cld_med"]
        ))
    fig.update_layout(
        title="Distribution spatiale de la contamination (Lambert RRAF)",
        xaxis_title="X (Lambert RRAF)", yaxis_title="Y (Lambert RRAF)",
        xaxis_scaleanchor="y",
        height=460,
        legend=dict(orientation="h", y=-0.15)
    )
    return apply_template(fig)


def fig_pluvio(dff):
    ordre = ["Très sec","Sec","Sub-humide","Humide","Très humide","Hyper-humide"]
    data  = (
        dff.dropna(subset=["pluvio_label"])
        .groupby("pluvio_label")["Taux_Chlordecone"]
        .agg(mediane="median", n="count")
        .reindex([p for p in ordre if p in dff["pluvio_label"].dropna().unique()])
        .reset_index()
    )
    if data.empty:
        return go.Figure()

    colors = [PLUVIO_COLORS.get(p,"#888") for p in data["pluvio_label"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["pluvio_label"], y=data["mediane"],
        marker_color=colors,
        text=[f"{v:.4f}" for v in data["mediane"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Médiane : %{y:.4f} mg/kg<br>n=%{customdata:,}",
        customdata=data["n"]
    ))
    fig.add_hline(y=SEUIL, line_dash="dash", line_color=PALETTE["seuil"],
                  annotation_text="Seuil ANSES")
    fig.update_layout(
        title="Contamination selon la classe de pluviométrie",
        xaxis_title="Classe pluviométrique", yaxis_title="Taux Chlordécone médian (mg/kg)",
        showlegend=False, height=360
    )
    return apply_template(fig)


def fig_histo_banane(dff):
    ordre    = ["Faible","Modéré","Intense"]
    hb_col   = {"Faible":"#A5D6A7","Modéré":"#FFB74D","Intense":"#EF5350"}
    df_hb    = dff[dff["histo_banane_cat"]!="Inconnu"].copy()
    if df_hb.empty:
        return go.Figure()

    data_stacked = []
    for h in ordre:
        sub = df_hb[df_hb["histo_banane_cat"]==h]; total = len(sub)
        if total == 0: continue
        data_stacked.append({
            "Historique": h,
            "Non contaminé":       (sub["classe_cld"]=="Non contaminé").sum()/total*100,
            "Contaminé modéré":    (sub["classe_cld"]=="Contaminé modéré").sum()/total*100,
            "Fortement contaminé": (sub["classe_cld"]=="Fortement contaminé").sum()/total*100,
        })
    if not data_stacked:
        return go.Figure()

    df_s = pd.DataFrame(data_stacked)
    fig  = go.Figure()
    for cls, col in zip(["Non contaminé","Contaminé modéré","Fortement contaminé"],
                        ["#2E7D32","#E65100","#B71C1C"]):
        if cls in df_s.columns:
            fig.add_trace(go.Bar(
                x=df_s["Historique"], y=df_s[cls],
                name=cls, marker_color=col,
                hovertemplate=f"<b>%{{x}}</b><br>{cls} : %{{y:.1f}}%"
            ))
    fig.update_layout(
        barmode="stack", title="Classes de contamination par historique bananier",
        xaxis_title="Intensité bananière", yaxis_title="%",
        yaxis=dict(range=[0,100]),
        legend=dict(orientation="h", y=-0.2),
        height=360
    )
    return apply_template(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Layout
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(title, value, subtitle="", color=PALETTE["blue"]):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted small mb-1"),
            html.H4(value, style={"color": color, "fontWeight": "700", "marginBottom": "2px"}),
            html.P(subtitle, className="text-muted small mb-0"),
        ]),
        className="h-100 border-0 shadow-sm",
        style={"borderLeft": f"4px solid {color} !important",
               "borderRadius": "8px"}
    )


sidebar = dbc.Card(
    dbc.CardBody([
        html.H6(" Filtres", className="fw-bold mb-3"),
        html.Label("Commune", className="small fw-bold"),
        dcc.Dropdown(
            id="filter-commune",
            options=[{"label": c, "value": c} for c in ALL_COMMUNES],
            value="Toutes", clearable=False, className="mb-3"
        ),
        html.Label("Type de sol", className="small fw-bold"),
        dcc.Dropdown(
            id="filter-sol",
            options=[{"label": s, "value": s} for s in ALL_SOLS],
            value="Tous", clearable=False, className="mb-3"
        ),
        html.Label("Années", className="small fw-bold"),
        dcc.RangeSlider(
            id="filter-years",
            min=YEARS[0], max=YEARS[-1],
            value=[YEARS[0], YEARS[-1]],
            marks={y: str(y) for y in YEARS},
            step=1, className="mb-3",
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        html.Label("Classe de contamination", className="small fw-bold"),
        dcc.Checklist(
            id="filter-classe",
            options=[
                {"label": " Non contaminé",       "value": "Non contaminé"},
                {"label": " Contaminé modéré",    "value": "Contaminé modéré"},
                {"label": " Fortement contaminé", "value": "Fortement contaminé"},
            ],
            value=["Non contaminé","Contaminé modéré","Fortement contaminé"],
            className="small mb-3"
        ),
        html.Hr(),
        html.Div(id="sidebar-stats", className="small text-muted"),
    ]),
    className="border-0 shadow-sm h-100",
    style={"borderRadius": "10px", "position": "sticky", "top": "10px"}
)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    ],
    title="Chlordécone — Martinique"
)
server = app.server  # Pour déploiement WSGI

app.layout = dbc.Container([
    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Contamination au Chlordécone — Martinique",
                        className="mb-0 fw-bold",
                        style={"color": PALETTE["dark"]}),
                html.P("DA03 – IDD | ENSAR | 2025-2026 | Données : 31 126 parcelles (2010–2019)",
                       className="text-muted small mb-0"),
            ], className="py-3")
        ])
    ]),
    html.Hr(className="mt-0 mb-3"),

    # ── Body ──────────────────────────────────────────────────────────────────
    dbc.Row([
        # Sidebar
        dbc.Col(sidebar, width=2),

        # Main content
        dbc.Col([
            # KPIs
            dbc.Row([
                dbc.Col(html.Div(id="kpi-total"),   width=2),
                dbc.Col(html.Div(id="kpi-nc"),      width=2),
                dbc.Col(html.Div(id="kpi-cm"),      width=2),
                dbc.Col(html.Div(id="kpi-fc"),      width=2),
                dbc.Col(html.Div(id="kpi-median"),  width=2),
                dbc.Col(html.Div(id="kpi-mean"),    width=2),
            ], className="mb-3 g-2"),

            # Tabs
            dbc.Tabs([
                dbc.Tab(label=" Distribution", tab_id="tab-dist"),
                dbc.Tab(label=" Temporelle",   tab_id="tab-time"),
                dbc.Tab(label=" Spatiale",     tab_id="tab-map"),
                dbc.Tab(label=" Sol & Pluie",  tab_id="tab-soil"),
                dbc.Tab(label=" Communes",     tab_id="tab-comm"),
                dbc.Tab(label=" Bananier",     tab_id="tab-histo"),
            ], id="tabs", active_tab="tab-dist", className="mb-3"),

            html.Div(id="tab-content")

        ], width=10),
    ]),

    # Footer
    html.Hr(className="mt-4"),
    html.P("Source : BaseCLD2026 — ENSAR DA03 IDD 2025-2026",
           className="text-center text-muted small pb-2"),

], fluid=True, style={"backgroundColor": PALETTE["bg"], "minHeight": "100vh",
                       "fontFamily": "Inter, system-ui, sans-serif"})


# ─────────────────────────────────────────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def filter_df(commune, sol, years, classes):
    dff = df.copy()
    if commune and commune != "Toutes":
        dff = dff[dff["COMMU_LAB"] == commune]
    if sol and sol != "Tous":
        dff = dff[dff["Sol_simple"] == sol]
    if years:
        dff = dff[dff["ANNEE"].between(years[0], years[1])]
    if classes:
        dff = dff[dff["classe_cld"].isin(classes)]
    return dff


@app.callback(
    [Output("kpi-total", "children"),
     Output("kpi-nc",    "children"),
     Output("kpi-cm",    "children"),
     Output("kpi-fc",    "children"),
     Output("kpi-median","children"),
     Output("kpi-mean",  "children"),
     Output("sidebar-stats", "children")],
    [Input("filter-commune", "value"),
     Input("filter-sol",     "value"),
     Input("filter-years",   "value"),
     Input("filter-classe",  "value")]
)
def update_kpis(commune, sol, years, classes):
    dff = filter_df(commune, sol, years, classes)
    total, nc, cm, fc, med, moy = fig_kpis(dff)

    cards = [
        kpi_card("Observations",          f"{total:,}",          color=PALETTE["dark"]),
        kpi_card("Non contaminées",        f"{nc:,}",             f"{nc/max(total,1)*100:.1f}%", PALETTE["green"]),
        kpi_card("Contaminées modérées",   f"{cm:,}",             f"{cm/max(total,1)*100:.1f}%", PALETTE["orange"]),
        kpi_card("Fortement contaminées",  f"{fc:,}",             f"{fc/max(total,1)*100:.1f}%", PALETTE["red"]),
        kpi_card("Médiane CLD",            f"{med:.4f} mg/kg",   color=PALETTE["blue"]),
        kpi_card("Moyenne CLD",            f"{moy:.4f} mg/kg",   color=PALETTE["blue"]),
    ]

    stats = [
        html.Strong("Sélection active"),
        html.P(f"{total:,} obs. sur {len(df):,}", className="mb-1"),
        html.P(f"≥ Seuil ANSES : {(dff['Taux_Chlordecone']>=SEUIL).sum():,} ({(dff['Taux_Chlordecone']>=SEUIL).mean()*100:.1f}%)",
               className="mb-0")
    ]
    return [*cards, stats]


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs",            "active_tab"),
     Input("filter-commune", "value"),
     Input("filter-sol",     "value"),
     Input("filter-years",   "value"),
     Input("filter-classe",  "value")]
)
def render_tab(tab, commune, sol, years, classes):
    dff = filter_df(commune, sol, years, classes)

    if tab == "tab-dist":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_distribution(dff)), width=12),
        ])

    elif tab == "tab-time":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_temporal(dff)), width=12),
        ])

    elif tab == "tab-map":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_spatial(dff)), width=12),
        ])

    elif tab == "tab-soil":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_by_sol(dff)),  width=6),
            dbc.Col(dcc.Graph(figure=fig_pluvio(dff)),  width=6),
        ])

    elif tab == "tab-comm":
        return dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_commune_lollipop(dff), style={"overflowY":"auto"}),
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=fig_heatmap_commune_annee(dff), style={"overflowY":"auto"}),
            ], width=6),
        ])

    elif tab == "tab-histo":
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_histo_banane(dff)), width=12),
        ])

    return html.P("Onglet non reconnu.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Lancement
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Tableau de bord Chlordécone — Martinique")
    print("  URL : http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)
