"""
Proyecto de Estadistica I
================================================
Estructura:
  0. Carga y cálculo de variables derivadas
  1. Header + intro
  2. Datos con sus tablas de frecuencias
  3. Análisis de Población
  4. Análisis de Área
  5. Análisis de Densidad
  6. Análisis de Porcentaje Mundial

CSV fuente: paises.csv  (columnas: country, pop2026, area)
Variables de densidad y porcentaje mundial se calculan en el codigo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc


# #AZUL: CONFIGURACIÓN DE NOMBRE EN NAVEGADOR Y PAGINA

st.set_page_config(
    page_title="Proyecto de Estadística I. Estudiante: Frank Marcelo Romero Villanueva",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# #VERDE: ESTILOS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
}

/* ── Títulos principales ── */
.main-header {
    background: linear-gradient(90deg, #00d4ff 0%, #0099cc 50%, #0066ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.sub-header {
    color: #8899aa;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}
.section-title {
    color: #cce4ff;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin-bottom: 0.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a5f;
}
.section-subtitle {
    color: #556677;
    font-size: 0.8rem;
    margin-bottom: 1.2rem;
}

/* ── Cuadros de texto informativos ── */
.info-box {
    background: linear-gradient(135deg, #0d2035 0%, #0a1a2e 100%);
    border-left: 4px solid #00d4ff;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 1.4rem;
    color: #aabbcc;
    font-size: 0.92rem;
    line-height: 1.75;
}
.info-box strong { color: #00d4ff; }

/* ── Métricas ── */
.metric-card {
    background: linear-gradient(145deg, #0d1f35, #112244);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 20px rgba(0,212,255,.05);
}
.metric-label {
    color: #6688aa;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.35rem;
}
.metric-value {
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    line-height: 1.2;
}
.metric-sub { color: #445566; font-size: 0.72rem; margin-top: 0.25rem; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: #0d2540;
    color: #00d4ff;
    border: 1px solid #00d4ff33;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    vertical-align: middle;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #080e1a;
    border-radius: 10px;
    gap: 3px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    color: #556677;
    border-radius: 7px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: .4px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #00d4ff !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e3a5f;
}

hr { border-color: #1e3a5f; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# #ROJO: CONSTANTES

WORLD_POP = 8_298_978_817

# #: base layout para gráficos Plotly (colores, fuentes, fondos, etc.)
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora, sans-serif", color="#8899aa", size=12),
    title_font=dict(color="#cce4ff", size=15),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899aa")),
)

# #VERDE: CARGA DEL ARCHIVO CSV Y CÁLCULO DE VARIABLES DERIVADAS

@st.cache_data
def cargar_datos(path: str = "paises.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"country", "pop2026", "area"}
    if not required.issubset(df.columns):
        st.error(f"El CSV debe contener: {required}")
        st.stop()

    df = df.dropna(subset=list(required))
    df["pop2026"] = pd.to_numeric(df["pop2026"], errors="coerce")
    df["area"]    = pd.to_numeric(df["area"],    errors="coerce")
    df = df.dropna(subset=["pop2026", "area"])

    # #: Variables derivadas (Las tenia el CSV original, pero las borre para crearlas usando los valores principales)
    df["density"]         = (df["pop2026"] / df["area"]).round(2)
    df["worldPercentage"] = ((df["pop2026"] / WORLD_POP) * 100).round(6)
    df["ranking"]         = df["pop2026"].rank(ascending=False, method="min").astype(int)

    return df.sort_values("ranking").reset_index(drop=True)


df = cargar_datos()

# #VERDE: HELPERS

def fmt(n, dec=0):
    return f"{n:,.{dec}f}"


def tabla_frecuencias(df_col: pd.Series, labels: list, bins) -> pd.DataFrame:
    """
    Esta parte construye tabla de frecuencias completa dado una columna y bins/labels.
    Devuelve: Intervalo, Marca de clase, fi, hi, hi%, Fi, Hi
    """
    cats = pd.cut(df_col, bins=bins, labels=labels, right=False, include_lowest=True)
    fi = cats.value_counts().reindex(labels, fill_value=0)

    # #: Marcas de clase
    midpoints = []
    for i in range(len(bins) - 1):
        lo, hi_ = bins[i], bins[i + 1]
        if hi_ == np.inf:
            # #: ACLARACION: Para el último bin abierto usamos lo + (lo - bins[i-2])/2 si hay anterior
            hi_ = lo * 1.5 if lo > 0 else lo + 1e9
        midpoints.append(round((lo + hi_) / 2, 2))

    tf = pd.DataFrame({
        "Intervalo":      labels,
        "Marca de clase": midpoints,
        "fi":             fi.values,
    })

    total = tf["fi"].sum()
    tf["hi"]  = (tf["fi"] / total).round(4)
    tf["hi%"] = (tf["hi"] * 100).round(2)
    tf["Fi"]  = tf["fi"].cumsum()
    tf["Hi"]  = tf["hi"].cumsum().round(4)

    # #: Fila TOTAL
    total_row = pd.DataFrame([{
        "Intervalo": "TOTAL",
        "Marca de clase": "—",
        "fi":  tf["fi"].sum(),
        "hi":  tf["hi"].sum().round(4),
        "hi%": tf["hi%"].sum().round(2),
        "Fi":  tf["Fi"].iloc[-1],
        "Hi":  tf["Hi"].iloc[-1].round(4),
    }])
    return pd.concat([tf, total_row], ignore_index=True)


# #ROJO: INTERVALOS FIJOS POR VARIABLE

# ── Población ────────────────────────────────
POP_BINS = [0, 100_000, 1_000_000, 5_000_000, 10_000_000,
            20_000_000, 50_000_000, 100_000_000, 1_000_000_000, np.inf]
POP_LABELS = [
    "< 100 mil",
    "100 mil – 1 M",
    "1 M – 5 M",
    "5 M – 10 M",
    "10 M – 20 M",
    "20 M – 50 M",
    "50 M – 100 M",
    "100 M – 1 B",
    "> 1 B",
]

# #: ── Área (km²) ───────────────────────────────
# #: Micro (<1k), Pequeño (1k-10k), Mediano (10k-100k),
# #: Grande (100k-500k), Muy grande (500k-2M), Gigante (>2M)
AREA_BINS = [0, 1_000, 10_000, 100_000, 500_000, 2_000_000, np.inf]
AREA_LABELS = [
    "< 1 000 km²",
    "1 000 – 10 000 km²",
    "10 000 – 100 000 km²",
    "100 000 – 500 000 km²",
    "500 000 – 2 000 000 km²",
    "> 2 000 000 km²",
]

# #: ── Densidad (hab/km²) ────────────────────────
# #: Despoblado, Bajo, Moderado, Alto, Muy alto, Extremo
DENS_BINS = [0, 10, 50, 100, 200, 500, np.inf]
DENS_LABELS = [
    "< 10 hab/km²",
    "10 – 50 hab/km²",
    "50 – 100 hab/km²",
    "100 – 200 hab/km²",
    "200 – 500 hab/km²",
    "> 500 hab/km²",
]

# #:  % Mundial
PCT_BINS = [0, 0.01, 0.1, 0.5, 1, 5, np.inf]
PCT_LABELS = [
    "< 0.01%",
    "0.01% – 0.1%",
    "0.1% – 0.5%",
    "0.5% – 1%",
    "1% – 5%",
    "> 5%",
]

# #: Asignar columnas de categoría (para uso en gráficas)
df["cat_pop"]  = pd.cut(df["pop2026"],         bins=POP_BINS,  labels=POP_LABELS,  right=False, include_lowest=True)
df["cat_area"] = pd.cut(df["area"],            bins=AREA_BINS, labels=AREA_LABELS, right=False, include_lowest=True)
df["cat_dens"] = pd.cut(df["density"],         bins=DENS_BINS, labels=DENS_LABELS, right=False, include_lowest=True)
df["cat_pct"]  = pd.cut(df["worldPercentage"], bins=PCT_BINS,  labels=PCT_LABELS,  right=False, include_lowest=True)

# #AZUL: HEADER

st.markdown('<p class="main-header">Proyecto de Estadística I. Estudiante: Frank Marcelo Romero Villanueva</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ANALISIS DE DATOS DE POBLACION DE PAISES Y ENTIDADES TERRITORIALES</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    Dashboard interactivo de análisis estadístico de la población mundial Een paises y territorios proyectada para 2026.
    El dataset fuente contiene únicamente <strong>país, población y área</strong>; todas las demás
    variables (densidad, % mundial, ranking) se calculan automáticamente.<br>
    Los datos fueron obtenidos de la pagina worldpopulationreview.com
</div>
""", unsafe_allow_html=True)

st.divider()

# #ROJO: SECCIÓN 0 — DATOS COMPLETOS (CSV + derivadas)

st.markdown('<p class="section-title">DATOS GENERALES</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Todas las columnas: CSV fuente + variables calculadas</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>ℹ️ Nota:</strong> El CSV original contiene <em>country</em>, <em>pop2026</em> y <em>area</em>.
    Las columnas <em>density</em>, <em>worldPercentage</em> y <em>ranking</em> se obienen calculadolas con los datos previos.
</div>
""", unsafe_allow_html=True)

tabla_full = df[["ranking", "country", "pop2026", "area", "density", "worldPercentage"]].copy()
tabla_full.columns = ["Ranking", "País", "Población 2026", "Área (km²)", "Densidad (hab/km²)", "% Mundial"]

st.dataframe(
    tabla_full,
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Ranking":           st.column_config.NumberColumn(format="%d",      width="small"),
        "País":              st.column_config.TextColumn(width="medium"),
        "Población 2026":    st.column_config.NumberColumn(format="%,d"),
        "Área (km²)":        st.column_config.NumberColumn(format="%,.0f"),
        "Densidad (hab/km²)":st.column_config.NumberColumn(format="%.2f"),
        "% Mundial":         st.column_config.ProgressColumn(
                                 format="%.4f%%",
                                 min_value=0,
                                 max_value=float(df["worldPercentage"].max()),
                             ),
    },
)

st.divider()

# #ROJO: SECCIÓN 1 — ANÁLISIS DE POBLACIÓN

st.markdown('<p class="section-title">👥 Análisis de Población</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    Se analizan los datos de poblacion de estos paises y territorios.
</div>
""", unsafe_allow_html=True)

# #: Métricas de población

c1, c2, c3, c4, c5 = st.columns(5)
pop_vals = df["pop2026"]

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">🌐 Países</div>
        <div class="metric-value">{len(df)}</div>
        <div class="metric-sub">en el dataset</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📊 Media</div>
        <div class="metric-value">{fmt(pop_vals.mean(), 0)}</div>
        <div class="metric-sub">habitantes</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📍 Mediana</div>
        <div class="metric-value">{fmt(pop_vals.median(), 0)}</div>
        <div class="metric-sub">habitantes</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Pais mas poblado (Máximo)</div>
        <div class="metric-value">{fmt(pop_vals.max(), 0)}</div>
        <div class="metric-sub">{df.loc[pop_vals.idxmax(), 'country']}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Pais menos poblado (Mínimo)</div>
        <div class="metric-value">{fmt(pop_vals.min(), 0)}</div>
        <div class="metric-sub">{df.loc[pop_vals.idxmin(), 'country']}</div>
    </div>""", unsafe_allow_html=True)

# #: Tabla de frecuencias — Población

st.markdown("#### 📋 Tabla de Frecuencias de Población")

st.markdown("""
<div class="info-box">
    Los países se agruparon en 9 intervalos que reflejan órdenes de magnitud poblacional,
    desde microestados con menos de 100 000 habitantes hasta las dos superpotencias demográficas
    que superan el billón.
</div>
""", unsafe_allow_html=True)

tf_pop = tabla_frecuencias(df["pop2026"], POP_LABELS, POP_BINS)
tf_pop_display = pd.concat([
    tf_pop[tf_pop["Intervalo"] != "TOTAL"].iloc[::-1].reset_index(drop=True),
    tf_pop[tf_pop["Intervalo"] == "TOTAL"]
], ignore_index=True)

st.dataframe(tf_pop_display, use_container_width=True, hide_index=True,
    column_config={
        "fi":  st.column_config.NumberColumn("fi",  format="%d"),
        "Fi":  st.column_config.NumberColumn("Fi",  format="%d"),
        "hi":  st.column_config.NumberColumn("hi",  format="%.4f"),
        "Hi":  st.column_config.NumberColumn("Hi",  format="%.4f"),
        "hi%": st.column_config.NumberColumn("hi%", format="%.2f%%"),
    }
)

# #: Gráficas — Población

st.markdown("#### 📊 Graficas de Población")

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    Las gráficas a continuación muestran la distribución por intervalos, el avance acumulado
    (ojiva) y la composición porcentual de la población mundial.
</div>
""", unsafe_allow_html=True)

tf_pop_data = tf_pop[tf_pop["Intervalo"] != "TOTAL"].copy()
# #VERDE: ACLARACION: Para gráficos EL orden de mayor a menor rango (> 1 B primero)
tf_pop_data_rev = tf_pop_data.iloc[::-1].reset_index(drop=True)
# #VERDE: Aqui en los colores, el mayor rango recibe el color más oscuro de la escala de azules
_blues_colors = pc.sample_colorscale("Blues", [i / (len(tf_pop_data_rev) - 1) for i in range(len(tf_pop_data_rev))])
_blues_colors_rev = list(reversed(_blues_colors))  # índice 0 = mayor rango = más oscuro

tab_p1, tab_p2, tab_p3, tab_p4 = st.tabs([
    "📊 Barras por intervalo",
    "🎯 Bastones (fi)",
    "🥧 Torta",
    "📈 Ojiva",
])

with tab_p1:
    fig = go.Figure(go.Bar(
        x=tf_pop_data_rev["Intervalo"],
        y=tf_pop_data_rev["fi"],
        marker=dict(color=_blues_colors_rev, showscale=False,
                    line=dict(color="#00d4ff", width=0.5)),
        text=tf_pop_data_rev["fi"],
        textposition="outside",
        textfont=dict(color="#8899aa", size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, title="Cantidad de países por intervalo de población",
                      xaxis_tickangle=-30, height=420,
                      xaxis_title="Intervalo de Población",
                      yaxis_title="Número de países (fi)")
    st.plotly_chart(fig, use_container_width=True)

with tab_p2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_pop_data_rev["Intervalo"], y=tf_pop_data_rev["fi"],
        mode="markers+lines",
        marker=dict(color="#00d4ff", size=10, symbol="circle"),
        line=dict(color="#00d4ff", width=1, dash="dot"),
    ))
    # Líneas verticales (bastones)
    for _, row in tf_pop_data_rev.iterrows():
        fig.add_shape(type="line",
            x0=row["Intervalo"], x1=row["Intervalo"],
            y0=0, y1=row["fi"],
            line=dict(color="#0099cc", width=2))
    fig.update_layout(**BASE_LAYOUT, title="Diagrama de bastones — Frecuencia absoluta",
                      xaxis_tickangle=-30, height=420,
                      xaxis_title="Intervalo de Población",
                      yaxis_title="fi")
    st.plotly_chart(fig, use_container_width=True)

with tab_p3:
    fig = go.Figure(go.Pie(
        labels=tf_pop_data_rev["Intervalo"],
        values=tf_pop_data_rev["fi"],
        marker=dict(colors=_blues_colors_rev),
        hole=0.4,
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, height=460,
                      title="Distribución porcentual de países por rango poblacional")
    st.plotly_chart(fig, use_container_width=True)

with tab_p4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_pop_data["Intervalo"], y=tf_pop_data["Fi"],
        mode="lines+markers",
        marker=dict(color="#00d4ff", size=8, symbol="square"),
        line=dict(color="#00aaee", width=2),
        fill="tozeroy", fillcolor="rgba(0,170,238,0.08)",
        name="Ojiva (Fi)",
    ))
    fig.update_layout(**BASE_LAYOUT, title="Ojiva — Frecuencia acumulada (Fi)",
                      xaxis_tickangle=-30, height=420,
                      xaxis_title="Intervalo de Población",
                      yaxis_title="Fi (acumulada)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# #ROJO: SECCIÓN 2 — ANÁLISIS DE ÁREA

st.markdown('<p class="section-title">🗺️ Análisis de Área Territorial</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    Para analizar la poblacion de los paises, tambien es importante conocer sus areas, 
            ya que esto nos permite entender la densidad poblacional.
</div>
""", unsafe_allow_html=True)

# #: Métricas de área

a1, a2, a3, a4, a5 = st.columns(5)
area_vals = df["area"]

with a1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📊 Media</div>
        <div class="metric-value">{fmt(area_vals.mean(), 0)}</div>
        <div class="metric-sub">km²</div>
    </div>""", unsafe_allow_html=True)
with a2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📍 Mediana</div>
        <div class="metric-value">{fmt(area_vals.median(), 0)}</div>
        <div class="metric-sub">km²</div>
    </div>""", unsafe_allow_html=True)
with a3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📈 Pais mas grande (Máximo)</div>
        <div class="metric-value">{fmt(area_vals.max(), 0)}</div>
        <div class="metric-sub">{df.loc[area_vals.idxmax(), 'country']}</div>
    </div>""", unsafe_allow_html=True)
with a4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📉 Pais mas pequeño (Mínimo)</div>
        <div class="metric-value">{fmt(area_vals.min(), 2)}</div>
        <div class="metric-sub">{df.loc[area_vals.idxmin(), 'country']}</div>
    </div>""", unsafe_allow_html=True)
with a5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📐 Desv. estándar</div>
        <div class="metric-value">{fmt(area_vals.std(), 0)}</div>
        <div class="metric-sub">km²</div>
    </div>""", unsafe_allow_html=True)

# #: Tabla de frecuencias — Área

st.markdown("#### 📋 Tabla de Frecuencias del Área Territorial")

st.markdown("""
<div class="info-box">
    Los países se clasificaron en 6 categorías según su extensión: desde microestados
    (&lt; 1 000 km²) hasta gigantes territoriales con más de 2 millones de km².
</div>
""", unsafe_allow_html=True)

tf_area = tabla_frecuencias(df["area"], AREA_LABELS, AREA_BINS)
tf_area_display = pd.concat([
    tf_area[tf_area["Intervalo"] != "TOTAL"].iloc[::-1].reset_index(drop=True),
    tf_area[tf_area["Intervalo"] == "TOTAL"]
], ignore_index=True)

st.dataframe(tf_area_display, use_container_width=True, hide_index=True,
    column_config={
        "fi":  st.column_config.NumberColumn("fi",  format="%d"),
        "Fi":  st.column_config.NumberColumn("Fi",  format="%d"),
        "hi":  st.column_config.NumberColumn("hi",  format="%.4f"),
        "Hi":  st.column_config.NumberColumn("Hi",  format="%.4f"),
        "hi%": st.column_config.NumberColumn("hi%", format="%.2f%%"),
    }
)

# #: Gráficas — Área

st.markdown("#### 📊 Graficas del Área Territorial")

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    Se visualiza la distribución de países por tamaño territorial, de la misma forma que en la poblacion.
</div>
""", unsafe_allow_html=True)

tf_area_data = tf_area[tf_area["Intervalo"] != "TOTAL"].copy()
# #VERDE: Para gráficos: orden de mayor a menor rango (> 2 000 000 km² primero)
tf_area_data_rev = tf_area_data.iloc[::-1].reset_index(drop=True)
_teal_colors = pc.sample_colorscale("Teal", [i / (len(tf_area_data_rev) - 1) for i in range(len(tf_area_data_rev))])
_teal_colors_rev = list(reversed(_teal_colors))  # índice 0 = mayor rango = más oscuro

tab_a1, tab_a2, tab_a3, tab_a4 = st.tabs([
    "📊 Barras por intervalo",
    "🎯 Bastones (fi)",
    "🥧 Torta",
    "📈 Ojiva",
])

with tab_a1:
    fig = go.Figure(go.Bar(
        x=tf_area_data_rev["Intervalo"],
        y=tf_area_data_rev["fi"],
        marker=dict(color=_teal_colors_rev, showscale=False,
                    line=dict(color="#00d4ff", width=0.5)),
        text=tf_area_data_rev["fi"],
        textposition="outside",
        textfont=dict(color="#8899aa", size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, title="Países por categoría de tamaño territorial",
                      xaxis_tickangle=-20, height=420,
                      xaxis_title="Intervalo de Área",
                      yaxis_title="Número de países (fi)")
    st.plotly_chart(fig, use_container_width=True)

with tab_a2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_area_data_rev["Intervalo"], y=tf_area_data_rev["fi"],
        mode="markers+lines",
        marker=dict(color="#00ccaa", size=10, symbol="circle"),
        line=dict(color="#00ccaa", width=1, dash="dot"),
    ))
    for _, row in tf_area_data_rev.iterrows():
        fig.add_shape(type="line",
            x0=row["Intervalo"], x1=row["Intervalo"],
            y0=0, y1=row["fi"],
            line=dict(color="#009988", width=2))
    fig.update_layout(**BASE_LAYOUT, title="Diagrama de bastones — Frecuencia absoluta",
                      xaxis_tickangle=-20, height=420,
                      xaxis_title="Intervalo de Área",
                      yaxis_title="fi")
    st.plotly_chart(fig, use_container_width=True)

with tab_a3:
    fig = go.Figure(go.Pie(
        labels=tf_area_data_rev["Intervalo"],
        values=tf_area_data_rev["fi"],
        marker=dict(colors=_teal_colors_rev),
        hole=0.4,
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, height=460,
                      title="Distribución porcentual de países por tamaño territorial")
    st.plotly_chart(fig, use_container_width=True)

with tab_a4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_area_data["Intervalo"], y=tf_area_data["Fi"],
        mode="lines+markers",
        marker=dict(color="#00ccaa", size=8, symbol="square"),
        line=dict(color="#00ccaa", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,170,0.08)",
        name="Ojiva (Fi)",
    ))
    fig.update_layout(**BASE_LAYOUT, title="Ojiva — Frecuencia acumulada (Fi)",
                      xaxis_tickangle=-20, height=420,
                      xaxis_title="Intervalo de Área",
                      yaxis_title="Fi (acumulada)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# #ROJO: SECCIÓN 3 — ANÁLISIS DE DENSIDAD

st.markdown('<p class="section-title">📐 Análisis de Densidad Poblacional</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    La densidad poblacional (hab/km²) relaciona la población con el territorio.
    Algunos territorios pueden tener gran poblacion en poco espacio, y otros, pueden 
    ser extensos pero con poca poblacion. 
</div>
""", unsafe_allow_html=True)

# #: Métricas de densidad

d1, d2, d3, d4, d5 = st.columns(5)
dens_vals = df["density"]

with d1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📊 Media</div>
        <div class="metric-value">{fmt(dens_vals.mean(), 2)}</div>
        <div class="metric-sub">hab/km²</div>
    </div>""", unsafe_allow_html=True)
with d2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📍 Mediana</div>
        <div class="metric-value">{fmt(dens_vals.median(), 2)}</div>
        <div class="metric-sub">hab/km²</div>
    </div>""", unsafe_allow_html=True)
with d3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📈 Mayor densidad</div>
        <div class="metric-value">{fmt(dens_vals.max(), 1)}</div>
        <div class="metric-sub">{df.loc[dens_vals.idxmax(), 'country']}</div>
    </div>""", unsafe_allow_html=True)
with d4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📉 Menor densidad</div>
        <div class="metric-value">{fmt(dens_vals.min(), 2)}</div>
        <div class="metric-sub">{df.loc[dens_vals.idxmin(), 'country']}</div>
    </div>""", unsafe_allow_html=True)
with d5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📐 Desv. estándar</div>
        <div class="metric-value">{fmt(dens_vals.std(), 2)}</div>
        <div class="metric-sub">hab/km²</div>
    </div>""", unsafe_allow_html=True)

# #: Tabla de frecuencias — Densidad

st.markdown("#### 📋 Tabla de Frecuencias de la Densidad Poblacional")

st.markdown("""
<div class="info-box">
    Se definen 6 categorías de densidad, desde los menos densos a los mas densos.
</div>
""", unsafe_allow_html=True)

tf_dens = tabla_frecuencias(df["density"], DENS_LABELS, DENS_BINS)
tf_dens_display = pd.concat([
    tf_dens[tf_dens["Intervalo"] != "TOTAL"].iloc[::-1].reset_index(drop=True),
    tf_dens[tf_dens["Intervalo"] == "TOTAL"]
], ignore_index=True)

st.dataframe(tf_dens_display, use_container_width=True, hide_index=True,
    column_config={
        "fi":  st.column_config.NumberColumn("fi",  format="%d"),
        "Fi":  st.column_config.NumberColumn("Fi",  format="%d"),
        "hi":  st.column_config.NumberColumn("hi",  format="%.4f"),
        "Hi":  st.column_config.NumberColumn("Hi",  format="%.4f"),
        "hi%": st.column_config.NumberColumn("hi%", format="%.2f%%"),
    }
)

# #: Gráficas — Densidad

st.markdown("#### 📊 Gráficas de la Densidad Poblacional")

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    Distribución de países por categoría de densidad.
</div>
""", unsafe_allow_html=True)

tf_dens_data = tf_dens[tf_dens["Intervalo"] != "TOTAL"].copy()
# #: Para gráficos: orden de mayor a menor rango (> 500 hab/km² primero)
tf_dens_data_rev = tf_dens_data.iloc[::-1].reset_index(drop=True)
_oranges_colors = pc.sample_colorscale("Oranges", [i / (len(tf_dens_data_rev) - 1) for i in range(len(tf_dens_data_rev))])
_oranges_colors_rev = list(reversed(_oranges_colors))  # índice 0 = mayor rango = más oscuro

tab_d1, tab_d2, tab_d3, tab_d4 = st.tabs([
    "📊 Barras por intervalo",
    "🎯 Bastones (fi)",
    "🥧 Torta",
    "📈 Ojiva",
])

with tab_d1:
    fig = go.Figure(go.Bar(
        x=tf_dens_data_rev["Intervalo"],
        y=tf_dens_data_rev["fi"],
        marker=dict(color=_oranges_colors_rev, showscale=False,
                    line=dict(color="#ff8800", width=0.5)),
        text=tf_dens_data_rev["fi"],
        textposition="outside",
        textfont=dict(color="#8899aa", size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, title="Países por categoría de densidad poblacional",
                      xaxis_tickangle=-20, height=420,
                      xaxis_title="Intervalo de Densidad",
                      yaxis_title="Número de países (fi)")
    st.plotly_chart(fig, use_container_width=True)

with tab_d2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_dens_data_rev["Intervalo"], y=tf_dens_data_rev["fi"],
        mode="markers+lines",
        marker=dict(color="#ff9944", size=10, symbol="circle"),
        line=dict(color="#ff9944", width=1, dash="dot"),
    ))
    for _, row in tf_dens_data_rev.iterrows():
        fig.add_shape(type="line",
            x0=row["Intervalo"], x1=row["Intervalo"],
            y0=0, y1=row["fi"],
            line=dict(color="#cc7722", width=2))
    fig.update_layout(**BASE_LAYOUT, title="Diagrama de bastones — Frecuencia absoluta",
                      xaxis_tickangle=-20, height=420,
                      xaxis_title="Intervalo de Densidad",
                      yaxis_title="fi")
    st.plotly_chart(fig, use_container_width=True)

with tab_d3:
    fig = go.Figure(go.Pie(
        labels=tf_dens_data_rev["Intervalo"],
        values=tf_dens_data_rev["fi"],
        marker=dict(colors=_oranges_colors_rev),
        hole=0.4,
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, height=460,
                      title="Distribución porcentual por categoría de densidad")
    st.plotly_chart(fig, use_container_width=True)

with tab_d4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_dens_data["Intervalo"], y=tf_dens_data["Fi"],
        mode="lines+markers",
        marker=dict(color="#ff9944", size=8, symbol="square"),
        line=dict(color="#ff9944", width=2),
        fill="tozeroy", fillcolor="rgba(255,153,68,0.08)",
        name="Ojiva (Fi)",
    ))
    fig.update_layout(**BASE_LAYOUT, title="Ojiva — Frecuencia acumulada (Fi)",
                      xaxis_tickangle=-20, height=420,
                      xaxis_title="Intervalo de Densidad",
                      yaxis_title="Fi (acumulada)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# #ROJO: SECCIÓN 4 — ANÁLISIS DE PORCENTAJE MUNDIAL

st.markdown('<p class="section-title">🌐 Análisis de Porcentaje Mundial</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>INFORMACIÓN:</strong><br>
    El porcentaje mundial expresa qué fracción de la población global representa cada país,
    usando como referencia una población mundial de <strong>8 298 978 817</strong> habitantes (estimación 2026).
    Este análisis se presenta en dos partes: distribución por intervalos y representación de los
    países más poblados respecto al "Resto del mundo".
</div>
""", unsafe_allow_html=True)

# #: Métricas de % mundiaL

p1, p2, p3, p4 = st.columns(4)
pct_vals = df["worldPercentage"]

with p1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📊 Media</div>
        <div class="metric-value">{fmt(pct_vals.mean(), 4)}%</div>
        <div class="metric-sub">por país</div>
    </div>""", unsafe_allow_html=True)
with p2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📍 Mediana</div>
        <div class="metric-value">{fmt(pct_vals.median(), 4)}%</div>
        <div class="metric-sub">por país</div>
    </div>""", unsafe_allow_html=True)
with p3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📈 Mayor %</div>
        <div class="metric-value">{fmt(pct_vals.max(), 2)}%</div>
        <div class="metric-sub">{df.loc[pct_vals.idxmax(), 'country']}</div>
    </div>""", unsafe_allow_html=True)
with p4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">🌍 Ref. mundial</div>
        <div class="metric-value" style="font-size:0.9rem;">{fmt(WORLD_POP)}</div>
        <div class="metric-sub">habitantes (2026)</div>
    </div>""", unsafe_allow_html=True)

# #: Tabla de frecuencias — % Mundial

st.markdown("#### 📋 Tabla de Frecuencias del Porcentaje Mundial")

st.markdown("""
<div class="info-box">
    Se definieron 6 intervalos que permiten distinguir desde países con presencia demográfica
    ínfima (&lt; 0.01% del total mundial) hasta las dos superpotencias que superan el 5%, claramente China e India, 
    las cuales se muestran en una tabla siguiente con su porcentaje respectivo.
</div>
""", unsafe_allow_html=True)

tf_pct = tabla_frecuencias(df["worldPercentage"], PCT_LABELS, PCT_BINS)
tf_pct_display = pd.concat([
    tf_pct[tf_pct["Intervalo"] != "TOTAL"].iloc[::-1].reset_index(drop=True),
    tf_pct[tf_pct["Intervalo"] == "TOTAL"]
], ignore_index=True)

# #: TABLA DE CHINA E INDIA
paises_over5 = df[df["worldPercentage"] > 5][["country", "worldPercentage"]].sort_values("worldPercentage", ascending=False)

st.dataframe(tf_pct_display, use_container_width=True, hide_index=True,
    column_config={
        "fi":  st.column_config.NumberColumn("fi",  format="%d"),
        "Fi":  st.column_config.NumberColumn("Fi",  format="%d"),
        "hi":  st.column_config.NumberColumn("hi",  format="%.4f"),
        "Hi":  st.column_config.NumberColumn("Hi",  format="%.4f"),
        "hi%": st.column_config.NumberColumn("hi%", format="%.2f%%"),
    }
)

if not paises_over5.empty:
    st.markdown("**Países en el intervalo > 5% (superan el 5% de la población mundial):**")
    paises_over5_display = paises_over5.rename(columns={"country": "País", "worldPercentage": "% Mundial"})
    st.dataframe(paises_over5_display, use_container_width=True, hide_index=True,
        column_config={"% Mundial": st.column_config.NumberColumn(format="%.4f%%")}
    )

# #: Gráficas — % Mundial

st.markdown("#### 📊 Graficas del Porcentaje Mundial")

# #VERDE: TIPO 1: distribución por intervalos

st.markdown("##### Tipo 1 · Distribución por intervalos")

st.markdown("""
<div class="info-box">
    Muestra cuántos países caen en cada rango de participación en la población mundial.
    La mayoría de los países representa menos del 0.1% de la población global.
</div>
""", unsafe_allow_html=True)

tf_pct_data = tf_pct[tf_pct["Intervalo"] != "TOTAL"].copy()
# #VERDE: Para gráficos: orden de mayor a menor rango (> 5% primero)
tf_pct_data_rev = tf_pct_data.iloc[::-1].reset_index(drop=True)
_purples_colors = pc.sample_colorscale("Purples", [i / (len(tf_pct_data_rev) - 1) for i in range(len(tf_pct_data_rev))])
_purples_colors_rev = list(reversed(_purples_colors))  # índice 0 = mayor rango = más oscuro

tab_pct1, tab_pct2, tab_pct3, tab_pct4 = st.tabs([
    "📊 Barras",
    "🎯 Bastones",
    "🥧 Torta",
    "📈 Ojiva",
])

with tab_pct1:
    fig = go.Figure(go.Bar(
        x=tf_pct_data_rev["Intervalo"],
        y=tf_pct_data_rev["fi"],
        marker=dict(color=_purples_colors_rev, showscale=False,
                    line=dict(color="#aa44ff", width=0.5)),
        text=tf_pct_data_rev["fi"],
        textposition="outside",
        textfont=dict(color="#8899aa", size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, title="Países por intervalo de % de población mundial",
                      xaxis_tickangle=-15, height=420,
                      xaxis_title="Intervalo de % Mundial",
                      yaxis_title="Número de países (fi)")
    st.plotly_chart(fig, use_container_width=True)

with tab_pct2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_pct_data_rev["Intervalo"], y=tf_pct_data_rev["fi"],
        mode="markers+lines",
        marker=dict(color="#bb66ff", size=10, symbol="circle"),
        line=dict(color="#bb66ff", width=1, dash="dot"),
    ))
    for _, row in tf_pct_data_rev.iterrows():
        fig.add_shape(type="line",
            x0=row["Intervalo"], x1=row["Intervalo"],
            y0=0, y1=row["fi"],
            line=dict(color="#9944cc", width=2))
    fig.update_layout(**BASE_LAYOUT, title="Diagrama de bastones — Frecuencia absoluta",
                      xaxis_tickangle=-15, height=420,
                      xaxis_title="Intervalo de % Mundial",
                      yaxis_title="fi")
    st.plotly_chart(fig, use_container_width=True)

with tab_pct3:
    fig = go.Figure(go.Pie(
        labels=tf_pct_data_rev["Intervalo"],
        values=tf_pct_data_rev["fi"],
        marker=dict(colors=_purples_colors_rev),
        hole=0.4,
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, height=460,
                      title="Distribución de países por intervalo de % mundial")
    st.plotly_chart(fig, use_container_width=True)

with tab_pct4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tf_pct_data["Intervalo"], y=tf_pct_data["Fi"],
        mode="lines+markers",
        marker=dict(color="#bb66ff", size=8, symbol="square"),
        line=dict(color="#bb66ff", width=2),
        fill="tozeroy", fillcolor="rgba(187,102,255,0.08)",
        name="Ojiva (Fi)",
    ))
    fig.update_layout(**BASE_LAYOUT, title="Ojiva — Frecuencia acumulada (Fi)",
                      xaxis_tickangle=-15, height=420,
                      xaxis_title="Intervalo de % Mundial",
                      yaxis_title="Fi (acumulada)")
    st.plotly_chart(fig, use_container_width=True)

# #: TIPO 2: Top 20

st.markdown("##### Tipo 2 · Top 20 países más poblados")

st.markdown("""
<div class="info-box">
    Los 20 países más poblados del mundo se muestran individualmente, mientras que todos los
    demás se agrupan en una sola categoría denominada "Resto del mundo", poniendo en perspectiva la
    concentración demográfica global.
</div>
""", unsafe_allow_html=True)

top20 = df.nlargest(20, "pop2026")[["country", "worldPercentage"]].copy()
resto_pct = df.nlargest(20, "pop2026", keep="last").copy()  # necesito los que NO están en top20
resto_pct = df[~df["country"].isin(top20["country"])]["worldPercentage"].sum()

pie_data = pd.concat([
    top20.rename(columns={"country": "Etiqueta", "worldPercentage": "Porcentaje"}),
    pd.DataFrame([{"Etiqueta": "Resto del mundo", "Porcentaje": round(resto_pct, 4)}]),
], ignore_index=True)

_dark24 = px.colors.qualitative.Dark24
_pie_colors = [_dark24[i % len(_dark24)] for i in range(len(pie_data))]
_color_map = {row["Etiqueta"]: _pie_colors[i] for i, row in pie_data.iterrows()}

col_pie, col_bar = st.columns([1, 1])

with col_pie:
    fig = go.Figure(go.Pie(
        labels=pie_data["Etiqueta"],
        values=pie_data["Porcentaje"],
        marker=dict(colors=_pie_colors),
        hole=0.35,
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        height=520,
        title="Top 20 países + Resto del mundo (% población mundial)",
    )

    fig.update_layout(
        legend=dict(
            font=dict(size=9, color="#8899aa"),
            bgcolor="rgba(0,0,0,0)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

with col_bar:
    # #VERDE: Ordenar: países por porcentaje, pero "Resto del mundo" al final
    top20_sorted = pie_data[pie_data["Etiqueta"] != "Resto del mundo"].sort_values("Porcentaje", ascending=False)
    resto_row = pie_data[pie_data["Etiqueta"] == "Resto del mundo"]
    pie_sorted = pd.concat([top20_sorted, resto_row], ignore_index=True)

    bar_colors = [_color_map[e] for e in pie_sorted["Etiqueta"]]

    fig = go.Figure(go.Bar(
        x=pie_sorted["Porcentaje"],
        y=pie_sorted["Etiqueta"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            showscale=False,
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        text=pie_sorted["Porcentaje"].apply(lambda x: f"{x:.2f}%"),
        textposition="outside",
        textfont=dict(size=9, color="#8899aa"),
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title="% Población mundial por país (Top 20 + Resto)",
        height=520
    )

    fig.update_yaxes(
        autorange="reversed",
        gridcolor="#1e3a5f"
    )

    fig.update_xaxes(
        title="% Población Mundial",
        gridcolor="#1e3a5f"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# #AZUL: FOOTER

st.markdown("""
<div style="text-align:center; color:#2a4060; font-size:0.75rem; padding:1rem 0 2rem;">
    Proyecto de Datos de Población de Países · 2026 ·
    Streamlit · Plotly · Pandas
</div>
""", unsafe_allow_html=True)
