# ============================================================
# SmarTouring · Dashboard (solo BERT) sobre comentarios_final
# ============================================================
# ------------------------------------------------------------

from pathlib import Path
from datetime import timedelta
import time
import os
import sys
import platform

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ------------------ CONFIGURACIÓN GLOBAL ---------------------
st.set_page_config(
    page_title="SmarTouring · Sentimiento (BERT)",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUILD = "bert-only-1.0"
st.title("🧭 SmarTouring — Sentimiento (BERT)")
st.caption(f"build: {BUILD}")

# Diagnóstico rápido en la barra lateral (útil si hay caché)
st.sidebar.markdown("---")
st.sidebar.caption("🔧 Diagnóstico")
st.sidebar.caption(f"Archivo: {Path(__file__).resolve()}")
st.sidebar.caption(f"Modificado: {time.ctime(os.path.getmtime(__file__))}")
st.sidebar.caption(f"Python: {sys.version.split()[0]} · Streamlit: {st.__version__} · OS: {platform.system()}")

PLOTLY_TEMPLATE = "plotly_white"

SENTIMENT_ORDER_DEFAULT = ["positivo", "neutro", "negativo"]
SENTIMENT_PALETTE = {
    "positivo": "#22c55e",  # green
    "neutro":   "#64748b",  # slate
    "negativo": "#ef4444",  # red
}

# Stopwords multilingües ampliadas para wordcloud
EXTRA_STOPS = {
    # Funcionales (ES)
    "y","e","o","u","el","la","los","las","un","una","unos","unas",
    "de","del","al","a","en","con","por","para","como","sobre","sin",
    "que","qué","porque","porqué","ya","pero","si","no","lo","su","sus",
    "es","son","ser","fue","era","está","esta","este","estos","estas",
    "hay","todo","toda","todos","todas","muy","más","mas","menos","nada",
    "algo","asi","así","aqui","aquí","alli","allí","etc","etc.","nos"
    # Comunes EN/FR/IT/DE/PT/CAT
    "the","and","or","of","to","in","with","for","on","at","from","this","that",
    "le","la","les","des","du","de","et","pour","avec","une","un",
    "il","lo","gli","le","con","per","che","del","della",
    "und","der","die","das","mit","ein","eine","im","am",
    "e","em","com","para","sem","uma","um","ao","às","os","as",
    "els","les","amb","per","del","de",
    # Palabras genéricas del dominio
    "hotel","hoteles","playa","viaje","viajes","turismo","turística","turistico",
    "bueno","buena","buenas","buenos","malo","mala","malas","malos",
    "genial","excelente","fantástico","fantastica","fantastico","fantásticas",
    "bonito","bonita","bonitas","bonitos","lugar","sitio","zona",
    "good","great","nice","bad","ok","well","really","very",
    # Días/meses (ES)
    "lunes","martes","miércoles","miercoles","jueves","viernes","sábado","sabado","domingo",
    "enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","setiembre","octubre","noviembre","diciembre",
    # Topónimos frecuentes (para evitar sesgo por ciudad)
    "barcelona","madrid","valencia","sevilla","málaga","malaga","palma","mallorca",
    "gran","canaria","gran canaria","tenerife","españa","spain",
}

# Helper para construir stopwords dinámicas a partir de los datos filtrados (ciudades/categorías/fuentes)
def build_dynamic_stopwords(df: pd.DataFrame, cols=("Ciudad","Categoria","Fuente")) -> set:
    stops = set()
    for c in cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).str.lower().unique().tolist()
            for v in vals:
                # dividir por espacios y guiones bajos para capturar tokens sueltos
                for tok in str(v).replace("_"," ").split():
                    tok = tok.strip()
                    if len(tok) >= 2:
                        stops.add(tok)
    return stops

# ------------------ COORDENADAS DE CIUDADES -------------------
CITY_COORDS = {
    # Núcleo TFM (8 ciudades)
    "madrid": (40.4168, -3.7038),
    "barcelona": (41.3874, 2.1686),
    "valencia": (39.4699, -0.3763),
    "sevilla": (37.3891, -5.9845),
    "málaga": (36.7213, -4.4217),
    "malaga": (36.7213, -4.4217),
    "palma de mallorca": (39.5696, 2.6502),
    "mallorca": (39.5696, 2.6502),
    "gran canaria": (28.1235, -15.4363),
    "las palmas de gran canaria": (28.1235, -15.4363),
    "tenerife": (28.4636, -16.2518),
    "santa cruz de tenerife": (28.4636, -16.2518),
    # Extras frecuentes
    "córdoba": (37.8882, -4.7794),
    "cordoba": (37.8882, -4.7794),
    "girona": (41.9794, 2.8214),
}
def city_key(x: str) -> str:
    return str(x).strip().lower()

# ------------------ UTILIDADES -------------------------------

def human_int(n):
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)

def normalize_text_col(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def unify_categories(df: pd.DataFrame):
    # Unificar variantes y capitalizar bonito
    if "Categoria" in df.columns:
        s = df["Categoria"].astype(str)
        s = s.str.replace("_", " ", regex=False).str.strip()
        syn = {
            "vida nocturna": "Vida nocturna",
            "vida_nocturna": "Vida nocturna",
            "nightlife": "Vida nocturna",
        }
        lower = s.str.lower()
        df["Categoria"] = lower.map(syn).fillna(s.str.title())
    return df

def best_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Intenta normalizar fechas con varias estrategias:
    1) to_datetime infer
    2) dayfirst=True
    3) yearfirst=True
    4) reemplazo de separadores y nuevo intento
    Devuelve pd.Series de dtype datetime64[ns].
    """
    s = series.astype(str).str.strip()
    # normalización rápida de separadores y eliminación de sufijos raros
    s = s.str.replace(r"[.]", "-", regex=True).str.replace("/", "-", regex=False)
    s = s.str.replace("Z", "", regex=False)

    # Intentos en cascada
    candidates = []
    for kwargs in (
        dict(errors="coerce", infer_datetime_format=True),
        dict(errors="coerce", dayfirst=True),
        dict(errors="coerce", yearfirst=True),
        dict(errors="coerce", dayfirst=True, yearfirst=True),
    ):
        candidates.append(pd.to_datetime(s, **kwargs))

    # Combinar por prioridad (primer acierto gana)
    out = candidates[0].copy()
    for cand in candidates[1:]:
        mask = out.isna() & cand.notna()
        if mask.any():
            out.loc[mask] = cand.loc[mask]

    # Segundo pase: si hay muchas NaT, intentar sin separadores de tiempo
    if out.isna().mean() > 0.05:
        s2 = s.str.replace("T", " ", regex=False)
        alt = pd.to_datetime(s2, errors="coerce", infer_datetime_format=True)
        mask = out.isna() & alt.notna()
        if mask.any():
            out.loc[mask] = alt.loc[mask]

    return out

def enforce_required_columns(df: pd.DataFrame):
    needed = ["Texto", "Ciudad", "Categoria", "Fecha", "Fuente", "sentimiento", "confianza"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas en el CSV: {missing}")
        st.stop()

def to_percent_df(counts_df, index_col, col_col, value_col="conteo"):
    if counts_df.empty:
        return counts_df
    pivot = counts_df.pivot(index=index_col, columns=col_col, values=value_col).fillna(0)
    row_sum = pivot.sum(axis=1)
    pct = pivot.div(row_sum, axis=0).fillna(0) * 100
    pct = pct.reset_index()
    return pct

def smart_sample(df: pd.DataFrame, n=1500):
    if len(df) <= n:
        return df
    return df.sample(n, random_state=42)

# ------------------ SIDEBAR: ARCHIVO Y CARGA -----------------

BASE_DIR = Path(__file__).parent
default_csv = (BASE_DIR / "Resultados Bert.csv").resolve()

st.sidebar.title("⚙️ Controles")
csv_path = st.sidebar.text_input("📄 CSV principal (BERT clasificado):", value=str(default_csv))
uploaded = st.sidebar.file_uploader("…o sube el CSV principal", type=["csv"])

if st.sidebar.button("♻️ Limpiar caché y recargar", use_container_width=True):
    st.cache_data.clear()
    st.experimental_rerun()

@st.cache_data(show_spinner=True)
def load_and_clean(csv_file):
    if isinstance(csv_file, (str, os.PathLike)) and Path(csv_file).exists():
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_csv(csv_file)  # file-like

    # Normalizaciones básicas
    for c in ["Ciudad", "Categoria", "Fuente", "sentimiento"]:
        df = normalize_text_col(df, c)

    # Unificación de categorías (vida nocturna, etc.)
    df = unify_categories(df)

    # Fechas robustas
    if "Fecha" in df.columns:
        before = df["Fecha"].isna().sum() if pd.api.types.is_datetime64_any_dtype(df["Fecha"]) else None
        df["Fecha"] = best_parse_datetime(df["Fecha"])
        bad = df["Fecha"].isna().sum()
        total = len(df)
        info = f"Fechas inválidas tras normalización: {bad}/{total} ({bad/total:.1%})"
    else:
        info = "No se encontró columna 'Fecha'. Se omiten filtros temporales."

    return df, info

# Cargar CSV
if uploaded is not None:
    df, date_info = load_and_clean(uploaded)
elif Path(csv_path).exists():
    df, date_info = load_and_clean(csv_path)
else:
    st.error("No se encontró el CSV principal. Revisa la ruta o sube el fichero.")
    st.stop()

enforce_required_columns(df)

st.success(date_info)

# ------------------ FILTROS ------------------------------

# Orden de sentimientos disponible
sent_unique = [s for s in SENTIMENT_ORDER_DEFAULT if s in df["sentimiento"].unique()]
SENTIMENT_COLOR = {s: SENTIMENT_PALETTE[s] for s in sent_unique}

min_date = pd.to_datetime(df["Fecha"].min()) if df["Fecha"].notna().any() else None
max_date = pd.to_datetime(df["Fecha"].max()) if df["Fecha"].notna().any() else None

st.sidebar.markdown("---")
st.sidebar.caption("🎛️ Filtros")

if min_date and max_date:
    dr = st.sidebar.date_input(
        "📅 Rango de fechas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
else:
    dr = None

def ms(label, col):
    vals = sorted([v for v in df[col].dropna().unique()])
    return st.sidebar.multiselect(label, options=vals, default=vals)

sel_ciudad = ms("🏙️ Ciudad", "Ciudad")
sel_cat    = ms("🏷️ Categoría", "Categoria")
sel_fuente = ms("🛰️ Fuente", "Fuente")

idioma_col = next((c for c in ["idioma","lang","language"] if c in df.columns), None)
if idioma_col:
    sel_idioma = ms("🗣️ Idioma", idioma_col)
else:
    sel_idioma = None

sel_sent = st.sidebar.multiselect("🙂 Sentimiento", options=sent_unique, default=sent_unique)

conf_thr = st.sidebar.slider("✅ Umbral mínimo de confianza", 0.0, 1.0, 0.50, 0.01)

# Aplicación de filtros
fdf = df.copy()
if dr and isinstance(dr, (list, tuple)) and len(dr) == 2 and fdf["Fecha"].notna().any():
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1]) + timedelta(days=1)
    fdf = fdf[(fdf["Fecha"] >= start) & (fdf["Fecha"] < end)]

fdf = fdf[
    fdf["Ciudad"].isin(sel_ciudad) &
    fdf["Categoria"].isin(sel_cat) &
    fdf["Fuente"].isin(sel_fuente) &
    fdf["sentimiento"].isin(sel_sent) &
    (fdf["confianza"] >= conf_thr)
]

if sel_idioma is not None and idioma_col:
    fdf = fdf[fdf[idioma_col].isin(sel_idioma)]

st.markdown(
    f"**Registros tras filtros:** {human_int(len(fdf))} · "
    f"Período: {fdf['Fecha'].min().date() if fdf['Fecha'].notna().any() else '—'} → "
    f"{fdf['Fecha'].max().date() if fdf['Fecha'].notna().any() else '—'}"
)

# ------------------ KPIs ------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
total = len(fdf)
pos = (fdf["sentimiento"] == "positivo").sum()
neg = (fdf["sentimiento"] == "negativo").sum()
neu = (fdf["sentimiento"] == "neutro").sum()
avg_conf = fdf["confianza"].mean() if total else 0.0

k1.metric("Reseñas", human_int(total))
k2.metric("Positivo", f"{(pos/total*100 if total else 0):.1f}%")
k3.metric("Negativo", f"{(neg/total*100 if total else 0):.1f}%")
k4.metric("Neutro", f"{(neu/total*100 if total else 0):.1f}%")
k5.metric("Confianza media", f"{avg_conf:.2f}")

# ------------------ DISTRIBUCIONES ----------------------
c1, c2 = st.columns((1,1))

with c1:
    st.subheader("Distribución de sentimiento")
    sent_counts = (
        fdf.groupby("sentimiento").size().reset_index(name="conteo")
    )
    if not sent_counts.empty:
        fig = px.bar(
            sent_counts, x="sentimiento", y="conteo", color="sentimiento",
            color_discrete_map=SENTIMENT_COLOR, template=PLOTLY_TEMPLATE,
            category_orders={"sentimiento": sent_unique},
            text_auto=True
        )
        fig.update_layout(yaxis_title="Nº reseñas", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos para la selección actual.")

with c2:
    st.subheader("Top 10 categorías por volumen")
    by_cat = (
        fdf.groupby("Categoria").size().reset_index(name="conteo")
        .sort_values("conteo", ascending=False).head(10)
    )
    figc = px.bar(by_cat, x="Categoria", y="conteo", template=PLOTLY_TEMPLATE)
    figc.update_layout(xaxis_title=None, yaxis_title="Nº reseñas")
    st.plotly_chart(figc, use_container_width=True)

# ------------------ MAPA POR CIUDAD ---------------------
st.markdown("---")
st.subheader("Mapa por ciudad (España)")

if "Ciudad" in fdf.columns and not fdf.empty:
    city_agg = (
        fdf.groupby("Ciudad")
           .agg(
               total=("sentimiento", "size"),
               pos=("sentimiento", lambda s: (s == "positivo").sum()),
               neg=("sentimiento", lambda s: (s == "negativo").sum()),
               neu=("sentimiento", lambda s: (s == "neutro").sum()),
           )
           .reset_index()
    )
    city_agg["key"] = city_agg["Ciudad"].map(city_key)
    coords = city_agg["key"].map(CITY_COORDS)
    city_agg["lat"] = coords.apply(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    city_agg["lon"] = coords.apply(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    city_agg = city_agg.dropna(subset=["lat", "lon"])

    if not city_agg.empty:
        city_agg["pos_pct"] = (city_agg["pos"] / city_agg["total"] * 100).round(1)
        city_agg["neg_pct"] = (city_agg["neg"] / city_agg["total"] * 100).round(1)
        city_agg["sentiment_index"] = (city_agg["pos_pct"] - city_agg["neg_pct"]).round(1)

        fig_map = px.scatter_geo(
            city_agg,
            lat="lat",
            lon="lon",
            size="total",
            color="sentiment_index",
            color_continuous_scale="RdYlGn",
            hover_name="Ciudad",
            hover_data={
                "total": ":,",
                "pos_pct": ":.1f",
                "neg_pct": ":.1f",
                "sentiment_index": ":.1f",
                "lat": False,
                "lon": False,
            },
            template=PLOTLY_TEMPLATE,
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(title="Índice (pos% − neg%)")
        )
        fig_map.update_geos(
            showcountries=True,
            countrycolor="lightgray",
            lataxis_range=[27, 44],
            lonaxis_range=[-18, 5],
            projection_type="natural earth",
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No hay coordenadas conocidas para las ciudades filtradas.")
else:
    st.info("No hay datos suficientes para el mapa por ciudad.")

# ------------------ SERIE TEMPORAL ---------------------
st.markdown("---")
st.subheader("Evolución semanal por clase (%)")

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    horizonte = st.selectbox("Horizonte", ["6 meses", "12 meses", "24 meses", "Todo"], index=2)
with col_b:
    suavizar = st.checkbox("Suavizar (media móvil 4 semanas)", value=True)
with col_c:
    tipo = st.radio("Tipo", ["Área apilada", "Líneas"], index=0, horizontal=True)

if fdf["Fecha"].notna().any():
    g = fdf.groupby([pd.Grouper(key="Fecha", freq="W"), "sentimiento"]).size().reset_index(name="conteo")
    pct = to_percent_df(g, "Fecha", "sentimiento", "conteo").sort_values("Fecha")
    for s in sent_unique:
        if s not in pct.columns:
            pct[s] = 0.0

    if horizonte != "Todo" and pct["Fecha"].notna().any():
        days = {"6 meses": 180, "12 meses": 365, "24 meses": 730}[horizonte]
        cut = pct["Fecha"].max() - pd.Timedelta(days=days)
        pct = pct[pct["Fecha"] >= cut]

    if suavizar:
        for s in sent_unique:
            if s in pct.columns:
                pct[s] = pct[s].rolling(4, min_periods=1).mean()

    if tipo == "Área apilada":
        long = pct.melt(id_vars="Fecha",
                        value_vars=[s for s in sent_unique if s in pct.columns],
                        var_name="sentimiento", value_name="pct")
        fig = px.area(
            long, x="Fecha", y="pct", color="sentimiento",
            color_discrete_map=SENTIMENT_COLOR, template=PLOTLY_TEMPLATE
        )
        fig.update_layout(yaxis_title="% semanal", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        figt = go.Figure()
        for s in sent_unique:
            if s in pct.columns:
                figt.add_trace(go.Scatter(
                    x=pct["Fecha"], y=pct[s], name=s, mode="lines",
                    line=dict(width=2, color=SENTIMENT_COLOR.get(s, None))
                ))
        figt.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="% semanal", xaxis_title=None)
        st.plotly_chart(figt, use_container_width=True)
else:
    st.info("No hay fechas válidas para graficar.")

 # ------------------ ESTACIONALIDAD POR MES ---------------------
st.markdown("---")
st.subheader("Estacionalidad por mes (todos los años combinados)")

if fdf["Fecha"].notna().any():
    mdf = fdf.copy()
    mdf["Mes"] = mdf["Fecha"].dt.month

    # Mapa de nombres de mes
    MONTH_ORDER = list(range(1, 13))
    MONTH_LABELS = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
    }

    # --- Barras apiladas en % por mes ---
    by_mes_sent = (
        mdf.groupby(["Mes", "sentimiento"]).size().reset_index(name="conteo")
    )
    if not by_mes_sent.empty:
        totals_mes = by_mes_sent.groupby("Mes")["conteo"].transform("sum")
        by_mes_sent["pct"] = by_mes_sent["conteo"] / totals_mes * 100

        order_sent = [s for s in ["positivo", "neutro", "negativo"] if s in by_mes_sent["sentimiento"].unique()]

        fig_month_stack = px.bar(
            by_mes_sent,
            x="Mes",
            y="pct",
            color="sentimiento",
            category_orders={"Mes": MONTH_ORDER, "sentimiento": order_sent},
            color_discrete_map=SENTIMENT_COLOR,
            template=PLOTLY_TEMPLATE,
            barmode="stack",
            text_auto=".1f",
        )
        fig_month_stack.update_layout(xaxis_title=None, yaxis_title="% dentro del mes")
        fig_month_stack.update_xaxes(tickmode="array", tickvals=MONTH_ORDER, ticktext=[MONTH_LABELS[m] for m in MONTH_ORDER])
        st.plotly_chart(fig_month_stack, use_container_width=True)
    else:
        st.info("No hay datos suficientes para el gráfico mensual apilado.")

    # --- Índice de sentimiento mensual (pos% - neg%) ---
    # --- Índice de sentimiento mensual (pos% - neg%) ---
    # Agregación robusta sin pivot(columns=None), evitando KeyError
    month_counts = (
        mdf.groupby(["Mes", "sentimiento"]).size().unstack(fill_value=0)
    )
    # Asegurar columnas esperadas aunque falten en los datos filtrados
    for col in ["positivo", "negativo"]:
        if col not in month_counts.columns:
            month_counts[col] = 0

    month_agg = month_counts.reset_index().sort_values("Mes")
    month_agg["total"] = month_agg.drop(columns=["Mes"]).sum(axis=1)

    month_agg["pos_pct"] = (month_agg["positivo"] / month_agg["total"].replace(0, np.nan) * 100).fillna(0)
    month_agg["neg_pct"] = (month_agg["negativo"] / month_agg["total"].replace(0, np.nan) * 100).fillna(0)
    month_agg["sentiment_index"] = (month_agg["pos_pct"] - month_agg["neg_pct"]).round(1)
    fig_month_idx = px.line(
        month_agg,
        x="Mes",
        y="sentiment_index",
        markers=True,
        template=PLOTLY_TEMPLATE,
    )
    fig_month_idx.update_traces(line=dict(width=3))
    fig_month_idx.update_layout(yaxis_title="Índice mensual (pos% − neg%)", xaxis_title=None)
    fig_month_idx.update_xaxes(tickmode="array", tickvals=MONTH_ORDER, ticktext=[MONTH_LABELS[m] for m in MONTH_ORDER])
    st.plotly_chart(fig_month_idx, use_container_width=True)
else:
    st.info("No hay fechas válidas para calcular estacionalidad mensual.")

# ------------------ SENTIMIENTO POR CATEGORÍA (APILADO) -------------------
st.markdown("---")
st.subheader("Sentimiento por categoría (apilado)")

if "Categoria" in fdf.columns and not fdf.empty:
    cat_totals = fdf.groupby("Categoria").size().sort_values(ascending=False)
    topN = st.slider("Número de categorías a mostrar", min_value=5, max_value=30, value=12, step=1)
    top_cats = cat_totals.head(topN).index.tolist()

    df_plot = fdf[fdf["Categoria"].isin(top_cats)].copy()
    by_cat_sent = (
        df_plot.groupby(["Categoria", "sentimiento"])
               .size()
               .reset_index(name="conteo")
    )

    show_pct = st.checkbox("Mostrar en % dentro de cada categoría", value=True)
    if show_pct:
        totals = by_cat_sent.groupby("Categoria")["conteo"].transform("sum")
        by_cat_sent["valor"] = by_cat_sent["conteo"] / totals * 100
        ylab = "% dentro de categoría"
        textfmt = ".1f"
    else:
        by_cat_sent["valor"] = by_cat_sent["conteo"]
        ylab = "Nº reseñas"
        textfmt = None

    order_sent = [s for s in ["positivo", "neutro", "negativo"] if s in by_cat_sent["sentimiento"].unique()]
    cat_order = df_plot.groupby("Categoria").size().sort_values(ascending=False).index.tolist()

    fig_stack = px.bar(
        by_cat_sent,
        x="Categoria",
        y="valor",
        color="sentimiento",
        category_orders={"Categoria": cat_order, "sentimiento": order_sent},
        color_discrete_map=SENTIMENT_COLOR,
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        text_auto=textfmt if textfmt else False,
    )
    fig_stack.update_layout(xaxis_title=None, yaxis_title=ylab)
    st.plotly_chart(fig_stack, use_container_width=True)
else:
    st.info("No hay datos de categoría para graficar.")

# ------------------ NUBE DE PALABRAS -------------------
st.markdown("---")
st.subheader("Nube de palabras (muestra)")
sel_sent_cloud = st.selectbox("Clase de sentimiento", options=sent_unique, index=0)
sample = smart_sample(fdf[fdf["sentimiento"] == sel_sent_cloud], n=2000)
texto_series = sample["Texto"].dropna().astype(str)
# Stopwords dinámicas basadas en los datos visibles (evita topónimos y etiquetas)
dyn_stops = build_dynamic_stopwords(fdf, cols=("Ciudad","Categoria"))
sw = set(STOPWORDS)
sw.update(EXTRA_STOPS)
sw.update(dyn_stops)
texto = " ".join(texto_series.tolist())
if texto.strip():
    wc = WordCloud(width=1100, height=420, background_color="white",
                   stopwords=sw, collocations=False).generate(texto)
    fig_wc, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc, use_container_width=True)
else:
    st.info("No hay texto suficiente para la nube.")

# ------------------ PREVIEW + EXPORT -------------------
st.markdown("---")
st.subheader("📋 Vista previa (muestra limitada) y exportación")
preview_cols = [c for c in ["Fecha","Ciudad","Categoria","Fuente","sentimiento","confianza","Texto"] if c in fdf.columns]
st.dataframe(smart_sample(fdf[preview_cols], n=1000), use_container_width=True)

st.download_button(
    "⬇️ Descargar CSV filtrado",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="comentarios_filtrados_bert.csv",
    mime="text/csv",
)

# Opción para guardar un CSV limpio de fechas (por si había problemas de formato)
st.markdown("**Guardado opcional**")
if st.button("💾 Guardar CSV con fechas normalizadas"):
    out = BASE_DIR / "comentarios_final_definitivo_clean.csv"
    df_out = df.copy()
    df_out.to_csv(out, index=False)
    st.success(f"Guardado en: {out}")
