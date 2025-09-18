# ===================================
# Librerías
# ===================================
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from streamlit_folium import st_folium
import folium
import pickle
import ast

# ===================================
# Función segura para cargar JSON
# ===================================
def load_json_safe(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    else:
        st.warning(f"⚠️ No se encontró {path}")
        return {}
    
# --- Utilidad: orden para estaciones (si aparece un día) ---
SEASON_ORDER = ["Invierno", "Primavera", "Verano", "Otoño"]
def order_season(s):
    try:
        return SEASON_ORDER.index(s)
    except Exception:
        return 99

# ===================================
# Configuración inicial de Streamlit
# ===================================
st.set_page_config(page_title="📊 Dashboard BERTopic", layout="wide")
st.title("📊 Dashboard de Resultados BERTopic")

# ===================================
# Cargar datasets
# ===================================
overview = load_json_safe("data/overview_metrics.json")
sentiment_data = load_json_safe("data/sentiment_analysis.json")
seasonal_data = load_json_safe("data/seasonal_analysis.json")
lang_analysis = load_json_safe("data/language_analysis.json")
multi_insights = load_json_safe("data/multilingual_insights.json")
topics_weather = load_json_safe("data/topics_with_weather_analysis.json")
weather_summary = load_json_safe("data/weather_insights_summary.json")
weather_sentiment = load_json_safe("data/weather_sentiment_analysis.json")


# CSVs
geo_df = pd.read_csv("data/geographic_data.csv") if os.path.exists("data/geographic_data.csv") else pd.DataFrame()
topic_df = pd.read_csv("data/topic_explorer.csv") if os.path.exists("data/topic_explorer.csv") else pd.DataFrame()
weather_matrix = pd.read_csv("data/weather_sentiment_matrix.csv") if os.path.exists("data/weather_sentiment_matrix.csv") else pd.DataFrame()
seasonal_weather = pd.read_csv("data/seasonal_weather_correlation.csv") if os.path.exists("data/seasonal_weather_correlation.csv") else pd.DataFrame()
monthly_summary = pd.read_csv("data/monthly_summary.csv") if os.path.exists("data/monthly_summary.csv") else pd.DataFrame()
time_series = pd.read_csv("data/time_series.csv")if os.path.exists("data/time_series.csv") else pd.DataFrame()

# PKL
time_series = pd.read_pickle("data/time_series.pkl")


# ===================================
# Tabs principales
# ===================================
tabs = st.tabs([
    "📌 Resumen", 
    "🗂️ Sentimientos", 
    "🌍 Idiomas & Multilingüe", 
    "📈 Temporalidad", 
    "📍 Geografía", 
    "🌦️ Clima", 

])

# ===================================
# TAB 1: Resumen
# ===================================
with tabs[0]:
    st.header("📌 Resumen Global")

    # === KPIs base del overview ===
    if overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documentos", f"{overview.get('total_documents', 0):,}")
        col2.metric("Tópicos", overview.get("total_topics", 0))
        col3.metric("Idiomas", overview.get("total_languages", 0))
        col4.metric("Ciudades", overview.get("total_cities", 0))

        st.subheader("📅 Rango temporal")
        if "date_range" in overview:
            st.write(f"De **{overview['date_range'].get('start','?')}** a **{overview['date_range'].get('end','?')}**")

        st.subheader("📊 Outliers")
        out_pct = overview.get("outliers_percentage", 0)
        out_cnt = overview.get("outliers_count", 0)
        st.progress(min(max(out_pct/100.0, 0), 1))
        st.caption(f"{out_pct}% ({out_cnt} docs)")

    else:
        st.error("No se pudo cargar overview_metrics.json")

    st.divider()

    # === Indicadores extra: Top ciudad & Top temporada ===
    c1, c2 = st.columns(2)

    # Top Ciudad (desde geographic_data.csv)
    with c1:
        st.subheader("🏙️ Top ciudad (por documentos)")
        if not geo_df.empty and "ciudad" in geo_df.columns and "total_documents" in geo_df.columns:
            top_city_row = geo_df.sort_values("total_documents", ascending=False).head(1)
            if not top_city_row.empty:
                city_name = top_city_row["ciudad"].iloc[0]
                city_docs = int(top_city_row["total_documents"].iloc[0])
                total_docs = int(overview.get("total_documents", city_docs))
                share = 100 * city_docs / total_docs if total_docs else 0
                st.metric(f"{city_name}", f"{city_docs:,} docs", f"{share:.1f}% del total")
            else:
                st.info("No hay filas en geographic_data.csv.")
        else:
            st.info("No se encontraron columnas 'ciudad' y 'total_documents' en geographic_data.csv.")

    # Top Temporada (desde seasonal_weather_correlation.csv)
    with c2:
        st.subheader("🗓️ Top temporada (por documentos)")
        if not seasonal_weather.empty and "season" in seasonal_weather.columns and "total_documents" in seasonal_weather.columns:
            top_season_row = seasonal_weather.groupby("season")["total_documents"].sum().reset_index()
            if not top_season_row.empty:
                top_season_row = top_season_row.sort_values("total_documents", ascending=False).head(1)
                season_name = str(top_season_row["season"].iloc[0])
                season_docs = int(top_season_row["total_documents"].iloc[0])
                total_docs = int(overview.get("total_documents", season_docs))
                share = 100 * season_docs / total_docs if total_docs else 0
                st.metric(f"{season_name}", f"{season_docs:,} docs", f"{share:.1f}% del total")
            else:
                st.info("No hay filas en seasonal_weather_correlation.csv.")
        else:
            st.info("No se encontraron columnas 'season' y 'total_documents' en seasonal_weather_correlation.csv.")

    st.divider()

    # === Sunburst ciudad → temporada → sentimiento ===
    st.subheader("🌞 Concentación: Ciudad → Temporada → Sentimiento")

    # Intento 1: si existe un dataset ya combinado con ciudad/season/sentiment
    # (no lo tenemos explícito; así que construimos con lo que haya)
    sunburst_df = pd.DataFrame()

    # Caso A: seasonal_weather_correlation.csv suele traer season + dominant_sentiment + total_documents
    #         pero NO nombres de ciudad. Entonces no da para ciudad→season→sentimiento.
    # Caso B: geographic_data.csv trae ciudad + (a veces) 'dominant_sentiment' y 'total_documents',
    #         pero NO season.
    # => Fallbacks:
    #   1) Si geo_df tiene 'dominant_sentiment' creamos Sunburst: Ciudad → Sentimiento (sin season)
    #   2) Si seasonal_weather tiene 'dominant_sentiment' hacemos Season → Sentimiento (sin ciudad)

    built = False

    # Fallback 1: Ciudad → Sentimiento (desde geo_df)
    if not geo_df.empty and all(col in geo_df.columns for col in ["ciudad", "total_documents"]) and "dominant_sentiment" in geo_df.columns:
        sb1 = geo_df[["ciudad", "dominant_sentiment", "total_documents"]].rename(
            columns={"ciudad": "Ciudad", "dominant_sentiment": "Sentimiento", "total_documents": "Documentos"}
        )
        fig = px.sunburst(sb1, path=["Ciudad", "Sentimiento"], values="Documentos",
                          color="Sentimiento", title="Concentración por Ciudad → Sentimiento")
        st.plotly_chart(fig, use_container_width=True)
        built = True

    # Fallback 2: Season → Sentimiento (desde seasonal_weather_correlation.csv)
    if not built and (not seasonal_weather.empty) and all(col in seasonal_weather.columns for col in ["season", "dominant_sentiment", "total_documents"]):
        sb2 = seasonal_weather[["season", "dominant_sentiment", "total_documents"]].rename(
            columns={"season": "Temporada", "dominant_sentiment": "Sentimiento", "total_documents": "Documentos"}
        )
        # orden estacional si aplica
        try:
            sb2["order"] = sb2["Temporada"].map(order_season)
            sb2 = sb2.sort_values("order")
        except Exception:
            pass
        fig = px.sunburst(sb2, path=["Temporada", "Sentimiento"], values="Documentos",
                          color="Sentimiento", title="Concentración por Temporada → Sentimiento")
        st.plotly_chart(fig, use_container_width=True)
        built = True

    if not built:
        st.info("Para el sunburst Ciudad → Temporada → Sentimiento necesitaríamos un dataset que contenga **ciudad y temporada juntas**. De momento muestro el mejor fallback disponible.")

# ===================================
# TAB 2: Sentimientos
# ===================================
with tabs[1]:
    st.header("🗂️ Análisis por Sentimiento")

    if sentiment_data:
        # --- Métricas globales ---
        total_docs = sum([s.get("total_documents",0) for s in sentiment_data.values()])
        total_topics = sum([s.get("total_topics",0) for s in sentiment_data.values()])
        c1, c2 = st.columns(2)
        c1.metric("Total documentos", f"{total_docs:,}")
        c2.metric("Total tópicos", total_topics)

        # --- Construir dataset ciudad × sentimiento ---
        rows = []
        for sent, block in sentiment_data.items():
            geo_sum = block.get("geographic_summary", {})
            for city, count in geo_sum.items():
                rows.append({"Ciudad": city, "Sentimiento": sent, "Documentos": count})
        df_geo_sent = pd.DataFrame(rows)

        if not df_geo_sent.empty:
            # --- Barras apiladas ---
            st.subheader("📍 Distribución de sentimientos por ciudad")
            topN = st.slider("Top ciudades a mostrar", 3, 5, 8)
            top_cities = (
                df_geo_sent.groupby("Ciudad")["Documentos"]
                .sum()
                .sort_values(ascending=False)
                .head(topN)
                .index
            )
            df_plot = df_geo_sent[df_geo_sent["Ciudad"].isin(top_cities)]
            fig = px.bar(
                df_plot,
                x="Ciudad", y="Documentos",
                color="Sentimiento", barmode="stack",
                title="Sentimientos por ciudad"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Ranking % negativos / positivos ---
            st.subheader("🏆 Rankings de ciudades por sentimiento")

            totals = df_geo_sent.groupby("Ciudad")["Documentos"].sum()
            negs = df_geo_sent[df_geo_sent["Sentimiento"]=="negativo"].groupby("Ciudad")["Documentos"].sum()
            poss = df_geo_sent[df_geo_sent["Sentimiento"]=="positivo"].groupby("Ciudad")["Documentos"].sum()

            ranking_neg = (negs / totals).fillna(0).sort_values(ascending=False).reset_index()
            ranking_pos = (poss / totals).fillna(0).sort_values(ascending=False).reset_index()

            col1, col2 = st.columns(2)
            col1.markdown("### 🚨 Ciudades con mayor % de negativos")
            col1.dataframe(ranking_neg.head(10))

            col2.markdown("### 🌟 Ciudades con mayor % de positivos")
            col2.dataframe(ranking_pos.head(10))
        else:
            st.info("No hay datos de sentimientos por ciudad en sentiment_analysis.json")
    else:
        st.error("No se pudo cargar sentiment_analysis.json")

    # --- Mapa de sentimientos por ciudad con Folium ---
    st.subheader("🗺️ Mapa de sentimientos por ciudad")

    try:
        geo_df
    except NameError:
        geo_df = pd.read_pickle("geographic_data.pkl")

    if not geo_df.empty and "ciudad" in geo_df.columns:
        geo_map = geo_df.copy()

        # Diccionario manual de coordenadas (añade las que falten)
        coords = {
            "Barcelona": (41.3851, 2.1734),
            "Madrid": (40.4168, -3.7038),
            "Sevilla": (37.3891, -5.9845),
            "Tenerife": (28.2916, -16.6291),
            "Malaga": (36.7213, -4.4214),
            "Gran Canaria": (28.1235, -15.4363),
            "Valencia": (39.4699, -0.3763),
            "Mallorca": (39.6953, 3.0176),
            "Bilbao": (43.2630, -2.9350),
            "Granada": (37.1773, -3.5986)
        }

        geo_map["lat"] = geo_map["ciudad"].map(lambda x: coords.get(x, (None,None))[0])
        geo_map["lon"] = geo_map["ciudad"].map(lambda x: coords.get(x, (None,None))[1])

        # Crear mapa Folium
        m = folium.Map(location=[40.0, -3.7], zoom_start=5)

        for _, row in geo_map.iterrows():
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                popup = f"""
                <b>{row['ciudad']}</b><br>
                Total documentos: {row['total_documents']}
                """
                color = "green" if row["dominant_sentiment"]=="positivo" else "red" if row["dominant_sentiment"]=="negativo" else "blue"
                folium.CircleMarker(
                    location=(row["lat"], row["lon"]),
                    radius=row["total_documents"]/50000,  # escala tamaño
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=popup
                ).add_to(m)

        st_folium(m, width=800, height=500)
    else:
        st.info("No hay datos de geographic_data.pkl disponibles.")

# ===================================
# TAB 3: Idiomas & Multilingüe
# ===================================
with tabs[2]:
    st.header("🌍 Análisis de Idiomas & Multilingüe")

    # -----------------------
    # Sección A: Zoom por idioma
    # -----------------------
    if lang_analysis:
        idioma_sel = st.selectbox("Selecciona idioma:", list(lang_analysis.keys()), key="zoom_idioma")
        lang_data = lang_analysis[idioma_sel]

        c1, c2 = st.columns(2)
        c1.metric("Documentos", f"{lang_data['total_documents']:,}")
        c2.metric("Tópicos únicos", lang_data["unique_topics"])

        # Pie chart sentimientos
        sentiments = {k: v["document_count"] for k, v in lang_data["sentiment_breakdown"].items()}
        fig = px.pie(values=sentiments.values(), names=sentiments.keys(),
                     title=f"Distribución de sentimientos en {idioma_sel}")
        st.plotly_chart(fig, use_container_width=True, key="pie_zoom_idioma")

    # -----------------------
    # Sección B: Comparativa global + comparador
    # -----------------------
    if lang_analysis:
        st.subheader("📊 Comparativa global de idiomas")

        # Armar dataset global
        rows = []
        for idioma, datos in lang_analysis.items():
            for sent, block in datos["sentiment_breakdown"].items():
                rows.append({
                    "Idioma": idioma,
                    "Sentimiento": sent,
                    "Documentos": block["document_count"]
                })
        df_lang = pd.DataFrame(rows)

        if not df_lang.empty:
            # Ranking top idiomas
            st.markdown("### 🏆 Ranking de idiomas por volumen total")
            top_langs = (
                df_lang.groupby("Idioma")["Documentos"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            st.dataframe(top_langs.head(10))

            # Barras apiladas sentimientos
            st.markdown("### 😊 Sentimientos por idioma (comparativa)")
            fig2 = px.bar(
                df_lang, x="Idioma", y="Documentos",
                color="Sentimiento", barmode="stack",
                title="Distribución de sentimientos por idioma"
            )
            st.plotly_chart(fig2, use_container_width=True, key="bar_global")

            # Comparador lado a lado (sustituye al treemap)
            st.markdown("### ⚔️ Comparador de Idiomas")
            col1, col2 = st.columns(2)

            with col1:
                idioma_a = st.selectbox("Idioma A", list(lang_analysis.keys()), key="idioma_a")
                data_a = lang_analysis[idioma_a]
                sentiments_a = {k: v["document_count"] for k, v in data_a["sentiment_breakdown"].items()}
                df_a = pd.DataFrame([{"Sentimiento": k, "Documentos": v} for k, v in sentiments_a.items()])
                fig_a = px.bar(df_a, x="Sentimiento", y="Documentos", color="Sentimiento",
                               title=f"Sentimientos en {idioma_a}")
                st.plotly_chart(fig_a, use_container_width=True, key="fig_idioma_a")

            with col2:
                idioma_b = st.selectbox("Idioma B", list(lang_analysis.keys()), key="idioma_b")
                data_b = lang_analysis[idioma_b]
                sentiments_b = {k: v["document_count"] for k, v in data_b["sentiment_breakdown"].items()}
                df_b = pd.DataFrame([{"Sentimiento": k, "Documentos": v} for k, v in sentiments_b.items()])
                fig_b = px.bar(df_b, x="Sentimiento", y="Documentos", color="Sentimiento",
                               title=f"Sentimientos en {idioma_b}")
                st.plotly_chart(fig_b, use_container_width=True, key="fig_idioma_b")

                # Aviso si eligieron el mismo idioma
                if idioma_a == idioma_b:
                    st.warning("⚠️ Has elegido el mismo idioma en ambos lados. Selecciona diferentes para comparar.")
        else:
            st.info("No hay datos suficientes para la comparativa global.")

    else:
        st.error("No se pudo cargar language_analysis.json")

    # -----------------------
    # Sección C: Insights Multilingües
    # -----------------------
    if multi_insights:
        st.subheader("🗣️ Insights Multilingües")
        df_multi = pd.DataFrame(multi_insights)

        if not df_multi.empty and "shannon_diversity" in df_multi.columns:
            sent_options = ["Todos"] + df_multi["sentiment"].dropna().unique().tolist()
            sel_sent = st.selectbox("Filtrar por sentimiento", sent_options, key="multi_sent")

            df_plot = df_multi.copy()
            if sel_sent != "Todos":
                df_plot = df_plot[df_plot["sentiment"] == sel_sent]

            fig4 = px.scatter(
                df_plot, x="shannon_diversity", y="document_count",
                size="language_count", color="sentiment",
                hover_data=["topic_id","dominant_language"],
                title="Diversidad lingüística (Shannon) por tópico"
            )
            st.plotly_chart(fig4, use_container_width=True, key="scatter_multi")

            # --- Leyenda interpretativa ---
            st.markdown("""
            📖 **Interpretación del índice de Shannon:**
            - **0.0** → Un solo idioma (sin diversidad).  
            - **0.2 – 0.5** → Baja diversidad (predomina un idioma, otros marginales).  
            - **0.6 – 1.0** → Diversidad media (varios idiomas presentes, uno dominante).  
            - **>1.0** → Alta diversidad (reparto equilibrado entre varios idiomas).  

            👉 Cuanto mayor el valor, más **multilingüe** es el tópico.
            """)


# ===================================
# TAB 4: Temporalidad
# ===================================
with tabs[3]:
    st.header("📈 Análisis Temporal y Estacional")

    # --- Evolución anual ---
    if seasonal_data:
        st.subheader("📅 Evolución por año")
        df_year = pd.DataFrame(seasonal_data["by_year"]).T.reset_index().rename(columns={"index": "Año"})
        fig = px.bar(df_year, x="Año", y=["positivo", "neutro", "negativo"],
                     barmode="group", title="Sentimientos por año")
        st.plotly_chart(fig, use_container_width=True)

        # --- Evolución mensual ---
        st.subheader("📆 Evolución por mes")
        df_month = pd.DataFrame(seasonal_data["by_month"]).T.reset_index().rename(columns={"index": "Mes"})
        fig = px.area(df_month, x="Mes", y=["positivo", "neutro", "negativo"],
                      groupnorm="percent", title="Proporción de sentimientos por mes")
        st.plotly_chart(fig, use_container_width=True)

    # --- Alertas de picos negativos ---
   
    if not time_series.empty and "date" in time_series.columns and "sentiment" in time_series.columns:
        df_time = time_series.copy()
        df_time["date"] = pd.to_datetime(df_time["date"])
        negs = df_time[df_time["sentiment"]=="negativo"].groupby("date")["document_count"].sum().reset_index()

        mean_neg = negs["document_count"].mean()
        std_neg = negs["document_count"].std()
        picos = negs[negs["document_count"] > mean_neg + 2*std_neg]

        if not picos.empty:
            st.subheader("🚨 Meses con picos de reseñas negativas")
            st.dataframe(picos.head(10))
    else:
        st.warning("`time_series` no tiene columnas esperadas.")


# ===================================
# TAB 5: Geografía
# ===================================
import ast

with tabs[4]:
    st.header("📍 Distribución Geográfica (enfoque descentralización)")

    if not geo_df.empty:
        # --- Expandir columna sentiment_breakdown ---
        if "sentiment_breakdown" in geo_df.columns:
            try:
                geo_df["sentiment_breakdown"] = geo_df["sentiment_breakdown"].apply(ast.literal_eval)
                sent_expanded = geo_df["sentiment_breakdown"].apply(pd.Series)
                geo_df = pd.concat([geo_df.drop(columns=["sentiment_breakdown"]), sent_expanded], axis=1)
            except Exception as e:
                st.warning(f"⚠️ No se pudo expandir sentiment_breakdown: {e}")

        # --- Gráfico barras: distribución general ---
        st.subheader("🏙️ Documentos por ciudad")
        fig = px.bar(
            geo_df.sort_values("total_documents", ascending=False),
            x="ciudad", y="total_documents",
            title="Distribución de documentos por ciudad"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- Top 3 vs resto (medida de concentración) ---
        st.subheader("📊 Concentración geográfica")
        total_docs = geo_df["total_documents"].sum()
        top3_docs = geo_df.sort_values("total_documents", ascending=False).head(3)["total_documents"].sum()
        resto_docs = total_docs - top3_docs

        conc_ratio = top3_docs / total_docs if total_docs else 0
        st.metric("Participación Top 3", f"{conc_ratio:.1%}", 
                  f"{top3_docs:,} docs frente a {resto_docs:,} en el resto")

        fig_conc = px.pie(
            values=[top3_docs, resto_docs],
            names=["Top 3 ciudades", "Resto"],
            title="Concentración de documentos"
        )
        st.plotly_chart(fig_conc, use_container_width=True)

        st.caption("👉 Un valor alto indica que las reseñas están muy concentradas en pocas ciudades (menos descentralización).")

        st.divider()

        # --- Índice de concentración (Herfindahl-Hirschman) ---
        st.subheader("📐 Índice de concentración geográfica")
        shares = geo_df["total_documents"] / total_docs if total_docs else 0
        hhi = (shares**2).sum()
        st.metric("HHI (Herfindahl-Hirschman)", f"{hhi:.3f}")

        st.caption("""
        - Valores cercanos a **1.0** → Alta concentración (centralización).  
        - Valores bajos (≈0.1–0.2) → Más descentralizado.  
        """)

        st.divider()

        # --- Mapa ratio positivos/negativos ---
        if {"positivo", "negativo"}.issubset(geo_df.columns):
            st.subheader("🌍 Mapa de ratio positivos/negativos")
            geo_df["ratio_pos_neg"] = geo_df["positivo"] / geo_df["negativo"].replace(0, 1)

            # Si no hay columnas de coordenadas, añadimos dummy
            if not {"lat", "lon"}.issubset(geo_df.columns):
                coords = {
                    "Barcelona": (41.3851, 2.1734),
                    "Madrid": (40.4168, -3.7038),
                    "Sevilla": (37.3891, -5.9845),
                    "Tenerife": (28.2916, -16.6291),
                    "Malaga": (36.7213, -4.4214),
                    "Gran Canaria": (28.1235, -15.4363),
                    "Valencia": (39.4699, -0.3763),
                    "Mallorca": (39.6953, 3.0176),
                    "Bilbao": (43.2630, -2.9350),
                    "Granada": (37.1773, -3.5986),
                }
                geo_df["lat"] = geo_df["ciudad"].map(lambda x: coords.get(x, (None, None))[0])
                geo_df["lon"] = geo_df["ciudad"].map(lambda x: coords.get(x, (None, None))[1])

            fig_map = px.scatter_geo(
                geo_df, lat="lat", lon="lon",
                size="total_documents", 
                color="ratio_pos_neg",
                hover_name="ciudad",
                hover_data=["total_documents","positivo","negativo"],
                projection="natural earth",
                title="Ratio positivos/negativos por ciudad"
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.caption("👉 Ciudades pequeñas con ratio positivo alto pueden ser oportunidades para descentralizar el turismo.")
        else:
            st.info("No se encontraron columnas 'positivo' y 'negativo' en geographic_data.csv para calcular el ratio.")

    else:
        st.error("No se pudo cargar geographic_data.csv")


# ===================================
# TAB 6: Clima
# ===================================
# ===================================
# TAB 6: Clima
# ===================================
with tabs[5]:
    st.header("🌦️ Clima y Sentimiento")

    # --- Top condiciones climáticas ---
    if weather_summary:
        freq_df = pd.DataFrame(weather_summary["weather_frequency"].items(),
                               columns=["Clima", "Docs"])
        fig = px.bar(freq_df.head(10),
                     x="Docs",
                     y="Clima",
                     orientation="h",
                     title="Top condiciones climáticas")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # --- Top 5 climas por sentimiento ---
    if not weather_matrix.empty:
        st.subheader("⭐ Top 5 climas por sentimiento")

        if all(col in weather_matrix.columns for col in ["descripcion_sencilla", "positivo", "negativo", "neutro"]):
            clima_stats = weather_matrix.copy()

            # --- Positivos ---
            top_pos = clima_stats.sort_values("positivo", ascending=False).head(5)
            fig_pos = px.bar(top_pos,
                             x="positivo",
                             y="descripcion_sencilla",
                             orientation="h",
                             title="Top 5 climas con más reseñas positivas",
                             color="positivo",
                             color_continuous_scale="Greens")
            fig_pos.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_pos, use_container_width=True)

            # --- Negativos ---
            top_neg = clima_stats.sort_values("negativo", ascending=False).head(5)
            fig_neg = px.bar(top_neg,
                             x="negativo",
                             y="descripcion_sencilla",
                             orientation="h",
                             title="Top 5 climas con más reseñas negativas",
                             color="negativo",
                             color_continuous_scale="Reds")
            fig_neg.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_neg, use_container_width=True)

            # --- Neutros ---
            top_neu = clima_stats.sort_values("neutro", ascending=False).head(5)
            fig_neu = px.bar(top_neu,
                             x="neutro",
                             y="descripcion_sencilla",
                             orientation="h",
                             title="Top 5 climas con más reseñas neutras",
                             color="neutro",
                             color_continuous_scale="Blues")
            fig_neu.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_neu, use_container_width=True)
        else:
            st.warning("⚠️ El CSV no tiene las columnas esperadas 'positivo', 'negativo' y 'neutro'.")
    else:
        st.error("No se pudo cargar weather_sentiment_matrix.csv")

        st.divider()

       # --- Distribución de sentimientos por temporada ---
    if not seasonal_weather.empty and "sentiment_breakdown" in seasonal_weather.columns:
        st.subheader("📊 Distribución de sentimientos por temporada")

        import ast
        # Convertir strings de diccionario a dict reales
        seasonal_weather["sentiment_breakdown"] = seasonal_weather["sentiment_breakdown"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Expandir a columnas separadas
        sentiments_expanded = seasonal_weather["sentiment_breakdown"].apply(pd.Series)
        seasonal_weather = pd.concat([seasonal_weather, sentiments_expanded], axis=1)

        # Agrupar por temporada
        df_prop = (
            seasonal_weather.groupby("season")[["positivo","negativo","neutro"]]
            .sum()
            .reset_index()
        )

        # Pasar a formato largo
        df_melt = df_prop.melt(
            id_vars="season",
            value_vars=["positivo","negativo","neutro"],
            var_name="sentimiento",
            value_name="documentos"
        )

        # Normalizar para proporción
        df_melt["proporcion"] = df_melt.groupby("season")["documentos"].transform(
            lambda x: x / x.sum()
        )

        # Gráfico área apilada
        fig = px.area(
            df_melt,
            x="season",
            y="proporcion",
            color="sentimiento",
            title="Proporción de sentimientos por temporada"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("👉 Aquí puedes ver cómo se distribuyen los sentimientos en cada temporada. Un área mayor indica más peso relativo de ese sentimiento en esa estación.")
    else:
        st.error("No se pudo cargar correctamente seasonal_weather_correlation.csv con breakdown de sentimientos.")
