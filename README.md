# SmarTouring — Dashboard (BERT)

Dashboard en **Streamlit** para explorar resultados de análisis de sentimiento (BERT) sobre reseñas turísticas en 8 ciudades de España.  
Incluye normalización robusta de fechas, filtros avanzados, **mapa por ciudad**, KPIs, **barras apiladas por categoría**, serie temporal y **nube de palabras** con stopwords estáticas y dinámicas.

## 🧩 Características
- Carga por defecto: `Resultados Bert.csv` (puedes subir otro desde la barra lateral).
- Canonización automática de columnas (acepta sinónimos):  
  `Fecha`, `Texto`, `Ciudad`, `Categoria`, `Fuente`, `sentimiento`, `confianza` (+ opcionales `idioma`, `topic`).
- Normalización de fechas: múltiples formatos (`dayfirst`, `yearfirst`, ISO con `T`/`Z`), epoch (s/ms/us/ns) y **Excel serial**.
- **Mapa por ciudad (España)**: tamaño = nº reseñas; color = índice de sentimiento *(pos% − neg%)*.
- **Barras apiladas** de **sentimiento por categoría**, conmutador **% / conteos** y Top-N.
- Serie temporal semanal (% por clase) con horizonte y suavizado.
- **Nube de palabras** con:
  - stopwords multilingües ampliadas (ES/EN/FR/IT/DE/PT/CAT),
  - stopwords **dinámicas** derivadas de las ciudades y categorías filtradas.
- Exportación del filtrado y guardado de CSV con **fechas normalizadas**.
- Expander para **auditar filas con fecha inválida** y descargarlas.

## 📁 Estructura recomendada
```
streamlit_data/
├─ dashboard.definitivo.py        # este dashboard (solo BERT)
├─ Resultados Bert.csv            # datos por defecto (opcional en el repo)
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 📊 Formato de datos esperado
Columnas mínimas (se renombran si llegan con sinónimos):
- **Fecha** (fecha/hora de la reseña),
- **Texto** (contenido de la reseña),
- **Ciudad**,
- **Categoria**,
- **Fuente** (plataforma),
- **sentimiento** (`positivo`, `neutro`, `negativo`),
- **confianza** (probabilidad/score ∈ [0,1]).

Opcionales: **idioma**, **topic**.

> Si alguna columna no existe, el panel lo indica y se detiene con un mensaje claro.

## ▶️ Ejecución local
```bash
# 1) Crear entorno (opcional)
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate                           # Windows

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Lanzar el dashboard
streamlit run dashboard.definitivo.py
```

## 🔧 Consejos de uso
- Desde la **barra lateral** puedes:
  - elegir/ subir el CSV,
  - limpiar caché (**♻️**) cuando cambies de archivo,
  - aplicar filtros por ciudad, categoría, fuente, idioma, sentimiento y confianza.
- En “Nube de palabras”, si aparecen términos que no quieres ver, añádelos a `EXTRA_STOPS`
  (o dímelos y los integro en el código).

## 🚀 Despliegue (Streamlit Community Cloud)
1. Sube este proyecto a GitHub (con `requirements.txt`).
2. En Streamlit Community Cloud: **New app** → selecciona repo/branch/`dashboard.definitivo.py`.
3. Define los **Secrets** si los necesitaras (no obligatorio en este proyecto).

> Si `Resultados Bert.csv` pesa >100 MB, usa **Git LFS** o sirve el archivo desde un almacenamiento externo.

## 🔒 Privacidad y datos
- Evita subir al repo datos sensibles (PII).
- Puedes trabajar localmente con el CSV real y publicar un **sample** en GitHub.

## 🧾 Licencia
MIT (puedes cambiarla si lo prefieres).
