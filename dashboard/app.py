"""
app.py — Backend del Dashboard de M&A Screening
------------------------------------------------
Sigue el patrón REST API de PAT:
  - @app.route = equivalente a @GetMapping / @PostMapping de Spring
  - Funciones de servicio separadas del controlador
  - Respuestas JSON = equivalente a @RestController en Spring

Para ejecutar:
    python app.py
"""

from flask import Flask, jsonify, request, render_template
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# RUTAS DE ARCHIVOS — edita estas dos líneas
# ─────────────────────────────────────────────

# Universo de empresas (Compustat raw, ~200k filas, SIN features derivadas)
UNIVERSE_PATH = "../data/raw/USCompanies.csv"

# Modelo entrenado (rf_smote, Notebook 04)
MODEL_PATH    = "../models/mejor_modelo.pkl"

# ─────────────────────────────────────────────
# FEATURES Y PARÁMETROS DEL MODELO
# (deben coincidir EXACTAMENTE con Notebook 04)
# ─────────────────────────────────────────────

# ORDEN EXACTO del Notebook 03/04 — no cambiar
FEATURES = [
    "exchg", "ap", "at", "che", "lt", "rect", "wcap", "dp",
    "ebitda", "ni", "sale", "xsga", "capx", "csho", "prcc_f", "sich",
    "total_debt", "ebitda_margin", "leverage", "capex_intensity",
    "roa", "current_ratio", "market_cap", "ev_ebitda",
    "asset_turnover", "cash_ratio"
]

WINSOR_LIMITS = {
    "ev_ebitda":     (-50, 100),
    "ebitda_margin": (-2,  1),
}

# Métricas del modelo rf_smote (Notebook 04)
MODEL_METRICS = {
    "modelo":    "Random Forest + SMOTE",
    "auc_roc":   0.726,
    "accuracy":  0.714,
    "precision": 0.596,   # 130/(130+88)
    "recall":    0.279,   # 130/(130+336)
    "f1":        0.376,   # 2*0.596*0.279/(0.596+0.279)
    "fecha":     datetime.now().strftime("%Y-%m-%d")
}

MODEL_COMPARISON = [
    {"modelo": "Logistic Regression",    "auc": 0.56,  "acc": 0.62,  "prec": 0.40,  "rec": 0.48,  "f1": 0.44,  "overfitting": "Bajo"},
    {"modelo": "Random Forest",          "auc": 0.68,  "acc": 0.70,  "prec": 0.58,  "rec": 0.42,  "f1": 0.49,  "overfitting": "Medio"},
    {"modelo": "Random Forest + SMOTE",  "auc": 0.726, "acc": 0.714, "prec": 0.596, "rec": 0.279, "f1": 0.376, "overfitting": "Moderado"},
    {"modelo": "Gradient Boosting",      "auc": 0.69,  "acc": 0.71,  "prec": 0.56,  "rec": 0.50,  "f1": 0.53,  "overfitting": "Alto"},
    {"modelo": "LightGBM",               "auc": 0.70,  "acc": 0.72,  "prec": 0.57,  "rec": 0.38,  "f1": 0.46,  "overfitting": "Alto"},
]

FEATURE_IMPORTANCE = [
    {"variable": "market_cap",      "importancia": 0.21},
    {"variable": "ev_ebitda",       "importancia": 0.15},
    {"variable": "che",             "importancia": 0.13},
    {"variable": "cash_ratio",      "importancia": 0.11},
    {"variable": "csho",            "importancia": 0.09},
    {"variable": "at",              "importancia": 0.08},
    {"variable": "leverage",        "importancia": 0.07},
    {"variable": "roa",             "importancia": 0.06},
    {"variable": "ebitda_margin",   "importancia": 0.05},
    {"variable": "sale",            "importancia": 0.05},
]


# ═══════════════════════════════════════════════════════
# CARGA Y PROCESAMIENTO DEL UNIVERSO
# Se ejecuta UNA SOLA VEZ al arrancar Flask.
# Las peticiones del frontend usan df_full / df_empresas ya listos.
# ═══════════════════════════════════════════════════════

def calcular_features(df):
    """
    Calcula las variables derivadas que usa el modelo.
    Las columnas raw de Compustat (exchg, ap, at, che, lt, rect, wcap,
    dp, ebitda, ni, sale, xsga, capx, csho, prcc_f, sich) ya vienen
    en USCompanies.csv. Solo hay que calcular las derivadas.
    """
    df = df.copy()

    # Deuda total = largo plazo + corto plazo
    df["total_debt"]      = df["dltt"].fillna(0) + df["dlc"].fillna(0)

    # Market cap = acciones * precio de cierre
    df["market_cap"]      = df["csho"] * df["prcc_f"]

    # Ratios financieros (misma lógica que build_dataset.py)
    df["ebitda_margin"]   = df["ebitda"] / df["sale"]
    df["roa"]             = df["ebitda"] / df["at"]
    df["leverage"]        = df["total_debt"] / df["at"]
    df["capex_intensity"] = df["capx"].fillna(0) / df["at"]
    df["asset_turnover"]  = df["sale"] / df["at"]
    df["ev_ebitda"]       = (df["market_cap"] + df["total_debt"] - df["che"]) / df["ebitda"]
    df["current_ratio"]   = df["wcap"].fillna(0) / df["at"]

    denominador           = (df["at"] - df["total_debt"]).replace(0, np.nan)
    df["cash_ratio"]      = df["che"] / denominador

    return df


def limpiar_universo(df):
    """
    Aplica los mismos filtros y transformaciones que Notebook 03:
    - Filtro mínimo de tamaño (at >= 1, sale >= 1)
    - Winsorización de ev_ebitda y ebitda_margin
    - Imputación de nulos con mediana
    """
    # Filtro mínimo (elimina observaciones triviales)
    df = df[(df["at"] >= 1) & (df["sale"] >= 1)].copy()

    # Winsorización (mismos límites que Notebook 03)
    for col, (lo, hi) in WINSOR_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Imputar nulos con mediana columna a columna
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def cargar_universo():
    """
    Carga USCompanies.csv, calcula features y aplica el modelo.
    Devuelve el dataframe listo con columna prob_target añadida.
    """
    print("⏳ Cargando universo de empresas...")
    df = pd.read_csv(UNIVERSE_PATH)
    print(f"   {len(df):,} filas cargadas")

    df = calcular_features(df)
    df = limpiar_universo(df)
    print(f"   {len(df):,} filas tras limpieza")

    # Mapeo básico de sector desde sich (SIC code)
    df["sector"] = df["sich"].apply(sic_a_sector)

    return df


def aplicar_modelo(df, modelo):
    """Aplica el modelo al universo y añade la columna prob_target."""
    X = df[FEATURES].values
    df["prob_target"] = modelo.predict_proba(X)[:, 1]
    return df


def construir_tabla_empresas(df):
    """
    Colapsa el dataframe empresa-año a una fila por empresa (por 'tic').
    Cada empresa lleva dos scores:
      - score_reciente: prob_target del año fiscal más alto disponible
      - score_max / año_max_score: el mayor prob_target histórico y el año
        en que ocurrió. En empate de score, se queda con el año más reciente
        de los empatados.
    El resto de columnas mostradas (sector, market_cap, etc.) se toman
    también del año más reciente, para que el perfil financiero del
    screener sea coherente con el score_reciente.
    """
    df = df.sort_values(["tic", "fyear"])

    # Año más reciente de cada empresa -> fila "base" con score_reciente
    idx_reciente = df.groupby("tic")["fyear"].idxmax()
    base = df.loc[idx_reciente].copy()
    base = base.rename(columns={"prob_target": "score_reciente"})

    # Año con mayor prob_target por empresa.
    # Ordenamos por (prob_target asc, fyear asc) y nos quedamos con el último
    # de cada grupo: así, en empate de prob_target, gana el fyear más alto.
    df_ordenado = df.sort_values(["tic", "prob_target", "fyear"])
    idx_max = df_ordenado.groupby("tic").tail(1).index
    maximos = df.loc[idx_max, ["tic", "fyear", "prob_target"]].rename(
        columns={"fyear": "año_max_score", "prob_target": "score_max"}
    )

    tabla = base.merge(maximos, on="tic", how="left")
    return tabla


def cargar_todo():
    """
    Intenta cargar el universo real + modelo.
    Si falla, genera datos de demo para poder arrancar igualmente.

    Devuelve:
      df_full     -> todas las filas empresa-año (para el histórico del deep dive)
      df_empresas -> una fila por empresa, con score_reciente y score_max/año_max_score
      modelo
      es_demo
    """
    try:
        df = cargar_universo()
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
        df = aplicar_modelo(df, modelo)
        df_empresas = construir_tabla_empresas(df)
        print("Modelo aplicado al universo completo")
        print(f"   {len(df):,} filas empresa-año -> {len(df_empresas):,} empresas únicas")
        return df, df_empresas, modelo, False   # False = no es demo
    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}")
        print("   Arrancando en modo DEMO con datos sinteticos")
        df_demo = generar_demo()
        df_empresas_demo = construir_tabla_empresas(df_demo)
        return df_demo, df_empresas_demo, None, True


def generar_demo():
    """Datos sintéticos cuando no hay archivos reales."""
    np.random.seed(42)
    n = 500
    sectores = ["Technology", "Healthcare", "Energy", "Financials",
                "Industrials", "Consumer Discretionary", "Materials"]
    df = pd.DataFrame({
        "tic":            [f"T{i:04d}" for i in range(n)],
        "conm":           [f"Company {i}" for i in range(n)],
        "sich":           np.random.randint(1000, 9999, n),
        "sector":         np.random.choice(sectores, n),
        "fyear":          np.random.choice([2018, 2019, 2020], n),
        "market_cap":     np.random.lognormal(7, 1.5, n),
        "ev_ebitda":      np.random.uniform(3, 25, n),
        "che":            np.random.lognormal(5, 1.2, n),
        "cash_ratio":     np.random.uniform(0.1, 2.0, n),
        "csho":           np.random.lognormal(5, 1, n),
        "at":             np.random.lognormal(8, 1.5, n),
        "sale":           np.random.lognormal(7, 1.5, n),
        "ebitda":         np.random.lognormal(5, 1.5, n),
        "roa":            np.random.uniform(-0.05, 0.15, n),
        "leverage":       np.random.uniform(0.1, 0.8, n),
        "current_ratio":  np.random.uniform(0.5, 3.0, n),
        "ebitda_margin":  np.random.uniform(-0.1, 0.4, n),
        "asset_turnover": np.random.uniform(0.2, 1.5, n),
        "capex_intensity":np.random.uniform(0.01, 0.15, n),
        "total_debt":     np.random.lognormal(6, 1.5, n),
        "prob_target":    np.random.beta(2, 5, n),
    })
    return df


def sic_a_sector(sich):
    """
    Mapeo básico de código SIC a sector (aproximación GICS).
    Útil para filtrar por sector en el screener.
    """
    try:
        s = int(sich)
    except (ValueError, TypeError):
        return "Other"
    if   s < 1000:  return "Agriculture"
    elif s < 1500:  return "Mining"
    elif s < 1800:  return "Construction"
    elif s < 2000:  return "Construction"
    elif s < 4000:  return "Manufacturing"
    elif s < 5000:  return "Transportation & Utilities"
    elif s < 5200:  return "Wholesale Trade"
    elif s < 5900:  return "Retail Trade"
    elif s < 6000:  return "Retail Trade"
    elif s < 6500:  return "Financials"
    elif s < 6800:  return "Real Estate"
    elif s < 7000:  return "Financials"
    elif s < 8000:  return "Services"
    elif s < 8700:  return "Services"
    elif s < 9000:  return "Technology"
    else:           return "Public Administration"


# Carga al arrancar (una sola vez)
# df_full     -> todas las filas empresa-año (para el historico del deep dive)
# df_empresas -> una fila por empresa (para summary y screener)
df_full, df_empresas, modelo_global, ES_DEMO = cargar_todo()
print(f"Dashboard listo. {len(df_empresas):,} empresas unicas, {len(df_full):,} filas empresa-año en memoria")


# ═══════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/summary", methods=["GET"])
def get_summary():
    # Trabajamos sobre df_empresas: una fila por empresa (score del año más reciente)
    n_empresas = len(df_empresas)

    probs = df_empresas["score_reciente"].dropna()

    # KPIs: score máximo y empresas con señal fuerte (>0.35), sobre el score reciente
    score_max    = round(float(probs.max()) * 100, 1)
    señal_fuerte = int((probs > 0.35).sum())

    top10 = (
        df_empresas.nlargest(10, "score_reciente")
        [["conm","tic","sector","fyear","market_cap","score_reciente"]]
        .rename(columns={"conm":"empresa","tic":"ticker",
                         "fyear":"año","score_reciente":"probabilidad"})
        .round({"probabilidad":3,"market_cap":1})
        .to_dict(orient="records")
    )

    counts, bins = np.histogram(probs, bins=20, range=(0,1))
    distribucion = {
        "bins":   [round(b,2) for b in bins[:-1].tolist()],
        "counts": counts.tolist()
    }

    por_sector = (
        df_empresas.groupby("sector")["score_reciente"]
        .mean().round(3).reset_index()
        .rename(columns={"score_reciente":"prob_media"})
        .sort_values("prob_media", ascending=False)
        .to_dict(orient="records")
    ) if "sector" in df_empresas.columns else []

    return jsonify({
        "kpis": {
            "n_empresas":    n_empresas,
            "score_max":     score_max,
            "señal_fuerte":  señal_fuerte,
            **MODEL_METRICS
        },
        "top10":             top10,
        "distribucion_probs": distribucion,
        "por_sector":         por_sector
    })


@app.route("/api/screener", methods=["GET"])
def get_screener():
    sector   = request.args.get("sector",   "")
    año      = request.args.get("año",      "")
    prob_min = float(request.args.get("prob_min", 0.0))
    # Paginación: el frontend pide páginas de 100
    pagina   = int(request.args.get("pagina", 1))
    por_pag  = 100

    # Trabajamos sobre df_empresas: una fila por empresa
    df = df_empresas.copy()
    if sector:   df = df[df["sector"] == sector]
    if año:      df = df[df["fyear"]  == int(año)]
    if prob_min: df = df[df["score_reciente"] >= prob_min]

    df = df.sort_values("score_reciente", ascending=False)
    total = len(df)

    # Devolver solo la página solicitada (100 filas)
    inicio = (pagina - 1) * por_pag
    df_pag = df.iloc[inicio: inicio + por_pag]

    cols = ["conm","tic","sector","fyear","market_cap",
            "ev_ebitda","ebitda_margin","leverage",
            "score_reciente","año_max_score","score_max"]
    cols_ok = [c for c in cols if c in df_pag.columns]

    resultado = (
        df_pag[cols_ok]
        .rename(columns={"conm":"empresa","tic":"ticker",
                         "fyear":"año","score_reciente":"probabilidad",
                         "año_max_score":"año_atractivo","score_max":"probabilidad_atractiva"})
        .round({"probabilidad":3,"market_cap":1,
                "ev_ebitda":2,"ebitda_margin":3,"leverage":3,
                "probabilidad_atractiva":3})
        .to_dict(orient="records")
    )

    sectores = sorted(df_empresas["sector"].dropna().unique().tolist())
    años     = sorted(df_empresas["fyear"].dropna().unique().astype(int).tolist())

    return jsonify({
        "empresas":  resultado,
        "total":     total,
        "pagina":    pagina,
        "paginas":   (total // por_pag) + 1,
        "filtros_disponibles": {"sectores": sectores, "años": años}
    })


@app.route("/api/empresa/<ticker>", methods=["GET"])
def get_empresa(ticker):
    # Para el deep dive usamos la fila colapsada (datos del año más reciente)
    fila = df_empresas[df_empresas["tic"] == ticker]
    if fila.empty:
        return jsonify({"error": f"Empresa '{ticker}' no encontrada"}), 404

    row  = fila.iloc[0]
    prob = float(row.get("score_reciente", 0))

    # Umbrales ajustados al rango real del modelo (máx ~40%)
    if prob >= 0.30:    señal, color_cls = "Señal fuerte",   "fuerte"
    elif prob >= 0.18:  señal, color_cls = "Señal moderada", "moderada"
    else:               señal, color_cls = "Señal débil",    "debil"

    financiero = {
        "market_cap":     round(float(row.get("market_cap",     0)), 1),
        "sale":           round(float(row.get("sale",           0)), 1),
        "ebitda":         round(float(row.get("ebitda",         0)), 1),
        "at":             round(float(row.get("at",             0)), 1),
        "che":            round(float(row.get("che",            0)), 1),
        "total_debt":     round(float(row.get("total_debt",     0)), 1),
        "ev_ebitda":      round(float(row.get("ev_ebitda",      0)), 2),
        "ebitda_margin":  round(float(row.get("ebitda_margin",  0)), 3),
        "roa":            round(float(row.get("roa",            0)), 3),
        "leverage":       round(float(row.get("leverage",       0)), 3),
        "asset_turnover": round(float(row.get("asset_turnover", 0)), 3),
        "cash_ratio":     round(float(row.get("cash_ratio",     0)), 3),
    }

    etiquetas = {
        "market_cap": "Market Cap (M$)", "sale": "Sales (M$)",
        "ebitda": "EBITDA (M$)", "at": "Total Assets (M$)",
        "che": "Cash & Equivalents (M$)", "total_debt": "Total Debt (M$)",
        "ev_ebitda": "EV/EBITDA", "ebitda_margin": "EBITDA Margin",
        "roa": "ROA", "leverage": "Leverage",
        "asset_turnover": "Asset Turnover", "cash_ratio": "Cash Ratio",
    }

    sector = row.get("sector", "")
    if sector and "sector" in df_empresas.columns:
        df_sec = df_empresas[df_empresas["sector"] == sector]
        medianas_sector = {
            k: round(float(df_sec[k].median()), 3)
            for k in financiero if k in df_sec.columns
        }
    else:
        medianas_sector = {}

    # Histórico de scores por año fiscal, para el gráfico de evolución
    # (usa df_full, que conserva todas las filas empresa-año)
    historico_df = (
        df_full[df_full["tic"] == ticker]
        .sort_values("fyear")
        [["fyear", "prob_target"]]
        .rename(columns={"fyear": "año", "prob_target": "score"})
        .round({"score": 3})
    )
    historico = historico_df.to_dict(orient="records")

    return jsonify({
        "info": {
            "empresa": str(row.get("conm", ticker)),
            "ticker":  ticker,
            "sector":  str(sector),
            "año":     int(row.get("fyear", 0))
        },
        "prediccion": {
            "probabilidad":  round(prob, 3),
            "score_pct":     round(prob * 100, 1),   # en % para mostrar
            "señal":         señal,
            "color_cls":     color_cls,
            # mantenemos "color" para compatibilidad con el JS antiguo
            "color":         "green" if prob >= 0.30 else "yellow" if prob >= 0.18 else "red"
        },
        "año_atractivo":       int(row.get("año_max_score", 0)) if pd.notna(row.get("año_max_score")) else None,
        "score_atractivo":     round(float(row.get("score_max", 0)), 3),
        "financiero":          financiero,
        "etiquetas":           etiquetas,
        "medianas_sector":     medianas_sector,
        "historico_scores":    historico,
        "feature_importance":  FEATURE_IMPORTANCE
    })


@app.route("/api/simular", methods=["POST"])
def simular():
    datos = request.get_json()
    if not datos:
        return jsonify({"error": "No se recibieron datos"}), 400

    try:
        # Inputs del formulario
        sale       = float(datos.get("sale",        0))
        ebitda     = float(datos.get("ebitda",      0))
        at         = float(datos.get("at",          0))
        che        = float(datos.get("che",          0))
        total_debt = float(datos.get("total_debt",   0))
        market_cap = float(datos.get("market_cap",   0))
        capex      = float(datos.get("capex",        0))
        csho       = float(datos.get("csho",         0))
        prcc_f     = float(datos.get("share_price",  0))

        # Columnas raw con valores por defecto razonables
        exchg = 11.0        # NYSE por defecto
        ap    = 0.0
        lt    = total_debt
        rect  = 0.0
        wcap  = at * 0.2    # aproximación
        dp    = ebitda * 0.1
        ni    = ebitda * 0.6
        xsga  = sale * 0.1
        sich  = 7372.0      # SIC genérico servicios informáticos

        # Calcular features derivadas (misma lógica que calcular_features())
        ev_ebitda       = (market_cap + total_debt - che) / ebitda if ebitda != 0 else 0
        ebitda_margin   = ebitda / sale    if sale        != 0 else 0
        roa             = ebitda / at      if at          != 0 else 0
        leverage        = total_debt / at  if at          != 0 else 0
        denominador     = at - total_debt
        cash_ratio      = che / denominador if denominador != 0 else 0
        asset_turnover  = sale / at         if at          != 0 else 0
        capex_intensity = capex / at        if at          != 0 else 0
        current_ratio   = wcap / at         if at          != 0 else 1.5

        # Vector en el MISMO ORDEN que FEATURES (26 columnas)
        # ['exchg','ap','at','che','lt','rect','wcap','dp',
        #  'ebitda','ni','sale','xsga','capx','csho','prcc_f','sich',
        #  'total_debt','ebitda_margin','leverage','capex_intensity',
        #  'roa','current_ratio','market_cap','ev_ebitda',
        #  'asset_turnover','cash_ratio']
        X = np.array([[
            exchg, ap, at, che, lt, rect, wcap, dp,
            ebitda, ni, sale, xsga, capex, csho, prcc_f, sich,
            total_debt, ebitda_margin, leverage, capex_intensity,
            roa, current_ratio, market_cap, ev_ebitda,
            asset_turnover, cash_ratio
        ]])

        if modelo_global is not None:
            prob = float(modelo_global.predict_proba(X)[0, 1])
        else:
            prob = min(max(0.1*cash_ratio + 0.3*(1-leverage) + 0.2*roa + 0.1, 0), 1)

        if prob >= 0.30:    clasificacion, color = "Señal fuerte",   "green"
        elif prob >= 0.18:  clasificacion, color = "Señal moderada", "yellow"
        else:               clasificacion, color = "Señal débil",    "red"

        percentil = float(np.mean(df_empresas["score_reciente"] <= prob) * 100)

        return jsonify({
            "probabilidad":  round(prob, 3),
            "clasificacion": clasificacion,
            "color":         color,
            "percentil":     round(percentil, 1),
            "variables_derivadas": {
                "ev_ebitda":       round(ev_ebitda, 2),
                "ebitda_margin":   round(ebitda_margin, 3),
                "roa":             round(roa, 3),
                "leverage":        round(leverage, 3),
                "cash_ratio":      round(cash_ratio, 3),
                "asset_turnover":  round(asset_turnover, 3),
                "capex_intensity": round(capex_intensity, 4),
            }
        })
    except (ValueError, ZeroDivisionError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model", methods=["GET"])
def get_model():
    probs = df_empresas["score_reciente"].dropna()
    counts, bins = np.histogram(probs, bins=20, range=(0,1))
    return jsonify({
        "metricas":            MODEL_METRICS,
        "comparativa_modelos": MODEL_COMPARISON,
        "feature_importance":  FEATURE_IMPORTANCE,
        "distribucion_probs": {
            "bins":   [round(b,2) for b in bins[:-1].tolist()],
            "counts": counts.tolist()
        }
    })


@app.route("/api/tickers", methods=["GET"])
def get_tickers():
    tickers = (
        df_empresas[["tic","conm"]].drop_duplicates()
        .rename(columns={"tic":"ticker","conm":"empresa"})
        .sort_values("empresa")
        .to_dict(orient="records")
    )
    return jsonify(tickers)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
