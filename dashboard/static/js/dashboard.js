/**
 * dashboard.js. Lógica del frontend
 * -----------------------------------
 * Patrón PAT Tema 5 (JavaScript):
 *   - fetch() equivale a las peticiones HTTP que vimos con telnet/curl
 *   - Los endpoints /api/... son nuestro propio servidor Flask
 *   - Separamos: navegación, llamadas API, y renderizado
 *
 * Patrón Tema 6 (Backend): el JS llama a nuestra API REST propia
 *   GET /api/summary    → Executive Summary
 *   GET /api/screener   → M&A Screener con filtros
 *   GET /api/empresa/TK → Company Deep Dive
 *   POST /api/simular   → Simulador (envía JSON en el body)
 *   GET /api/model      → Model Performance
 */

"use strict";

// ─── Variables globales de estado ───
let datosScreener = [];          // Cache de la página actual del screener
let datosSimulacion = null;      // Último resultado del simulador
let datosEmpresa = null;         // Último resultado de deep dive
let paginaActual = 1;            // Paginación del screener
let totalPaginas = 1;            // Total de páginas del screener
let totalResultadosActual = 0;   // Total de resultados de la última carga del screener


// ═══════════════════════════════════════════
// NAVEGACIÓN. mostrar/ocultar secciones
// Equivalente al target de un <a href="#id"> en PAT CSS (pseudo-clases)
// ═══════════════════════════════════════════

function mostrarSeccion(id) {
  // Ocultar todas las secciones
  document.querySelectorAll(".seccion").forEach(s => s.classList.remove("activa"));
  document.querySelectorAll("nav a").forEach(a => a.classList.remove("activo"));

  // Mostrar la seleccionada
  document.getElementById(id).classList.add("activa");
  document.getElementById(`nav-${id}`).classList.add("activo");

  // Cargar datos si la sección los necesita y no están ya cargados
  if (id === "summary"   && document.getElementById("kpi-empresas").textContent === "—") cargarSummary();
  if (id === "screener"  && datosScreener.length === 0) cargarScreener();
  if (id === "deepdive"  && document.getElementById("selector-empresa").options.length <= 1) cargarTickers();
}


// ═══════════════════════════════════════════
// SECCIÓN 1: EXECUTIVE SUMMARY
// GET /api/summary
// ═══════════════════════════════════════════

async function cargarSummary() {
  // fetch() = petición GET al servidor (patrón PAT)
  const res  = await fetch("/api/summary");
  const data = await res.json();

  // Rellenar KPI cards
  const k = data.kpis;
  setText("kpi-empresas",    k.n_empresas.toLocaleString());
  setText("kpi-score-max",   k.score_max + "%");
  setText("kpi-señal-fuerte", k.señal_fuerte.toLocaleString());
  setText("kpi-modelo",      k.modelo);
  setText("kpi-auc",         k.auc_roc.toFixed(3));
  setText("kpi-precision",   (k.precision * 100).toFixed(1) + "%");
  setText("footer-fecha",    `Modelo: ${k.modelo}`);

  // Histograma de probabilidades (Plotly)
  Plotly.newPlot("grafico-dist-summary", [{
    type: "bar",
    x: data.distribucion_probs.bins,
    y: data.distribucion_probs.counts,
    marker: { color: generarGradienteColores(data.distribucion_probs.bins) },
    hovertemplate: "Prob: %{x:.2f}<br>Empresas: %{y}<extra></extra>"
  }], layoutBase("Probabilidad de adquisición", "Nº empresas"), configBase());

  // Gráfico de barras por sector
  if (data.por_sector && data.por_sector.length > 0) {
    const sectores = data.por_sector.map(s => s.sector);
    const probs    = data.por_sector.map(s => s.prob_media);
    Plotly.newPlot("grafico-sectores-summary", [{
      type: "bar",
      x: probs,
      y: sectores,
      orientation: "h",
      marker: { color: "#2a6496" },
      hovertemplate: "%{y}: %{x:.3f}<extra></extra>"
    }], { ...layoutBase("Prob. media", ""), margin: { l: 160, r: 20, t: 20, b: 40 } }, configBase());
  }

  // Top 10 tabla
  const tbody = document.getElementById("tabla-top10");
  tbody.innerHTML = data.top10.map((e, i) => `
    <tr>
      <td>${i + 1}</td>
      <td><strong>${e.empresa || "—"}</strong></td>
      <td>${e.ticker || "—"}</td>
      <td>${e.sector || "—"}</td>
      <td>${e.año || "—"}</td>
      <td>${(e.market_cap || 0).toLocaleString()}</td>
      <td>${badgeProbabilidad(e.probabilidad)}</td>
    </tr>`).join("");
}


// ═══════════════════════════════════════════
// SECCIÓN 2: M&A SCREENER
// GET /api/screener?sector=...&prob_min=...
// ═══════════════════════════════════════════

async function cargarScreener() {
  // Primera carga sin filtros. también trae los valores para los selects
  const res  = await fetch("/api/screener?pagina=1");
  const data = await res.json();

  datosScreener = data.empresas;
  totalPaginas  = data.paginas;
  paginaActual  = 1;

  // Rellenar filtros con valores del servidor (solo la primera vez)
  rellenarSelect("filtro-sector", data.filtros_disponibles.sectores);
  rellenarSelect("filtro-año",    data.filtros_disponibles.años.map(String));

  renderizarScreener(data);
}

async function aplicarFiltros() {
  // Al filtrar siempre volvemos a página 1
  paginaActual = 1;
  await fetchScreener();
}

async function cambiarPagina(delta) {
  const nueva = paginaActual + delta;
  if (nueva < 1 || nueva > totalPaginas) return;
  paginaActual = nueva;
  await fetchScreener();
}

async function fetchScreener() {
  // Leer filtros actuales del HTML
  const sector  = document.getElementById("filtro-sector").value;
  const año     = document.getElementById("filtro-año").value;
  const probMin = document.getElementById("filtro-prob").value;

  // Construir URL con query params + paginación
  // Equivale a @RequestParam en Spring: ?sector=X&pagina=2
  const params = new URLSearchParams({ pagina: paginaActual });
  if (sector)  params.append("sector",   sector);
  if (año)     params.append("año",      año);
  if (probMin) params.append("prob_min", probMin);

  const res  = await fetch(`/api/screener?${params}`);
  const data = await res.json();

  datosScreener = data.empresas;
  totalPaginas  = data.paginas;
  renderizarScreener(data);
}

function renderizarScreener(data) {
  const empresas = data.empresas;
  const total    = data.total;
  totalResultadosActual = total;

  // Info de resultados y paginación
  setText("total-resultados",
    `${total.toLocaleString()} resultados. página ${paginaActual} de ${totalPaginas}`);

  // Botones de paginación
  document.getElementById("btn-prev").disabled = (paginaActual <= 1);
  document.getElementById("btn-next").disabled = (paginaActual >= totalPaginas);

  // Tabla: año y score más reciente + año y score más atractivo histórico
  const tbody = document.getElementById("tabla-screener-body");
  tbody.innerHTML = empresas.map(e => `
    <tr>
      <td>${e.empresa || "—"}</td>
      <td>${e.ticker  || "—"}</td>
      <td>${e.sector  || "—"}</td>
      <td>${(e.market_cap    || 0).toFixed(0)}</td>
      <td>${(e.ev_ebitda     || 0).toFixed(1)}</td>
      <td>${((e.ebitda_margin || 0) * 100).toFixed(1)}%</td>
      <td>${(e.leverage      || 0).toFixed(2)}</td>
      <td>${e.año     || "—"}</td>
      <td>${badgeProbabilidad(e.probabilidad)}</td>
      <td>${e.año_atractivo || "—"}</td>
      <td>${badgeProbabilidad(e.probabilidad_atractiva)}</td>
    </tr>`).join("");
}


// ═══════════════════════════════════════════
// SECCIÓN 3: COMPANY DEEP DIVE
// GET /api/empresa/<ticker>
// ═══════════════════════════════════════════

async function cargarTickers() {
  // GET /api/tickers → lista de empresas para el select
  const res     = await fetch("/api/tickers");
  const tickers = await res.json();
  const sel     = document.getElementById("selector-empresa");

  tickers.forEach(t => {
    const opt = document.createElement("option");
    opt.value       = t.ticker;
    opt.textContent = `${t.empresa} (${t.ticker})`;
    sel.appendChild(opt);
  });
}

async function cargarEmpresa() {
  const ticker = document.getElementById("selector-empresa").value;
  if (!ticker) { alert("Selecciona una empresa primero"); return; }

  // GET /api/empresa/AAPL (patrón @PathVariable en PAT)
  const res  = await fetch(`/api/empresa/${ticker}`);
  if (!res.ok) { alert("Empresa no encontrada"); return; }

  datosEmpresa = await res.json();
  renderizarEmpresa(datosEmpresa);
}

function renderizarEmpresa(data) {
  document.getElementById("resultado-empresa").style.display = "block";

  // Info general
  document.getElementById("info-empresa").innerHTML = `
    <b>Empresa:</b> ${data.info.empresa}<br>
    <b>Ticker:</b>  ${data.info.ticker}<br>
    <b>Sector:</b>  ${data.info.sector}<br>
    <b>Año fiscal:</b> ${data.info.año}`;

  // Predicción
  const prob    = data.prediccion.probabilidad;
  const clasif  = data.prediccion.clasificacion;
  const colHex  = { green: "#27ae60", yellow: "#f39c12", red: "#e74c3c" }[data.prediccion.color];

  document.getElementById("prob-valor").textContent = `${(prob * 100).toFixed(1)}%`;
  document.getElementById("prob-valor").style.color = colHex;
  document.getElementById("prob-clasif").textContent = clasif;

  // Gauge (medidor) con Plotly
  Plotly.newPlot("grafico-gauge", [{
    type: "indicator", mode: "gauge+number",
    value: prob * 100,
    number: { suffix: "%", font: { size: 28 } },
    gauge: {
      axis: { range: [0, 100] },
      bar: { color: colHex },
      steps: [
        { range: [0, 35],  color: "#fce4e4" },
        { range: [35, 60], color: "#fef9e7" },
        { range: [60, 100], color: "#d5f5e3" }
      ]
    }
  }], { height: 200, margin: { t: 20, b: 10, l: 30, r: 30 } }, configBase());

  // Perfil financiero
  const f = data.financiero;
  const etiquetas = {
    market_cap: "Market Cap (M$)", sale: "Sales (M$)", ebitda: "EBITDA (M$)",
    at: "Assets (M$)", che: "Cash (M$)", total_debt: "Debt (M$)",
    ev_ebitda: "EV/EBITDA", ebitda_margin: "EBITDA Margin",
    roa: "ROA", leverage: "Leverage",
    asset_turnover: "Asset Turnover", cash_ratio: "Cash Ratio"
  };
  document.getElementById("perfil-financiero").innerHTML = Object.entries(f)
    .map(([k, v]) => `
      <div class="perfil-item">
        <div class="pval">${typeof v === "number" && v > 100 ? v.toLocaleString() : v}</div>
        <div class="plabel">${etiquetas[k] || k}</div>
      </div>`).join("");

  // Evolución del score por año fiscal (histórico completo de la empresa)
  const historico = data.historico_scores || [];
  if (historico.length > 0) {
    const años   = historico.map(h => h.año);
    const scores = historico.map(h => h.score * 100);

    Plotly.newPlot("grafico-evolucion", [{
      type: "scatter", mode: "lines+markers",
      x: años, y: scores,
      line: { color: "#2a6496" },
      marker: {
        size: 8,
        color: años.map(a => a === data.año_atractivo ? "#e74c3c" : "#2a6496")
      },
      hovertemplate: "Año %{x}: %{y:.1f}%<extra></extra>"
    }], { ...layoutBase("Año fiscal", "Score (%)"), height: 220 }, configBase());

    setText("evolucion-nota",
      `Año más atractivo histórico: ${data.año_atractivo} (score ${data.score_atractivo ? (data.score_atractivo * 100).toFixed(1) : "—"}%). El punto en rojo lo marca en el gráfico.`);
  } else {
    document.getElementById("grafico-evolucion").innerHTML = "";
    setText("evolucion-nota", "No hay histórico de varios años disponible para esta empresa.");
  }

  // Radar chart empresa vs sector
  const metricas     = ["ebitda_margin", "roa", "leverage", "cash_ratio", "asset_turnover"];
  const labelsMet    = ["EBITDA Margin", "ROA", "Leverage", "Cash Ratio", "Asset Turnover"];
  const valEmpresa   = metricas.map(m => f[m] || 0);
  const valSector    = metricas.map(m => data.medianas_sector[m] || 0);

  Plotly.newPlot("grafico-radar", [
    { type: "scatterpolar", r: valEmpresa, theta: labelsMet, fill: "toself",
      name: data.info.ticker, line: { color: "#2a6496" } },
    { type: "scatterpolar", r: valSector, theta: labelsMet, fill: "toself",
      name: "Mediana sector", line: { color: "#e74c3c", dash: "dot" }, opacity: 0.5 }
  ], { polar: { radialaxis: { visible: true } }, legend: { x: 0 },
       height: 280, margin: { t: 20, b: 20, l: 20, r: 20 } }, configBase());

  // Feature importance
  const fi   = data.feature_importance.slice(0, 8);
  Plotly.newPlot("grafico-importance", [{
    type: "bar", orientation: "h",
    x: fi.map(f => f.importancia),
    y: fi.map(f => f.variable),
    marker: { color: "#2a6496" },
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>"
  }], { ...layoutBase("Importancia", ""), margin: { l: 120, r: 20, t: 20, b: 40 }, height: 280 }, configBase());
}


// ═══════════════════════════════════════════
// SECCIÓN 4: SIMULADOR
// POST /api/simular (envía JSON en el body)
// ═══════════════════════════════════════════

async function calcularSimulacion() {
  // Leer inputs del formulario HTML
  const payload = {
    sale:       getNum("sim-sale"),
    ebitda:     getNum("sim-ebitda"),
    at:         getNum("sim-at"),
    che:        getNum("sim-che"),
    total_debt: getNum("sim-debt"),
    market_cap: getNum("sim-mcap"),
    capex:      getNum("sim-capex"),
    csho:       getNum("sim-csho"),
    share_price: getNum("sim-price")
  };

  // Validación básica
  if (!payload.sale || !payload.ebitda || !payload.at) {
    alert("Sales, EBITDA y Assets son obligatorios"); return;
  }

  // POST /api/simular. envía JSON en el body
  // Equivale a una petición POST con body en PAT (telnet, chunked, etc.)
  const res = await fetch("/api/simular", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },   // cabecera obligatoria
    body:    JSON.stringify(payload)                    // body con JSON
  });

  if (!res.ok) {
    const err = await res.json();
    alert("Error: " + err.error);
    return;
  }

  datosSimulacion = await res.json();
  renderizarSimulacion(datosSimulacion);
}

function renderizarSimulacion(data) {
  document.getElementById("resultado-simulacion").style.display = "block";

  const prob   = data.probabilidad;
  const colHex = { green: "#27ae60", yellow: "#f39c12", red: "#e74c3c" }[data.color];

  document.getElementById("sim-prob-valor").textContent = `${(prob * 100).toFixed(1)}%`;
  document.getElementById("sim-prob-valor").style.color = colHex;
  document.getElementById("sim-clasif").textContent     = data.clasificacion;
  document.getElementById("sim-percentil").textContent  = `${data.percentil}%`;

  // Gauge
  Plotly.newPlot("sim-gauge", [{
    type: "indicator", mode: "gauge+number",
    value: prob * 100,
    number: { suffix: "%", font: { size: 32 } },
    gauge: {
      axis: { range: [0, 100] },
      bar: { color: colHex },
      steps: [
        { range: [0,  35], color: "#fce4e4" },
        { range: [35, 60], color: "#fef9e7" },
        { range: [60, 100], color: "#d5f5e3" }
      ]
    }
  }], { height: 220, margin: { t: 20, b: 10, l: 30, r: 30 } }, configBase());

  // Variables derivadas
  const etiq = {
    ev_ebitda: "EV/EBITDA", ebitda_margin: "EBITDA Margin",
    roa: "ROA", leverage: "Leverage",
    cash_ratio: "Cash Ratio", asset_turnover: "Asset Turnover",
    capex_intensity: "Capex Intensity"
  };
  document.getElementById("sim-derivadas").innerHTML = Object.entries(data.variables_derivadas)
    .map(([k, v]) => `<div><strong>${etiq[k] || k}:</strong> ${v}</div>`).join("");
}


// ═══════════════════════════════════════════
// SECCIÓN 5: MODEL PERFORMANCE
// GET /api/model
// ═══════════════════════════════════════════



// ═══════════════════════════════════════════
// GENERACIÓN DE INFORMES HTML
// (descarga un archivo HTML con el resumen)
// ═══════════════════════════════════════════

function generarInformeEmpresa() {
  if (!datosEmpresa) return;
  const d = datosEmpresa;
  const html = plantillaInforme(
    `Informe M&A. ${d.info.empresa} (${d.info.ticker})`,
    `<h2>${d.info.empresa} (${d.info.ticker})</h2>
     <p><b>Sector:</b> ${d.info.sector} | <b>Año:</b> ${d.info.año}</p>
     <h3>Resultado del modelo</h3>
     <p>Probabilidad: <strong>${(d.prediccion.probabilidad * 100).toFixed(1)}%</strong>
        . ${d.prediccion.clasificacion}</p>
     <h3>Perfil financiero</h3>
     <ul>${Object.entries(d.financiero).map(([k,v]) => `<li><b>${k}:</b> ${v}</li>`).join("")}</ul>
     <h3>Variables más relevantes</h3>
     <ol>${d.feature_importance.slice(0,5).map(f => `<li>${f.variable} (${f.importancia})</li>`).join("")}</ol>
     <h3>Conclusión automática</h3>
     <p>${conclusionAutomatica(d)}</p>`
  );
  descargarHTML(html, `informe_${d.info.ticker}.html`);
}

function generarInformeSimulacion() {
  if (!datosSimulacion) return;
  const d = datosSimulacion;
  const html = plantillaInforme(
    "Informe M&A. Simulación",
    `<h2>Resultado de la simulación</h2>
     <p>Probabilidad: <strong>${(d.probabilidad * 100).toFixed(1)}%</strong>
        . ${d.clasificacion}</p>
     <p>Percentil en el universo: <strong>${d.percentil}%</strong></p>
     <h3>Variables derivadas</h3>
     <ul>${Object.entries(d.variables_derivadas).map(([k,v]) => `<li><b>${k}:</b> ${v}</li>`).join("")}</ul>
     <h3>Interpretación</h3>
     <p>${d.probabilidad >= 0.6
       ? "La empresa presenta características financieras consistentes con un perfil de target de adquisición."
       : d.probabilidad >= 0.35
       ? "La empresa muestra algunos factores de atracción para adquisiciones, pero sin perfil claro de target."
       : "La empresa no presenta el perfil típico de target de adquisición en el universo analizado."
     }</p>`
  );
  descargarHTML(html, "informe_simulacion.html");
}

function conclusionAutomatica(data) {
  const prob   = data.prediccion.probabilidad;
  const ticker = data.info.ticker;
  if (prob >= 0.6)
    return `${ticker} presenta una probabilidad alta de convertirse en objetivo de adquisición (${(prob*100).toFixed(1)}%). Su perfil de tamaño, liquidez y valoración la sitúan como target atractivo.`;
  if (prob >= 0.35)
    return `${ticker} muestra algunos factores de interés para potenciales adquirentes, pero sin un perfil definitivo de target (${(prob*100).toFixed(1)}%).`;
  return `${ticker} no presenta el perfil típico de objetivo de adquisición según el modelo (${(prob*100).toFixed(1)}%). Su estructura financiera o valoración no es consistente con el universo de targets históricos.`;
}

function plantillaInforme(titulo, contenido) {
  return `<!DOCTYPE html><html lang="es"><head>
    <meta charset="UTF-8"><title>${titulo}</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; color: #2c3e50; }
      h1 { color: #0d1b2a; border-bottom: 2px solid #2a6496; padding-bottom: 8px; }
      h2,h3 { color: #1b3a5c; }
      li { margin: 4px 0; }
    </style></head><body>
    <h1>${titulo}</h1>
    <p style="color:#7f8c8d; font-size:0.85rem;">
      Generado el ${new Date().toLocaleDateString("es-ES")} —
      TFG · M&A Target Screening · ICAI
    </p>
    ${contenido}
    </body></html>`;
}

function descargarHTML(htmlStr, nombre) {
  // Crea un blob y lo descarga como archivo
  const blob = new Blob([htmlStr], { type: "text/html" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = nombre;
  a.click();
}


// ═══════════════════════════════════════════
// UTILIDADES
// ═══════════════════════════════════════════

/** Escribe texto en un elemento por id */
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

/** Lee el valor numérico de un input */
function getNum(id) {
  return parseFloat(document.getElementById(id).value) || 0;
}

/** Rellena un <select> con opciones */
function rellenarSelect(id, opciones) {
  const sel = document.getElementById(id);
  // Mantener la opción vacía (Todos)
  while (sel.options.length > 1) sel.remove(1);
  opciones.forEach(op => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = op;
    sel.appendChild(opt);
  });
}

/** Devuelve un badge HTML según la probabilidad */
function badgeProbabilidad(prob) {
  if (prob === undefined || prob === null || isNaN(prob)) return "—";
  const pct  = (prob * 100).toFixed(1);
  const cls  = prob >= 0.6 ? "green" : prob >= 0.35 ? "yellow" : "red";
  return `<span class="badge ${cls}">${pct}%</span>`;
}

/** Color hex según probabilidad */
function colorPorProb(prob) {
  if (prob >= 0.6)  return "#27ae60";
  if (prob >= 0.35) return "#f39c12";
  return "#e74c3c";
}

/** Genera array de colores para histograma según valor del bin */
function generarGradienteColores(bins) {
  return bins.map(b => b >= 0.6 ? "#27ae60" : b >= 0.35 ? "#f39c12" : "#5a8fc5");
}

/** Calcula histograma manual (array de conteos y bins) */
function histograma(data, nBins) {
  const min = 0, max = 1, step = (max - min) / nBins;
  const bins   = Array.from({ length: nBins }, (_, i) => +(min + i * step).toFixed(2));
  const counts = new Array(nBins).fill(0);
  data.forEach(v => {
    const i = Math.min(Math.floor((v - min) / step), nBins - 1);
    counts[i]++;
  });
  return [counts, bins];
}

/** Ordena la tabla del screener por columna (patrón click en <th>) */
let _sortAsc = true;
function ordenarTabla(colIdx) {
  datosScreener.sort((a, b) => {
    const keys = ["empresa","ticker","sector","market_cap","ev_ebitda",
                  "ebitda_margin","leverage","año","probabilidad",
                  "año_atractivo","probabilidad_atractiva"];
    const k = keys[colIdx];
    const va = a[k], vb = b[k];
    if (typeof va === "string") return _sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return _sortAsc ? va - vb : vb - va;
  });
  _sortAsc = !_sortAsc;
  // datosScreener es el array de empresas de la página actual;
  // renderizarScreener espera {empresas, total} como lo manda el API
  renderizarScreener({ empresas: datosScreener, total: totalResultadosActual });
}

/** Configuración base de layout para gráficos Plotly */
function layoutBase(xtitle, ytitle) {
  return {
    xaxis: { title: xtitle, color: "#5a6a7a" },
    yaxis: { title: ytitle, color: "#5a6a7a" },
    paper_bgcolor: "white",
    plot_bgcolor:  "#f8f9fb",
    font:   { family: "Segoe UI, Arial", size: 11 },
    height: 260,
    margin: { t: 20, b: 50, l: 50, r: 20 }
  };
}

/** Config Plotly: sin toolbar salvo guardar imagen */
function configBase() {
  return { displayModeBar: true, displaylogo: false,
           modeBarButtonsToRemove: ["zoom2d","pan2d","select2d","lasso2d","resetScale2d"] };
}


// ─── Carga inicial al arrancar la página ───
// La sección Summary se carga automáticamente al entrar
window.addEventListener("DOMContentLoaded", () => {
  cargarSummary();
});
