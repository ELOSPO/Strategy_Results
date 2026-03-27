"""
deploy/app.py — Quant Terminal (versión pública)
Lee datos pre-computados desde data/ (generados por generate_data.py).
NO contiene credenciales, modelos ni lógica propietaria.

Ejecutar:
    streamlit run deploy/app.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models

# ── Constantes (sin importar data_engine) ─────────────────────────────────────
MOMENTUM_WEIGHT   = 0.70
PREDICTION_WEIGHT = 0.30
RISK_FREE_RATE    = 0.05
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Carga y deserialización de CSVs ───────────────────────────────────────────

_JSON_COLS = [
    "forecast_dates", "forecast_prices",
    "garch_forecast", "garch_vol",
    "model_weights",  "entry_point",
]


def _deserialize(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte JSON strings de vuelta a objetos Python."""
    out = df.copy()
    for col in _JSON_COLS:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_market(prefix: str):
    base = os.path.join(DATA_DIR, prefix)

    with open(f"{base}_meta.json") as f:
        meta = json.load(f)

    prices      = pd.read_csv(f"{base}_prices.csv",      index_col=0, parse_dates=True)
    predictions = _deserialize(pd.read_csv(f"{base}_predictions.csv", index_col=0))
    momentum    = pd.read_csv(f"{base}_momentum.csv",    index_col=0)
    ranking     = pd.read_csv(f"{base}_ranking.csv",     index_col=0)

    # Restaurar tipos numéricos en predicciones
    num_cols = ["current_price", "predicted_price", "predicted_return_pct",
                "mape", "ml_mape", "arima_mape", "garch_mape"]
    for c in num_cols:
        if c in predictions.columns:
            predictions[c] = pd.to_numeric(predictions[c], errors="coerce")

    pred_tickers = meta.get("pred_tickers", meta.get("tickers", []))
    all_tickers  = meta.get("all_tickers",  pred_tickers)
    return prices, predictions, momentum, ranking, pred_tickers, all_tickers, meta


# ── Optimizador de portafolio (PyPortfolioOpt — sin credenciales) ─────────────

def optimize_portfolio(selected, predictions_df, momentum_df, prices, horizon, weekly):
    sub_prices = prices[selected].dropna()
    S   = risk_models.sample_cov(sub_prices)
    ann = 52 / horizon if weekly else 252 / horizon
    # Retorno: ML si tiene predicción, momentum 3M como proxy si no
    ret = {}
    for t in selected:
        if t in predictions_df.index:
            ret[t] = predictions_df.loc[t, "predicted_return_pct"] / 100 * ann
        elif t in momentum_df.index:
            ret[t] = momentum_df.loc[t, "momentum_3m"] * (252 / 63)
        else:
            ret[t] = 0.0
    mu = pd.Series(ret).reindex(sub_prices.columns)

    def _frontier(mu_v, S_v, n):
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(600):
            w = rng.dirichlet(np.ones(n))
            r = float(np.dot(w, mu_v))
            v = float(np.sqrt(w @ S_v @ w))
            rows.append({"return": r, "volatility": v, "sharpe": (r - RISK_FREE_RATE) / (v + 1e-9)})
        return pd.DataFrame(rows)

    n = len(selected)
    for method in ("max_sharpe", "min_volatility"):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            getattr(ef, method)(**({} if method == "min_volatility"
                                    else {"risk_free_rate": RISK_FREE_RATE}))
            cw = ef.clean_weights()
            r, v, sh = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
            return {
                "weights": (pd.DataFrame.from_dict(cw, orient="index", columns=["weight"])
                            .query("weight > 0.005").sort_values("weight", ascending=False)),
                "expected_return": r, "volatility": v, "sharpe": sh,
                "frontier": _frontier(mu.values, S.values, n),
                "mu": mu, "success": True, "method": method,
            }
        except Exception:
            continue

    w_eq = np.ones(n) / n
    return {
        "weights": pd.DataFrame({"weight": [1/n]*n}, index=selected),
        "expected_return": float(np.dot(w_eq, mu.values)),
        "volatility": float(np.sqrt(w_eq @ S.values @ w_eq)),
        "sharpe": 0.0, "frontier": pd.DataFrame(),
        "mu": mu, "success": False, "method": "equal_weight",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quant Terminal | HedgeFund",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #080808 !important; color: #d4d4d4;
    }
    [data-testid="stHeader"] { background: #080808; }
    [data-testid="stSidebar"] { background: #0e0e0e; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111; border-bottom: 1px solid #2a2a2a;
    }
    .stTabs [data-baseweb="tab"] { color: #777; font-weight: 600; letter-spacing: .5px; }
    .stTabs [aria-selected="true"] {
        color: #ff6b00 !important; border-bottom: 2px solid #ff6b00 !important;
        background-color: #111 !important;
    }
    [data-testid="stMetricValue"] { color: #ff6b00; font-size: 1.5rem; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #666; font-size: .78rem; }
    [data-testid="stMetricDelta"]  { font-size: .85rem; }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #888 !important; font-size: .82rem;
    }
    [data-testid="stDataFrame"] { background: #0d0d0d; }
    hr { border-color: #222; }
    .bb-card {
        background: #111; border: 1px solid #1e1e1e;
        border-radius: 3px; padding: 12px 16px; margin-bottom: 8px;
    }
    .bb-card-buy  { border-left: 3px solid #00c853; }
    .bb-card-sell { border-left: 3px solid #ff1744; }
    .stButton > button[kind="primary"] {
        background-color: #ff6b00; color: white; border: none;
        font-weight: bold; letter-spacing: .5px;
    }
    .stButton > button[kind="primary"]:hover { background-color: #e05a00; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(90deg,#131313,#0a0a0a);
            padding:14px 24px; border-left:4px solid #ff6b00;
            margin-bottom:12px; border-radius:2px;">
  <span style="color:#ff6b00;font-size:1.45rem;font-weight:bold;
               letter-spacing:3px;">▶ QUANT TERMINAL</span>
  <span style="color:#333;font-size:1rem;"> | </span>
  <span style="color:#aaa;font-size:.95rem;letter-spacing:1px;">
    QUANT HEDGE FUND &nbsp;•&nbsp; BETA v1.2
  </span>
</div>
""", unsafe_allow_html=True)

# ── Selector de mercado ────────────────────────────────────────────────────────

_market = st.radio(
    "Mercado", options=["🇺🇸 S&P 500", "🇨🇴 Colombia BVC"],
    horizontal=True, label_visibility="collapsed",
)
MARKET = "sp500" if "S&P" in _market else "colombia"
st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

# ── Carga de datos ─────────────────────────────────────────────────────────────

prefix = "sp500" if MARKET == "sp500" else "col"

_data_ok = os.path.exists(os.path.join(DATA_DIR, f"{prefix}_predictions.csv"))
if not _data_ok:
    st.error(
        f"No se encontraron datos para **{_market}** en `data/`. "
        "Ejecuta `generate_data.py` primero."
    )
    st.stop()

with st.spinner("⏳ Cargando datos…"):
    prices, predictions_df, momentum_df, ranking_df, top_tickers, all_tickers, meta = _load_market(prefix)

horizon  = meta["horizon"]
weekly   = meta.get("weekly", False)
gen_date = meta.get("generated_at", "—")
_horizon_label = f"{horizon} semanas" if weekly else f"{horizon}d hábiles"

# ── KPIs globales ──────────────────────────────────────────────────────────────

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
best_ticker  = ranking_df.index[0]
worst_ticker = ranking_df.index[-1]
avg_pred_ret = predictions_df["predicted_return_pct"].mean()

_universe_label = f"S&P 500 ({len(top_tickers)})" if MARKET == "sp500" else f"BVC Colombia ({len(top_tickers)})"
_pred_label     = f"Generado: {gen_date}"

col_k1.metric("Mejor señal",  best_ticker,
              f"{predictions_df.loc[best_ticker,'predicted_return_pct']:+.2f}%")
col_k2.metric("Peor señal",   worst_ticker,
              f"{predictions_df.loc[worst_ticker,'predicted_return_pct']:+.2f}%")
col_k3.metric("Retorno promedio esperado",
              f"{avg_pred_ret:+.2f}%", _horizon_label)
col_k4.metric("Universo", _universe_label, _pred_label)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📊  Stock Predictions",
    "🏆  Top Buy / Sell",
    "🔧  Portfolio Builder",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STOCK PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    _mkt_label = (f"Top {len(top_tickers)} del S&P 500"
                  if MARKET == "sp500"
                  else f"{len(top_tickers)} acciones BVC 🇨🇴")
    st.markdown(
        f"### Predicciones — próximos **{_horizon_label}**  "
        f"<span style='color:#555;font-size:.85rem;'>{_mkt_label}</span>",
        unsafe_allow_html=True,
    )

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    tbl = predictions_df[
        ["current_price", "predicted_price", "predicted_return_pct", "mape"]
    ].copy()
    tbl.index.name = "Ticker"
    tbl.columns    = [
        "Precio Actual",
        "Precio Predicho",
        "Retorno Esperado (%)",
        "Error Modelo (MAPE %)",
    ]
    tbl["Señal"] = tbl["Retorno Esperado (%)"].apply(
        lambda x: "🟢 COMPRAR" if x > 0 else "🔴 VENDER"
    )
    tbl = tbl.sort_values("Retorno Esperado (%)", ascending=False)

    st.dataframe(
        tbl.style
           .background_gradient(subset=["Retorno Esperado (%)"], cmap="RdYlGn")
           .format(
               {
                   "Precio Actual":       "${:.2f}",
                   "Precio Predicho":     "${:.2f}",
                   "Retorno Esperado (%)": "{:+.2f}%",
                   "Error Modelo (MAPE %)": "{:.1f}%",
               }
           ),
        use_container_width=True,
        height=300,
    )

    st.markdown("---")

    # ── Vista de detalle por acción ────────────────────────────────────────────
    st.markdown("#### Vista de detalle por acción")

    col_sel, col_chart = st.columns([1, 3])

    with col_sel:
        stock_detail = st.selectbox("Selecciona una acción", sorted(all_tickers), key="detail")
        has_pred = stock_detail in predictions_df.index
        last_px = float(prices[stock_detail].dropna().iloc[-1]) if stock_detail in prices.columns else None

        if has_pred:
            row       = predictions_df.loc[stock_detail]
            ret       = float(row["predicted_return_pct"])
            entry     = row["entry_point"]
            ret_color = "#00c853" if ret >= 0 else "#ff1744"
            signal    = "COMPRAR" if ret >= 0 else "VENDER"
            best_model = row.get("best_model", "ml")
            w         = row.get("model_weights") or {}
            ml_mape   = row.get("ml_mape")
            garch_mape = row.get("garch_mape")
            mape_str  = f"{row['mape']:.1f}%" if row["mape"] else "N/A"

            def _fmt_model(name, label, mape_val):
                star  = " ★" if name == best_model else ""
                wt    = w.get(name, 0)
                mape_s = f"MAPE {mape_val:.1f}%" if mape_val else ""
                color = "#ffab00" if name == best_model else "#555"
                return (f"<span style='color:{color};font-weight:bold;'>{label}{star} {int(wt*100)}%</span>"
                        f"<span style='color:#444;font-size:.75rem;'> {mape_s}</span>")

            model_line = (f"{_fmt_model('ml','ML',ml_mape)} &nbsp;|&nbsp; "
                          f"{_fmt_model('garch','GARCH',garch_mape)}")

            st.markdown(f"""
            <div class="bb-card">
              <div style="color:#666;font-size:.75rem;">SEÑAL &nbsp;·&nbsp;
                <span style="color:#444;">Modelo ganador: {best_model.upper()} ★</span></div>
              <div style="color:{ret_color};font-size:1.4rem;font-weight:bold;">{signal}</div>
              <div style="color:#aaa;font-size:.82rem;margin-top:6px;">
                Actual: <b>${row['current_price']:.2f}</b>&nbsp;
                Pred.: <b style="color:{ret_color};">${row['predicted_price']:.2f} ({ret:+.2f}%)</b><br>
                MAPE ensemble: {mape_str}
              </div>
              <div style="font-size:.75rem;margin-top:6px;">{model_line}</div>
            </div>""", unsafe_allow_html=True)

            if entry["type"] == "v_shape":
                st.markdown(f"""
                <div class="bb-card" style="border-left:3px solid #ffab00;margin-top:8px;">
                  <div style="color:#ffab00;font-size:.75rem;">⚡ V-SHAPE DETECTADO</div>
                  <div style="color:#aaa;font-size:.82rem;margin-top:4px;">
                    Entrada: <b style="color:#ffab00;">{entry['entry_date']}</b> @ ${entry['entry_price']:.2f}<br>
                    Caída: <b style="color:#ff1744;">{entry['drawdown_pct']:.1f}%</b> &nbsp;
                    Upside: <b style="color:#00c853;">+{entry['upside_pct']:.1f}%</b>
                  </div>
                </div>""", unsafe_allow_html=True)
            elif entry["type"] == "uptrend":
                st.markdown(f"""
                <div class="bb-card" style="border-left:3px solid #00c853;margin-top:8px;">
                  <div style="color:#00c853;font-size:.75rem;">▲ ALCISTA — ENTRAR YA</div>
                  <div style="color:#aaa;font-size:.82rem;margin-top:4px;">
                    Upside: <b style="color:#00c853;">+{entry['upside_pct']:.1f}%</b> en {_horizon_label}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                lbl = "▼ BAJISTA" if entry["type"] == "downtrend" else "→ SIN SEÑAL"
                st.markdown(f"""
                <div class="bb-card" style="border-left:3px solid #555;margin-top:8px;">
                  <div style="color:#888;font-size:.75rem;">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bb-card" style="border-left:3px solid #333;margin-top:12px;">
              <div style="color:#555;font-size:.75rem;">SIN PREDICCIÓN</div>
              <div style="color:#888;font-size:.9rem;margin-top:4px;">
                Fuera del top {len(top_tickers)} por momentum.<br>Solo se muestra historial.
              </div>
              <div style="color:#aaa;font-size:.82rem;margin-top:6px;">
                Precio actual: <b style="color:#d4d4d4;">${last_px:.2f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        mom_row = momentum_df.loc[stock_detail] if stock_detail in momentum_df.index else None
        if mom_row is not None:
            st.markdown(
            f"""
            <div class="bb-card" style="margin-top:8px;">
              <div style="color:#666;font-size:.75rem;">MOMENTUM</div>
              <div style="color:#aaa;font-size:.82rem;margin-top:4px;">
                15d: <b>{mom_row['momentum_15d']*100:+.2f}%</b><br>
                3M:  <b>{mom_row['momentum_3m']*100:+.2f}%</b><br>
                6M:  <b>{mom_row['momentum_6m']*100:+.2f}%</b><br>
                1Y:  <b>{mom_row['momentum_1y']*100:+.2f}%</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        hist_close = prices[stock_detail].dropna().tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_close.index, y=hist_close.values,
            name="Histórico", line=dict(color="#ff6b00", width=2),
        ))

        if has_pred:
            fc_dates   = [pd.Timestamp(d) for d in row["forecast_dates"]]
            best_model = row.get("best_model", "ml")

            # ML always drives the price trajectory; GARCH provides the vol band only
            winner_fc    = row.get("forecast_prices", [])   # always ML prices
            winner_label = f"ML Forecast (GARCH★ vol)" if best_model == "garch" else "ML ★"

            last_price = float(hist_close.iloc[-1])
            wx = [hist_close.index[-1]] + fc_dates[: len(winner_fc)]
            wy = [last_price] + [float(p) for p in winner_fc]
            line_color = "#ffab00" if wy[-1] > wy[0] else "#e05a00"

            # CI band: prefer GARCH ±1σ, fall back to ML ±MAPE%
            gv = row.get("garch_vol", [])
            if winner_fc and gv:
                gx = fc_dates[: len(winner_fc)]
                fig.add_trace(go.Scatter(
                    x=gx + gx[::-1],
                    y=[float(p)*(1+float(v)) for p,v in zip(winner_fc,gv)] +
                      [float(p)*(1-float(v)) for p,v in zip(winner_fc,gv)][::-1],
                    fill="toself", fillcolor="rgba(255,107,0,.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="GARCH ±1σ",
                ))
            elif winner_fc:
                ml_m = (row.get("ml_mape") or 5.0) / 100
                mx = fc_dates[: len(winner_fc)]
                fig.add_trace(go.Scatter(
                    x=mx + mx[::-1],
                    y=[float(p)*(1+ml_m) for p in winner_fc] +
                      [float(p)*(1-ml_m) for p in winner_fc][::-1],
                    fill="toself", fillcolor="rgba(255,107,0,.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"ML ±{ml_m*100:.1f}% CI",
                ))

            # Línea forecast
            fig.add_trace(go.Scatter(
                x=wx, y=wy, name=winner_label,
                line=dict(color=line_color, width=2.5),
                mode="lines+markers", marker=dict(size=5),
            ))

            if entry["type"] in ("v_shape", "uptrend") and entry["entry_date"]:
                ets   = pd.Timestamp(entry["entry_date"])
                emark = "⚡ Entrada" if entry["type"] == "v_shape" else "▲ Entrada"
                fig.add_trace(go.Scatter(
                    x=[ets], y=[entry["entry_price"]],
                    mode="markers+text", name=emark, text=[emark],
                    textposition="top center",
                    textfont=dict(color="#ffab00", size=11),
                    marker=dict(color="#ffab00", size=12, symbol="star",
                                line=dict(color="white", width=1)),
                ))
                fig.add_vline(x=ets.timestamp()*1000,
                              line=dict(color="#ffab00", width=1, dash="dot"), opacity=.4)

        chart_title = (f"{stock_detail}  —  Historial + Forecast {winner_label} ({_horizon_label})"
                       if has_pred else f"{stock_detail}  —  Historial de precios")
        tick_prefix = "$" if MARKET == "sp500" else ""
        fig.update_layout(
            title=chart_title,
            template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
            height=400, legend=dict(bgcolor="#1a1a1a", font_size=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e1e1e", tickprefix=tick_prefix),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TOP BUY / SELL
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        f"### Sugerencias de portafolio  —  "
        f"Score = **{int(MOMENTUM_WEIGHT*100)}%** Momentum  +  "
        f"**{int(PREDICTION_WEIGHT*100)}%** ML Prediction"
    )

    n_show   = 10 if MARKET == "colombia" else 5
    top_buy  = ranking_df.head(n_show)
    top_sell = ranking_df.tail(n_show).sort_values("score")

    col_buy, col_sell = st.columns(2)

    with col_buy:
        st.markdown(
            "#### 🟢 TOP COMPRAR &nbsp;<span style='color:#555;font-size:.75rem;'>"
            "(Long candidates)</span>",
            unsafe_allow_html=True,
        )
        for ticker, row in top_buy.iterrows():
            rc = "#00c853" if row["predicted_return_pct"] >= 0 else "#ff1744"
            st.markdown(
                f"""
<div class="bb-card bb-card-buy">
  <span style="color:#eee;font-weight:bold;font-size:.95rem;">{ticker}</span>
  &nbsp;&nbsp;
  <span style="color:#555;font-size:.75rem;">Score: {row['score']:.3f}</span><br>
  <span style="color:#bbb;">
    ${row['current_price']:.2f} →
    <b style="color:{rc};">${row['predicted_price']:.2f}
    ({row['predicted_return_pct']:+.2f}%)</b>
  </span><br>
  <span style="color:#555;font-size:.77rem;">
    15d: {row['momentum_15d']*100:+.1f}% &nbsp;|&nbsp;
    3M: {row['momentum_3m']*100:+.1f}% &nbsp;|&nbsp;
    1Y: {row['momentum_1y']*100:+.1f}%
  </span>
</div>""",
                unsafe_allow_html=True,
            )

    with col_sell:
        sell_label = ("(Short candidates — S&P 500)"
                      if MARKET == "sp500"
                      else "(Menor momentum — BVC)")
        st.markdown(
            f"#### 🔴 TOP VENDER &nbsp;<span style='color:#555;font-size:.75rem;'>"
            f"{sell_label}</span>",
            unsafe_allow_html=True,
        )
        for ticker, row in top_sell.iterrows():
            rc = "#00c853" if row["predicted_return_pct"] >= 0 else "#ff1744"
            st.markdown(
                f"""
<div class="bb-card bb-card-sell">
  <span style="color:#eee;font-weight:bold;font-size:.95rem;">{ticker}</span>
  &nbsp;&nbsp;
  <span style="color:#555;font-size:.75rem;">Score: {row['score']:.3f}</span><br>
  <span style="color:#bbb;">
    ${row['current_price']:.2f} →
    <b style="color:{rc};">${row['predicted_price']:.2f}
    ({row['predicted_return_pct']:+.2f}%)</b>
  </span><br>
  <span style="color:#555;font-size:.77rem;">
    15d: {row['momentum_15d']*100:+.1f}% &nbsp;|&nbsp;
    3M: {row['momentum_3m']*100:+.1f}% &nbsp;|&nbsp;
    1Y: {row['momentum_1y']*100:+.1f}%
  </span>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    rank_plot = ranking_df.reset_index().rename(columns={"index": "ticker"}).sort_values("score", ascending=True)
    rank_plot["color_flag"] = rank_plot["predicted_return_pct"].apply(
        lambda x: "Alcista" if x > 0 else "Bajista"
    )
    fig_rank = px.bar(
        rank_plot, x="score", y="ticker", orientation="h",
        color="predicted_return_pct", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=f"Ranking combinado  ({int(MOMENTUM_WEIGHT*100)}% Momentum + {int(PREDICTION_WEIGHT*100)}% Prediction)",
        labels={"score": "Score", "ticker": "", "predicted_return_pct": "Retorno predicho (%)"},
        text=rank_plot["predicted_return_pct"].apply(lambda x: f"{x:+.1f}%"),
    )
    fig_rank.update_traces(textposition="outside", textfont_size=11)
    fig_rank.update_layout(
        template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
        height=400, xaxis=dict(showgrid=True, gridcolor="#1e1e1e"),
        yaxis=dict(showgrid=False), coloraxis_showscale=False,
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("#### Momentum por ventana temporal")
    mom_plot = momentum_df.copy() * 100
    mom_melt = mom_plot.reset_index().rename(columns={"index": "ticker"}).melt(
        id_vars="ticker",
        value_vars=["momentum_15d","momentum_3m","momentum_6m","momentum_1y"],
        var_name="Ventana", value_name="Retorno (%)",
    )
    mom_melt["Ventana"] = mom_melt["Ventana"].map(
        {"momentum_15d":"15d","momentum_3m":"3M","momentum_6m":"6M","momentum_1y":"1Y"}
    )
    mom_melt.rename(columns={"ticker":"Ticker"}, inplace=True)
    fig_mom = px.bar(
        mom_melt, x="Ticker", y="Retorno (%)", color="Retorno (%)",
        facet_row="Ventana", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0, title="Momentum rolling por ventana", height=600,
    )
    fig_mom.update_layout(
        template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
        coloraxis_showscale=False,
    )
    fig_mom.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_mom, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Portfolio Builder — Markowitz sobre retornos esperados")
    st.caption(
        "μ (retorno esperado) = predicción anualizada  |  "
        "Σ (riesgo) = covarianza histórica  |  "
        f"Objetivo: Max Sharpe  (rf = {RISK_FREE_RATE*100:.0f}%)"
    )

    inp_col, res_col = st.columns([1, 2])

    with inp_col:
        selected = st.multiselect(
            "Selecciona acciones para tu portafolio",
            options=sorted(all_tickers), default=top_tickers[:5],
        )
        capital = st.slider(
            "Capital inicial (USD)",
            min_value=10_000, max_value=1_000_000,
            value=100_000, step=10_000, format="$%d",
        )

        if len(selected) < 2:
            st.warning("Selecciona al menos 2 acciones.")
        else:
            st.markdown("---")

            @st.cache_data(ttl=3600, show_spinner=False)
            def _run_opt(tickers_tuple, _horizon, _weekly):
                return optimize_portfolio(
                    list(tickers_tuple), predictions_df, momentum_df, prices, _horizon, _weekly
                )

            with st.spinner("Optimizando portafolio…"):
                result = _run_opt(tuple(sorted(selected)), horizon, weekly)

            exp_r   = result["expected_return"] * 100
            vol     = result["volatility"]      * 100
            sh      = result["sharpe"]
            exp_pnl = capital * result["expected_return"]
            r_color = "#00c853" if exp_r > 0 else "#ff1744"

            st.markdown(
                f"""
<div class="bb-card" style="margin-bottom:6px;">
  <div style="color:#555;font-size:.75rem;">RETORNO ESPERADO (anual)</div>
  <div style="color:{r_color};font-size:1.6rem;font-weight:bold;">{exp_r:+.1f}%</div>
</div>
<div class="bb-card" style="margin-bottom:6px;">
  <div style="color:#555;font-size:.75rem;">VOLATILIDAD ESPERADA (anual)</div>
  <div style="color:#ff6b00;font-size:1.6rem;font-weight:bold;">{vol:.1f}%</div>
</div>
<div class="bb-card" style="margin-bottom:6px;">
  <div style="color:#555;font-size:.75rem;">SHARPE RATIO</div>
  <div style="color:#ff6b00;font-size:1.6rem;font-weight:bold;">{sh:.2f}</div>
</div>
<div class="bb-card">
  <div style="color:#555;font-size:.75rem;">P&L ESPERADO (USD)</div>
  <div style="color:{r_color};font-size:1.6rem;font-weight:bold;">${exp_pnl:+,.0f}</div>
</div>""",
                unsafe_allow_html=True,
            )

            method = result.get("method", "")
            if method == "min_volatility":
                st.info("ℹ️ Retornos negativos — se usó **Min Volatility** en lugar de Max Sharpe.")
            elif method == "equal_weight":
                st.warning(f"⚠️ Optimización falló — pesos iguales.")

    if len(selected) >= 2:
        with res_col:
            w_df   = result["weights"]
            mu_ser = result["mu"]

            chart_col, table_col = st.columns([1, 1])

            with chart_col:
                fig_pie = go.Figure(go.Pie(
                    labels=w_df.index, values=w_df["weight"], hole=0.42,
                    marker=dict(colors=px.colors.qualitative.Bold,
                                line=dict(color="#080808", width=2)),
                    textinfo="label+percent", textfont=dict(color="white", size=11),
                ))
                fig_pie.update_layout(
                    title="Pesos óptimos", template="plotly_dark",
                    paper_bgcolor="#111", height=310, showlegend=False,
                    annotations=[dict(text="Weights", x=.5, y=.5,
                                      font_size=12, showarrow=False, font_color="#555")],
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with table_col:
                alloc = w_df.copy()
                alloc["Capital ($)"]      = (alloc["weight"] * capital).round(0)
                alloc["Pred. Ret. (%)"]   = predictions_df.loc[alloc.index, "predicted_return_pct"]
                alloc["P&L esperado ($)"] = (alloc["Capital ($)"] * alloc["Pred. Ret. (%)"] / 100).round(0)
                alloc["Peso"] = (alloc["weight"] * 100).round(1).astype(str) + "%"
                alloc = alloc[["Peso", "Capital ($)", "Pred. Ret. (%)", "P&L esperado ($)"]]
                st.markdown("##### Asignación de capital")
                st.dataframe(
                    alloc.style
                         .background_gradient(subset=["Pred. Ret. (%)"], cmap="RdYlGn")
                         .format({"Capital ($)": "${:,.0f}",
                                  "P&L esperado ($)": "${:+,.0f}",
                                  "Pred. Ret. (%)": "{:+.2f}%"}),
                    use_container_width=True, height=300,
                )

            st.markdown("##### Frontera eficiente (Monte Carlo)")
            frontier = result["frontier"]
            fig_ef   = go.Figure()

            if not frontier.empty:
                fig_ef.add_trace(go.Scatter(
                    x=frontier["volatility"]*100, y=frontier["return"]*100,
                    mode="markers", name="Portafolios simulados",
                    marker=dict(color=frontier["sharpe"], colorscale="RdYlGn",
                                size=4, opacity=.5,
                                colorbar=dict(title="Sharpe", thickness=10)),
                ))

            fig_ef.add_trace(go.Scatter(
                x=[result["volatility"]*100], y=[result["expected_return"]*100],
                mode="markers+text", name="Óptimo",
                text=["★ Óptimo"], textposition="top center",
                textfont=dict(color="white", size=11),
                marker=dict(color="#ff6b00", size=16, symbol="star",
                            line=dict(color="white", width=1)),
            ))

            for tk in selected:
                tk_vol = float(prices[tk].dropna().pct_change().std() * np.sqrt(252) * 100)
                tk_ret = float(mu_ser.loc[tk] * 100) if tk in mu_ser.index else 0
                fig_ef.add_trace(go.Scatter(
                    x=[tk_vol], y=[tk_ret], mode="markers+text", text=[tk],
                    textposition="top center",
                    textfont=dict(size=9, color="#aaa"),
                    marker=dict(color="#4fc3f7", size=8), showlegend=False,
                ))

            fig_ef.update_layout(
                xaxis_title="Volatilidad anual (%)", yaxis_title="Retorno anual esperado (%)",
                template="plotly_dark", paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
                height=400, legend=dict(bgcolor="#1a1a1a", font_size=10),
                xaxis=dict(showgrid=True, gridcolor="#1e1e1e"),
                yaxis=dict(showgrid=True, gridcolor="#1e1e1e"),
                hovermode="closest",
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            st.markdown("##### Retorno esperado vs Momentum")
            comp_df = pd.DataFrame({
                "Pred. Return (%)": predictions_df.loc[selected, "predicted_return_pct"],
                "Momentum 15d (%)": momentum_df.loc[selected, "momentum_15d"] * 100,
                "Momentum 3M (%)":  momentum_df.loc[selected, "momentum_3m"]  * 100,
                "Score":            ranking_df.loc[selected, "score"],
            }).round(2).sort_values("Score", ascending=False)
            st.dataframe(
                comp_df.style
                       .background_gradient(subset=["Pred. Return (%)"], cmap="RdYlGn")
                       .background_gradient(subset=["Score"], cmap="Blues")
                       .format("{:+.2f}%", subset=["Pred. Return (%)","Momentum 15d (%)","Momentum 3M (%)"])
                       .format("{:.3f}", subset=["Score"]),
                use_container_width=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<span style='color:#333;font-size:.75rem;'>"
    "⚡ Quant Hedge Fund Terminal v1.2 beta &nbsp;|&nbsp; "
    "S&P 500 + Colombia BVC &nbsp;|&nbsp; "
    f"Datos: {gen_date}"
    "</span>",
    unsafe_allow_html=True,
)
