"""
Profit Radar Pro — Clean AI Server v5.0
=========================================
Server Flask interamente riprogettato e ripulito per la nuova architettura.
Fornisce SOLO i parametri del nuovo EA "Executor" e i dati del "Data Collector".
"""

import os
import json
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ============================================================
#  CONFIGURAZIONE PATH
# ============================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
EA_CONFIG_PATH = os.path.join(DATA_DIR, "ea_config.json")
EA_STATUS_PATH = os.path.join(DATA_DIR, "ea_status.json")

# Inizializzazione Flask
app = Flask(__name__)
CORS(app)

# ============================================================
#  DEFAULT CONFIG — SOLO VALORI DEL NUOVO EA EXECUTOR + COLLECTOR
# ============================================================
DEFAULT_EA_CONFIG = {
    # --- File e Sincronizzazione ---
    "csv_file" : "PRP_TrustedLatest.csv",      # Nome file latest
    "ready_file" : "PRP_TrustedReady.csv",      # Nome file ready
    "history_file" : "PRP_TrustedHistory.csv",  # Nome file storico
    "timer_seconds" : 10,                       # Timer EA (secondi)
    "verbose_journal" : True,                   # Stampa log in MT4
    "process_current_init" : False,             # Rielabora barra READY all'avvio
    
    # --- Generali e Rischio ---
    "magic_number" : 270202,                    # Magic Number Executor
    "fixed_lots" : 0.07,                        # Lotto base (Rischio 1% su 1.000€)
    "max_concurrent" : 3,                       # Massimo trade contemporanei
    "max_per_pair" : 1,                         # Massimo trade per cross
    "max_entries_bar" : 2,                      # Massimo ingressi per singola barra
    "max_spread_points" : 30,                   # Spread massimo in punti (3 pip)
    "slippage" : 3,                             # Slippage massimo
    "allow_trend" : True,                       # Abilita modulo standard (trend)
    "allow_reversal" : True,                    # Abilita modulo reversal
    
    # --- Modulo Trend ---
    "trend_min_rv" : 5.0,                       # Radar Value minimo di ingresso
    "trend_max_adr" : 70.0,                     # ADR% massimo accettato
    "trend_sl_mult" : 1.5,                      # Moltiplicatore ATR per lo SL
    "trend_tp_pct" : 80.0,                      # TP come % dell'ADR residuo (70-80%)
    "trend_min_rr" : 1.5,                       # Rapporto R:R minimo accettato
    
    # --- Modulo Reversal ---
    "rev_min_rv" : 70.0,                        # Radar Value minimo (eccesso)
    "rev_min_adr" : 100.0,                      # ADR% minimo (oggi > media)
    "rev_min_ema_dist" : 20.0,                  # Distanza EMA minima (elastico teso)
    "rev_sl_mult" : 1.5,                        # Moltiplicatore ATR per lo SL
    "rev_min_rr" : 1.5,                         # Rapporto R:R minimo accettato
    
    # --- Gestione Post-Trade (Uscite) ---
    "profit_fade_r" : 0.70,                     # Chiusura a +0.7R se trend cala
    "loss_cut_r" : 0.60,                        # Chiusura a -0.6R se trend degrada
    "close_on_opposite" : True,                 # Chiusura immediata su stato opposto
    "close_on_gray" : True,                     # Chiusura su GRAY in profitto
    "close_on_weak" : True,                     # Chiusura su stato debole in profitto

    # --- Parametri del Collettore ---
    "ai_url" : "https://profit-radar-ai.onrender.com/predict",
    "send_feedback" : True,
    "use_ai" : True,
    "ai_min_conf" : 70,
    "executor_magic" : 270202
}

ea_status = {
    "last_update": None, "balance": 0, "equity": 0, "open_trades": 0, "daily_pnl": 0,
    "daily_wins": 0, "daily_losses": 0, "daily_win_amount": 0, "daily_loss_amount": 0,
    "ai_calls": 0, "ai_confirm": 0, "ai_reject": 0, "ai_errors": 0, "ai_missed_trades": 0,
    "warmup_ok": False, "warmup_last": None, "data_source": "LIVE", "cross_active": 0, "cross_total": 0,
    "ea_version": "2.00 (Dual)", "peaks": {}
}

# ============================================================
#  FUNZIONI UTILI
# ============================================================
def load_ea_config():
    if os.path.exists(EA_CONFIG_PATH):
        try:
            with open(EA_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_EA_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except: pass
    return dict(DEFAULT_EA_CONFIG)

def save_ea_config(cfg):
    with open(EA_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def get_trade_stats():
    trade_stats = {}
    path = FEEDBACK_PATH if os.path.exists(FEEDBACK_PATH) else os.path.join(DATA_DIR, "PRP_TradeLog.csv")
    if os.path.exists(path):
        try:
            sep = ";" if path.endswith("PRP_TradeLog.csv") else ","
            df = pd.read_csv(path, sep=sep, on_bad_lines="skip")
            df.columns = [c.lower() for c in df.columns]
            if "symbol" in df.columns and "rv" in df.columns:
                df["symbol_clean"] = df["symbol"].str.upper().str.strip()
                df["rv"] = pd.to_numeric(df["rv"], errors="coerce").fillna(0)
                df["rv_abs"] = df["rv"].abs()
                
                for sym, group in df.groupby("symbol_clean"):
                    count = len(group)
                    avg_rv = float(group["rv_abs"].mean())
                    max_rv = float(group["rv_abs"].max())
                    
                    win_rate = 0.0
                    if "won" in group.columns:
                        won_col = group["won"].astype(str).str.lower().str.strip()
                        win_rate = float((won_col == "true").mean() * 100)
                    
                    trade_stats[sym] = {
                        "count": count,
                        "avg_rv": round(avg_rv, 1),
                        "max_rv": round(max_rv, 1),
                        "win_rate": round(win_rate, 1)
                    }
        except Exception as e: pass
    return trade_stats

# ============================================================
#  ENDPOINTS API (EA)
# ============================================================

@app.route("/ea_status", methods=["POST"])
def receive_ea_status():
    global ea_status
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        for key in ea_status:
            if key in data:
                ea_status[key] = data[key]
        ea_status["last_update"] = datetime.now(timezone.utc).isoformat()

        with open(EA_STATUS_PATH, "w") as f:
            json.dump(ea_status, f, indent=2)

        return jsonify({"status": "ok", "config": load_ea_config()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/feedback", methods=["POST"])
def receive_feedback():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        new_row = pd.DataFrame([data])
        header_needed = not os.path.exists(FEEDBACK_PATH)
        new_row.to_csv(FEEDBACK_PATH, mode="a", header=header_needed, index=False)

        total_fb = len(pd.read_csv(FEEDBACK_PATH))
        return jsonify({"status": "ok", "total_feedback": total_fb})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0}), 200

        direction = data.get("direction", "").upper()
        rv = float(data.get("rv", 0))
        adx = float(data.get("adx", 0))
        adr_pct = float(data.get("adr_pct", 0))

        # Regola basata su regole semplice e sicura
        conf = 50
        if direction == "BUY" and rv > 0: conf += 15
        elif direction == "SELL" and rv < 0: conf += 15
        if adx > 25: conf += 10
        if adr_pct < 50: conf += 5

        signal = "HOLD"
        cfg = load_ea_config()
        min_conf = cfg.get("ai_min_conf", 70)
        
        if conf >= min_conf:
            signal = direction

        return jsonify({"signal": signal, "confidence": conf, "method": "rules_v1"})
    except Exception as e:
        return jsonify({"signal": "HOLD", "confidence": 0, "error": str(e)}), 200


@app.route("/ea_config", methods=["GET"])
def get_ea_config():
    return jsonify(load_ea_config())


@app.route("/ea_config", methods=["POST"])
def update_ea_config():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        cfg = load_ea_config()
        updatable = list(DEFAULT_EA_CONFIG.keys())
        bool_keys = {"verbose_journal", "process_current_init", "allow_trend", "allow_reversal", 
                     "close_on_opposite", "close_on_gray", "close_on_weak", "send_feedback", "use_ai"}

        for key in updatable:
            if key in data:
                val = data[key]
                if key in bool_keys:
                    cfg[key] = (str(val).lower() == 'true') or (val is True)
                elif isinstance(DEFAULT_EA_CONFIG[key], float):
                    cfg[key] = float(val)
                elif isinstance(DEFAULT_EA_CONFIG[key], int):
                    cfg[key] = int(val)
                else:
                    cfg[key] = str(val).strip()

        save_ea_config(cfg)
        return jsonify({"status": "ok", "message": "Configurazione salvata con successo!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    cfg = load_ea_config()
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Radar Executor Dashboard</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a1a;color:#e0e0e0;min-height:100vh}}
.container{{max-width:960px;margin:0 auto;padding:12px}}
h1{{text-align:center;font-size:1.3em;padding:12px 0;color:#4fc3f7;border-bottom:1px solid #1a1a3a;margin-bottom:10px}}
h1 span{{color:#81c784}}
.section{{background:#12122a;border-radius:10px;padding:14px;margin-bottom:12px;border:1px solid #1e1e40}}
.section h2{{font-size:0.95em;color:#4fc3f7;margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.section h2::before{{content:'';width:4px;height:16px;background:#4fc3f7;border-radius:2px}}
.row{{display:flex;flex-wrap:wrap;gap:8px}}
.card{{flex:1;min-width:140px;background:#1a1a35;border-radius:8px;padding:10px;text-align:center}}
.card .val{{font-size:1.6em;font-weight:700;line-height:1.3}}
.card .lbl{{font-size:0.7em;color:#888;text-transform:uppercase;margin-top:2px}}
.green{{color:#81c784}}.red{{color:#ef5350}}.yellow{{color:#ffd54f}}.blue{{color:#4fc3f7}}.white{{color:#e0e0e0}}
.status-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
.dot-green{{background:#81c784;box-shadow:0 0 6px #81c784}}
.dot-red{{background:#ef5350;box-shadow:0 0 6px #ef5350}}
.dot-yellow{{background:#ffd54f;box-shadow:0 0 6px #ffd54f}}
.dot-gray{{background:#666}}
table{{width:100%;border-collapse:collapse;font-size:0.78em}}
th{{text-align:left;padding:6px 8px;background:#1a1a35;color:#888;text-transform:uppercase;font-size:0.85em;border-bottom:1px solid #2a2a50}}
td{{padding:5px 8px;border-bottom:1px solid #15152a}}
tr:hover{{background:#1a1a35}}
.btn{{display:inline-block;padding:8px 16px;border:none;border-radius:6px;cursor:pointer;font-size:0.85em;font-weight:600;transition:all .2s}}
.btn:hover{{transform:translateY(-1px);opacity:0.9}}
.btn-blue{{background:#1565c0;color:#fff}}
.btn-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
.config-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:8px}}
.cfg-item{{background:#1a1a35;border-radius:6px;padding:10px}}
.cfg-item label{{display:block;font-size:0.75em;color:#888;margin-bottom:4px;text-transform:uppercase}}
.cfg-item input,.cfg-item select{{width:100%;padding:6px 8px;background:#0a0a1a;border:1px solid #2a2a50;border-radius:4px;color:#e0e0e0;font-size:0.9em}}
.refresh-bar{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:0.8em;color:#666}}
@media(max-width:600px){{.card{{min-width:100px}}.card .val{{font-size:1.3em}}.config-grid{{grid-template-columns:1fr}}}}

/* === TOOLTIP === */
.tooltip{{position:relative;display:inline-block;margin-left:5px;cursor:help;color:#4fc3f7;font-weight:700}}
.tooltip .tooltiptext{{visibility:hidden;width:260px;background:#1a1a35;color:#e0e0e0;text-align:left;border-radius:8px;padding:10px;border:1px solid #4fc3f7;position:absolute;z-index:100;bottom:125%;left:50%;margin-left:-130px;opacity:0;transition:opacity .2s;font-size:.85em;line-height:1.4;text-transform:none;box-shadow:0 4px 12px rgba(0,0,0,.5)}}
.tooltip .tooltiptext::after{{content:'';position:absolute;top:100%;left:50%;margin-left:-5px;border-width:5px;border-style:solid;border-color:#4fc3f7 transparent transparent transparent}}
.tooltip:hover .tooltiptext{{visibility:visible;opacity:1}}
</style>
</head>
<body>
<div class="container">
<h1>📡 Profit Radar <span>Pro</span> — Executor Dashboard</h1>

<div class="refresh-bar">
  <span id="lastUpdate">Caricamento...</span>
  <span><span class="status-dot dot-gray" id="eaDot"></span><span id="eaStatus">-</span></span>
</div>

<div class="section"><h2>Account Live (dal Collettore)</h2>
<div class="row">
  <div class="card"><div class="val white" id="balance">-</div><div class="lbl">Balance EUR</div></div>
  <div class="card"><div class="val white" id="equity">-</div><div class="lbl">Equity EUR</div></div>
  <div class="card"><div class="val" id="dailyPnl">-</div><div class="lbl">P&L Oggi</div></div>
  <div class="card"><div class="val" id="openTrades">-</div><div class="lbl">Trade Aperti</div></div>
</div></div>

<!-- === SEZIONE RV / PEAKS APERTA DI DEFAULT === -->
<div class="section"><h2>📊 Analisi Picchi e Statistiche Cross</h2>
  <div style="font-size: 0.80em; color: #aaa; margin-bottom: 12px; line-height: 1.4;">
    Questa tabella interattiva mostra l'analisi statistica dei picchi di tendenza (Radar Value) e del Win Rate registrato storicamente per ciascuno dei 28 cross. Unifica automaticamente i simboli con e senza il suffisso +.
  </div>
  <div style="overflow-y:auto; max-height: 250px; border: 1px solid #1e1e40; border-radius: 6px;">
    <table>
      <thead>
        <tr>
          <th>Simbolo</th>
          <th>Trade Totali</th>
          <th>Win Rate</th>
          <th>Avg RV (Entrata)</th>
          <th>Max RV (Entrata)</th>
          <th>Picco Dinamico (EA)</th>
        </tr>
      </thead>
      <tbody id="statsTable">
        <tr><td colspan="6" style="text-align:center;color:#666;padding:12px;">In attesa dei dati dall'EA...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- === SEZIONI DI CONFIGURAZIONE DELL'ESECUTORE === -->
<div class="section"><h2>📁 File e Sincronizzazione</h2>
<div class="config-grid">
  <div class="cfg-item"><label>File snapshot latest</label>
    <input type="text" id="cfgCsvFile" value="{cfg.get('csv_file', 'PRP_TrustedLatest.csv')}"></div>
  <div class="cfg-item"><label>File ready flag</label>
    <input type="text" id="cfgReadyFile" value="{cfg.get('ready_file', 'PRP_TrustedReady.csv')}"></div>
  <div class="cfg-item"><label>File history storico</label>
    <input type="text" id="cfgHistoryFile" value="{cfg.get('history_file', 'PRP_TrustedHistory.csv')}"></div>
  <div class="cfg-item"><label>Timer EA (sec)</label>
    <input type="number" id="cfgTimerSeconds" value="{cfg.get('timer_seconds', 10)}" min="1" max="60" step="1"></div>
  <div class="cfg-item"><label>Stampa log MT4</label>
    <select id="cfgVerboseJournal"><option value="true" {"selected" if cfg.get('verbose_journal') else ""}>Attivo</option><option value="false" {"" if cfg.get('verbose_journal') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Rielabora avvio</label>
    <select id="cfgProcessCurrentInit"><option value="true" {"selected" if cfg.get('process_current_init') else ""}>Attivo</option><option value="false" {"" if cfg.get('process_current_init') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>⚙️ Generali e Rischio</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Magic Number</label>
    <input type="number" id="cfgMagicNumber" value="{cfg.get('magic_number', 270202)}" min="1000" max="999999" step="1"></div>
  <div class="cfg-item"><label>Lotto base</label>
    <input type="number" id="cfgFixedLots" value="{cfg.get('fixed_lots', 0.07)}" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Max trade aperti</label>
    <input type="number" id="cfgMaxConcurrent" value="{cfg.get('max_concurrent', 3)}" min="1" max="28" step="1"></div>
  <div class="cfg-item"><label>Max trade per coppia</label>
    <input type="number" id="cfgMaxPerPair" value="{cfg.get('max_per_pair', 1)}" min="1" max="5" step="1"></div>
  <div class="cfg-item"><label>Max ingressi per barra</label>
    <input type="number" id="cfgMaxEntriesBar" value="{cfg.get('max_entries_bar', 2)}" min="1" max="5" step="1"></div>
  <div class="cfg-item"><label>Spread max (punti)</label>
    <input type="number" id="cfgMaxSpreadPoints" value="{cfg.get('max_spread_points', 30)}" min="5" max="100" step="5"></div>
  <div class="cfg-item"><label>Slippage</label>
    <input type="number" id="cfgSlippage" value="{cfg.get('slippage', 3)}" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>Modulo Trend (Std)</label>
    <select id="cfgAllowTrend"><option value="true" {"selected" if cfg.get('allow_trend') else ""}>Attivo</option><option value="false" {"" if cfg.get('allow_trend') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Modulo Reversal</label>
    <select id="cfgAllowReversal"><option value="true" {"selected" if cfg.get('allow_reversal') else ""}>Attivo</option><option value="false" {"" if cfg.get('allow_reversal') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>🎯 Modulo Trend-Following</h2>
<div class="config-grid">
  <div class="cfg-item"><label>RV minimo</label>
    <input type="number" id="cfgTrendMinRV" value="{cfg.get('trend_min_rv', 5.0)}" min="1.0" max="50.0" step="0.5"></div>
  <div class="cfg-item"><label>ADR% massimo</label>
    <input type="number" id="cfgTrendMaxADR" value="{cfg.get('trend_max_adr', 70.0)}" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>ATR mult per SL</label>
    <input type="number" id="cfgTrendSL_ATR_Mult" value="{cfg.get('trend_sl_mult', 1.5)}" min="0.5" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>TP % ADR residuo</label>
    <input type="number" id="cfgTrendTP_ADR_Pct" value="{cfg.get('trend_tp_pct', 80.0)}" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>R:R minimo</label>
    <input type="number" id="cfgMinRR_Trend" value="{cfg.get('trend_min_rr', 1.5)}" min="0.5" max="3.0" step="0.1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>🔄 Modulo Reversal (Inversione)</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Reversal dinamico (Picchi)</label>
    <select id="cfgDynamicReversalOn"><option value="true" {"selected" if cfg.get('dynamic_reversal_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('dynamic_reversal_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal RV minimo</label>
    <input type="number" id="cfgRevMinRV" value="{cfg.get('rev_min_rv', 70.0)}" min="30" max="150" step="1"></div>
  <div class="cfg-item"><label>Reversal ADR% minimo</label>
    <input type="number" id="cfgRevMinADR" value="{cfg.get('rev_min_adr', 100.0)}" min="50" max="150" step="1"></div>
  <div class="cfg-item"><label>Distanza EMA min (pip)</label>
    <input type="number" id="cfgRevMinEMADistPips" value="{cfg.get('rev_min_ema_dist', 20.0)}" min="5" max="100" step="1"></div>
  <div class="cfg-item"><label>ATR mult per SL</label>
    <input type="number" id="cfgRevSL_ATR_Mult" value="{cfg.get('rev_sl_mult', 1.5)}" min="0.5" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>R:R minimo Reversal</label>
    <input type="number" id="cfgMinRR_Reversal" value="{cfg.get('rev_min_rr', 1.5)}" min="0.5" max="3.0" step="0.1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>🛡️ Gestione Post-Trade (Uscite)</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Profit Fade soglia R</label>
    <input type="number" id="cfgProfitFadeR" value="{cfg.get('profit_fade_r', 0.70)}" min="0.1" max="1.0" step="0.05"></div>
  <div class="cfg-item"><label>Loss Cut soglia R</label>
    <input type="number" id="cfgLossCutR" value="{cfg.get('loss_cut_r', 0.60)}" min="0.1" max="1.0" step="0.05"></div>
  <div class="cfg-item"><label>Chiudi su stato opposto</label>
    <select id="cfgCloseOnOpposite"><option value="true" {"selected" if cfg.get('close_on_opposite') else ""}>Attivo</option><option value="false" {"" if cfg.get('close_on_opposite') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Chiudi su GRAY in profit</label>
    <select id="cfgCloseOnGrayProfit"><option value="true" {"selected" if cfg.get('close_on_gray') else ""}>Attivo</option><option value="false" {"" if cfg.get('close_on_gray') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Chiudi su debole in profit</label>
    <select id="cfgCloseOnWeakProfit"><option value="true" {"selected" if cfg.get('close_on_weak') else ""}>Attivo</option><option value="false" {"" if cfg.get('close_on_weak') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>Ultimi 20 Trade</h2>
<div style="overflow-x:auto"><table>
<thead><tr><th>Simbolo</th><th>Dir</th><th>Modulo</th><th>Pips</th><th>Profitto</th><th>Risultato</th></tr></thead>
<tbody id="tradeTable"><tr><td colspan="6" style="text-align:center;color:#666">Nessun trade nel database</td></tr></tbody>
</table></div></div>

<div class="section"><h2>Azioni</h2>
<div class="btn-row">
  <button class="btn btn-green" onclick="retrain()">🔄 Riaddestra</button>
  <button class="btn btn-blue" onclick="refresh()">🔃 Aggiorna</button>
</div>
<div id="actionMsg" style="margin-top:8px;font-size:0.8em;color:#ffd54f"></div>
</div>

<div style="text-align:center;padding:16px 0;font-size:0.7em;color:#444">
  Profit Radar Pro v5.0 — Giovanni Mori
</div>
</div>

<span id="cfgMsg" style="display:none"></span>

<script>
const API=window.location.origin;
function fmt(v,d=2){{return v!=null?v.toFixed(d):'-'}}
function pnlClass(v){{return v>0?'green':v<0?'red':'white'}}
function refresh(){{
  fetch(API+'/dashboard_data').then(r=>r.json()).then(d=>{{
    const ea=d.ea,srv=d.server,cfg=d.config;
    const dot=document.getElementById('eaDot');
    const age=ea.last_update?((Date.now()-new Date(ea.last_update))/1000/60):999;
    if(age<5){{dot.className='status-dot dot-green';document.getElementById('eaStatus').textContent='EA Connesso'}}
    else if(age<30){{dot.className='status-dot dot-yellow';document.getElementById('eaStatus').textContent='EA '+Math.round(age)+'m fa'}}
    else{{dot.className='status-dot dot-gray';document.getElementById('eaStatus').textContent='EA offline'}}
    document.getElementById('lastUpdate').textContent='Aggiornato: '+new Date().toLocaleTimeString('it-IT');
    document.getElementById('balance').textContent=fmt(ea.balance);
    document.getElementById('equity').textContent=fmt(ea.equity);
    const pnl=ea.daily_pnl||0;const pe=document.getElementById('dailyPnl');
    pe.textContent=(pnl>=0?'+':'')+fmt(pnl);pe.className='val '+pnlClass(pnl);
    document.getElementById('openTrades').textContent=ea.open_trades+'/'+(cfg.max_concurrent||3);

    // --- Popola Tabella Statistiche e Picchi Dinamici ---
    const statsTable = document.getElementById('statsTable');
    if (statsTable) {{
      const stats = d.trade_stats || {{}};
      const peaks = (d.ea && d.ea.peaks) ? d.ea.peaks : {{}};
      const unified = {{}};
      
      Object.keys(stats).forEach(rawSym => {{
        const clean = rawSym.replace('+', '').toUpperCase().trim();
        if (!unified[clean]) {{
          unified[clean] = {{ count: 0, win_rate: 0, avg_rv: 0, max_rv: 0, peak: '-' }};
        }}
        const s = stats[rawSym];
        unified[clean].count = s.count || 0;
        unified[clean].win_rate = s.win_rate != null ? s.win_rate : 0;
        unified[clean].avg_rv = s.avg_rv != null ? s.avg_rv : 0;
        unified[clean].max_rv = s.max_rv != null ? s.max_rv : 0;
      }});
      
      Object.keys(peaks).forEach(rawSym => {{
        const clean = rawSym.replace('+', '').toUpperCase().trim();
        if (!unified[clean]) {{
          unified[clean] = {{ count: 0, win_rate: 0, avg_rv: 0, max_rv: 0, peak: '-' }};
        }}
        unified[clean].peak = peaks[rawSym] != null ? peaks[rawSym] : '-';
      }});
      
      const sortedSymbols = Object.keys(unified).sort();
      
      if (sortedSymbols.length > 0) {{
        statsTable.innerHTML = sortedSymbols.map(sym => {{
          const u = unified[sym];
          let wrColor = u.win_rate >= 50 ? '#81c784' : u.count > 0 ? '#ef5350' : '#888';
          return '<tr>' +
            '<td><strong>' + sym + '</strong></td>' +
            '<td>' + u.count + '</td>' +
            '<td style="color:' + wrColor + '">' + (u.count > 0 ? u.win_rate.toFixed(1) + '%' : '-') + '</td>' +
            '<td>' + (u.count > 0 ? u.avg_rv.toFixed(1) : '-') + '</td>' +
            '<td>' + (u.count > 0 ? u.max_rv.toFixed(1) : '-') + '</td>' +
            '<td style="color:#4fc3f7"><strong>' + u.peak + '</strong></td>' +
            '</tr>';
        }}).join('');
      }} else {{
        statsTable.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;padding:12px;">In attesa del primo sync dell&apos;EA...</td></tr>';
      }}
    }}

    // --- Daily Stop W/L pesato ---
    const lw=cfg.loss_weight||1.5;
    const win=ea.daily_win_amount||0;
    const loss=ea.daily_loss_amount||0;
    const limit=win*lw;
    document.getElementById('dWin').textContent='+'+fmt(win);
    document.getElementById('dLoss').textContent='-'+fmt(loss);
    document.getElementById('dConsec').textContent=(ea.consecutive_losses||0)+'/'+(cfg.max_consec_loss||2);
    const stopped=ea.daily_stopped;
    const ss=document.getElementById('dStopState');
    ss.textContent=stopped?'🛑 STOP':'🟢 Attivo';ss.className='val '+(stopped?'red':'green');
    const bar=document.getElementById('dStopBar');
    if(limit>0){{
      let pct=Math.min(100,Math.round(loss/limit*100));
      bar.style.width=pct+'%';
      bar.style.background=pct<50?'#81c784':pct<80?'#ffd54f':'#ef5350';
      document.getElementById('dStopPct').textContent=pct+'%';
      document.getElementById('dStopDetail').textContent='persi '+fmt(loss)+' / soglia '+fmt(limit)+' EUR (peso x'+lw+') | margine '+fmt(limit-loss);
    }} else {{
      bar.style.width='0%';
      document.getElementById('dStopPct').textContent='-';
      document.getElementById('dStopDetail').textContent='In attesa della 1a vincita (per ora conta solo lo stop loss di fila)';
    }}

    document.getElementById('cfgCsvFile').value=cfg.csv_file||'PRP_TrustedLatest.csv';
    document.getElementById('cfgReadyFile').value=cfg.ready_file||'PRP_TrustedReady.csv';
    document.getElementById('cfgHistoryFile').value=cfg.history_file||'PRP_TrustedHistory.csv';
    document.getElementById('cfgTimerSeconds').value=cfg.timer_seconds||10;
    document.getElementById('cfgVerboseJournal').value=cfg.verbose_journal?'true':'false';
    document.getElementById('cfgProcessCurrentInit').value=cfg.process_current_init?'true':'false';
    document.getElementById('cfgMagicNumber').value=cfg.magic_number||270202;
    document.getElementById('cfgFixedLots').value=cfg.fixed_lots||0.07;
    document.getElementById('cfgMaxConcurrent').value=cfg.max_concurrent||3;
    document.getElementById('cfgMaxPerPair').value=cfg.max_per_pair||1;
    document.getElementById('cfgMaxEntriesBar').value=cfg.max_entries_bar||2;
    document.getElementById('cfgMaxSpreadPoints').value=cfg.max_spread_points||30;
    document.getElementById('cfgSlippage').value=cfg.slippage||3;
    document.getElementById('cfgAllowTrend').value=cfg.allow_trend?'true':'false';
    document.getElementById('cfgAllowReversal').value=cfg.allow_reversal?'true':'false';
    document.getElementById('cfgTrendMinRV').value=cfg.trend_min_rv||5.0;
    document.getElementById('cfgTrendMaxADR').value=cfg.trend_max_adr||70.0;
    document.getElementById('cfgTrendSL_ATR_Mult').value=cfg.trend_sl_mult||1.5;
    document.getElementById('cfgTrendTP_ADR_Pct').value=cfg.trend_tp_pct||80.0;
    document.getElementById('cfgMinRR_Trend').value=cfg.trend_min_rr||1.5;
    document.getElementById('cfgDynamicReversalOn').value=cfg.dynamic_reversal_on?'true':'false';
    document.getElementById('cfgRevMinRV').value=cfg.rev_min_rv||70.0;
    document.getElementById('cfgRevMinADR').value=cfg.rev_min_adr||100.0;
    document.getElementById('cfgRevMinEMADistPips').value=cfg.rev_min_ema_dist||20.0;
    document.getElementById('cfgRevSL_ATR_Mult').value=cfg.rev_sl_mult||1.5;
    document.getElementById('cfgMinRR_Reversal').value=cfg.rev_min_rr||1.5;
    document.getElementById('cfgProfitFadeR').value=cfg.profit_fade_r||0.70;
    document.getElementById('cfgLossCutR').value=cfg.loss_cut_r||0.60;
    document.getElementById('cfgCloseOnOpposite').value=cfg.close_on_opposite?'true':'false';
    document.getElementById('cfgCloseOnGrayProfit').value=cfg.close_on_gray?'true':'false';
    document.getElementById('cfgCloseOnWeakProfit').value=cfg.close_on_weak?'true':'false';

    const tb=document.getElementById('tradeTable');
    if(d.trade_history&&d.trade_history.length>0){{
      tb.innerHTML=d.trade_history.reverse().map(t=>{{
        const p=t.profit||0,w=t.won;
        return '<tr><td>'+(t.symbol||'-')+'</td><td>'+(t.direction||'-')+'</td><td>'+(t.module||'-')+'</td><td>'+fmt(t.pips,1)+'</td><td class="'+pnlClass(p)+'">'+(p>=0?'+':'')+fmt(p)+'€</td><td><span style="color:'+(w?'#81c784':'#ef5350')+'">'+(w?'WIN':'LOSS')+'</span></td></tr>'
      }}).join('');
    }}
  }}).catch(e=>{{
    console.error('Dashboard fetch error:',e);
    document.getElementById('lastUpdate').textContent='❌ Errore connessione al server';
    document.getElementById('lastUpdate').style.color='#ef5350';
  }});
}}
function saveAllConfig(btn = null){{
  const cfg={{
    csv_file:document.getElementById('cfgCsvFile').value,
    ready_file:document.getElementById('cfgReadyFile').value,
    history_file:document.getElementById('cfgHistoryFile').value,
    timer_seconds:parseInt(document.getElementById('cfgTimerSeconds').value),
    verbose_journal:document.getElementById('cfgVerboseJournal').value==='true',
    process_current_init:document.getElementById('cfgProcessCurrentInit').value==='true',
    magic_number:parseInt(document.getElementById('cfgMagicNumber').value),
    fixed_lots:parseFloat(document.getElementById('cfgFixedLots').value),
    max_concurrent:parseInt(document.getElementById('cfgMaxConcurrent').value),
    max_per_pair:parseInt(document.getElementById('cfgMaxPerPair').value),
    max_entries_bar:parseInt(document.getElementById('cfgMaxEntriesBar').value),
    max_spread_points:parseInt(document.getElementById('cfgMaxSpreadPoints').value),
    slippage:parseInt(document.getElementById('cfgSlippage').value),
    allow_trend:document.getElementById('cfgAllowTrend').value==='true',
    allow_reversal:document.getElementById('cfgAllowReversal').value==='true',
    trend_min_rv:parseFloat(document.getElementById('cfgTrendMinRV').value),
    trend_max_adr:parseFloat(document.getElementById('cfgTrendMaxADR').value),
    trend_sl_mult:parseFloat(document.getElementById('cfgTrendSL_ATR_Mult').value),
    trend_tp_pct:parseFloat(document.getElementById('cfgTrendTP_ADR_Pct').value),
    trend_min_rr:parseFloat(document.getElementById('cfgMinRR_Trend').value),
    dynamic_reversal_on:document.getElementById('cfgDynamicReversalOn').value==='true',
    rev_min_rv:parseFloat(document.getElementById('cfgRevMinRV').value),
    rev_min_adr:parseFloat(document.getElementById('cfgRevMinADR').value),
    rev_min_ema_dist:parseFloat(document.getElementById('cfgRevMinEMADistPips').value),
    rev_sl_mult:parseFloat(document.getElementById('cfgRevSL_ATR_Mult').value),
    rev_min_rr:parseFloat(document.getElementById('cfgMinRR_Reversal').value),
    profit_fade_r:parseFloat(document.getElementById('cfgProfitFadeR').value),
    loss_cut_r:parseFloat(document.getElementById('cfgLossCutR').value),
    close_on_opposite:document.getElementById('cfgCloseOnOpposite').value==='true',
    close_on_gray:document.getElementById('cfgCloseOnGrayProfit').value==='true',
    close_on_weak:document.getElementById('cfgCloseOnWeakProfit').value==='true',
  }};
  fetch(API+'/ea_config',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(cfg)}}).then(r=>r.json()).then(d=>{{
    const m_text = d.status==='ok'?'✅ '+d.message:'❌ '+(d.message||'errore');
    const m_color = d.status==='ok'?'#81c784':'#ef5350';
    if(btn) {{
      let m = btn.parentNode.querySelector('.cfg-msg');
      if(!m) {{
        m = document.createElement('span');
        m.className = 'cfg-msg';
        m.style.fontSize = '0.85em';
        m.style.marginLeft = '12px';
        m.style.alignSelf = 'center';
        btn.parentNode.appendChild(m);
      }}
      m.textContent = m_text;
      m.style.color = m_color;
      setTimeout(()=>{{m.textContent=''}},5000);
    }}
  }}).catch(()=>{{
    const err_text = '❌ Errore connessione';
    if(btn) {{
      let m = btn.parentNode.querySelector('.cfg-msg');
      if(!m) {{
        m = document.createElement('span');
        m.className = 'cfg-msg';
        m.style.fontSize = '0.85em';
        m.style.marginLeft = '12px';
        m.style.alignSelf = 'center';
        btn.parentNode.appendChild(m);
      }}
      m.textContent = err_text;
      m.style.color = '#ef5350';
      setTimeout(()=>{{m.textContent=''}},5000);
    }}
  }});
}}
function retrain() {{
  document.getElementById('actionMsg').textContent='⏳ Riaddestramento...';
  fetch(API+'/retrain',{{method:'POST'}}).then(r=>r.json()).then(d=>{{
    document.getElementById('actionMsg').innerHTML=d.status==='trained'?'✅ Trained! Samples: '+d.samples+' | WR: '+d.win_rate+'%':'⚠️ '+(d.error||'Errore');
  }}).catch(()=>{{document.getElementById('actionMsg').textContent='❌ Errore connessione'}});
}}
setInterval(refresh,5000);
refresh();
</script>
</body>
</html>"""
    return html

# ============================================================
#  AVVIO SERVER
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
