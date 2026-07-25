"""
Profit Radar Pro — Simplified AI Server v4.5
============================================
Server Flask ottimizzato per architettura sdoppiata "Data Collector + Executor".
Gestisce:
- Sincronizzazione EA e configurazione remota (/ea_status)
- Ricezione feedback di trade chiusi (/feedback) con auto-training
- Predizioni predittive LightGBM + GPT (/predict)
- Dashboard di controllo web (/dashboard) con menu collassabili
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
#  CONFIGURAZIONE DIRECTORY E PATH
# ============================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
EA_CONFIG_PATH = os.path.join(DATA_DIR, "ea_config.json")
EA_STATUS_PATH = os.path.join(DATA_DIR, "ea_status.json")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

MIN_FEEDBACK_FOR_TRAIN = 50

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("profit_radar")

# Inizializzazione Flask
app = Flask(__name__)
CORS(app)

# ============================================================
#  CONFIGURAZIONE REMOTE DI DEFAULT (OTTIMIZZATA V2)
# ============================================================
DEFAULT_EA_CONFIG = {
    # --- Configurazione principale ---
    "aggressiveness" : 1,
    "use_ai" : True,
    "ai_min_conf" : 75,
    "send_feedback" : True,
    "daily_stop_on" : True,
    "max_consec_loss" : 2,
    "loss_weight" : 1.5,
    "max_concurrent" : 3,
    "max_per_pair" : 1,
    # --- Lotto, rischio e fasce ---
    "fixed_lots" : 0.07,
    "max_lots_cap" : 0.14,
    "max_lots_safety" : 0.15,
    "dynamic_lots_on" : False,
    "dynamic_lookback" : 20,
    "friday_lots" : 0.00,
    "afternoon_lots" : 0.00,
    # --- Filtri giorno e direzione ---
    "no_monday_trade" : True,
    "no_buy" : False,
    "symbol_blacklist" : 'EURAUD,GBPAUD,USDJPY,AUDJPY,AUDUSD',
    "hyper_on" : True,
    "hyper_symbols" : 'EURCAD,EURUSD,GBPJPY,EURJPY,NZDJPY,CHFJPY,EURNZD,USDCAD',
    # --- TP, RR, Trailing e Break-Even ---
    "tp_percent" : 80,
    "tp_percent_min" : 50,
    "tp_adaptive" : True,
    "max_tp_pips" : 0,
    "min_rr" : 1.5,
    "be_pips" : 15,
    "be_profit" : 0,
    "trailing_on" : True,
    "trail_activate" : 1.0,
    "trail_atr_mult" : 1.5,
    "trail_step_pips" : 5,
    # --- Filtri standard ---
    "rv_max" : 20,
    "adr_max" : 50.0,
    "max_consecutive" : 10,
    "min_ema_gap_pct" : 0.1,
    "rev_min_ema_gap_pct" : 0.5,
    # --- Filtro RX ---
    "rx_required" : False,
    "rx_max_age" : 20,
    "rx_bonus_score" : True,
    # --- Modulo Breakout ---
    "breakout_on" : False,
    "breakout_min_light" : 2,
    "breakout_ema_gap_pct" : 0.2,
    "breakout_max_rv" : 15,
    "breakout_max_adx" : 25,
    "breakout_max_adr" : 50,
    "breakout_min_rr" : 1.8,
    "breakout_req_rx" : False,
    "breakout_max_rx_age" : 2,
    "breakout_atr_exp" : True,
    "breakout_price_ema" : True,
    "breakout_min_body" : 0.3,
    "breakout_score_bonus" : -80,
    # --- Modulo Reversal ---
    "reversal_on" : True,
    "dynamic_reversal_on" : True,
    "reversal_observe" : False,
    "rev_lots" : 0.05,
    "reversal_rv" : 70,
    "reversal_rv_max" : 120,
    "reversal_adr" : 100.0,
    "rev_req_decel" : True,
    "rev_min_decel" : 1.5,
    "rev_req_rx" : True,
    "rev_rx_bonus" : True,
    "rev_req_diverg" : False,
    "rev_diverg_bars" : 8,
    "rev_req_hist_flip" : True,
    "rev_max_hist_age" : 5,
    # --- Orari e sessione ---
    "session_filter_on" : True,
    "session_start_utc" : 7,
    "session_end_utc" : 17,
    "time_offset" : 0,
    "no_night_trade" : True,
    "night_start_h" : 23,
    "night_end_h" : 7,
    "sunday_start_h" : 23,
    "fri_close_profit_h" : 21,
    "fri_close_profit_m" : 0,
    "fri_close_loss_h" : 22,
    "fri_close_loss_m" : 0,
    "fri_force_close_h" : 23,
    "fri_force_close_m" : 0,
    # --- Dati, AI e log ---
    "data_mode" : 1,
    "csv_file" : 'PRP_TrustedLatest.csv',
    "csv_max_age_sec" : 0,
    "radar_indicator" : 'THE_PROFIT_RADAR_PRO_by_ULTIMA_MARKETS_v2_7',
    "export_csv" : True,
    "auto_fallback" : True,
    "fallback_after" : 3,
    "show_export_btn" : True,
    "auto_export" : True,
    "strategy_test" : False,
    "ai_url" : 'https://profit-radar-ai.onrender.com/predict',
    "ai_timeout" : 5000,
    "ai_log" : True,
    "test_trade" : False,
    # --- Tecnici e sicurezza ---
    "magic_number" : 270101,
    "max_slippage" : 30,
    "max_spread" : 20,
    "spread_dyn_mult" : 2.0,
    "atr_mult" : 1.5,
    "atr_period" : 14,
    "fractal_bars" : 5,
    # --- Dashboard grafico MT4 ---
    "dash_x" : 10,
    "dash_y" : 10,
    "dash_font_size" : 10,
    "dash_color" : 16777215,
    "dash_bg_color" : 3100495,
    "dash_bg" : True,
}

# Stato live del server e dell'EA
stats = {
    "total_predict_calls": 0, "total_feedback_calls": 0, "total_errors": 0,
    "last_predict_time": None, "last_retrain_time": None, "started": datetime.now(timezone.utc).isoformat(),
    "model_is_trained": False, "model_loaded": False, "model_version": 0
}

ea_status = {
    "last_update": None, "balance": 0, "equity": 0, "open_trades": 0, "daily_pnl": 0,
    "daily_wins": 0, "daily_losses": 0, "daily_win_amount": 0, "daily_loss_amount": 0,
    "ai_calls": 0, "ai_confirm": 0, "ai_reject": 0, "ai_errors": 0, "ai_missed_trades": 0,
    "warmup_ok": False, "warmup_last": None, "data_source": "LIVE", "cross_active": 0, "cross_total": 0,
    "ea_version": "2.00 (Dual)", "peaks": {}
}

# ============================================================
#  CARICAMENTO MODELLO E FEATURES
# ============================================================
model = None
TRAIN_FEATURES = None

def load_model_from_disk():
    global model, TRAIN_FEATURES, stats
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
            stats["model_loaded"] = True
            stats["model_is_trained"] = True
            stats["model_version"] = 1
            
            feat_path = MODEL_PATH.replace(".pkl", "_features.json")
            if os.path.exists(feat_path):
                with open(feat_path, "r") as f:
                    TRAIN_FEATURES = json.load(f)
            logger.info("[ML] Modello caricato con successo dal disco.")
        except Exception as e:
            logger.error(f"[ML] Errore caricamento modello: {e}")

# Inizializza
load_model_from_disk()

# ============================================================
#  FUNZIONI DI CONFIGURAZIONE
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

# ============================================================
#  PULIZIA JSON (Anti NaN/Inf)
# ============================================================
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

# ============================================================
#  ANALISI STATISTICA STORICO TRADE
# ============================================================
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
        except Exception as e:
            logger.error(f"[STATS ERROR] {e}")
    return trade_stats

# ============================================================
#  ADDESTRAMENTO MACCHINE (LightGBM)
# ============================================================
def train_model():
    global model, stats, TRAIN_FEATURES

    path = FEEDBACK_PATH
    fallback_log_path = os.path.join(DATA_DIR, "PRP_TradeLog.csv")
    use_fallback_csv = False

    if not os.path.exists(path):
        if os.path.exists(fallback_log_path):
            path = fallback_log_path
            use_fallback_csv = True
        else:
            return {"error": "Nessun dato log trade disponibile per l'addestramento"}
    else:
        try:
            temp_df = pd.read_csv(path)
            if len(temp_df) < MIN_FEEDBACK_FOR_TRAIN and os.path.exists(fallback_log_path):
                path = fallback_log_path
                use_fallback_csv = True
        except:
            if os.path.exists(fallback_log_path):
                path = fallback_log_path
                use_fallback_csv = True

    try:
        import joblib
        import lightgbm as lgb

        if use_fallback_csv:
            fb_df = pd.read_csv(path, sep=";", on_bad_lines="skip")
            fb_df.columns = [c.lower() for c in fb_df.columns]
            if "adr%" in fb_df.columns:
                fb_df["adr_pct"] = fb_df["adr%"]
        else:
            fb_df = pd.read_csv(path)

        if len(fb_df) < MIN_FEEDBACK_FOR_TRAIN:
            return {"error": f"Dati insufficienti: servono almeno {MIN_FEEDBACK_FOR_TRAIN} record."}

        df = fb_df.copy()

        # Numeric coercion
        df["rv"] = pd.to_numeric(df["rv"], errors="coerce").fillna(0)
        df["adx"] = pd.to_numeric(df["adx"], errors="coerce").fillna(0)
        df["adr_pct"] = pd.to_numeric(df["adr_pct"], errors="coerce").fillna(0)

        # Feature derivate
        df["rv_abs"] = df["rv"].abs()
        df["adr_residual_pct"] = (100 - df["adr_pct"]).clip(lower=0)

        # rv_decel
        if "rv_prev" in df.columns:
            rv_prev_num = pd.to_numeric(df["rv_prev"], errors="coerce").fillna(0)
            df["rv_decel"] = rv_prev_num.abs() - df["rv"].abs()
        else:
            df["rv_decel"] = 0.0

        # Set di feature base
        feature_cols = ["rv", "adx", "adr_pct", "rv_abs", "adr_residual_pct", "rv_decel"]

        # Feature opzionali arricchite
        OPTIONAL_FEATURES = [
            "nm", "nm_accel", "nm_dist", "nm_signal", "is_compressing",
            "ema_pos", "ema_gap_pct", "rv_prev", "rv_prev2"
        ]
        for feat in OPTIONAL_FEATURES:
            if feat in df.columns:
                df[feat] = df[feat].replace({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
                df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)
                if df[feat].nunique() > 1:
                    feature_cols.append(feat)

        if df["won"].dtype == object:
            df["won"] = df["won"].astype(str).str.lower().str.strip() == "true"
        else:
            df["won"] = df["won"].astype(bool)

        X = df[feature_cols].values
        y = df["won"].astype(int).values

        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        if pos_count < 3 or neg_count < 3:
            return {"error": f"Classi troppo sbilanciate (won={pos_count}, lost={neg_count})."}

        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 15, "learning_rate": 0.05, "verbose": -1, "seed": 42
        }

        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=80)

        # Salva
        joblib.dump(model, MODEL_PATH)
        TRAIN_FEATURES = list(feature_cols)
        
        with open(MODEL_PATH.replace(".pkl", "_features.json"), "w") as f:
            json.dump(TRAIN_FEATURES, f)

        stats["model_is_trained"] = True
        stats["model_loaded"] = True
        stats["model_version"] += 1
        stats["last_retrain_time"] = datetime.now(timezone.utc).isoformat()

        return {
            "status": "trained", "samples": len(df),
            "won": pos_count, "lost": neg_count,
            "win_rate": round(pos_count / len(y) * 100, 1)
        }
    except Exception as e:
        logger.error(f"[TRAIN ERROR] {e}")
        return {"error": str(e)}

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
    global stats
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        stats["total_feedback_calls"] += 1
        new_row = pd.DataFrame([data])
        
        # Scrivi feedback.csv
        header_needed = not os.path.exists(FEEDBACK_PATH)
        new_row.to_csv(FEEDBACK_PATH, mode="a", header=header_needed, index=False)

        # Auto-training se raggiungiamo la soglia
        total_fb = len(pd.read_csv(FEEDBACK_PATH))
        train_res = None
        if total_fb >= MIN_FEEDBACK_FOR_TRAIN and total_fb % 10 == 0:
            train_res = train_model()

        res = {"status": "ok", "total_feedback": total_fb}
        if train_res:
            res["train"] = train_res
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    global stats, model, TRAIN_FEATURES
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0}), 200

        stats["total_predict_calls"] += 1
        stats["last_predict_time"] = datetime.now(timezone.utc).isoformat()

        direction = data.get("direction", "").upper()
        rv = float(data.get("rv", 0))
        adx = float(data.get("adx", 0))
        adr_pct = float(data.get("adr_pct", 0))

        features_row = {
            "rv": rv, "adx": adx, "adr_pct": adr_pct,
            "rv_abs": abs(rv), "adr_residual_pct": max(0, 100 - adr_pct),
            "rv_decel": abs(float(data.get("rv_prev", 0))) - abs(rv)
        }

        # Carica features opzionali inviate
        for k, v in data.items():
            if k not in features_row and isinstance(v, (int, float, bool)):
                features_row[k] = float(v)

        lgbm_conf = 0
        if stats["model_is_trained"] and model is not None:
            try:
                features_df = pd.DataFrame([features_row])
                # Ensure all features match
                if TRAIN_FEATURES:
                    for f in TRAIN_FEATURES:
                        if f not in features_df.columns:
                            features_df[f] = 0.0
                    features_df = features_df[TRAIN_FEATURES]
                preds = model.predict(features_df.values)
                prob = float(preds[0])
                
                # Traduzione probabilità binary in confidenza (0-100) per direzione proposta
                if direction == "BUY":
                    lgbm_conf = int(prob * 100)
                else:
                    lgbm_conf = int((1 - prob) * 100)
            except: pass

        # Regola basata su regole se il modello non è pronto o fallisce
        if lgbm_conf == 0:
            # Semplice rule score
            lgbm_conf = 50
            if direction == "BUY" and rv > 0: lgbm_conf += 10
            elif direction == "SELL" and rv < 0: lgbm_conf += 10
            if adx > 25: lgbm_conf += 10
            if adr_pct < 50: lgbm_conf += 5

        # Decisione
        signal = "HOLD"
        conf = lgbm_conf
        cfg = load_ea_config()
        min_conf = cfg.get("ai_min_conf", 70)
        
        if conf >= min_conf:
            signal = direction

        return jsonify({"signal": signal, "confidence": conf, "method": "ml_hybrid"})
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
        
        # Filtro ed estrazione parametri validi
        updatable = [
            "aggressiveness", "use_ai", "ai_min_conf", "send_feedback", "daily_stop_on",
            "max_consec_loss", "loss_weight", "max_concurrent", "max_per_pair",
            "fixed_lots", "max_lots_cap", "max_lots_safety", "dynamic_lots_on", "dynamic_lookback",
            "friday_lots", "afternoon_lots", "no_monday_trade", "no_buy", "symbol_blacklist",
            "hyper_on", "hyper_symbols", "tp_percent", "tp_percent_min", "tp_adaptive",
            "max_tp_pips", "min_rr", "be_pips", "be_profit", "trailing_on", "trail_activate",
            "trail_atr_mult", "trail_step_pips", "rv_max", "adr_max", "max_consecutive",
            "min_ema_gap_pct", "rev_min_ema_gap_pct", "rx_required", "rx_max_age", "rx_bonus_score",
            "breakout_on", "reversal_on", "dynamic_reversal_on", "reversal_observe", "rev_lots",
            "reversal_rv", "reversal_rv_max", "reversal_adr", "rev_req_decel", "rev_min_decel",
            "rev_req_rx", "rev_rx_bonus", "rev_req_diverg", "rev_req_hist_flip", "rev_max_hist_age",
            "session_filter_on", "session_start_utc", "session_end_utc", "time_offset", "no_night_trade",
            "night_start_h", "night_end_h", "sunday_start_h", "fri_close_profit_h", "fri_close_profit_m",
            "fri_close_loss_h", "fri_close_loss_m", "fri_force_close_h", "fri_force_close_m",
            "data_mode", "csv_file", "csv_max_age_sec", "radar_indicator", "export_csv",
            "auto_fallback", "fallback_after", "show_export_btn", "auto_export", "strategy_test",
            "ai_url", "ai_timeout", "ai_log", "test_trade", "magic_number", "max_slippage",
            "max_spread", "spread_dyn_mult", "atr_mult", "atr_period", "fractal_bars",
            "dash_x", "dash_y", "dash_font_size", "dash_color", "dash_bg_color", "dash_bg"
        ]

        bool_keys = {
            "use_ai", "send_feedback", "daily_stop_on", "dynamic_lots_on", "no_monday_trade", "no_buy",
            "hyper_on", "tp_adaptive", "trailing_on", "rx_required", "rx_bonus_score", "breakout_on",
            "reversal_on", "dynamic_reversal_on", "reversal_observe", "rev_req_decel", "rev_req_rx",
            "rev_rx_bonus", "rev_req_diverg", "rev_req_hist_flip", "session_filter_on", "no_night_trade",
            "export_csv", "auto_fallback", "show_export_btn", "auto_export", "strategy_test", "ai_log",
            "test_trade", "dash_bg"
        }

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


@app.route("/retrain", methods=["POST"])
def retrain():
    result = train_model()
    return jsonify(result)

# ============================================================
#  ENDPOINTS DASHBOARD (HTML / JS)
# ============================================================

@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    ea = dict(ea_status)
    srv = dict(stats)
    fb_count = 0
    trade_history = []
    
    if os.path.exists(FEEDBACK_PATH):
        try:
            fb_df = pd.read_csv(FEEDBACK_PATH)
            fb_count = len(fb_df)
            for t in fb_df.tail(20).to_dict("records"):
                t["profit"] = float(t.get("profit", 0)) if not pd.isna(t.get("profit", 0)) else 0.0
                t["pips"] = float(t.get("pips", 0)) if not pd.isna(t.get("pips", 0)) else 0.0
                t["won"] = bool(t.get("won", False)) if not pd.isna(t.get("won", False)) else False
                trade_history.append(t)
        except: pass

    result = {
        "ea": ea, "server": srv, "config": load_ea_config(),
        "feedback_count": fb_count, "trade_history": trade_history,
        "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
        "trade_stats": get_trade_stats()
    }
    return jsonify(sanitize_for_json(result))


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    cfg = load_ea_config()
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Radar AI Dashboard</title>
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
.btn-green{{background:#2e7d32;color:#fff}}.btn-blue{{background:#1565c0;color:#fff}}
.btn-red{{background:#b71c1c;color:#fff}}.btn-gray{{background:#333;color:#ccc}}
.btn-yellow{{background:#f57f17;color:#fff}}
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

/* === SEZIONI COLLASSABILI === */
details.section {{
  transition: all 0.3s ease;
}}
details.section summary {{
  list-style: none;
  cursor: pointer;
  outline: none;
  user-select: none;
}}
details.section summary::-webkit-details-marker {{
  display: none;
}}
details.section summary h2 {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: 0 !important;
}}
details.section summary h2::after {{
  content: '▼';
  font-size: 0.75em;
  color: #4fc3f7;
  transition: transform 0.2s ease;
  margin-left: auto;
}}
details[open].section summary h2::after {{
  transform: rotate(180deg);
}}
details.section > :not(summary) {{
  margin-top: 14px;
}}
</style>
</head>
<body>
<div class="container">
<h1>📡 Profit Radar <span>Pro</span> — Dashboard</h1>

<div class="refresh-bar">
  <span id="lastUpdate">Caricamento...</span>
  <span><span class="status-dot dot-gray" id="eaDot"></span><span id="eaStatus">-</span></span>
</div>

<div class="section"><h2>Account</h2>
<div class="row">
  <div class="card"><div class="val white" id="balance">-</div><div class="lbl">Balance EUR</div></div>
  <div class="card"><div class="val white" id="equity">-</div><div class="lbl">Equity EUR</div></div>
  <div class="card"><div class="val" id="dailyPnl">-</div><div class="lbl">P&L Oggi</div></div>
  <div class="card"><div class="val" id="openTrades">-</div><div class="lbl">Trade Aperti</div></div>
</div></div>

<div class="section"><h2>AI Engine</h2>
<div class="row">
  <div class="card"><div class="val white" id="aiCalls">-</div><div class="lbl">Chiamate</div></div>
  <div class="card"><div class="val green" id="aiConfirm">-</div><div class="lbl">Confermati</div></div>
  <div class="card"><div class="val red" id="aiReject">-</div><div class="lbl">Scartati</div></div>
  <div class="card"><div class="val" id="aiErrors">-</div><div class="lbl">Errori / Miss</div></div>
</div></div>

<div class="section"><h2>Mercato</h2>
<div class="row">
  <div class="card"><div class="val white" id="crossTotal">-</div><div class="lbl">Cross Monitorati</div></div>
  <div class="card"><div class="val green" id="crossActive">-</div><div class="lbl">Trend Attivi</div></div>
  <div class="card"><div class="val white" id="dailyWL">-</div><div class="lbl">Wins / Losses</div></div>
</div></div>

<div class="section"><h2>Daily Stop (W/L pesato)</h2>
<div class="row">
  <div class="card"><div class="val green" id="dWin">-</div><div class="lbl">Wins Odierne</div></div>
  <div class="card"><div class="val red" id="dLoss">-</div><div class="lbl">Losses Odierne</div></div>
  <div class="card"><div class="val white" id="dConsec">-</div><div class="lbl">Losses Fila</div></div>
  <div class="card"><div class="val" id="dStopState">-</div><div class="lbl">Stato Stop</div></div>
</div>
<div style="margin-top:10px">
  <div style="display:flex;justify-content:space-between;font-size:0.78em;color:#888;margin-bottom:4px">
    <span>Margine prima dello stop</span><span id="dStopPct">-</span>
  </div>
  <div style="background:#0a0a1a;border-radius:6px;height:18px;overflow:hidden;border:1px solid #2a2a50">
    <div id="dStopBar" style="height:100%;width:0%;background:#81c784;transition:width .3s"></div>
  </div>
  <div style="font-size:0.75em;color:#666;margin-top:4px" id="dStopDetail">-</div>
</div></div>

<!-- === SEZIONE RV / PEAKS APERTA DI DEFAULT === -->
<details class="section" open><summary><h2>📊 Analisi Picchi e Statistiche Cross</h2></summary>
  <div style="font-size: 0.8em; color: #aaa; margin-bottom: 12px; line-height: 1.4;">
    Questa tabella interattiva mostra l'analisi statistica dei picchi di tendenza (Radar Value) e del Win Rate registrato storicamente per ciascuno dei 28 cross. Unifica automaticamente i simboli con e senza il suffisso +.
  </div>
  <div style="overflow-y:auto; max-height: 300px; border: 1px solid #1e1e40; border-radius: 6px;">
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
</details>

<!-- === SEZIONI DI CONFIGURAZIONE CHIUSE DI DEFAULT === -->
<details class="section"><summary><h2>Configurazione principale</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Stile (aggressività)<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto il robot è esigente. 1=Conservativo, 2=Moderato, 3=Aggressivo, 4=Iperconservativo.</span></span></label>
    <input type="number" id="cfgAggressiveness" value="{cfg.get('aggressiveness', 1)}" min="1" max="4" step="1"></div>
  <div class="cfg-item"><label>AI Attiva<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, l'IA decide se il trade è buono. Se DISATTIVO, il robot decide da solo.</span></span></label>
    <select id="cfgUseAi"><option value="true" {"selected" if cfg.get('use_ai') else ""}>Attivo</option><option value="false" {"" if cfg.get('use_ai') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Confidenza minima %<span class="tooltip"> ⓘ<span class="tooltiptext">Sotto questa %, l'IA scarta il trade.</span></span></label>
    <input type="number" id="cfgAiMinConf" value="{cfg.get('ai_min_conf', 75)}" min="50" max="95" step="1"></div>
  <div class="cfg-item"><label>Invia Feedback<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, invia i dati di esito trade a Render per addestrare l'IA.</span></span></label>
    <select id="cfgSendFeedback"><option value="true" {"selected" if cfg.get('send_feedback') else ""}>Attivo</option><option value="false" {"" if cfg.get('send_feedback') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>🛑 Daily Stop<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, blocca nuovi trade in caso di perdite massime consecutive raggiunte.</span></span></label>
    <select id="cfgDailyStopOn"><option value="true" {"selected" if cfg.get('daily_stop_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('daily_stop_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Max loss consecutivi<span class="tooltip"> ⓘ<span class="tooltiptext">Perdite consecutive massime consentite in una giornata.</span></span></label>
    <input type="number" id="cfgMaxConsecLoss" value="{cfg.get('max_consec_loss', 2)}" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>Peso perdite (x vincite)<span class="tooltip"> ⓘ<span class="tooltiptext">Peso moltiplicatore perdite vs vincite.</span></span></label>
    <input type="number" id="cfgLossWeight" value="{cfg.get('loss_weight', 1.5)}" min="1.0" max="5.0" step="0.1"></div>
  <div class="cfg-item"><label>Max trade aperti<span class="tooltip"> ⓘ<span class="tooltiptext">Numero massimo di trade aperti contemporaneamente.</span></span></label>
    <input type="number" id="cfgMaxConcurrent" value="{cfg.get('max_concurrent', 3)}" min="1" max="28" step="1"></div>
  <div class="cfg-item"><label>Max trade per coppia<span class="tooltip"> ⓘ<span class="tooltiptext">Massimo 1 trade per singola coppia.</span></span></label>
    <input type="number" id="cfgMaxPerPair" value="{cfg.get('max_per_pair', 1)}" min="1" max="5" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Lotto, rischio e fasce</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Lotto base<span class="tooltip"> ⓘ<span class="tooltiptext">Lotto iniziale utilizzato per i trade.</span></span></label>
    <input type="number" id="cfgFixedLots" value="{cfg.get('fixed_lots', 0.07)}" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto max cap<span class="tooltip"> ⓘ<span class="tooltiptext">Lotto massimo impostabile in assoluto.</span></span></label>
    <input type="number" id="cfgMaxLotsCap" value="{cfg.get('max_lots_cap', 0.14)}" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto max di sicurezza<span class="tooltip"> ⓘ<span class="tooltiptext">Protezione di sicurezza massima.</span></span></label>
    <input type="number" id="cfgMaxLotsSafety" value="{cfg.get('max_lots_safety', 0.15)}" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto dinamico<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, adatta il lotto in base al win rate.</span></span></label>
    <select id="cfgDynamicLotsOn"><option value="true" {"selected" if cfg.get('dynamic_lots_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('dynamic_lots_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Lotto Venerdì<span class="tooltip"> ⓘ<span class="tooltiptext">Lotto usato il venerdì. 0.00 = venerdì chiuso.</span></span></label>
    <input type="number" id="cfgFridayLots" value="{cfg.get('friday_lots', 0.00)}" min="0.00" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto Pomeriggio<span class="tooltip"> ⓘ<span class="tooltiptext">Lotto usato nel pomeriggio. 0.00 = pomeriggio chiuso.</span></span></label>
    <input type="number" id="cfgAfternoonLots" value="{cfg.get('afternoon_lots', 0.00)}" min="0.00" max="1.0" step="0.01"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Filtri giorno e direzione</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>🚫 Filtro Lunedì<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, il lunedì il bot non opera.</span></span></label>
    <select id="cfgNoMondayTrade"><option value="true" {"selected" if cfg.get('no_monday_trade') else ""}>Attivo</option><option value="false" {"" if cfg.get('no_monday_trade') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>🚫 Filtro BUY<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, blocca i trade BUY e fa solo SELL.</span></span></label>
    <select id="cfgNoBuy"><option value="true" {"selected" if cfg.get('no_buy') else ""}>Attivo</option><option value="false" {"" if cfg.get('no_buy') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Blacklist simboli<span class="tooltip"> ⓘ<span class="tooltiptext">Simboli da evitare separati da virgole.</span></span></label>
    <input type="text" id="cfgSymbolBlacklist" value="{cfg.get('symbol_blacklist', 'EURAUD,GBPAUD,USDJPY,AUDJPY,AUDUSD')}"></div>
  <div class="cfg-item"><label>Iperconservativo ON<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, opera solo sulle coppie in whitelist con regole rigidissime.</span></span></label>
    <select id="cfgHyperOn"><option value="true" {"selected" if cfg.get('hyper_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('hyper_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Whitelist ipercons.<span class="tooltip"> ⓘ<span class="tooltiptext">Coppie permesse in modalità iperconservativa.</span></span></label>
    <input type="text" id="cfgHyperSymbols" value="{cfg.get('hyper_symbols', 'EURCAD,EURUSD,GBPJPY,EURJPY,NZDJPY,CHFJPY,EURNZD,USDCAD')}"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>TP, RR, Trailing e Break-Even</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>TP % ADR max<span class="tooltip"> ⓘ<span class="tooltiptext">Limite massimo Take Profit come % dell'ADR residuo.</span></span></label>
    <input type="number" id="cfgTpPercent" value="{cfg.get('tp_percent', 80)}" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>TP % ADR min<span class="tooltip"> ⓘ<span class="tooltiptext">Limite minimo Take Profit come % dell'ADR.</span></span></label>
    <input type="number" id="cfgTpPercentMin" value="{cfg.get('tp_percent_min', 50)}" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>TP adattivo<span class="tooltip"> ⓘ<span class="tooltiptext">Adatta il TP alla forza del trend.</span></span></label>
    <select id="cfgTpAdaptive"><option value="true" {"selected" if cfg.get('tp_adaptive') else ""}>Attivo</option><option value="false" {"" if cfg.get('tp_adaptive') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>TP max in pip (0=off)<span class="tooltip"> ⓘ<span class="tooltiptext">0 = usa % ADR.</span></span></label>
    <input type="number" id="cfgMaxTpPips" value="{cfg.get('max_tp_pips', 0)}" min="0" max="200" step="1"></div>
  <div class="cfg-item"><label>R:R minimo<span class="tooltip"> ⓘ<span class="tooltiptext">Rapporto Rischio/Rendimento minimo accettato.</span></span></label>
    <input type="number" id="cfgMinRr" value="{cfg.get('min_rr', 1.5)}" min="0.5" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Break-Even pip<span class="tooltip"> ⓘ<span class="tooltiptext">Pip di profitto a cui scatta lo spostamento a pareggio dello stop loss.</span></span></label>
    <input type="number" id="cfgBePips" value="{cfg.get('be_pips', 15)}" min="0" max="100" step="1"></div>
  <div class="cfg-item"><label>Break-Even profitto bloccato<span class="tooltip"> ⓘ<span class="tooltiptext">Quanti pip blinda a profitto quando scatta il BE (0 = pareggio).</span></span></label>
    <input type="number" id="cfgBeProfit" value="{cfg.get('be_profit', 0)}" min="0" max="50" step="1"></div>
  <div class="cfg-item"><label>Trailing Stop<span class="tooltip"> ⓘ<span class="tooltiptext">Abilita il trailing stop ad inseguimento barra per barra.</span></span></label>
    <select id="cfgTrailingOn"><option value="true" {"selected" if cfg.get('trailing_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('trailing_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Trail attiva a R<span class="tooltip"> ⓘ<span class="tooltiptext">Dopo quante R di profitto attiva il trailing stop.</span></span></label>
    <input type="number" id="cfgTrailActivate" value="{cfg.get('trail_activate', 1.0)}" min="0.5" max="5.0" step="0.1"></div>
  <div class="cfg-item"><label>Trail ATR mult<span class="tooltip"> ⓘ<span class="tooltiptext">Moltiplicatore ATR per la distanza del trailing stop.</span></span></label>
    <input type="number" id="cfgTrailAtrMult" value="{cfg.get('trail_atr_mult', 1.5)}" min="0.1" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Trail step pip<span class="tooltip"> ⓘ<span class="tooltiptext">Sposta lo stop solo se il miglioramento è almeno di questi pip.</span></span></label>
    <input type="number" id="cfgTrailStepPips" value="{cfg.get('trail_step_pips', 5)}" min="0" max="50" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Filtri standard</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>RV massimo<span class="tooltip"> ⓘ<span class="tooltiptext">Radar Value massimo per entrare a favore di trend (STD).</span></span></label>
    <input type="number" id="cfgRvMax" value="{cfg.get('rv_max', 20)}" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>ADR% massimo<span class="tooltip"> ⓘ<span class="tooltiptext">Percentuale di corsa giornaliera massima per entrare a favore di trend (STD).</span></span></label>
    <input type="number" id="cfgAdrMax" value="{cfg.get('adr_max', 50.0)}" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>Max candele consecutive<span class="tooltip"> ⓘ<span class="tooltiptext">Numero massimo di candele consecutive dello stesso colore.</span></span></label>
    <input type="number" id="cfgMaxConsecutive" value="{cfg.get('max_consecutive', 10)}" min="5" max="50" step="1"></div>
  <div class="cfg-item"><label>Min gap EMA %<span class="tooltip"> ⓘ<span class="tooltiptext">Gap minimo tra EMA21 ed EMA200 per convalidare il trend.</span></span></label>
    <input type="number" id="cfgMinEmaGapPct" value="{cfg.get('min_ema_gap_pct', 0.1)}" min="0.0" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Min gap EMA % (Reversal)<span class="tooltip"> ⓘ<span class="tooltiptext">Gap minimo EMA richiesto per sbloccare il Reversal.</span></span></label>
    <input type="number" id="cfgRevMinEmaGapPct" value="{cfg.get('rev_min_ema_gap_pct', 0.5)}" min="0.0" max="1.0" step="0.01"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Filtro RX</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>RX richiesto<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, richiede obbligatoriamente un segnale RX per il modulo Standard.</span></span></label>
    <select id="cfgRxRequired"><option value="true" {"selected" if cfg.get('rx_required') else ""}>Attivo</option><option value="false" {"" if cfg.get('rx_required') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>RX eta max (candele)<span class="tooltip"> ⓘ<span class="tooltiptext">Età massima del segnale RX per essere considerato valido.</span></span></label>
    <input type="number" id="cfgRxMaxAge" value="{cfg.get('rx_max_age', 20)}" min="1" max="50" step="1"></div>
  <div class="cfg-item"><label>RX bonus punteggio<span class="tooltip"> ⓘ<span class="tooltiptext">Aumenta il punteggio interno se c'è un segnale RX valido.</span></span></label>
    <select id="cfgRxBonusScore"><option value="true" {"selected" if cfg.get('rx_bonus_score') else ""}>Attivo</option><option value="false" {"" if cfg.get('rx_bonus_score') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Modulo Breakout</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Breakout attivo<span class="tooltip"> ⓘ<span class="tooltiptext">Abilita o disabilita il modulo Breakout.</span></span></label>
    <select id="cfgBreakoutOn"><option value="true" {"selected" if cfg.get('breakout_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('breakout_on') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Modulo Reversal</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Reversal dinamico (Picchi)<span class="tooltip"> ⓘ<span class="tooltiptext">EA calcola in tempo reale la media dei 4 picchi storici maggiori invece di usare un valore fisso.</span></span></label>
    <select id="cfgDynamicReversalOn"><option value="true" {"selected" if cfg.get('dynamic_reversal_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('dynamic_reversal_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal attivo<span class="tooltip"> ⓘ<span class="tooltiptext">Abilita o disabilita le operazioni contrarie al trend in eccesso.</span></span></label>
    <select id="cfgReversalOn"><option value="true" {"selected" if cfg.get('reversal_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('reversal_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal solo osservazione<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, logga i segnali ma non apre trade.</span></span></label>
    <select id="cfgReversalObserve"><option value="true" {"selected" if cfg.get('reversal_observe') else ""}>Attivo</option><option value="false" {"" if cfg.get('reversal_observe') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal lotto<span class="tooltip"> ⓘ<span class="tooltiptext">Lotto specifico per il Reversal. 0.00 = usa lotto base.</span></span></label>
    <input type="number" id="cfgRevLots" value="{cfg.get('rev_lots', 0.05)}" min="0.00" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Reversal RV minimo<span class="tooltip"> ⓘ<span class="tooltiptext">Radar Value minimo per considerare un trend maturo pronto ad invertire.</span></span></label>
    <input type="number" id="cfgReversalRv" value="{cfg.get('reversal_rv', 70)}" min="30" max="150" step="1"></div>
  <div class="cfg-item"><label>Reversal RV massimo<span class="tooltip"> ⓘ<span class="tooltiptext">RV massimo per un'entrata Reversal sicura.</span></span></label>
    <input type="number" id="cfgReversalRvMax" value="{cfg.get('reversal_rv_max', 120)}" min="50" max="200" step="1"></div>
  <div class="cfg-item"><label>Reversal ADR% minimo<span class="tooltip"> ⓘ<span class="tooltiptext">L'ADR giornaliero deve aver superato questa percentuale per attivare il Reversal.</span></span></label>
    <input type="number" id="cfgReversalAdr" value="{cfg.get('reversal_adr', 100.0)}" min="50" max="150" step="1"></div>
  <div class="cfg-item"><label>Reversal richiede decelerazione<span class="tooltip"> ⓘ<span class="tooltiptext">Richiede decelerazione dell'istogramma prima dell'ingresso.</span></span></label>
    <select id="cfgRevReqDecel"><option value="true" {"selected" if cfg.get('rev_req_decel') else ""}>Attivo</option><option value="false" {"" if cfg.get('rev_req_decel') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal decelerazione min<span class="tooltip"> ⓘ<span class="tooltiptext">Differenza minima tra istogrammi consecutivi (es. 1.5).</span></span></label>
    <input type="number" id="cfgRevMinDecel" value="{cfg.get('rev_min_decel', 1.5)}" min="0.1" max="50.0" step="0.1"></div>
  <div class="cfg-item"><label>Reversal richiede RX<span class="tooltip"> ⓘ<span class="tooltiptext">Richiede un segnale RX (nuovo max/min a 10 giorni).</span></span></label>
    <select id="cfgRevReqRx"><option value="true" {"selected" if cfg.get('rev_req_rx') else ""}>Attivo</option><option value="false" {"" if cfg.get('rev_req_rx') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal RX bonus<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, aumenta il punteggio se c'è RX.</span></span></label>
    <select id="cfgRevRxBonus"><option value="true" {"selected" if cfg.get('rev_rx_bonus') else ""}>Attivo</option><option value="false" {"" if cfg.get('rev_rx_bonus') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal richiede divergenza<span class="tooltip"> ⓘ<span class="tooltiptext">Richiede una divergenza confermata del Radar Value.</span></span></label>
    <select id="cfgRevReqDiverg"><option value="true" {"selected" if cfg.get('rev_req_diverg') else ""}>Attivo</option><option value="false" {"" if cfg.get('rev_req_diverg') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal richiede flip istogramma<span class="tooltip"> ⓘ<span class="tooltiptext">Richiede il primo cambio colore da chiaro a scuro dell'istogramma.</span></span></label>
    <select id="cfgRevReqHistFlip"><option value="true" {"selected" if cfg.get('rev_req_hist_flip') else ""}>Attivo</option><option value="false" {"" if cfg.get('rev_req_hist_flip') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal eta max flip<span class="tooltip"> ⓘ<span class="tooltiptext">Età massima del flip (candele M15).</span></span></label>
    <input type="number" id="cfgRevMaxHistAge" value="{cfg.get('rev_max_hist_age', 5)}" min="1" max="10" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Orari e sessione</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Filtro sessione<span class="tooltip"> ⓘ<span class="tooltiptext">Abilita o disabilita il filtro temporale della sessione operativa.</span></span></label>
    <select id="cfgSessionFilterOn"><option value="true" {"selected" if cfg.get('session_filter_on') else ""}>Attivo</option><option value="false" {"" if cfg.get('session_filter_on') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Sessione inizio UTC<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di inizio sessione operativa principale (es. 7 = 07:00 UTC, cioè 08:00 o 09:00 Italia).</span></span></label>
    <input type="number" id="cfgSessionStartUtc" value="{cfg.get('session_start_utc', 7)}" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Sessione fine UTC<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di fine sessione operativa (es. 17 = 17:00 UTC, cioè 18:00 o 19:00 Italia).</span></span></label>
    <input type="number" id="cfgSessionEndUtc" value="{cfg.get('session_end_utc', 17)}" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Fuso orario broker vs Italia<span class="tooltip"> ⓘ<span class="tooltiptext">Differenza oraria tra la MT4 ed l'ora italiana. (0 se identica, -1 se broker è UTC+2).</span></span></label>
    <input type="number" id="cfgTimeOffset" value="{cfg.get('time_offset', 0)}" min="-5" max="5" step="1"></div>
  <div class="cfg-item"><label>Blocco notte<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, blocca e chiude tutti i trade prima della notte.</span></span></label>
    <select id="cfgNoNightTrade"><option value="true" {"selected" if cfg.get('no_night_trade') else ""}>Attivo</option><option value="false" {"" if cfg.get('no_night_trade') else "selected"}>Disattivo</option></select></div>
  <div class="cfg-item"><label>Notte inizio (Italia)<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di inizio del blocco notte (es. 23).</span></span></label>
    <input type="number" id="cfgNightStartH" value="{cfg.get('night_start_h', 23)}" min="18" max="23" step="1"></div>
  <div class="cfg-item"><label>Notte fine (Italia)<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di fine del blocco notte (es. 7).</span></span></label>
    <input type="number" id="cfgNightEndH" value="{cfg.get('night_end_h', 7)}" min="0" max="12" step="1"></div>
  <div class="cfg-item"><label>Domenica inizio (Italia)<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di riapertura dei mercati la domenica sera.</span></span></label>
    <input type="number" id="cfgSundayStartH" value="{cfg.get('sunday_start_h', 23)}" min="18" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven chiudi profitto ora<span class="tooltip"> ⓘ<span class="tooltiptext">Ora del venerdì a cui chiudere i trade in profitto (es. 21).</span></span></label>
    <input type="number" id="cfgFriCloseProfitH" value="{cfg.get('fri_close_profit_h', 21)}" min="12" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven chiudi perdita ora<span class="tooltip"> ⓘ<span class="tooltiptext">Ora del venerdì a cui chiudere i trade in perdita (es. 22).</span></span></label>
    <input type="number" id="cfgFriCloseLossH" value="{cfg.get('fri_close_loss_h', 22)}" min="12" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven forza chiusura ora<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di chiusura forzata totale del venerdì sera (es. 23).</span></span></label>
    <input type="number" id="cfgFriForceCloseH" value="{cfg.get('fri_force_close_h', 23)}" min="12" max="23" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Dati, AI e log</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Fonte dati<span class="tooltip"> ⓘ<span class="tooltiptext">1=Auto (Live con fallback CSV).</span></span></label>
    <input type="number" id="cfgDataMode" value="{cfg.get('data_mode', 1)}" min="0" max="2" step="1"></div>
  <div class="cfg-item"><label>Nome file CSV<span class="tooltip"> ⓘ<span class="tooltiptext">Nome del file CSV scritto dal Collettore.</span></span></label>
    <input type="text" id="cfgCsvFile" value="{cfg.get('csv_file', 'PRP_TrustedLatest.csv')}"></div>
  <div class="cfg-item"><label>CSV eta max (sec)<span class="tooltip"> ⓘ<span class="tooltiptext">Età massima del file prima di considerarlo obsoleto. 0 = infinita.</span></span></label>
    <input type="number" id="cfgCsvMaxAgeSec" value="{cfg.get('csv_max_age_sec', 0)}" min="0" max="3600" step="60"></div>
  <div class="cfg-item"><label>Nome indicatore Radar<span class="tooltip"> ⓘ<span class="tooltiptext">Nome esatto del file dell'indicatore.</span></span></label>
    <input type="text" id="cfgRadarIndicator" value="{cfg.get('radar_indicator', 'THE_PROFIT_RADAR_PRO_by_ULTIMA_MARKETS_v2_7')}"></div>
  <div class="cfg-item"><label>Timeout AI (ms)<span class="tooltip"> ⓘ<span class="tooltiptext">Limite di attesa della risposta dall'IA.</span></span></label>
    <input type="number" id="cfgAiTimeout" value="{cfg.get('ai_timeout', 5000)}" min="1000" max="30000" step="1000"></div>
  <div class="cfg-item"><label>Apri trade di test<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, all'avvio apre un trade finto. Solo debug.</span></span></label>
    <select id="cfgTestTrade"><option value="true" {"selected" if cfg.get('test_trade') else ""}>Attivo</option><option value="false" {"" if cfg.get('test_trade') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Tecnici e sicurezza</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Magic Number<span class="tooltip"> ⓘ<span class="tooltiptext">ID univoco dell'EA per identificare i suoi ordini.</span></span></label>
    <input type="number" id="cfgMagicNumber" value="{cfg.get('magic_number', 270101)}" min="100000" max="999999" step="1"></div>
  <div class="cfg-item"><label>Max slippage (points)<span class="tooltip"> ⓘ<span class="tooltiptext">Slippage massimo consentito in punti (30 points = 3 pips).</span></span></label>
    <input type="number" id="cfgMaxSlippage" value="{cfg.get('max_slippage', 30)}" min="1" max="100" step="1"></div>
  <div class="cfg-item"><label>Max spread (points)<span class="tooltip"> ⓘ<span class="tooltiptext">Spread massimo consentito (20 points = 2 pips).</span></span></label>
    <input type="number" id="cfgMaxSpread" value="{cfg.get('max_spread', 20)}" min="1" max="100" step="1"></div>
  <div class="cfg-item"><label>Spread dinamico mult<span class="tooltip"> ⓘ<span class="tooltiptext">Blocca l'ingresso se lo spread supera N volte la media recente.</span></span></label>
    <input type="number" id="cfgSpreadDynMult" value="{cfg.get('spread_dyn_mult', 2.0)}" min="1.0" max="5.0" step="0.1"></div>
  <div class="cfg-item"><label>ATR mult per SL<span class="tooltip"> ⓘ<span class="tooltiptext">Moltiplicatore dell'ATR per calcolare lo Stop Loss.</span></span></label>
    <input type="number" id="cfgAtrMult" value="{cfg.get('atr_mult', 1.5)}" min="0.5" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Periodo ATR<span class="tooltip"> ⓘ<span class="tooltiptext">Periodo dell'ATR per SL e Trailing.</span></span></label>
    <input type="number" id="cfgAtrPeriod" value="{cfg.get('atr_period', 14)}" min="5" max="50" step="1"></div>
  <div class="cfg-item"><label>Candele fractal SL<span class="tooltip"> ⓘ<span class="tooltiptext">Numero di candele per cercare il fractal per lo SL (es. 5).</span></span></label>
    <input type="number" id="cfgFractalBars" value="{cfg.get('fractal_bars', 5)}" min="3" max="50" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<details class="section"><summary><h2>Dashboard grafico MT4</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Dashboard X (pixel)<span class="tooltip"> ⓘ<span class="tooltiptext">Posizione orizzontale del pannello grafico su MT4.</span></span></label>
    <input type="number" id="cfgDashX" value="{cfg.get('dash_x', 10)}" min="0" max="3000" step="10"></div>
  <div class="cfg-item"><label>Dashboard Y (pixel)<span class="tooltip"> ⓘ<span class="tooltiptext">Posizione verticale del pannello grafico su MT4.</span></span></label>
    <input type="number" id="cfgDashY" value="{cfg.get('dash_y', 10)}" min="0" max="2000" step="10"></div>
  <div class="cfg-item"><label>Dashboard font size<span class="tooltip"> ⓘ<span class="tooltiptext">Dimensione del testo del pannello MT4.</span></span></label>
    <input type="number" id="cfgDashFontSize" value="{cfg.get('dash_font_size', 10)}" min="6" max="20" step="1"></div>
  <div class="cfg-item"><label>Dashboard colore testo<span class="tooltip"> ⓘ<span class="tooltiptext">Colore del testo della dashboard su MT4.</span></span></label>
    <select id="cfgDashColor"><option value="16777215" {"selected" if cfg.get('dash_color') == 16777215 else ""}>Bianco</option><option value="0" {"selected" if cfg.get('dash_color') == 0 else ""}>Nero</option></select></div>
  <div class="cfg-item"><label>Dashboard colore sfondo<span class="tooltip"> ⓘ<span class="tooltiptext">Colore dello sfondo della dashboard su MT4 (es. Grigio ardesia).</span></span></label>
    <select id="cfgDashBgColor"><option value="3100495" {"selected" if cfg.get('dash_bg_color') == 3100495 else ""}>Grigio ardesia</option><option value="0" {"selected" if cfg.get('dash_bg_color') == 0 else ""}>Nero</option></select></div>
  <div class="cfg-item"><label>Dashboard sfondo<span class="tooltip"> ⓘ<span class="tooltiptext">Se attivo, mostra lo sfondo del pannello su MT4.</span></span></label>
    <select id="cfgDashBg"><option value="true" {"selected" if cfg.get('dash_bg') else ""}>Attivo</option><option value="false" {"" if cfg.get('dash_bg') else "selected"}>Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>

<div class="section"><h2>Ultimi 20 Trade</h2>
<div style="overflow-x:auto"><table>
<thead><tr><th>Simbolo</th><th>Dir</th><th>Modulo</th><th>Pips</th><th>Profitto</th><th>Risultato</th><th>AI Conf</th></tr></thead>
<tbody id="tradeTable"><tr><td colspan="7" style="text-align:center;color:#666">Nessun trade</td></tr></tbody>
</table></div></div>

<div class="section"><h2>Azioni</h2>
<div class="btn-row">
  <button class="btn btn-green" onclick="retrain()">🔄 Riaddestra</button>
  <button class="btn btn-gray" onclick="refresh()">🔃 Aggiorna</button>
</div>
<div id="actionMsg" style="margin-top:8px;font-size:0.8em;color:#ffd54f"></div>
</div>

<div style="text-align:center;padding:16px 0;font-size:0.7em;color:#444">
  Profit Radar Pro v4.5 — Giovanni Mori
</div>
</div>

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
    document.getElementById('openTrades').textContent=ea.open_trades+'/'+(cfg.max_concurrent||10);
    document.getElementById('aiCalls').textContent=ea.ai_calls||0;
    document.getElementById('aiConfirm').textContent=ea.ai_confirm||0;
    document.getElementById('aiReject').textContent=ea.ai_reject||0;
    document.getElementById('aiErrors').textContent=ea.ai_errors||0;
    document.getElementById('crossTotal').textContent=ea.cross_total||0;
    document.getElementById('crossActive').textContent=ea.cross_active||0;
    document.getElementById('dailyWL').textContent=(ea.daily_wins||0)+' / '+(ea.daily_losses||0);

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

    document.getElementById('cfgDailyStopOn').value=(cfg.daily_stop_on!==false)?'true':'false';
    document.getElementById('cfgAggressiveness').value=cfg.aggressiveness||1;
    document.getElementById('cfgSendFeedback').value=cfg.send_feedback?'true':'false';
    document.getElementById('cfgUseAi').value=cfg.use_ai?'true':'false';
    document.getElementById('cfgAiMinConf').value=cfg.ai_min_conf||75;
    document.getElementById('cfgMaxConsecLoss').value=cfg.max_consec_loss||2;
    document.getElementById('cfgLossWeight').value=cfg.loss_weight||1.5;
    document.getElementById('cfgRvMax').value=cfg.rv_max||20;
    document.getElementById('cfgAdrMax').value=cfg.adr_max||50.0;
    document.getElementById('cfgMinRr').value=cfg.min_rr||1.5;
    document.getElementById('cfgTpPercent').value=cfg.tp_percent||80;
    document.getElementById('cfgTpPercentMin').value=cfg.tp_percent_min||50;
    document.getElementById('cfgMaxTpPips').value=cfg.max_tp_pips||0;
    document.getElementById('cfgFixedLots').value=cfg.fixed_lots||0.07;
    document.getElementById('cfgMaxLotsCap').value=cfg.max_lots_cap||0.14;
    document.getElementById('cfgDynamicLotsOn').value=cfg.dynamic_lots_on?'true':'false';
    document.getElementById('cfgDynamicLookback').value=cfg.dynamic_lookback||20;
    document.getElementById('cfgFridayLots').value=(cfg.friday_lots!=null?cfg.friday_lots.toFixed(2):'0.00');
    document.getElementById('cfgAfternoonLots').value=(cfg.afternoon_lots!=null?cfg.afternoon_lots.toFixed(2):'0.00');
    document.getElementById('cfgNoMondayTrade').value=cfg.no_monday_trade?'true':'false';
    document.getElementById('cfgNoBuy').value=cfg.no_buy?'true':'false';
    document.getElementById('cfgSymbolBlacklist').value=cfg.symbol_blacklist||'';
    document.getElementById('cfgTrailingOn').value=cfg.trailing_on?'true':'false';
    document.getElementById('cfgTrailActivate').value=cfg.trail_activate||1.0;
    document.getElementById('cfgTrailAtrMult').value=cfg.trail_atr_mult||1.5;
    document.getElementById('cfgTrailStepPips').value=cfg.trail_step_pips||5;
    document.getElementById('cfgHyperOn').value=cfg.hyper_on?'true':'false';
    document.getElementById('cfgBreakoutOn').value=cfg.breakout_on?'true':'false';
    document.getElementById('cfgReversalOn').value=cfg.reversal_on?'true':'false';
    document.getElementById('cfgDynamicReversalOn').value=cfg.dynamic_reversal_on?'true':'false';
    document.getElementById('cfgMaxConcurrent').value=cfg.max_concurrent||3;
    const tb=document.getElementById('tradeTable');
    if(d.trade_history&&d.trade_history.length>0){{
      tb.innerHTML=d.trade_history.reverse().map(t=>{{
        const p=t.profit||0,w=t.won;
        return '<tr><td>'+(t.symbol||'-')+'</td><td>'+(t.direction||'-')+'</td><td>'+(t.module||'-')+'</td><td>'+fmt(t.pips,1)+'</td><td class="'+pnlClass(p)+'">'+(p>=0?'+':'')+fmt(p)+'€</td><td><span style="color:'+(w?'#81c784':'#ef5350')+'">'+(w?'WIN':'LOSS')+'</span></td><td>'+(t.ai_confidence||'-')+'%</td></tr>'
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
    aggressiveness:parseInt(document.getElementById('cfgAggressiveness').value),
    use_ai:document.getElementById('cfgUseAi').value==='true',
    ai_min_conf:parseInt(document.getElementById('cfgAiMinConf').value),
    send_feedback:document.getElementById('cfgSendFeedback').value==='true',
    daily_stop_on:document.getElementById('cfgDailyStopOn').value==='true',
    max_consec_loss:parseInt(document.getElementById('cfgMaxConsecLoss').value),
    loss_weight:parseFloat(document.getElementById('cfgLossWeight').value),
    max_concurrent:parseInt(document.getElementById('cfgMaxConcurrent').value),
    max_per_pair:parseInt(document.getElementById('cfgMaxPerPair').value),
    fixed_lots:parseFloat(document.getElementById('cfgFixedLots').value),
    max_lots_cap:parseFloat(document.getElementById('cfgMaxLotsCap').value),
    max_lots_safety:parseFloat(document.getElementById('cfgMaxLotsSafety').value),
    dynamic_lots_on:document.getElementById('cfgDynamicLotsOn').value==='true',
    dynamic_lookback:parseInt(document.getElementById('cfgDynamicLookback').value),
    friday_lots:parseFloat(document.getElementById('cfgFridayLots').value),
    afternoon_lots:parseFloat(document.getElementById('cfgAfternoonLots').value),
    no_monday_trade:document.getElementById('cfgNoMondayTrade').value==='true',
    no_buy:document.getElementById('cfgNoBuy').value==='true',
    symbol_blacklist:document.getElementById('cfgSymbolBlacklist').value,
    hyper_on:document.getElementById('cfgHyperOn').value==='true',
    hyper_symbols:document.getElementById('cfgHyperSymbols').value,
    tp_percent:parseInt(document.getElementById('cfgTpPercent').value),
    tp_percent_min:parseInt(document.getElementById('cfgTpPercentMin').value),
    tp_adaptive:document.getElementById('cfgTpAdaptive').value==='true',
    max_tp_pips:parseInt(document.getElementById('cfgMaxTpPips').value),
    min_rr:parseFloat(document.getElementById('cfgMinRr').value),
    be_pips:parseInt(document.getElementById('cfgBePips').value),
    be_profit:parseInt(document.getElementById('cfgBeProfit').value),
    trailing_on:document.getElementById('cfgTrailingOn').value==='true',
    trail_activate:parseFloat(document.getElementById('cfgTrailActivate').value),
    trail_atr_mult:parseFloat(document.getElementById('cfgTrailAtrMult').value),
    trail_step_pips:parseInt(document.getElementById('cfgTrailStepPips').value),
    rv_max:parseInt(document.getElementById('cfgRvMax').value),
    adr_max:parseFloat(document.getElementById('cfgAdrMax').value),
    max_consecutive:parseInt(document.getElementById('cfgMaxConsecutive').value),
    min_ema_gap_pct:parseFloat(document.getElementById('cfgMinEmaGapPct').value),
    rev_min_ema_gap_pct:parseFloat(document.getElementById('cfgRevMinEmaGapPct').value),
    rx_required:document.getElementById('cfgRxRequired').value==='true',
    rx_max_age:parseInt(document.getElementById('cfgRxMaxAge').value),
    rx_bonus_score:document.getElementById('cfgRxBonusScore').value==='true',
    breakout_on:document.getElementById('cfgBreakoutOn').value==='true',
    reversal_on:document.getElementById('cfgReversalOn').value==='true',
    dynamic_reversal_on:document.getElementById('cfgDynamicReversalOn').value==='true',
    reversal_observe:document.getElementById('cfgReversalObserve').value==='true',
    rev_lots:parseFloat(document.getElementById('cfgRevLots').value),
    reversal_rv:parseInt(document.getElementById('cfgReversalRv').value),
    reversal_rv_max:parseInt(document.getElementById('cfgReversalRvMax').value),
    reversal_adr:parseFloat(document.getElementById('cfgReversalAdr').value),
    rev_req_decel:document.getElementById('cfgRevReqDecel').value==='true',
    rev_min_decel:parseFloat(document.getElementById('cfgRevMinDecel').value),
    rev_req_rx:document.getElementById('cfgRevReqRx').value==='true',
    rev_rx_bonus:document.getElementById('cfgRevRxBonus').value==='true',
    rev_req_diverg:document.getElementById('cfgRevReqDiverg').value==='true',
    rev_req_hist_flip:document.getElementById('cfgRevReqHistFlip').value==='true',
    rev_max_hist_age:parseInt(document.getElementById('cfgRevMaxHistAge').value),
    session_filter_on:document.getElementById('cfgSessionFilterOn').value==='true',
    session_start_utc:parseInt(document.getElementById('cfgSessionStartUtc').value),
    session_end_utc:parseInt(document.getElementById('cfgSessionEndUtc').value),
    time_offset:parseInt(document.getElementById('cfgTimeOffset').value),
    no_night_trade:document.getElementById('cfgNoNightTrade').value==='true',
    night_start_h:parseInt(document.getElementById('cfgNightStartH').value),
    night_end_h:parseInt(document.getElementById('cfgNightEndH').value),
    sunday_start_h:parseInt(document.getElementById('cfgSundayStartH').value),
    fri_close_profit_h:parseInt(document.getElementById('cfgFriCloseProfitH').value),
    fri_close_profit_m:parseInt(document.getElementById('cfgFriCloseProfitM').value),
    fri_close_loss_h:parseInt(document.getElementById('cfgFriCloseLossH').value),
    fri_close_loss_m:parseInt(document.getElementById('cfgFriCloseLossM').value),
    fri_force_close_h:parseInt(document.getElementById('cfgFriForceCloseH').value),
    fri_force_close_m:parseInt(document.getElementById('cfgFriForceCloseM').value),
    data_mode:parseInt(document.getElementById('cfgDataMode').value),
    csv_file:document.getElementById('cfgCsvFile').value,
    csv_max_age_sec:parseInt(document.getElementById('cfgCsvMaxAgeSec').value),
    radar_indicator:document.getElementById('cfgRadarIndicator').value,
    export_csv:document.getElementById('cfgExportCsv').value==='true',
    auto_fallback:document.getElementById('cfgAutoFallback').value==='true',
    fallback_after:parseInt(document.getElementById('cfgFallbackAfter').value),
    show_export_btn:document.getElementById('cfgShowExportBtn').value==='true',
    auto_export:document.getElementById('cfgAutoExport').value==='true',
    strategy_test:document.getElementById('cfgStrategyTest').value==='true',
    ai_url:document.getElementById('cfgAiUrl').value,
    ai_timeout:parseInt(document.getElementById('cfgAiTimeout').value),
    ai_log:document.getElementById('cfgAiLog').value==='true',
    test_trade:document.getElementById('cfgTestTrade').value==='true',
    magic_number:parseInt(document.getElementById('cfgMagicNumber').value),
    max_slippage:parseInt(document.getElementById('cfgMaxSlippage').value),
    max_spread:parseInt(document.getElementById('cfgMaxSpread').value),
    spread_dyn_mult:parseFloat(document.getElementById('cfgSpreadDynMult').value),
    atr_mult:parseFloat(document.getElementById('cfgAtrMult').value),
    atr_period:parseInt(document.getElementById('cfgAtrPeriod').value),
    fractal_bars:parseInt(document.getElementById('cfgFractalBars').value),
    dash_x:parseInt(document.getElementById('cfgDashX').value),
    dash_y:parseInt(document.getElementById('cfgDashY').value),
    dash_font_size:parseInt(document.getElementById('cfgDashFontSize').value),
    dash_color:parseInt(document.getElementById('cfgDashColor').value),
    dash_bg_color:parseInt(document.getElementById('cfgDashBgColor').value),
    dash_bg:document.getElementById('cfgDashBg').value==='true',}};
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
    const g=document.getElementById('cfgMsg');
    if(g){{g.textContent=m_text;g.style.color=m_color;setTimeout(()=>{{g.textContent=''}},5000);}}
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
    const g=document.getElementById('cfgMsg');
    if(g){{g.textContent=err_text;g.style.color='#ef5350';setTimeout(()=>{{g.textContent=''}},5000);}}
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
