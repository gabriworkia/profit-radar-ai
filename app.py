"""
Profit Radar Pro — AI Server
==============================
Server Flask con LightGBM per conferma trade EA MetaTrader 4.

Endpoint:
  POST /predict   → riceve JSON dal EA, ritorna {signal, confidence}
  POST /feedback  → riceve esito trade per accumulo dati
  GET  /health    → verifica stato server
  GET  /stats     → statistiche chiamate e modello
  POST /retrain   → riaddestra il modello con i dati accumulati

Deploy: Render.com (free tier, 512MB RAM)
"""

import os
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Config ---
DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
REQUESTS_PATH = os.path.join(DATA_DIR, "requests_log.csv")
EA_CONFIG_PATH = os.path.join(DATA_DIR, "ea_config.json")
EA_STATUS_PATH = os.path.join(DATA_DIR, "ea_status.json")
AB_RESULTS_PATH = os.path.join(DATA_DIR, "ab_results.csv")
MIN_FEEDBACK_FOR_TRAIN = int(os.environ.get("MIN_FEEDBACK_FOR_TRAIN", "50"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")

# --- App ---
app = Flask(__name__)
CORS(app)

# --- Stats in memoria ---
stats = {
    "started": datetime.now(timezone.utc).isoformat(),
    "total_predict_calls": 0,
    "total_feedback_calls": 0,
    "total_errors": 0,
    "last_predict_time": None,
    "last_retrain_time": None,
    "model_version": 0,
    "model_loaded": False,
    "model_is_trained": False,
}

# --- Modello ---
model = None
feature_names = [
    "rv", "adx", "adr_pct", "adr_pip", "adr_media",
    "ema_pos", "ema_gap_pct",
    "rv_prev", "rv_prev2", "light_streak", "was_gray", "hist_flip_bar",
    # Contesto mercato
    "ctx_total", "ctx_non_gray", "ctx_green", "ctx_red",
    "ctx_avg_abs_rv", "ctx_extreme_rv",
    # Feature derivate
    "rv_decel", "adr_residual_pct",
    # Momentum metrics
    "nm", "nm_signal", "nm_accel", "nm_dist", "is_compressing",
]

# ============================================================
#  RULES-BASED SCORER (usato finché il modello non è addestrato)
# ============================================================
def rules_based_score(data):
    """
    Sistema a punteggio basato su regole esperte.
    Ritorna (signal, confidence, details).
    
    Logica:
    - Parte da confidenza 50%
    - Aggiunge/sottrae punti in base a fattori tecnici
    - Range finale: 0-100
    """
    score = 50  # base neutrale
    
    direction = data.get("direction", "").upper()
    rv = float(data.get("rv", 0))
    adx = float(data.get("adx", 0))
    adr_pct = float(data.get("adr_pct", 0))
    adr_pip = float(data.get("adr_pip", 0))
    adr_media = float(data.get("adr_media", 0))
    ema_pos = int(data.get("ema_pos", 0))
    ema_gap_pct = float(data.get("ema_gap_pct", 0))
    hist = data.get("hist", "UNKNOWN").upper()
    rv_prev = float(data.get("rv_prev", 0))
    rv_prev2 = float(data.get("rv_prev2", 0))
    light_streak = int(data.get("light_streak", 0))
    was_gray = data.get("was_gray", False)
    hist_flip_bar = int(data.get("hist_flip_bar", 999))
    
    # Contesto
    ctx = data.get("context", {})
    ctx_non_gray = int(ctx.get("non_gray", 0))
    ctx_green = int(ctx.get("green", 0))
    ctx_red = int(ctx.get("red", 0))
    ctx_avg_rv = float(ctx.get("avg_abs_rv", 0))
    ctx_extreme = int(ctx.get("extreme_rv", 0))
    
    module = data.get("module", "STD").upper()
    
    # --- Momentum metrics ---
    nm = float(data.get("nm", 0))
    nm_signal = float(data.get("nm_signal", 0))
    nm_accel = float(data.get("nm_accel", 0))
    nm_dist = float(data.get("nm_dist", 0))
    is_compressing = data.get("is_compressing", False)
    
    # --- Fattori positivi (+confidenza) ---
    
    # EMA concorde con direzione (+8)
    if direction == "BUY" and ema_pos == 1:
        score += 8
    elif direction == "SELL" and ema_pos == -1:
        score += 8
    
    # RV moderato (non estremo) (+5 a +10)
    abs_rv = abs(rv)
    if 5 <= abs_rv <= 15:
        score += 10  # sweet spot
    elif 15 < abs_rv <= 25:
        score += 7
    elif 25 < abs_rv <= 35:
        score += 3
    elif abs_rv > 50:
        score -= 5  # troppo estremo
    
    # ADX nella zona giusta (+5 a +8)
    if 15 <= adx <= 25:
        score += 8  # trend presente ma non esagerato
    elif 25 < adx <= 40:
        score += 5
    elif adx > 50:
        score -= 3  # trend troppo maturo (per standard/breakout)
    
    # ADR con spazio residuo (+5 a +8)
    if adr_pct < 40:
        score += 8  # molto spazio
    elif adr_pct < 55:
        score += 5
    elif adr_pct > 80:
        score -= 10  # giorno quasi esaurito
    
    # Residuo ADR in pip
    if adr_media > 0:
        residual_pct = (adr_media - adr_pip) / adr_media * 100
    else:
        residual_pct = 0
    
    if residual_pct > 50:
        score += 5
    elif residual_pct < 20:
        score -= 8
    
    # EMA gap piccolo = trend giovane (+5)
    if ema_gap_pct < 0.10:
        score += 5
    elif ema_gap_pct > 0.30:
        score -= 3
    
    # Histogram LIGHT (più forte di DARK) (+5)
    if "LIGHT" in hist:
        score += 5
    elif "DARK" in hist:
        score += 2
    elif hist == "GRAY":
        score -= 5
    
    # --- Fattori per BREAKOUT ---
    if module == "BRK":
        if was_gray:
            score += 8  # transizione da GRAY = buon breakout
        if light_streak >= 2:
            score += 5  # conferma continuity
        if light_streak > 5:
            score -= 5  # troppo tardi
    
    # --- Fattori per REVERSAL ---
    if module == "REV":
        # Decelerazione RV
        if abs(rv_prev) > 0:
            decel = abs(rv_prev) - abs_rv
            if decel > 10:
                score += 10
            elif decel > 5:
                score += 5
            elif decel < 0:
                score -= 5  # accelerando = non reversal
        
        # Histogram flip recente
        if hist_flip_bar <= 2:
            score += 8
        elif hist_flip_bar <= 5:
            score += 3
        
        # ADX alto è buono per reversal
        if adx >= 40:
            score += 5
        if adx >= 50:
            score += 5
        
        # ADR alto = trend maturo
        if adr_pct >= 80:
            score += 5
    
    # --- Momentum Normalized System ---
    
    # NM concorde con direzione (+8)
    if direction == "BUY" and nm > 0:
        score += 8
    elif direction == "SELL" and nm < 0:
        score += 8
    elif direction == "BUY" and nm < -0.3:
        score -= 10  # Contro-trend forte
    elif direction == "SELL" and nm > 0.3:
        score -= 10
    
    # Acceleration concorde (+6)
    if direction == "BUY" and nm_accel > 0:
        score += 6
    elif direction == "SELL" and nm_accel < 0:
        score += 6
    elif direction == "BUY" and nm_accel < -0.1:
        score -= 5  # Momentum che decelera
    elif direction == "SELL" and nm_accel > 0.1:
        score -= 5
    
    # NM sopra Signal = trend vivo (+5)
    if direction == "BUY" and nm > nm_signal:
        score += 5
    elif direction == "SELL" and nm < nm_signal:
        score += 5
    
    # Compressione = potenziale breakout (+7 per BRK, -3 per STD)
    if is_compressing:
        if module == "BRK":
            score += 7  # Breakout imminente!
        else:
            score -= 3  # Aspetta il breakout
    
    # Distance alta = trend forte (+3)
    if nm_dist > 0.5:
        if (direction == "BUY" and nm > 0) or (direction == "SELL" and nm < 0):
            score += 3
    
    # Distance bassa ma non compressione = trend debole (-4)
    if nm_dist < 0.15 and not is_compressing:
        score -= 4
    
    # --- Contesto mercato ---
    # Tanti cross nella stessa direzione = conferma (+3)
    if direction == "BUY" and ctx_green > 10:
        score += 3
    elif direction == "SELL" and ctx_red > 10:
        score += 3
    
    # Troppi extreme RV = mercato volatile (-5)
    if ctx_extreme > 5:
        score -= 5
    
    # --- Clamp ---
    score = max(0, min(100, score))
    
    # Soglia: sotto 60% la confidenza è bassa
    signal = direction  # conferma la direzione proposta
    
    details = {
        "base": 50,
        "final": score,
        "method": "rules_v1",
    }
    
    return signal, score, details


# ============================================================
#  PREDICT con modello o regole
# ============================================================
def predict_with_model(features_df):
    """Usa il modello LightGBM addestrato se disponibile."""
    global model
    if model is None:
        return None, 0
    try:
        proba = model.predict_proba(features_df[feature_names])
        confidence = int(proba[0][1] * 100)  # probabilità classe positiva
        signal = "BUY" if proba[0][1] >= 0.5 else "SELL"
        return signal, confidence
    except Exception as e:
        print(f"[MODEL ERROR] {e}")
        return None, 0


# ============================================================
#  LOGGING & DATA ACCUMULATION
# ============================================================
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def log_request(data, result):
    """Salva ogni richiesta per futuro training."""
    ensure_data_dir()
    try:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": data.get("symbol", ""),
            "module": data.get("module", ""),
            "direction": data.get("direction", ""),
            "rv": data.get("rv", 0),
            "adx": data.get("adx", 0),
            "adr_pct": data.get("adr_pct", 0),
            "adr_pip": data.get("adr_pip", 0),
            "adr_media": data.get("adr_media", 0),
            "ema_pos": data.get("ema_pos", 0),
            "hist": data.get("hist", ""),
            "ema_gap_pct": data.get("ema_gap_pct", 0),
            "rv_prev": data.get("rv_prev", 0),
            "rv_prev2": data.get("rv_prev2", 0),
            "light_streak": data.get("light_streak", 0),
            "was_gray": data.get("was_gray", False),
            "hist_flip_bar": data.get("hist_flip_bar", 999),
            "ctx_total": data.get("context", {}).get("total", 0),
            "ctx_non_gray": data.get("context", {}).get("non_gray", 0),
            "ctx_green": data.get("context", {}).get("green", 0),
            "ctx_red": data.get("context", {}).get("red", 0),
            "ctx_avg_abs_rv": data.get("context", {}).get("avg_abs_rv", 0),
            "ctx_extreme_rv": data.get("context", {}).get("extreme_rv", 0),
            "ai_signal": result.get("signal", ""),
            "ai_confidence": result.get("confidence", 0),
            "method": result.get("method", ""),
        }
        
        df = pd.DataFrame([row])
        header = not os.path.exists(REQUESTS_PATH)
        df.to_csv(REQUESTS_PATH, mode='a', header=header, index=False)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def log_feedback(fb_data):
    """Salva feedback trade per training futuro."""
    ensure_data_dir()
    try:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticket": fb_data.get("ticket", 0),
            "symbol": fb_data.get("symbol", ""),
            "direction": fb_data.get("direction", ""),
            "module": fb_data.get("module", ""),
            "entry_price": fb_data.get("entry_price", 0),
            "exit_price": fb_data.get("exit_price", 0),
            "profit": fb_data.get("profit", 0),
            "pips": fb_data.get("pips", 0),
            "won": fb_data.get("won", False),
            "ai_confidence": fb_data.get("ai_confidence", 0),
            "ai_signal": fb_data.get("ai_signal", ""),
            "rv": fb_data.get("rv", 0),
            "adx": fb_data.get("adx", 0),
            "adr_pct": fb_data.get("adr_pct", 0),
            "hist": fb_data.get("hist", ""),
        }
        
        df = pd.DataFrame([row])
        header = not os.path.exists(FEEDBACK_PATH)
        df.to_csv(FEEDBACK_PATH, mode='a', header=header, index=False)
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")


# ============================================================
#  LOAD / TRAIN MODEL
# ============================================================
def load_model():
    """Carica modello salvato se esiste."""
    global model, stats
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
            stats["model_loaded"] = True
            stats["model_is_trained"] = True
            print(f"[MODEL] Caricato da {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[MODEL] Errore caricamento: {e}")
    return False


def train_model():
    """Addestra LightGBM con i dati accumulati."""
    global model, stats
    
    if not os.path.exists(FEEDBACK_PATH):
        return {"error": "Nessun dato feedback disponibile"}
    
    try:
        import joblib
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score
        
        # Carica feedback
        fb_df = pd.read_csv(FEEDBACK_PATH)
        if len(fb_df) < MIN_FEEDBACK_FOR_TRAIN:
            return {"error": f"Servono almeno {MIN_FEEDBACK_FOR_TRAIN} feedback, attuali: {len(fb_df)}"}
        
        # Carica richieste
        req_df = pd.read_csv(REQUESTS_PATH) if os.path.exists(REQUESTS_PATH) else pd.DataFrame()
        
        # Merge su timestamp più vicino (semplificato: merge su symbol+direction)
        # Per ora usiamo direttamente i feedback che contengono i dati
        df = fb_df.copy()
        
        # Feature engineering
        df["rv_abs"] = df["rv"].abs()
        df["adr_residual_pct"] = 100 - df["adr_pct"]
        
        feature_cols = ["rv", "adx", "adr_pct", "rv_abs", "adr_residual_pct",
                        "nm", "nm_accel", "nm_dist", "is_compressing"]
        
        # Solo righe con feature valide
        df = df.dropna(subset=feature_cols + ["won"])
        
        if len(df) < MIN_FEEDBACK_FOR_TRAIN:
            return {"error": f"Dati puliti insufficienti: {len(df)}"}
        
        X = df[feature_cols].values
        y = df["won"].astype(int).values
        
        # Check class balance
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        if pos_count < 5 or neg_count < 5:
            return {"error": f"Classi sbilanciate: won={pos_count}, lost={neg_count}"}
        
        # Train LightGBM
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "verbose": -1,
            "n_jobs": 1,
            "seed": 42,
        }
        
        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)],
        )
        
        # Salva modello
        import joblib
        joblib.dump(model, MODEL_PATH)
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importance().tolist()))
        
        stats["model_is_trained"] = True
        stats["model_version"] += 1
        stats["last_retrain_time"] = datetime.now(timezone.utc).isoformat()
        
        return {
            "status": "trained",
            "samples": len(df),
            "won": int(pos_count),
            "lost": int(neg_count),
            "win_rate": round(pos_count / len(y) * 100, 1),
            "features": importance,
            "version": stats["model_version"],
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  FLASK ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": stats["model_loaded"],
        "model_is_trained": stats["model_is_trained"],
        "model_version": stats["model_version"],
        "uptime_since": stats["started"],
        "total_predict": stats["total_predict_calls"],
        "total_feedback": stats["total_feedback_calls"],
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    """Statistiche dettagliate."""
    fb_count = 0
    req_count = 0
    if os.path.exists(FEEDBACK_PATH):
        try:
            fb_count = len(pd.read_csv(FEEDBACK_PATH))
        except:
            pass
    if os.path.exists(REQUESTS_PATH):
        try:
            req_count = len(pd.read_csv(REQUESTS_PATH))
        except:
            pass
    
    return jsonify({
        "server": stats,
        "data": {
            "feedback_rows": fb_count,
            "request_rows": req_count,
            "min_for_train": MIN_FEEDBACK_FOR_TRAIN,
            "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principale chiamato dall'EA.
    
    Input JSON dal EA:
    {
        "symbol": "EURUSD",
        "module": "STD",
        "direction": "BUY",
        "rv": 12.5,
        "adx": 22.1,
        "adr_pct": 45.2,
        ...
        "context": { "total": 28, "non_gray": 15, ... }
    }
    
    Output JSON:
    {
        "signal": "BUY",
        "confidence": 78,
        "method": "rules_v1"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0, "method": "error", "error": "No JSON"}), 200
        
        stats["total_predict_calls"] += 1
        stats["last_predict_time"] = datetime.now(timezone.utc).isoformat()
        
        direction = data.get("direction", "").upper()
        
        # --- Prepara features per modello ---
        ctx = data.get("context", {})
        adr_media = float(data.get("adr_media", 0))
        adr_pip = float(data.get("adr_pip", 0))
        adr_pct = float(data.get("adr_pct", 0))
        rv = float(data.get("rv", 0))
        rv_prev = float(data.get("rv_prev", 0))
        
        features_row = {
            "rv": rv,
            "adx": float(data.get("adx", 0)),
            "adr_pct": adr_pct,
            "adr_pip": adr_pip,
            "adr_media": adr_media,
            "ema_pos": int(data.get("ema_pos", 0)),
            "ema_gap_pct": float(data.get("ema_gap_pct", 0)),
            "rv_prev": rv_prev,
            "rv_prev2": float(data.get("rv_prev2", 0)),
            "light_streak": int(data.get("light_streak", 0)),
            "was_gray": 1 if data.get("was_gray", False) else 0,
            "hist_flip_bar": int(data.get("hist_flip_bar", 999)),
            "ctx_total": int(ctx.get("total", 0)),
            "ctx_non_gray": int(ctx.get("non_gray", 0)),
            "ctx_green": int(ctx.get("green", 0)),
            "ctx_red": int(ctx.get("red", 0)),
            "ctx_avg_abs_rv": float(ctx.get("avg_abs_rv", 0)),
            "ctx_extreme_rv": int(ctx.get("extreme_rv", 0)),
            "rv_decel": abs(rv_prev) - abs(rv) if abs(rv_prev) > 0 else 0,
            "adr_residual_pct": max(0, 100 - adr_pct),
            # Momentum metrics
            "nm": float(data.get("nm", 0)),
            "nm_signal": float(data.get("nm_signal", 0)),
            "nm_accel": float(data.get("nm_accel", 0)),
            "nm_dist": float(data.get("nm_dist", 0)),
            "is_compressing": 1 if data.get("is_compressing", False) else 0,
        }
        
        # --- Prova modello prima, poi regole ---
        signal = direction
        confidence = 0
        method = "rules_v1"
        
        if stats["model_is_trained"] and model is not None:
            try:
                features_df = pd.DataFrame([features_row])
                ml_signal, ml_conf = predict_with_model(features_df)
                if ml_conf > 0:
                    signal = ml_signal
                    confidence = ml_conf
                    method = f"lgbm_v{stats['model_version']}"
            except:
                pass
        
        if confidence == 0:
            # Fallback a regole esperte
            signal, confidence, details = rules_based_score(data)
            method = "rules_v1"
        
        # --- Costruisci risposta ---
        result = {
            "signal": signal,
            "confidence": confidence,
            "method": method,
            "symbol": data.get("symbol", ""),
            "direction_proposed": direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # --- Logga richiesta ---
        log_request(data, result)
        
        return jsonify(result)
        
    except Exception as e:
        stats["total_errors"] += 1
        traceback.print_exc()
        return jsonify({
            "signal": "HOLD",
            "confidence": 0,
            "method": "error",
            "error": str(e),
        }), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Riceve esito trade dall'EA per accumulo dati.
    
    Input:
    {
        "ticket": 12345,
        "symbol": "EURUSD",
        "direction": "BUY",
        "module": "STD",
        "entry_price": 1.08500,
        "exit_price": 1.08650,
        "profit": 1.50,
        "pips": 15,
        "won": true,
        "ai_confidence": 78,
        "ai_signal": "BUY",
        "rv": 12.5,
        "adx": 22.1,
        "adr_pct": 45.2,
        "hist": "GREEN_LIGHT"
    }
    """
    try:
        fb_data = request.get_json(force=True)
        if not fb_data:
            return jsonify({"status": "error", "message": "No JSON"}), 200
        
        stats["total_feedback_calls"] += 1
        log_feedback(fb_data)
        
        fb_count = 0
        if os.path.exists(FEEDBACK_PATH):
            try:
                fb_count = len(pd.read_csv(FEEDBACK_PATH))
            except:
                pass
        
        return jsonify({
            "status": "ok",
            "logged": True,
            "total_feedback": fb_count,
            "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Forza riaddestramento del modello.
    """
    result = train_model()
    return jsonify(result)


@app.route("/import_csv", methods=["POST"])
def import_csv():
    """
    Importa trade log da un URL (GitHub raw) e salva come feedback.
    
    Input JSON:
    {
        "url": "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/main/data/PRP_TradeLog.csv",
        "auto_train": true
    }
    
    Oppure passa i dati direttamente:
    {
        "trades": [
            {"symbol":"EURUSD","direction":"BUY","module":"STD","profit":1.5,"pips":15,"won":true,...},
            ...
        ]
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200
        
        auto_train = data.get("auto_train", False)
        imported = 0
        skipped = 0
        
        # --- Modalita' 1: Importa da URL ---
        csv_url = data.get("url", "")
        if csv_url:
            try:
                import urllib.request
                print(f"[IMPORT] Scaricando CSV da: {csv_url}")
                req = urllib.request.Request(csv_url, headers={"User-Agent": "ProfitRadarAI/1.0"})
                with urllib.request.urlopen(req, timeout=15) as response:
                    csv_content = response.read().decode("utf-8")
                
                if not csv_content or len(csv_content) < 50:
                    return jsonify({"status": "error", "message": "CSV vuoto o troppo piccolo"}), 200
                
                # --- Parse CSV (supporta sia ; che , come separatore) ---
                import io
                import csv as csv_module
                
                # Rileva separatore
                first_line = csv_content.split("\n")[0]
                sep = ";" if first_line.count(";") > first_line.count(",") else ","
                
                reader = csv_module.DictReader(io.StringIO(csv_content), delimiter=sep)
                
                existing_tickets = set()
                if os.path.exists(FEEDBACK_PATH):
                    try:
                        existing_df = pd.read_csv(FEEDBACK_PATH)
                        if "ticket" in existing_df.columns:
                            existing_tickets = set(existing_df["ticket"].astype(str).tolist())
                    except:
                        pass
                
                new_rows = []
                for row in reader:
                    # --- Cerca ticket (diversi possibili nomi colonna) ---
                    ticket = str(row.get("Ticket", row.get("ticket", row.get("Order", "")))).strip()
                    
                    # Salta se gia' presente
                    if ticket and ticket in existing_tickets:
                        skipped += 1
                        continue
                    
                    # --- Estrai dati con fallback per diversi formati ---
                    symbol = row.get("Symbol", row.get("symbol", ""))
                    direction = row.get("Direction", row.get("direction", row.get("Type", ""))).upper()
                    if "BUY" in direction:
                        direction = "BUY"
                    elif "SELL" in direction:
                        direction = "SELL"
                    
                    module = row.get("Module", row.get("module", "STD"))
                    profit_str = row.get("Profit", row.get("profit", row.get("Profit$", "0")))
                    pips_str = row.get("Pips", row.get("pips", "0"))
                    won_str = row.get("Won", row.get("won", ""))
                    
                    # Determina won da profit o da campo esplicito
                    try:
                        profit_val = float(str(profit_str).replace(",", "."))
                    except:
                        profit_val = 0
                    
                    if won_str in ("true", "True", "1", "TRUE"):
                        won = True
                    elif won_str in ("false", "False", "0", "FALSE"):
                        won = False
                    else:
                        won = profit_val > 0
                    
                    try:
                        pips_val = float(str(pips_str).replace(",", "."))
                    except:
                        pips_val = 0
                    
                    # Feature
                    rv = row.get("RV", row.get("rv", "0"))
                    adx = row.get("ADX", row.get("adx", "0"))
                    adr_pct = row.get("ADR%", row.get("adr_pct", row.get("ADR", "0")))
                    hist = row.get("Hist", row.get("hist", row.get("HistogramState", "")))
                    ai_conf = row.get("AI_Conf", row.get("ai_confidence", "0"))
                    ai_signal = row.get("AI_Signal", row.get("ai_signal", ""))
                    entry = row.get("EntryPrice", row.get("entry_price", "0"))
                    exit_prc = row.get("ExitPrice", row.get("exit_price", "0"))
                    open_time = row.get("OpenTime", row.get("open_time", ""))
                    close_time = row.get("CloseTime", row.get("close_time", ""))
                    
                    fb_row = {
                        "timestamp": close_time if close_time else datetime.now(timezone.utc).isoformat(),
                        "ticket": ticket,
                        "symbol": symbol,
                        "direction": direction,
                        "module": module,
                        "entry_price": float(str(entry).replace(",", ".") or "0"),
                        "exit_price": float(str(exit_prc).replace(",", ".") or "0"),
                        "profit": profit_val,
                        "pips": pips_val,
                        "won": won,
                        "ai_confidence": int(str(ai_conf) or "0"),
                        "ai_signal": str(ai_signal),
                        "rv": float(str(rv).replace(",", ".") or "0"),
                        "adx": float(str(adx).replace(",", ".") or "0"),
                        "adr_pct": float(str(adr_pct).replace(",", ".") or "0"),
                        "hist": str(hist),
                    }
                    
                    new_rows.append(fb_row)
                    imported += 1
                
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    header = not os.path.exists(FEEDBACK_PATH)
                    new_df.to_csv(FEEDBACK_PATH, mode="a", header=header, index=False)
                    print(f"[IMPORT] Salvati {imported} trade ({skipped} gia' esistenti)")
                
            except Exception as e:
                traceback.print_exc()
                return jsonify({"status": "error", "message": f"Errore download/parse: {str(e)}"}), 200
        
        # --- Modalita' 2: Importa da array JSON ---
        trades = data.get("trades", [])
        if trades:
            for t in trades:
                ticket = str(t.get("ticket", ""))
                
                existing_tickets = set()
                if os.path.exists(FEEDBACK_PATH):
                    try:
                        existing_df = pd.read_csv(FEEDBACK_PATH)
                        if "ticket" in existing_df.columns:
                            existing_tickets = set(existing_df["ticket"].astype(str).tolist())
                    except:
                        pass
                
                if ticket and ticket in existing_tickets:
                    skipped += 1
                    continue
                
                fb_row = {
                    "timestamp": t.get("close_time", datetime.now(timezone.utc).isoformat()),
                    "ticket": ticket,
                    "symbol": t.get("symbol", ""),
                    "direction": t.get("direction", ""),
                    "module": t.get("module", "STD"),
                    "entry_price": float(t.get("entry_price", 0)),
                    "exit_price": float(t.get("exit_price", 0)),
                    "profit": float(t.get("profit", 0)),
                    "pips": float(t.get("pips", 0)),
                    "won": bool(t.get("won", False)),
                    "ai_confidence": int(t.get("ai_confidence", 0)),
                    "ai_signal": t.get("ai_signal", ""),
                    "rv": float(t.get("rv", 0)),
                    "adx": float(t.get("adx", 0)),
                    "adr_pct": float(t.get("adr_pct", 0)),
                    "hist": t.get("hist", ""),
                }
                
                new_rows = new_rows if csv_url else []
                if not csv_url:
                    # Non sovrascrivere new_rows se gia' popolato da CSV
                    df_row = pd.DataFrame([fb_row])
                    header = not os.path.exists(FEEDBACK_PATH)
                    df_row.to_csv(FEEDBACK_PATH, mode="a", header=header, index=False)
                
                imported += 1
            
            if not csv_url:
                print(f"[IMPORT] Salvati {imported} trade da JSON ({skipped} gia' esistenti)")
        
        if imported == 0 and skipped == 0:
            return jsonify({"status": "error", "message": "Nessun dato da importare"}), 200
        
        # --- Conta totali ---
        total_feedback = 0
        if os.path.exists(FEEDBACK_PATH):
            try:
                total_feedback = len(pd.read_csv(FEEDBACK_PATH))
            except:
                pass
        
        result = {
            "status": "ok",
            "imported": imported,
            "skipped_duplicates": skipped,
            "total_feedback": total_feedback,
            "ready_to_train": total_feedback >= MIN_FEEDBACK_FOR_TRAIN,
        }
        
        # --- Auto-train se richiesto e pronto ---
        if auto_train and total_feedback >= MIN_FEEDBACK_FOR_TRAIN:
            train_result = train_model()
            result["train_result"] = train_result
            result["model_trained"] = train_result.get("status") == "trained"
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/train_from_github", methods=["POST"])
def train_from_github():
    """
    Importa CSV da GitHub e addestra il modello in un colpo solo.
    
    Input JSON:
    {
        "csv_url": "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/main/data/PRP_TradeLog.csv"
    }
    
    Se non viene specificato un URL, usa il default dal repo.
    """
    try:
        data = request.get_json(force=True) or {}
        
        default_url = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/main/data/PRP_TradeLog.csv"
        csv_url = data.get("csv_url", default_url)
        
        # Step 1: Importa
        import urllib.request
        print(f"[IMPORT] Scaricando CSV da: {csv_url}")
        req = urllib.request.Request(csv_url, headers={"User-Agent": "ProfitRadarAI/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                csv_content = response.read().decode("utf-8")
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Impossibile scaricare CSV: {str(e)}",
                "hint": "Verifica che il file esista su GitHub e il repo sia public",
            }), 200
        
        # Step 2: Salva anche come file raw sul server
        raw_path = os.path.join(DATA_DIR, "imported_tradelog.csv")
        with open(raw_path, "w") as f:
            f.write(csv_content)
        
        # Step 3: Converti nel formato feedback
        import io
        import csv as csv_module
        
        first_line = csv_content.split("\n")[0]
        sep = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv_module.DictReader(io.StringIO(csv_content), delimiter=sep)
        
        # Leggi feedback esistente per evitare duplicati
        existing_tickets = set()
        if os.path.exists(FEEDBACK_PATH):
            try:
                existing_df = pd.read_csv(FEEDBACK_PATH)
                if "ticket" in existing_df.columns:
                    existing_tickets = set(existing_df["ticket"].astype(str).tolist())
            except:
                pass
        
        new_rows = []
        for row in reader:
            ticket = str(row.get("Ticket", row.get("ticket", ""))).strip()
            if ticket and ticket in existing_tickets:
                continue
            
            profit_str = row.get("Profit", row.get("profit", "0"))
            try:
                profit_val = float(str(profit_str).replace(",", "."))
            except:
                profit_val = 0
            
            won_str = row.get("Won", row.get("won", ""))
            if won_str in ("true", "True", "1"):
                won = True
            elif won_str in ("false", "False", "0"):
                won = False
            else:
                won = profit_val > 0
            
            direction = row.get("Direction", row.get("direction", "")).upper()
            if "BUY" in direction: direction = "BUY"
            elif "SELL" in direction: direction = "SELL"
            
            pips_str = row.get("Pips", row.get("pips", "0"))
            try:
                pips_val = float(str(pips_str).replace(",", "."))
            except:
                pips_val = 0
            
            rv_str = row.get("RV", row.get("rv", "0"))
            adx_str = row.get("ADX", row.get("adx", "0"))
            adr_str = row.get("ADR%", row.get("adr_pct", "0"))
            
            fb_row = {
                "timestamp": row.get("CloseTime", row.get("close_time", datetime.now(timezone.utc).isoformat())),
                "ticket": ticket,
                "symbol": row.get("Symbol", row.get("symbol", "")),
                "direction": direction,
                "module": row.get("Module", row.get("module", "STD")),
                "entry_price": float(str(row.get("EntryPrice", row.get("entry_price", "0"))).replace(",", ".") or "0"),
                "exit_price": float(str(row.get("ExitPrice", row.get("exit_price", "0"))).replace(",", ".") or "0"),
                "profit": profit_val,
                "pips": pips_val,
                "won": won,
                "ai_confidence": int(str(row.get("AI_Conf", row.get("ai_confidence", "0"))) or "0"),
                "ai_signal": str(row.get("AI_Signal", row.get("ai_signal", ""))),
                "rv": float(str(rv_str).replace(",", ".") or "0"),
                "adx": float(str(adx_str).replace(",", ".") or "0"),
                "adr_pct": float(str(adr_str).replace(",", ".") or "0"),
                "hist": str(row.get("Hist", row.get("hist", ""))),
            }
            new_rows.append(fb_row)
        
        imported = len(new_rows)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            header = not os.path.exists(FEEDBACK_PATH)
            new_df.to_csv(FEEDBACK_PATH, mode="a", header=header, index=False)
        
        # Step 4: Conta totali
        total_feedback = 0
        if os.path.exists(FEEDBACK_PATH):
            try:
                total_feedback = len(pd.read_csv(FEEDBACK_PATH))
            except:
                pass
        
        # Step 5: Train se possibile
        train_result = None
        if total_feedback >= MIN_FEEDBACK_FOR_TRAIN:
            train_result = train_model()
        
        result = {
            "status": "ok",
            "csv_rows_found": imported + len(existing_tickets),
            "new_imported": imported,
            "total_feedback": total_feedback,
            "ready_to_train": total_feedback >= MIN_FEEDBACK_FOR_TRAIN,
        }
        
        if train_result:
            result["train_result"] = train_result
            result["model_trained"] = train_result.get("status") == "trained"
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 200


# ============================================================
#  EA STATUS & REMOTE CONFIG
# ============================================================

# --- Default EA config ---
DEFAULT_EA_CONFIG = {
    "aggressiveness": 2,
    "use_ai": True,
    "ai_min_conf": 70,
    "max_consec_loss": 2,
    "max_daily_loss": 3,
    "max_daily_profit": 3.0,
    "rv_max": 30,
    "adr_max": 60.0,
    "min_rr": 1.5,
    "breakout_on": True,
    "reversal_on": False,
    "fixed_lots": 0.01,
    "max_concurrent": 10,
}

ea_status = {
    "last_update": None,
    "balance": 0,
    "equity": 0,
    "open_trades": 0,
    "daily_pnl": 0,
    "daily_wins": 0,
    "daily_losses": 0,
    "consecutive_losses": 0,
    "ai_calls": 0,
    "ai_confirm": 0,
    "ai_reject": 0,
    "ai_errors": 0,
    "ai_missed_trades": 0,
    "warmup_ok": False,
    "warmup_last": None,
    "data_source": "",
    "cross_active": 0,
    "cross_total": 0,
    "daily_stopped": False,
    "account_currency": "EUR",
    "ea_version": "",
}


def load_ea_config():
    """Carica config EA da file."""
    if os.path.exists(EA_CONFIG_PATH):
        try:
            with open(EA_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            # Merge con defaults per parametri mancanti
            for k, v in DEFAULT_EA_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except:
            pass
    return dict(DEFAULT_EA_CONFIG)


def save_ea_config(cfg):
    """Salva config EA su file."""
    ensure_data_dir()
    with open(EA_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


@app.route("/ea_status", methods=["POST"])
def receive_ea_status():
    """L'EA manda i suoi stats ogni candela M15. Riceve indietro la config."""
    global ea_status
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        # Aggiorna status
        for key in ea_status:
            if key in data:
                ea_status[key] = data[key]
        ea_status["last_update"] = datetime.now(timezone.utc).isoformat()

        # Salva su file
        ensure_data_dir()
        with open(EA_STATUS_PATH, "w") as f:
            json.dump(ea_status, f, indent=2)

        # Ritorna la config corrente (l'EA la applica)
        cfg = load_ea_config()
        return jsonify({"status": "ok", "config": cfg})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/ea_config", methods=["GET"])
def get_ea_config():
    """Leggi la config EA corrente."""
    return jsonify(load_ea_config())


@app.route("/ea_config", methods=["POST"])
def update_ea_config():
    """Aggiorna parametri EA dalla dashboard."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        cfg = load_ea_config()

        # Aggiorna solo i campi validi
        updatable = [
            "aggressiveness", "use_ai", "ai_min_conf",
            "max_consec_loss", "max_daily_loss", "max_daily_profit",
            "rv_max", "adr_max", "min_rr",
            "breakout_on", "reversal_on",
            "fixed_lots", "max_concurrent",
        ]
        updated = []
        for key in updatable:
            if key in data:
                old_val = cfg.get(key)
                new_val = data[key]
                # Tipo corretto
                if key in ("use_ai", "breakout_on", "reversal_on"):
                    new_val = bool(new_val)
                elif key in ("adr_max", "max_daily_profit", "min_rr", "fixed_lots"):
                    new_val = float(new_val)
                else:
                    new_val = int(new_val)
                cfg[key] = new_val
                if old_val != new_val:
                    updated.append(f"{key}: {old_val} → {new_val}")

        save_ea_config(cfg)

        return jsonify({
            "status": "ok",
            "config": cfg,
            "updated": updated,
            "message": f"Aggiornati {len(updated)} parametri. L'EA li applicherà alla prossima candela M15."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """API per la dashboard web - tutti i dati in una chiamata."""
    # EA status
    ea = dict(ea_status)

    # Server stats
    srv = dict(stats)

    # Feedback count
    fb_count = 0
    trade_history = []
    if os.path.exists(FEEDBACK_PATH):
        try:
            fb_df = pd.read_csv(FEEDBACK_PATH)
            fb_count = len(fb_df)
            # Ultimi 20 trade
            recent = fb_df.tail(20).to_dict("records")
            for t in recent:
                t["profit"] = float(t.get("profit", 0))
                t["pips"] = float(t.get("pips", 0))
                t["won"] = bool(t.get("won", False))
                trade_history.append(t)
        except:
            pass

    # Config
    cfg = load_ea_config()

    return jsonify({
        "ea": ea,
        "server": srv,
        "config": cfg,
        "feedback_count": fb_count,
        "trade_history": trade_history,
        "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
    })


# ============================================================
#  GPT A/B TEST MODULE
# ============================================================

GPT_SYSTEM_PROMPT = """Sei un analista forex quantitativo esperto. Valuta trade candidate ricevendo dati tecnici completi.

Il trader ha queste caratteristiche storiche:
- Win rate: ~41%
- Account: piccolo ( EUR), lotto fisso 0.01
- Usa un EA su M15 con 3 moduli: Standard, Breakout, Reversal
- Il modello LightGBM interno ha imparato:
  - ADX alto (>50) + ADR estremo (>100%) = spesso LOSS
  - ADR basso (<55%) + RV moderato (5-25) = più probabile WIN
  - ADR > 77% = trade rischiosi, spesso rifiutati

Indicatori chiave:
- Radar Value (RV): forza/direzione trend. Estremo se |RV|>40
- ADX: forza trend. >40 forte, <20 debole
- ADR%: range giornaliero usato. <50% molto spazio, >100% esausto
- Histogram: GREEN_LIGHT=rialzo forte, RED_LIGHT=ribasso forte, GRAY=neutrale
- Normalized Momentum: positivo=rialzista, negativo=ribassista
- Compression: mercato pronto per breakout

REGOLE:
- Se i dati sono insufficienti o ambigui, dai confidenza bassa (<50)
- Non avere paura di dire HOLD se il trade non è chiaro
- Considera sempre il contesto: quanti cross sono attivi, quanti nella stessa direzione

Rispondi SOLO in JSON valido, nient'altro:
{"signal":"BUY" o "SELL" o "HOLD","confidence":0-100,"reasoning":"motivo in 1 frase"}"""


def call_gpt(data):
    """Chiama GPT-4o-mini per valutare un trade."""
    if not OPENAI_API_KEY:
        return {"signal": "HOLD", "confidence": 0, "reasoning": "API key non configurata", "error": True}

    try:
        import urllib.request

        # Costruisci il messaggio utente con tutti i dati
        rv = float(data.get("rv", 0))
        adx = float(data.get("adx", 0))
        adr_pct = float(data.get("adr_pct", 0))
        adr_pip = float(data.get("adr_pip", 0))
        adr_media = float(data.get("adr_media", 0))
        direction = data.get("direction", "BUY")
        module = data.get("module", "STD")
        hist = data.get("hist", "UNKNOWN")
        ema_pos = int(data.get("ema_pos", 0))
        symbol = data.get("symbol", "")
        nm = float(data.get("nm", 0))
        nm_accel = float(data.get("nm_accel", 0))
        nm_dist = float(data.get("nm_dist", 0))
        is_compress = data.get("is_compressing", False)

        ctx = data.get("context", {})
        ctx_total = int(ctx.get("total", 0))
        ctx_active = int(ctx.get("non_gray", 0))
        ctx_green = int(ctx.get("green", 0))
        ctx_red = int(ctx.get("red", 0))

        user_msg = f"""Valuta questo trade:
Simbolo: {symbol}
Direzione proposta: {direction}
Modulo: {module}
Radar Value: {rv}
ADX(14): {adx}
ADR%: {adr_pct}% ({adr_pip} pip fatti su {adr_media} media)
Histogram: {hist}
EMA Position: {ema_pos} ({'rialzista' if ema_pos == 1 else 'ribassista'})
Normalized Momentum: {nm}
Momentum Acceleration: {nm_accel}
Momentum Distance: {nm_dist}
Compression: {'Sì' if is_compress else 'No'}
Contesto: {ctx_active}/{ctx_total} cross attivi, {ctx_green} green, {ctx_red} red"""

        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "response_format": {"type": "json_object"}
        }

        payload_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"]
        gpt_response = json.loads(content)

        return {
            "signal": gpt_response.get("signal", "HOLD").upper(),
            "confidence": min(100, max(0, int(gpt_response.get("confidence", 0)))),
            "reasoning": gpt_response.get("reasoning", ""),
            "model": GPT_MODEL,
            "error": False
        }

    except Exception as e:
        print(f"[GPT ERROR] {e}")
        traceback.print_exc()
        return {"signal": "HOLD", "confidence": 0, "reasoning": str(e), "error": True}


@app.route("/predict_gpt", methods=["POST"])
def predict_gpt():
    """Endpoint A/B test: chiama GPT e logga il risultato."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0, "reasoning": "No JSON"}), 200

        # Chiama GPT
        gpt_result = call_gpt(data)

        # Logga il risultato A/B
        ensure_data_dir()
        ab_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": data.get("symbol", ""),
            "direction": data.get("direction", ""),
            "module": data.get("module", ""),
            "rv": data.get("rv", 0),
            "adx": data.get("adx", 0),
            "adr_pct": data.get("adr_pct", 0),
            "hist": data.get("hist", ""),
            "lgbm_signal": data.get("direction", ""),
            "lgbm_conf": data.get("ai_confidence", 0),
            "gpt_signal": gpt_result.get("signal", ""),
            "gpt_conf": gpt_result.get("confidence", 0),
            "gpt_reasoning": gpt_result.get("reasoning", ""),
            "agreement": "SAME" if data.get("direction", "").upper() == gpt_result.get("signal", "").upper() else "DIFF",
        }

        df_row = pd.DataFrame([ab_row])
        header = not os.path.exists(AB_RESULTS_PATH)
        df_row.to_csv(AB_RESULTS_PATH, mode="a", header=header, index=False)

        return jsonify(gpt_result)

    except Exception as e:
        return jsonify({"signal": "HOLD", "confidence": 0, "reasoning": str(e), "error": True}), 200


@app.route("/ab_stats", methods=["GET"])
def ab_stats():
    """Statistiche A/B test."""
    result = {
        "total": 0,
        "agreement_same": 0,
        "agreement_diff": 0,
        "agreement_pct": 0,
        "gpt_enabled": bool(OPENAI_API_KEY),
        "gpt_model": GPT_MODEL,
        "gpt_key_preview": OPENAI_API_KEY[:8] + "..." if len(OPENAI_API_KEY) > 8 else "",
    }

    if not os.path.exists(AB_RESULTS_PATH):
        return jsonify(result)

    try:
        df = pd.read_csv(AB_RESULTS_PATH)
        total = len(df)
        same = len(df[df["agreement"] == "SAME"])
        diff = len(df[df["agreement"] == "DIFF"])

        stats = {
            "total": total,
            "agreement_same": same,
            "agreement_diff": diff,
            "agreement_pct": round(same / total * 100, 1) if total > 0 else 0,
            "gpt_enabled": bool(OPENAI_API_KEY),
            "gpt_model": GPT_MODEL,
        }

        # Se abbiamo colonne won nei risultati (dopo feedback)
        if "won" in df.columns:
            won_df = df.dropna(subset=["won"])
            if len(won_df) > 0:
                # GPT agree + trade won
                agree_won = len(won_df[(won_df["agreement"] == "SAME") & (won_df["won"] == True)])
                agree_total = len(won_df[won_df["agreement"] == "SAME"])
                # GPT disagree + trade won
                disagree_won = len(won_df[(won_df["agreement"] == "DIFF") & (won_df["won"] == True)])
                disagree_total = len(won_df[won_df["agreement"] == "DIFF"])

                stats["agree_winrate"] = round(agree_won / agree_total * 100, 1) if agree_total > 0 else 0
                stats["disagree_winrate"] = round(disagree_won / disagree_total * 100, 1) if disagree_total > 0 else 0
                stats["feedback_received"] = len(won_df)

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/ab_feedback", methods=["POST"])
def ab_feedback():
    """Aggiorna i risultati A/B con l'esito del trade."""
    try:
        data = request.get_json(force=True)
        if not data or not os.path.exists(AB_RESULTS_PATH):
            return jsonify({"status": "error", "message": "No data or no AB file"}), 200

        symbol = data.get("symbol", "")
        direction = data.get("direction", "")
        timestamp_approx = data.get("timestamp", "")

        df = pd.read_csv(AB_RESULTS_PATH)

        # Trova la riga più recente per questo simbolo+direzione
        mask = (df["symbol"] == symbol) & (df["direction"] == direction)
        candidates = df[mask]

        if len(candidates) == 0:
            return jsonify({"status": "not_found"}), 200

        # Prendi l'ultima riga
        last_idx = candidates.index[-1]

        df.loc[last_idx, "won"] = data.get("won", False)
        df.loc[last_idx, "profit"] = data.get("profit", 0)
        df.loc[last_idx, "pips"] = data.get("pips", 0)

        df.to_csv(AB_RESULTS_PATH, index=False)

        return jsonify({"status": "ok", "updated": True})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


# ============================================================
#  DASHBOARD HTML
# ============================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Profit Radar Pro - Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a1a;color:#e0e0e0;min-height:100vh}
.container{max-width:960px;margin:0 auto;padding:12px}
h1{text-align:center;font-size:1.3em;padding:12px 0;color:#4fc3f7;border-bottom:1px solid #1a1a3a;margin-bottom:10px}
h1 span{color:#81c784}
.section{background:#12122a;border-radius:10px;padding:14px;margin-bottom:12px;border:1px solid #1e1e40}
.section h2{font-size:0.95em;color:#4fc3f7;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.section h2::before{content:'';width:4px;height:16px;background:#4fc3f7;border-radius:2px}
.row{display:flex;flex-wrap:wrap;gap:8px}
.card{flex:1;min-width:140px;background:#1a1a35;border-radius:8px;padding:10px;text-align:center}
.card .val{font-size:1.6em;font-weight:700;line-height:1.3}
.card .lbl{font-size:0.7em;color:#888;text-transform:uppercase;margin-top:2px}
.green{color:#81c784}.red{color:#ef5350}.yellow{color:#ffd54f}.blue{color:#4fc3f7}.white{color:#e0e0e0}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot-green{background:#81c784;box-shadow:0 0 6px #81c784}
.dot-red{background:#ef5350;box-shadow:0 0 6px #ef5350}
.dot-yellow{background:#ffd54f;box-shadow:0 0 6px #ffd54f}
.dot-gray{background:#666}
table{width:100%;border-collapse:collapse;font-size:0.78em}
th{text-align:left;padding:6px 8px;background:#1a1a35;color:#888;text-transform:uppercase;font-size:0.85em;border-bottom:1px solid #2a2a50}
td{padding:5px 8px;border-bottom:1px solid #15152a}
tr:hover{background:#1a1a35}
.btn{display:inline-block;padding:8px 16px;border:none;border-radius:6px;cursor:pointer;font-size:0.85em;font-weight:600;transition:all .2s}
.btn:hover{transform:translateY(-1px);opacity:0.9}
.btn-green{background:#2e7d32;color:#fff}.btn-blue{background:#1565c0;color:#fff}
.btn-red{background:#b71c1c;color:#fff}.btn-gray{background:#333;color:#ccc}
.btn-yellow{background:#f57f17;color:#fff}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.config-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:8px}
.cfg-item{background:#1a1a35;border-radius:6px;padding:10px}
.cfg-item label{display:block;font-size:0.75em;color:#888;margin-bottom:4px;text-transform:uppercase}
.cfg-item input,.cfg-item select{width:100%;padding:6px 8px;background:#0a0a1a;border:1px solid #2a2a50;border-radius:4px;color:#e0e0e0;font-size:0.9em}
.cfg-item input[type=checkbox]{width:auto}
.refresh-bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:0.8em;color:#666}
@media(max-width:600px){.card{min-width:100px}.card .val{font-size:1.3em}.config-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<h1>📡 Profit Radar <span>Pro</span> — Dashboard</h1>

<div class="refresh-bar">
  <span id="lastUpdate">Caricamento...</span>
  <span><span class="status-dot dot-gray" id="eaDot"></span><span id="eaStatus">-</span></span>
</div>

<!-- ACCOUNT -->
<div class="section">
<h2>Account</h2>
<div class="row">
  <div class="card"><div class="val white" id="balance">-</div><div class="lbl">Balance EUR</div></div>
  <div class="card"><div class="val white" id="equity">-</div><div class="lbl">Equity EUR</div></div>
  <div class="card"><div class="val" id="dailyPnl">-</div><div class="lbl">P&L Oggi</div></div>
  <div class="card"><div class="val" id="openTrades">-</div><div class="lbl">Trade Aperti</div></div>
</div>
</div>

<!-- AI STATS -->
<div class="section">
<h2>AI Engine</h2>
<div class="row">
  <div class="card"><div class="val blue" id="aiCalls">-</div><div class="lbl">Chiamate AI</div></div>
  <div class="card"><div class="val green" id="aiConfirm">-</div><div class="lbl">Confermati</div></div>
  <div class="card"><div class="val yellow" id="aiReject">-</div><div class="lbl">Rifiutati</div></div>
  <div class="card"><div class="val red" id="aiErrors">-</div><div class="lbl">Errori</div></div>
  <div class="card"><div class="val red" id="aiMissed">-</div><div class="lbl">Trade Persi</div></div>
</div>
</div>

<!-- MERCATO -->
<div class="section">
<h2>Mercato</h2>
<div class="row">
  <div class="card"><div class="val white" id="crossTotal">-</div><div class="lbl">Cross Totali</div></div>
  <div class="card"><div class="val blue" id="crossActive">-</div><div class="lbl">Cross Attivi</div></div>
  <div class="card"><div class="val" id="dailyWL">-</div><div class="lbl">W / L Oggi</div></div>
  <div class="card"><div class="val" id="consecLoss">-</div><div class="lbl">Loss Consec.</div></div>
</div>
</div>

<!-- CONFIGURAZIONE EA -->
<div class="section">
<h2>Configurazione EA</h2>
<div class="config-grid">
  <div class="cfg-item">
    <label>Aggressivita'</label>
    <select id="cfgAggr" onchange="updateConfig()">
      <option value="1">1 - Conservativo</option>
      <option value="2" selected>2 - Moderato</option>
      <option value="3">3 - Aggressivo</option>
    </select>
  </div>
  <div class="cfg-item">
    <label>AI Attiva</label>
    <select id="cfgAI" onchange="updateConfig()">
      <option value="true">SI - Obbligatoria</option>
      <option value="false">NO - Disattivata</option>
    </select>
  </div>
  <div class="cfg-item">
    <label>Confidenza minima %</label>
    <input type="number" id="cfgMinConf" value="70" min="50" max="95" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>Max loss consecutivi</label>
    <input type="number" id="cfgMaxCLoss" value="2" min="1" max="5" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>Max loss giornalieri</label>
    <input type="number" id="cfgMaxDLoss" value="3" min="1" max="8" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>Max profitto giornaliero %</label>
    <input type="number" id="cfgMaxProf" value="3.0" min="1" max="10" step="0.5" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>RV massimo</label>
    <input type="number" id="cfgRVMax" value="30" min="10" max="50" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>ADR% massimo</label>
    <input type="number" id="cfgADRMax" value="60" min="30" max="90" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>R:R minimo</label>
    <input type="number" id="cfgMinRR" value="1.5" min="1.0" max="3.0" step="0.1" onchange="updateConfig()">
  </div>
  <div class="cfg-item">
    <label>Breakout</label>
    <select id="cfgBrk" onchange="updateConfig()">
      <option value="true">Attivo</option>
      <option value="false">Disattivo</option>
    </select>
  </div>
  <div class="cfg-item">
    <label>Reversal</label>
    <select id="cfgRev" onchange="updateConfig()">
      <option value="true">Attivo</option>
      <option value="false" selected>Disattivo</option>
    </select>
  </div>
</div>
<div class="btn-row">
  <button class="btn btn-blue" onclick="saveAllConfig()">💾 Salva Configurazione</button>
  <span id="cfgMsg" style="color:#81c784;font-size:0.8em;align-self:center"></span>
</div>
</div>

<!-- TRADE HISTORY -->
<div class="section">
<h2>Ultimi 20 Trade</h2>
<div style="overflow-x:auto">
<table>
<thead><tr><th>Simbolo</th><th>Dir</th><th>Modulo</th><th>Pips</th><th>Profitto</th><th>Risultato</th><th>AI Conf</th></tr></thead>
<tbody id="tradeTable"><tr><td colspan="7" style="text-align:center;color:#666">Nessun trade</td></tr></tbody>
</table>
</div>
</div>

<!-- A/B TEST GPT vs LIGHTGBM -->
<div class="section">
<h2>🧪 A/B Test: LightGBM vs GPT</h2>
<div class="row">
  <div class="card"><div class="val blue" id="abTotal">-</div><div class="lbl">Trade Testati</div></div>
  <div class="card"><div class="val green" id="abAgree">-</div><div class="lbl">D'Accordo</div></div>
  <div class="card"><div class="val red" id="abDisagree">-</div><div class="lbl">In Disaccordo</div></div>
  <div class="card"><div class="val" id="abAgreePct">-</div><div class="lbl">% Accordo</div></div>
</div>
<div class="row" style="margin-top:8px">
  <div class="card"><div class="val" id="abGptStatus">-</div><div class="lbl">GPT Status</div></div>
  <div class="card"><div class="val white" id="abModel">-</div><div class="lbl">Modello</div></div>
</div>
</div>

<!-- AZIONI -->
<div class="section">
<h2>Azioni</h2>
<div class="btn-row">
  <button class="btn btn-green" onclick="retrain()">🔄 Riaddestra Modello</button>
  <button class="btn btn-blue" onclick="trainGithub()">📥 Importa da GitHub + Train</button>
  <button class="btn btn-yellow" onclick="loadAB()">📊 Carica A/B Stats</button>
  <button class="btn btn-gray" onclick="refresh()">🔃 Aggiorna Adesso</button>
</div>
<div id="actionMsg" style="margin-top:8px;font-size:0.8em;color:#ffd54f"></div>
</div>

<div style="text-align:center;padding:16px 0;font-size:0.7em;color:#444">
  Profit Radar Pro v3.0 — Giovanni Mori — Account 22157346
</div>
</div>

<script>
const API=window.location.origin;
let pendingConfig={};

function fmt(v,d=2){return v!=null?v.toFixed(d):'-'}
function pnlClass(v){return v>0?'green':v<0?'red':'white'}

function refresh(){
  fetch(API+'/dashboard_data').then(r=>r.json()).then(d=>{
    const ea=d.ea,srv=d.server,cfg=d.config;

    // Connection status
    const dot=document.getElementById('eaDot');
    const age=ea.last_update?((Date.now()-new Date(ea.last_update))/1000/60):999;
    if(age<5){dot.className='status-dot dot-green';document.getElementById('eaStatus').textContent='EA Connesso'}
    else if(age<30){dot.className='status-dot dot-yellow';document.getElementById('eaStatus').textContent='EA connesso '+Math.round(age)+'m fa'}
    else{dot.className='status-dot dot-gray';document.getElementById('eaStatus').textContent='EA offline'}

    document.getElementById('lastUpdate').textContent='Aggiornato: '+new Date().toLocaleTimeString('it-IT');

    // Account
    document.getElementById('balance').textContent=fmt(ea.balance);
    document.getElementById('equity').textContent=fmt(ea.equity);
    const pnl=ea.daily_pnl||0;
    const pnlEl=document.getElementById('dailyPnl');
    pnlEl.textContent=(pnl>=0?'+':'')+fmt(pnl);
    pnlEl.className='val '+pnlClass(pnl);
    document.getElementById('openTrades').textContent=ea.open_trades+'/'+(cfg.max_concurrent||10);

    // AI
    document.getElementById('aiCalls').textContent=ea.ai_calls||0;
    document.getElementById('aiConfirm').textContent=ea.ai_confirm||0;
    document.getElementById('aiReject').textContent=ea.ai_reject||0;
    document.getElementById('aiErrors').textContent=ea.ai_errors||0;
    const missEl=document.getElementById('aiMissed');
    missEl.textContent=ea.ai_missed_trades||0;
    missEl.className='val '+(ea.ai_missed_trades>0?'red':'green');

    // Market
    document.getElementById('crossTotal').textContent=ea.cross_total||0;
    document.getElementById('crossActive').textContent=ea.cross_active||0;
    document.getElementById('dailyWL').textContent=(ea.daily_wins||0)+' / '+(ea.daily_losses||0);
    document.getElementById('dailyWL').className='val';
    const clEl=document.getElementById('consecLoss');
    clEl.textContent=ea.consecutive_losses||0;
    clEl.className='val '+((ea.consecutive_losses||0)>=2?'red':'white');

    // Config form
    document.getElementById('cfgAggr').value=cfg.aggressiveness||2;
    document.getElementById('cfgAI').value=cfg.use_ai?'true':'false';
    document.getElementById('cfgMinConf').value=cfg.ai_min_conf||70;
    document.getElementById('cfgMaxCLoss').value=cfg.max_consec_loss||2;
    document.getElementById('cfgMaxDLoss').value=cfg.max_daily_loss||3;
    document.getElementById('cfgMaxProf').value=cfg.max_daily_profit||3.0;
    document.getElementById('cfgRVMax').value=cfg.rv_max||30;
    document.getElementById('cfgADRMax').value=cfg.adr_max||60;
    document.getElementById('cfgMinRR').value=cfg.min_rr||1.5;
    document.getElementById('cfgBrk').value=cfg.breakout_on?'true':'false';
    document.getElementById('cfgRev').value=cfg.reversal_on?'true':'false';

    // Trade history
    const tb=document.getElementById('tradeTable');
    if(d.trade_history&&d.trade_history.length>0){
      tb.innerHTML=d.trade_history.reverse().map(t=>{
        const p=t.profit||0;
        const w=t.won;
        return '<tr>'+
          '<td>'+(t.symbol||'-')+'</td>'+
          '<td>'+(t.direction||'-')+'</td>'+
          '<td>'+(t.module||'-')+'</td>'+
          '<td>'+fmt(t.pips,1)+'</td>'+
          '<td class="'+pnlClass(p)+'">'+(p>=0?'+':'')+fmt(p)+'€</td>'+
          '<td><span style="color:'+(w?'#81c784':'#ef5350')+'">'+(w?'WIN':'LOSS')+'</span></td>'+
          '<td>'+(t.ai_confidence||'-')+'%</td>'+
          '</tr>'
      }).join('');
    }

  }).catch(e=>console.error('Fetch error:',e));
}

function updateConfig(){
  // Just store pending, don't save yet
}

function saveAllConfig(){
  const cfg={
    aggressiveness:parseInt(document.getElementById('cfgAggr').value),
    use_ai:document.getElementById('cfgAI').value==='true',
    ai_min_conf:parseInt(document.getElementById('cfgMinConf').value),
    max_consec_loss:parseInt(document.getElementById('cfgMaxCLoss').value),
    max_daily_loss:parseInt(document.getElementById('cfgMaxDLoss').value),
    max_daily_profit:parseFloat(document.getElementById('cfgMaxProf').value),
    rv_max:parseInt(document.getElementById('cfgRVMax').value),
    adr_max:parseFloat(document.getElementById('cfgADRMax').value),
    min_rr:parseFloat(document.getElementById('cfgMinRR').value),
    breakout_on:document.getElementById('cfgBrk').value==='true',
    reversal_on:document.getElementById('cfgRev').value==='true',
  };
  fetch(API+'/ea_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
    .then(r=>r.json()).then(d=>{
      const msg=document.getElementById('cfgMsg');
      if(d.status==='ok'){
        msg.textContent='✅ '+d.message;
        msg.style.color='#81c784';
      }else{
        msg.textContent='❌ Errore: '+(d.message||'unknown');
        msg.style.color='#ef5350';
      }
      setTimeout(()=>{msg.textContent=''},5000);
    }).catch(e=>{
      document.getElementById('cfgMsg').textContent='❌ Errore connessione';
      document.getElementById('cfgMsg').style.color='#ef5350';
    });
}

function retrain(){
  document.getElementById('actionMsg').textContent='⏳ Riaddestramento in corso...';
  fetch(API+'/retrain',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.status==='trained'){
      document.getElementById('actionMsg').innerHTML='✅ Modello addestrato! Samples: '+d.samples+
        ' | Win rate: '+d.win_rate+'% | Versione: '+d.version;
    }else{
      document.getElementById('actionMsg').innerHTML='⚠️ '+(d.error||'Non abbastanza dati');
    }
  }).catch(e=>{document.getElementById('actionMsg').textContent='❌ Errore'});
}

function trainGithub(){
  document.getElementById('actionMsg').textContent='⏳ Importazione da GitHub + Training...';
  fetch(API+'/train_from_github',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
    .then(r=>r.json()).then(d=>{
      let msg='✅ Importati '+d.new_imported+' trade | Totale: '+d.total_feedback;
      if(d.train_result)msg+=' | Train: '+JSON.stringify(d.train_result).substring(0,100);
      else if(d.ready_to_train)msg+=' | Pronto per training!';
      document.getElementById('actionMsg').innerHTML=msg;
    }).catch(e=>{document.getElementById('actionMsg').textContent='❌ Errore'});
}

function loadAB(){
  fetch(API+'/ab_stats').then(r=>r.json()).then(d=>{
    document.getElementById('abTotal').textContent=d.total||0;
    document.getElementById('abAgree').textContent=d.agreement_same||0;
    document.getElementById('abDisagree').textContent=d.agreement_diff||0;
    const pct=d.agreement_pct||0;
    const el=document.getElementById('abAgreePct');
    el.textContent=pct+'%';
    el.className='val '+(pct>70?'green':pct>50?'yellow':'red');
    document.getElementById('abGptStatus').textContent=d.gpt_enabled?'🟢 Attivo':'🔴 No API Key';
    document.getElementById('abGptStatus').className='val '+(d.gpt_enabled?'green':'red');
    document.getElementById('abModel').textContent=d.gpt_model||'-';
    document.getElementById('actionMsg').innerHTML='✅ A/B stats aggiornati: '+d.total+' trade testati';
  }).catch(e=>{document.getElementById('actionMsg').textContent='❌ Errore A/B stats'});
}

// Auto-refresh ogni 30 secondi
refresh();
setInterval(refresh,30000);
</script>
</body>
</html>"""


@app.route("/diag", methods=["GET"])
def diag():
    """Diagnostica completa — verifica configurazione server."""
    return jsonify({
        "status": "ok",
        "openai_key_set": bool(OPENAI_API_KEY),
        "openai_key_preview": OPENAI_API_KEY[:10] + "..." if len(OPENAI_API_KEY) > 10 else "VUOTA",
        "openai_key_length": len(OPENAI_API_KEY),
        "gpt_model": GPT_MODEL,
        "data_dir": os.path.abspath(DATA_DIR),
        "files_in_data": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else [],
        "ea_status_exists": os.path.exists(EA_STATUS_PATH),
        "ea_status": json.load(open(EA_STATUS_PATH)) if os.path.exists(EA_STATUS_PATH) else None,
        "uptime": stats["started"],
        "total_predict": stats["total_predict_calls"],
        "total_feedback": stats["total_feedback_calls"],
    })


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Dashboard web completa."""
    from flask import Response
    return Response(DASHBOARD_HTML, mimetype="text/html")


# ============================================================
#  INIT
# ============================================================
#  INIT — eseguito direttamente all'avvio del modulo
# ============================================================
ensure_data_dir()
load_model()
print(f"[INIT] Server avviato | Data dir: {DATA_DIR}")
print(f"[INIT] Modello: {'LOADED' if stats['model_is_trained'] else 'REGOLE (nessun modello)'}")
print(f"[INIT] Min feedback per training: {MIN_FEEDBACK_FOR_TRAIN}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[START] Profit Radar Pro AI Server on port {port}")
    ensure_data_dir()
    load_model()
    app.run(host="0.0.0.0", port=port, debug=False)
