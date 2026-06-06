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
MIN_FEEDBACK_FOR_TRAIN = int(os.environ.get("MIN_FEEDBACK_FOR_TRAIN", "50"))

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
    # Feature derivata
    "rv_decel", "adr_residual_pct",
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
        
        feature_cols = ["rv", "adx", "adr_pct", "rv_abs", "adr_residual_pct"]
        
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
