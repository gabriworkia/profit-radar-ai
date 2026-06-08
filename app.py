"""
Profit Radar Pro — AI Server v4.0
===================================
Server Flask con LightGBM + GPT A/B test per conferma trade EA MT4.

Deploy: Render.com (free tier)
Repo: https://github.com/gabriworkia/profit-radar-ai
"""

import os
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ============================================================
#  CONFIGURAZIONE
# ============================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
REQUESTS_PATH = os.path.join(DATA_DIR, "requests_log.csv")
EA_CONFIG_PATH = os.path.join(DATA_DIR, "ea_config.json")
EA_STATUS_PATH = os.path.join(DATA_DIR, "ea_status.json")
AB_RESULTS_PATH = os.path.join(DATA_DIR, "ab_results.csv")

# GitHub URLs per ripristino post-deploy
GITHUB_AB_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/main/Data/ab_results.csv"
GITHUB_REQUESTS_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/main/Data/requests_log.csv"

MIN_FEEDBACK_FOR_TRAIN = int(os.environ.get("MIN_FEEDBACK_FOR_TRAIN", "50"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5-nano"

GITHUB_CSV_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/main/Data/PRP_TradeLog.csv"

# ============================================================
#  APP
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
#  STATS IN MEMORIA
# ============================================================
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

# ============================================================
#  MODELLO
# ============================================================
model = None
feature_names = [
    "rv", "adx", "adr_pct", "adr_pip", "adr_media",
    "ema_pos", "ema_gap_pct",
    "rv_prev", "rv_prev2", "light_streak", "was_gray", "hist_flip_bar",
    "ctx_total", "ctx_non_gray", "ctx_green", "ctx_red",
    "ctx_avg_abs_rv", "ctx_extreme_rv",
    "rv_decel", "adr_residual_pct",
    "nm", "nm_signal", "nm_accel", "nm_dist", "is_compressing",
]


# ============================================================
#  UTILITIES
# ============================================================
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_float(s, default=0.0):
    """Converte in float in modo sicuro."""
    try:
        return float(str(s).replace(",", ".") or str(default))
    except:
        return default


# ============================================================
#  RULES-BASED SCORER (fallback se modello non trainato)
# ============================================================
def rules_based_score(data):
    """Sistema a punteggio basato su regole esperte. Range 0-100."""
    score = 50

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
    module = data.get("module", "STD").upper()

    nm = float(data.get("nm", 0))
    nm_accel = float(data.get("nm_accel", 0))
    nm_dist = float(data.get("nm_dist", 0))
    is_compressing = data.get("is_compressing", False)

    # EMA concorde (+8)
    if direction == "BUY" and ema_pos == 1: score += 8
    elif direction == "SELL" and ema_pos == -1: score += 8

    # RV moderato (+10 sweet spot)
    abs_rv = abs(rv)
    if 5 <= abs_rv <= 15: score += 10
    elif 15 < abs_rv <= 25: score += 7
    elif 25 < abs_rv <= 35: score += 3
    elif abs_rv > 50: score -= 5

    # ADX nella zona giusta (+8)
    if 15 <= adx <= 25: score += 8
    elif 25 < adx <= 40: score += 5
    elif adx > 50: score -= 3

    # ADR con spazio residuo (+8)
    if adr_pct < 40: score += 8
    elif adr_pct < 55: score += 5
    elif adr_pct > 80: score -= 10

    # Residuo ADR
    if adr_media > 0:
        residual_pct = (adr_media - adr_pip) / adr_media * 100
        if residual_pct > 50: score += 5
        elif residual_pct < 20: score -= 8

    # EMA gap
    if ema_gap_pct < 0.10: score += 5
    elif ema_gap_pct > 0.30: score -= 3

    # Histogram
    if "LIGHT" in hist: score += 5
    elif "DARK" in hist: score += 2
    elif hist == "GRAY": score -= 5

    # Module-specific
    was_gray = data.get("was_gray", False)
    light_streak = int(data.get("light_streak", 0))
    if module == "BRK" and was_gray: score += 8
    if module == "BRK" and light_streak >= 2: score += 5
    if module == "BRK" and light_streak > 5: score -= 5

    hist_flip_bar = int(data.get("hist_flip_bar", 999))
    if module == "REV":
        if abs(rv_prev) > 0:
            decel = abs(rv_prev) - abs_rv
            if decel > 10: score += 10
            elif decel > 5: score += 5
            elif decel < 0: score -= 5
        if hist_flip_bar <= 2: score += 8
        elif hist_flip_bar <= 5: score += 3
        if adx >= 40: score += 5
        if adr_pct >= 80: score += 5

    # Momentum
    if direction == "BUY" and nm > 0: score += 8
    elif direction == "SELL" and nm < 0: score += 8
    elif direction == "BUY" and nm < -0.3: score -= 10
    elif direction == "SELL" and nm > 0.3: score -= 10

    if direction == "BUY" and nm_accel > 0: score += 6
    elif direction == "SELL" and nm_accel < 0: score += 6

    if direction == "BUY" and nm > float(data.get("nm_signal", 0)): score += 5
    elif direction == "SELL" and nm < float(data.get("nm_signal", 0)): score += 5

    if is_compressing:
        if module == "BRK": score += 7
        else: score -= 3

    if nm_dist > 0.5:
        if (direction == "BUY" and nm > 0) or (direction == "SELL" and nm < 0): score += 3
    if nm_dist < 0.15 and not is_compressing: score -= 4

    # Contesto
    ctx = data.get("context", {})
    ctx_green = int(ctx.get("green", 0))
    ctx_red = int(ctx.get("red", 0))
    ctx_extreme = int(ctx.get("extreme_rv", 0))
    if direction == "BUY" and ctx_green > 10: score += 3
    elif direction == "SELL" and ctx_red > 10: score += 3
    if ctx_extreme > 5: score -= 5

    score = max(0, min(100, score))
    return direction, score, {"base": 50, "final": score, "method": "rules_v1"}


TRAIN_FEATURES = None  # Salvate al momento del training

# ============================================================
#  PREDICT CON MODELLO
# ============================================================
def predict_with_model(features_df):
    global model, TRAIN_FEATURES
    if model is None:
        return None, 0
    try:
        # Aggiungi features derivate se mancanti
        if "rv_abs" not in features_df.columns and "rv" in features_df.columns:
            features_df["rv_abs"] = features_df["rv"].abs()
        if "adr_residual_pct" not in features_df.columns and "adr_pct" in features_df.columns:
            features_df["adr_residual_pct"] = (100 - features_df["adr_pct"]).clip(lower=0)
        
        # Usa le features con cui il modello è stato addestrato
        if TRAIN_FEATURES is not None:
            cols = [f for f in TRAIN_FEATURES if f in features_df.columns]
        else:
            cols = [f for f in feature_names if f in features_df.columns]
        
        if not cols:
            return None, 0
        
        # LightGBM Booster usa predict() non predict_proba()
        import lightgbm as lgb
        raw_pred = model.predict(features_df[cols])
        if raw_pred is None or len(raw_pred) == 0:
            return None, 0
        
        proba = float(raw_pred[0])  # probabilità classe positiva [0..1]
        confidence = int(proba * 100)
        signal = "BUY" if proba >= 0.5 else "SELL"
        return signal, confidence
    except Exception as e:
        print(f"[MODEL ERROR] {e}")
        return None, 0


# ============================================================
#  LOGGING
# ============================================================
def log_request(data, result):
    ensure_data_dir()
    try:
        ctx = data.get("context", {})
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": data.get("symbol", ""),
            "module": data.get("module", ""),
            "direction": data.get("direction", ""),
            "rv": data.get("rv", 0), "adx": data.get("adx", 0),
            "adr_pct": data.get("adr_pct", 0),
            "hist": data.get("hist", ""),
            "ai_signal": result.get("signal", ""),
            "ai_confidence": result.get("confidence", 0),
            "method": result.get("method", ""),
        }
        pd.DataFrame([row]).to_csv(REQUESTS_PATH, mode='a',
            header=not os.path.exists(REQUESTS_PATH), index=False)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def log_feedback(fb_data):
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
            "nm": fb_data.get("nm", 0),
            "nm_accel": fb_data.get("nm_accel", 0),
            "nm_dist": fb_data.get("nm_dist", 0),
            "is_compressing": fb_data.get("is_compressing", False),
        }
        pd.DataFrame([row]).to_csv(FEEDBACK_PATH, mode='a',
            header=not os.path.exists(FEEDBACK_PATH), index=False)
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")


# ============================================================
#  LOAD / TRAIN MODEL
# ============================================================
def load_model():
    global model, stats, TRAIN_FEATURES
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
            stats["model_loaded"] = True
            stats["model_is_trained"] = True
            # Carica le features salvate
            feat_path = MODEL_PATH.replace(".pkl", "_features.json")
            if os.path.exists(feat_path):
                with open(feat_path, "r") as f:
                    TRAIN_FEATURES = json.load(f)
                print(f"[MODEL] Features caricate: {TRAIN_FEATURES}")
            print(f"[MODEL] Caricato da {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[MODEL] Errore caricamento: {e}")
            # Prova a riaddestrare se ci sono abbastanza feedback
            print(f"[MODEL] Tentativo retrain automatico...")
            result = train_model()
            if result.get("status") == "trained":
                print(f"[MODEL] Retrain automatico riuscito! v{stats['model_version']}")
                return True
            else:
                print(f"[MODEL] Retrain automatico fallito: {result}")
    return False


def train_model():
    global model, stats, TRAIN_FEATURES

    if not os.path.exists(FEEDBACK_PATH):
        return {"error": "Nessun dato feedback disponibile"}

    try:
        import joblib
        import lightgbm as lgb

        fb_df = pd.read_csv(FEEDBACK_PATH)
        if len(fb_df) < MIN_FEEDBACK_FOR_TRAIN:
            return {"error": f"Servono almeno {MIN_FEEDBACK_FOR_TRAIN} feedback, attuali: {len(fb_df)}"}

        df = fb_df.copy()

        # Feature engineering
        df["rv_abs"] = df["rv"].astype(float).abs()
        df["adr_residual_pct"] = 100 - df["adr_pct"].astype(float)

        feature_cols = ["rv", "adx", "adr_pct", "rv_abs", "adr_residual_pct"]

        # Optional features
        for feat in ["nm", "nm_accel", "nm_dist", "is_compressing"]:
            if feat in df.columns:
                df[feat] = df[feat].fillna(0)
                feature_cols.append(feat)

        df["rv"] = pd.to_numeric(df["rv"], errors="coerce").fillna(0)
        df["adx"] = pd.to_numeric(df["adx"], errors="coerce").fillna(0)
        df["adr_pct"] = pd.to_numeric(df["adr_pct"], errors="coerce").fillna(0)
        df["won"] = df["won"].astype(bool)
        # Usa TUTTI i trade, anche quelli con features=0
        # Il modello impara che "no features = meno info = più cautela"

        if len(df) < MIN_FEEDBACK_FOR_TRAIN:
            return {"error": f"Dati puliti insufficienti: {len(df)}"}

        X = df[feature_cols].values
        y = df["won"].astype(int).values

        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        if pos_count < 3 or neg_count < 3:
            return {"error": f"Classi sbilanciate: won={pos_count}, lost={neg_count}"}

        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 15, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_child_samples": 5, "verbose": -1, "n_jobs": 1, "seed": 42,
        }

        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=100,
                          valid_sets=[train_data], callbacks=[lgb.log_evaluation(0)])

        joblib.dump(model, MODEL_PATH)
        TRAIN_FEATURES = list(feature_cols)  # Salva per predict
        # Salva features su disco per sopravvivere ai restart
        feat_path = MODEL_PATH.replace(".pkl", "_features.json")
        with open(feat_path, "w") as f:
            json.dump(TRAIN_FEATURES, f)
        importance = dict(zip(feature_cols, model.feature_importance().tolist()))

        stats["model_is_trained"] = True
        stats["model_loaded"] = True
        stats["model_version"] += 1
        stats["last_retrain_time"] = datetime.now(timezone.utc).isoformat()

        return {
            "status": "trained", "samples": len(df),
            "won": pos_count, "lost": neg_count,
            "win_rate": round(pos_count / len(y) * 100, 1),
            "features": importance, "version": stats["model_version"],
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
#  FLASK ROUTES — CORE
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
    fb_count = 0
    req_count = 0
    if os.path.exists(FEEDBACK_PATH):
        try: fb_count = len(pd.read_csv(FEEDBACK_PATH))
        except: pass
    if os.path.exists(REQUESTS_PATH):
        try: req_count = len(pd.read_csv(REQUESTS_PATH))
        except: pass
    return jsonify({
        "server": stats,
        "data": {
            "feedback_rows": fb_count, "request_rows": req_count,
            "min_for_train": MIN_FEEDBACK_FOR_TRAIN,
            "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0, "method": "error"}), 200

        stats["total_predict_calls"] += 1
        stats["last_predict_time"] = datetime.now(timezone.utc).isoformat()

        direction = data.get("direction", "").upper()
        ctx = data.get("context", {})

        features_row = {
            "rv": float(data.get("rv", 0)),
            "adx": float(data.get("adx", 0)),
            "adr_pct": float(data.get("adr_pct", 0)),
            "adr_pip": float(data.get("adr_pip", 0)),
            "adr_media": float(data.get("adr_media", 0)),
            "ema_pos": int(data.get("ema_pos", 0)),
            "ema_gap_pct": float(data.get("ema_gap_pct", 0)),
            "rv_prev": float(data.get("rv_prev", 0)),
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
            "rv_decel": abs(float(data.get("rv_prev", 0))) - abs(float(data.get("rv", 0))),
            "adr_residual_pct": max(0, 100 - float(data.get("adr_pct", 0))),
            "nm": float(data.get("nm", 0)),
            "nm_signal": float(data.get("nm_signal", 0)),
            "nm_accel": float(data.get("nm_accel", 0)),
            "nm_dist": float(data.get("nm_dist", 0)),
            "is_compressing": 1 if data.get("is_compressing", False) else 0,
        }

        signal, confidence, method = direction, 0, "rules_v1"

        if stats["model_is_trained"] and model is not None:
            try:
                features_df = pd.DataFrame([features_row])
                ml_signal, ml_conf = predict_with_model(features_df)
                if ml_conf > 0:
                    signal, confidence, method = ml_signal, ml_conf, f"lgbm_v{stats['model_version']}"
            except: pass

        if confidence == 0:
            signal, confidence, details = rules_based_score(data)
            method = "rules_v1"

        result = {
            "signal": signal, "confidence": confidence, "method": method,
            "symbol": data.get("symbol", ""),
            "direction_proposed": direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log_request(data, result)
        return jsonify(result)

    except Exception as e:
        stats["total_errors"] += 1
        traceback.print_exc()
        return jsonify({"signal": "HOLD", "confidence": 0, "method": "error", "error": str(e)}), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        fb_data = request.get_json(force=True)
        if not fb_data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        stats["total_feedback_calls"] += 1
        log_feedback(fb_data)

        fb_count = 0
        if os.path.exists(FEEDBACK_PATH):
            try: fb_count = len(pd.read_csv(FEEDBACK_PATH))
            except: pass

        return jsonify({
            "status": "ok", "logged": True,
            "total_feedback": fb_count,
            "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/retrain", methods=["POST"])
def retrain():
    result = train_model()
    return jsonify(result)


# ============================================================
#  IMPORT FROM GITHUB — SUPPORTA ENTRAMBI I FORMATI
# ============================================================

@app.route("/import_csv", methods=["POST"])
def import_csv():
    return train_from_github()


@app.route("/train_from_github", methods=["POST"])
def train_from_github():
    """Importa CSV da GitHub e addestra il modello."""
    try:
        data = request.get_json(force=True) or {}
        csv_url = data.get("csv_url", GITHUB_CSV_URL)

        import urllib.request
        import io, csv as csv_module

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

        if not csv_content or len(csv_content) < 50:
            return jsonify({"status": "error", "message": "CSV vuoto"}), 200

        # Salva raw
        ensure_data_dir()
        raw_path = os.path.join(DATA_DIR, "imported_tradelog.csv")
        with open(raw_path, "w") as f:
            f.write(csv_content)

        # Rileva separatore
        first_line = csv_content.split("\n")[0]
        sep = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv_module.DictReader(io.StringIO(csv_content), delimiter=sep)

        headers = reader.fieldnames or []
        print(f"[IMPORT] Headers ({len(headers)}): {headers[:10]}...")

        # Leggi feedback esistente per evitare duplicati
        existing_tickets = set()
        if os.path.exists(FEEDBACK_PATH):
            try:
                existing_df = pd.read_csv(FEEDBACK_PATH)
                if "ticket" in existing_df.columns:
                    existing_tickets = set(existing_df["ticket"].astype(str).tolist())
            except: pass

        # Parole chiave per simboli
        symbols_set = {"eurusd","gbpusd","usdchf","usdcad","audusd","nzdusd","usdjpy",
            "eurjpy","gbpjpy","eurgbp","audcad","audchf","audnzd","audjpy","cadchf",
            "cadjpy","chfjpy","eurchf","eurcad","euraud","eurnzd","gbpaud","gbpcad",
            "gbpchf","gbpnzd","nzdcad","nzdchf","nzdjpy"}

        new_rows = []
        skipped = 0

        for row in reader:
            # --- Estrai ticket ---
            ticket = ""
            for key in ["Ticket", "ticket", "Order"]:
                val = str(row.get(key, "")).strip()
                if val and val.isdigit() and len(val) >= 7:
                    ticket = val
                    break
            if not ticket:
                continue

            if ticket in existing_tickets:
                skipped += 1
                continue

            # --- Estrai simbolo ---
            symbol = ""
            for key in ["Symbol", "symbol"]:
                val = str(row.get(key, "")).strip().lower().replace("+", "")
                if val in symbols_set:
                    symbol = val.upper()
                    break
            if not symbol:
                # Cerca in tutti i valori della riga
                for v in row.values():
                    v2 = str(v).strip().lower().replace("+", "")
                    if v2 in symbols_set:
                        symbol = v2.upper()
                        break
            if not symbol:
                continue  # skip se non troviamo il simbolo

            # --- Direzione ---
            direction = ""
            raw_dir = str(row.get("Direction", row.get("direction", ""))).upper()
            if "BUY" in raw_dir: direction = "BUY"
            elif "SELL" in raw_dir: direction = "SELL"
            if not direction:
                for v in row.values():
                    v2 = str(v).strip().upper()
                    if v2 in ("BUY", "SELL"):
                        direction = v2
                        break
            if not direction:
                continue

            # --- Module ---
            module = "STD"
            for v in row.values():
                v2 = str(v).strip().upper()
                if v2 in ("STD", "BRK", "REV", "MANUAL"):
                    module = v2
                    break

            # --- Profit ---
            profit_val = 0
            for key in ["Profit", "profit", "Profit$"]:
                try:
                    profit_val = float(str(row.get(key, "0")).replace(",", "."))
                    break
                except: pass

            # --- Won ---
            won_str = str(row.get("Won", row.get("won", ""))).strip().lower()
            if won_str in ("true", "1"): won = True
            elif won_str in ("false", "0"): won = False
            else: won = profit_val > 0

            # --- Pips ---
            pips_val = 0
            for key in ["Pips", "pips"]:
                try:
                    pips_val = float(str(row.get(key, "0")).replace(",", "."))
                    break
                except: pass

            # --- RV (prova Calc, poi Original) ---
            rv = 0
            for key in ["RV_Calc", "rv", "RV", "RV_Original"]:
                v = safe_float(row.get(key, "0"))
                if abs(v) > 0.01 and abs(v) < 200:
                    rv = v
                    break

            # --- ADX ---
            adx = 0
            for key in ["ADX_Calc", "adx", "ADX", "ADX_Original"]:
                v = safe_float(row.get(key, "0"))
                if v > 0 and v < 100:
                    adx = v
                    break

            # --- ADR% ---
            adr_pct = 0
            for key in ["ADR_Pct_Calc", "adr_pct", "ADR%"]:
                v = safe_float(row.get(key, "0"))
                if v > 0 and v < 300:
                    adr_pct = v
                    break

            # --- Histogram ---
            hist = ""
            for key in ["Hist_Calc", "hist", "HistogramState", "Hist_Original", "Hist"]:
                v = str(row.get(key, "")).strip()
                if v and v != "UNKNOWN" and v in ("GREEN_LIGHT", "GREEN_DARK", "RED_LIGHT", "RED_DARK", "GRAY"):
                    hist = v
                    break

            # --- AI Conf ---
            ai_conf = 0
            for key in ["AI_Conf", "ai_confidence"]:
                try:
                    ai_conf = int(safe_float(row.get(key, "0")))
                    break
                except: pass

            # --- Momentum ---
            nm = safe_float(row.get("NM_Calc", row.get("nm", "0")))
            nm_accel = safe_float(row.get("NM_Accel_Calc", row.get("nm_accel", "0")))
            nm_dist = safe_float(row.get("NM_Dist_Calc", row.get("nm_dist", "0")))
            is_compressing = bool(safe_float(row.get("Compression_Calc", row.get("is_compressing", "0"))))

            # --- Timestamp ---
            close_time = str(row.get("CloseTime", row.get("close_time", ""))).strip()

            fb_row = {
                "timestamp": close_time if close_time else datetime.now(timezone.utc).isoformat(),
                "ticket": ticket,
                "symbol": symbol,
                "direction": direction,
                "module": module,
                "entry_price": safe_float(row.get("EntryPrice", row.get("entry_price", "0"))),
                "exit_price": safe_float(row.get("ExitPrice", row.get("exit_price", "0"))),
                "profit": profit_val,
                "pips": pips_val,
                "won": won,
                "ai_confidence": ai_conf,
                "ai_signal": str(row.get("AI_Signal", row.get("ai_signal", ""))),
                "rv": rv,
                "adx": adx,
                "adr_pct": adr_pct,
                "hist": hist,
                "nm": nm,
                "nm_accel": nm_accel,
                "nm_dist": nm_dist,
                "is_compressing": is_compressing,
            }

            new_rows.append(fb_row)

        imported = len(new_rows)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(FEEDBACK_PATH, mode="a",
                         header=not os.path.exists(FEEDBACK_PATH), index=False)

        # Conta totali
        total_feedback = 0
        if os.path.exists(FEEDBACK_PATH):
            try: total_feedback = len(pd.read_csv(FEEDBACK_PATH))
            except: pass

        # Train se possibile
        train_result = None
        if total_feedback >= MIN_FEEDBACK_FOR_TRAIN:
            train_result = train_model()

        result = {
            "status": "ok",
            "new_imported": imported,
            "skipped_duplicates": skipped,
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
    "last_update": None, "balance": 0, "equity": 0,
    "open_trades": 0, "daily_pnl": 0, "daily_wins": 0,
    "daily_losses": 0, "consecutive_losses": 0,
    "ai_calls": 0, "ai_confirm": 0, "ai_reject": 0,
    "ai_errors": 0, "ai_missed_trades": 0,
    "warmup_ok": False, "warmup_last": None,
    "data_source": "", "cross_active": 0, "cross_total": 0,
    "daily_stopped": False, "account_currency": "EUR", "ea_version": "",
}


def load_ea_config():
    if os.path.exists(EA_CONFIG_PATH):
        try:
            with open(EA_CONFIG_PATH, "r") as f: cfg = json.load(f)
            for k, v in DEFAULT_EA_CONFIG.items():
                if k not in cfg: cfg[k] = v
            return cfg
        except: pass
    return dict(DEFAULT_EA_CONFIG)


def save_ea_config(cfg):
    ensure_data_dir()
    with open(EA_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


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

        ensure_data_dir()
        with open(EA_STATUS_PATH, "w") as f:
            json.dump(ea_status, f, indent=2)

        cfg = load_ea_config()
        return jsonify({"status": "ok", "config": cfg})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


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
        updatable = [
            "aggressiveness", "use_ai", "ai_min_conf",
            "max_consec_loss", "max_daily_loss", "max_daily_profit",
            "rv_max", "adr_max", "min_rr", "breakout_on", "reversal_on",
            "fixed_lots", "max_concurrent",
        ]
        updated = []
        for key in updatable:
            if key in data:
                old_val = cfg.get(key)
                new_val = data[key]
                if key in ("use_ai", "breakout_on", "reversal_on"):
                    new_val = bool(new_val)
                elif key in ("adr_max", "max_daily_profit", "min_rr", "fixed_lots"):
                    new_val = float(new_val)
                else:
                    new_val = int(new_val)
                cfg[key] = new_val
                if old_val != new_val:
                    updated.append(f"{key}: {old_val} -> {new_val}")

        save_ea_config(cfg)
        return jsonify({
            "status": "ok", "config": cfg, "updated": updated,
            "message": f"Aggiornati {len(updated)} parametri."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


def sanitize_for_json(obj):
    """Ricorsivamente pulisce NaN/Inf per JSON valido (i browser non li accettano)."""
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
                t["ai_signal"] = str(t.get("ai_signal", "")) if not pd.isna(t.get("ai_signal", "")) else ""
                t["hist"] = str(t.get("hist", "")) if not pd.isna(t.get("hist", "")) else ""
                t["rv"] = float(t.get("rv", 0)) if not pd.isna(t.get("rv", 0)) else 0.0
                t["adx"] = float(t.get("adx", 0)) if not pd.isna(t.get("adx", 0)) else 0.0
                t["adr_pct"] = float(t.get("adr_pct", 0)) if not pd.isna(t.get("adr_pct", 0)) else 0.0
                trade_history.append(t)
        except: pass

    result = {
        "ea": ea, "server": srv, "config": load_ea_config(),
        "feedback_count": fb_count, "trade_history": trade_history,
        "ready_to_train": fb_count >= MIN_FEEDBACK_FOR_TRAIN,
    }
    # Doppia sicurezza: pulisci tutto da NaN/Inf
    result = sanitize_for_json(result)
    return jsonify(result)


# ============================================================
#  GPT A/B TEST
# ============================================================

GPT_SYSTEM_PROMPT = """Sei un analista forex quantitativo esperto. Valuta trade candidate ricevendo dati tecnici completi.

Il trader ha queste caratteristiche storiche:
- Win rate: ~41%
- Account: piccolo (EUR), lotto fisso 0.01
- Usa un EA su M15 con 3 moduli: Standard, Breakout, Reversal

Indicatori chiave:
- Radar Value (RV): forza/direzione trend. Estremo se |RV|>40
- ADX: forza trend. >40 forte, <20 debole
- ADR%: range giornaliero usato. <50% molto spazio, >100% esausto
- Histogram: GREEN_LIGHT=rialzo forte, RED_LIGHT=ribasso forte, GRAY=neutrale
- Normalized Momentum: positivo=rialzista, negativo=ribassista

REGOLE:
- Se i dati sono insufficienti o ambigui, dai confidenza bassa (<50)
- Non avere paura di dire HOLD se il trade non e' chiaro

Rispondi SOLO in JSON valido: {"signal":"BUY" o "SELL" o "HOLD","confidence":0-100,"reasoning":"motivo in 1 frase"}"""


def call_gpt(data):
    if not OPENAI_API_KEY:
        return {"signal": "HOLD", "confidence": 0, "reasoning": "API key non configurata", "error": True}

    try:
        import urllib.request

        rv = float(data.get("rv", 0))
        adx = float(data.get("adx", 0))
        adr_pct = float(data.get("adr_pct", 0))
        direction = data.get("direction", "BUY")
        module = data.get("module", "STD")
        hist = data.get("hist", "UNKNOWN")
        symbol = data.get("symbol", "")
        nm = float(data.get("nm", 0))

        user_msg = f"""Valuta questo trade:
Simbolo: {symbol}
Direzione proposta: {direction}
Modulo: {module}
Radar Value: {rv}
ADX(14): {adx}
ADR%: {adr_pct}%
Histogram: {hist}
Normalized Momentum: {nm}"""

        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.3, "max_tokens": 150,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
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
            "model": GPT_MODEL, "error": False
        }
    except Exception as e:
        print(f"[GPT ERROR] {e}")
        traceback.print_exc()
        return {"signal": "HOLD", "confidence": 0, "reasoning": str(e), "error": True}


@app.route("/predict_gpt", methods=["POST"])
def predict_gpt():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0, "reasoning": "No JSON"}), 200

        gpt_result = call_gpt(data)

        ensure_data_dir()
        ab_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": data.get("symbol", ""),
            "direction": data.get("direction", ""),
            "module": data.get("module", ""),
            "rv": data.get("rv", 0), "adx": data.get("adx", 0),
            "adr_pct": data.get("adr_pct", 0), "hist": data.get("hist", ""),
            "lgbm_signal": data.get("direction", ""),
            "lgbm_conf": data.get("ai_confidence", 0),
            "gpt_signal": gpt_result.get("signal", ""),
            "gpt_conf": gpt_result.get("confidence", 0),
            "gpt_reasoning": gpt_result.get("reasoning", ""),
            "agreement": "SAME" if data.get("direction", "").upper() == gpt_result.get("signal", "").upper() else "DIFF",
        }
        pd.DataFrame([ab_row]).to_csv(AB_RESULTS_PATH, mode="a",
            header=not os.path.exists(AB_RESULTS_PATH), index=False)

        return jsonify(gpt_result)
    except Exception as e:
        return jsonify({"signal": "HOLD", "confidence": 0, "reasoning": str(e), "error": True}), 200


@app.route("/ab_stats", methods=["GET"])
def ab_stats():
    result = {
        "total": 0, "agreement_same": 0, "agreement_diff": 0, "agreement_pct": 0,
        "gpt_enabled": bool(OPENAI_API_KEY), "gpt_model": GPT_MODEL,
        "gpt_key_preview": OPENAI_API_KEY[:8] + "..." if len(OPENAI_API_KEY) > 8 else "",
    }
    if not os.path.exists(AB_RESULTS_PATH):
        return jsonify(result)
    try:
        df = pd.read_csv(AB_RESULTS_PATH)
        total = len(df)
        same = len(df[df["agreement"] == "SAME"]) if "agreement" in df.columns else 0
        diff = len(df[df["agreement"] == "DIFF"]) if "agreement" in df.columns else 0
        return jsonify({
            "total": total, "agreement_same": same, "agreement_diff": diff,
            "agreement_pct": round(same / total * 100, 1) if total > 0 else 0,
            "gpt_enabled": bool(OPENAI_API_KEY), "gpt_model": GPT_MODEL,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================
#  DIAGNOSTICA
# ============================================================

@app.route("/diag", methods=["GET"])
def diag():
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
        "min_feedback_for_train": MIN_FEEDBACK_FOR_TRAIN,
        "uptime": stats["started"],
        "total_predict": stats["total_predict_calls"],
        "total_feedback": stats["total_feedback_calls"],
    })


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

<div class="section"><h2>Account</h2>
<div class="row">
  <div class="card"><div class="val white" id="balance">-</div><div class="lbl">Balance EUR</div></div>
  <div class="card"><div class="val white" id="equity">-</div><div class="lbl">Equity EUR</div></div>
  <div class="card"><div class="val" id="dailyPnl">-</div><div class="lbl">P&L Oggi</div></div>
  <div class="card"><div class="val" id="openTrades">-</div><div class="lbl">Trade Aperti</div></div>
</div></div>

<div class="section"><h2>AI Engine</h2>
<div class="row">
  <div class="card"><div class="val blue" id="aiCalls">-</div><div class="lbl">Chiamate AI</div></div>
  <div class="card"><div class="val green" id="aiConfirm">-</div><div class="lbl">Confermati</div></div>
  <div class="card"><div class="val yellow" id="aiReject">-</div><div class="lbl">Rifiutati</div></div>
  <div class="card"><div class="val red" id="aiErrors">-</div><div class="lbl">Errori</div></div>
</div></div>

<div class="section"><h2>Mercato</h2>
<div class="row">
  <div class="card"><div class="val white" id="crossTotal">-</div><div class="lbl">Cross Totali</div></div>
  <div class="card"><div class="val blue" id="crossActive">-</div><div class="lbl">Cross Attivi</div></div>
  <div class="card"><div class="val" id="dailyWL">-</div><div class="lbl">W / L Oggi</div></div>
</div></div>

<div class="section"><h2>Configurazione EA</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Aggressivita'</label>
    <select id="cfgAggr"><option value="1">1 - Conservativo</option><option value="2" selected>2 - Moderato</option><option value="3">3 - Aggressivo</option></select></div>
  <div class="cfg-item"><label>AI Attiva</label>
    <select id="cfgAI"><option value="true">SI</option><option value="false">NO</option></select></div>
  <div class="cfg-item"><label>Confidenza minima %</label>
    <input type="number" id="cfgMinConf" value="70" min="50" max="95"></div>
  <div class="cfg-item"><label>Max loss consecutivi</label>
    <input type="number" id="cfgMaxCLoss" value="2" min="1" max="5"></div>
  <div class="cfg-item"><label>Max loss giornalieri</label>
    <input type="number" id="cfgMaxDLoss" value="3" min="1" max="8"></div>
  <div class="cfg-item"><label>Max profitto %</label>
    <input type="number" id="cfgMaxProf" value="3.0" min="1" max="10" step="0.5"></div>
  <div class="cfg-item"><label>RV massimo</label>
    <input type="number" id="cfgRVMax" value="30" min="10" max="50"></div>
  <div class="cfg-item"><label>ADR% massimo</label>
    <input type="number" id="cfgADRMax" value="60" min="30" max="90"></div>
  <div class="cfg-item"><label>R:R minimo</label>
    <input type="number" id="cfgMinRR" value="1.5" min="1.0" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Breakout</label>
    <select id="cfgBrk"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal</label>
    <select id="cfgRev"><option value="true">Attivo</option><option value="false" selected>Disattivo</option></select></div>
</div>
<div class="btn-row"><button class="btn btn-blue" onclick="saveAllConfig()">💾 Salva Configurazione</button>
  <span id="cfgMsg" style="color:#81c784;font-size:0.8em;align-self:center"></span></div>
</div>

<div class="section"><h2>Ultimi 20 Trade</h2>
<div style="overflow-x:auto"><table>
<thead><tr><th>Simbolo</th><th>Dir</th><th>Modulo</th><th>Pips</th><th>Profitto</th><th>Risultato</th><th>AI Conf</th></tr></thead>
<tbody id="tradeTable"><tr><td colspan="7" style="text-align:center;color:#666">Nessun trade</td></tr></tbody>
</table></div></div>

<div class="section"><h2>🧪 A/B Test: LightGBM vs GPT</h2>
<div class="row">
  <div class="card"><div class="val blue" id="abTotal">-</div><div class="lbl">Trade Testati</div></div>
  <div class="card"><div class="val green" id="abAgree">-</div><div class="lbl">D'Accordo</div></div>
  <div class="card"><div class="val red" id="abDisagree">-</div><div class="lbl">In Disaccordo</div></div>
  <div class="card"><div class="val" id="abGptStatus">-</div><div class="lbl">GPT Status</div></div>
  <div class="card"><div class="val white" id="abModel">-</div><div class="lbl">Modello</div></div>
</div></div>

<div class="section"><h2>Azioni</h2>
<div class="btn-row">
  <button class="btn btn-green" onclick="retrain()">🔄 Riaddestra</button>
  <button class="btn btn-blue" onclick="trainGithub()">📥 Importa da GitHub + Train</button>
  <button class="btn btn-yellow" onclick="loadAB()">📊 A/B Stats</button>
  <button class="btn btn-gray" onclick="refresh()">🔃 Aggiorna</button>
</div>
<div id="actionMsg" style="margin-top:8px;font-size:0.8em;color:#ffd54f"></div>
</div>

<div style="text-align:center;padding:16px 0;font-size:0.7em;color:#444">
  Profit Radar Pro v4.0 — Giovanni Mori
</div>
</div>

<script>
const API=window.location.origin;
function fmt(v,d=2){return v!=null?v.toFixed(d):'-'}
function pnlClass(v){return v>0?'green':v<0?'red':'white'}
function refresh(){
  fetch(API+'/dashboard_data').then(r=>r.json()).then(d=>{
    const ea=d.ea,srv=d.server,cfg=d.config;
    const dot=document.getElementById('eaDot');
    const age=ea.last_update?((Date.now()-new Date(ea.last_update))/1000/60):999;
    if(age<5){dot.className='status-dot dot-green';document.getElementById('eaStatus').textContent='EA Connesso'}
    else if(age<30){dot.className='status-dot dot-yellow';document.getElementById('eaStatus').textContent='EA '+Math.round(age)+'m fa'}
    else{dot.className='status-dot dot-gray';document.getElementById('eaStatus').textContent='EA offline'}
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
    const tb=document.getElementById('tradeTable');
    if(d.trade_history&&d.trade_history.length>0){
      tb.innerHTML=d.trade_history.reverse().map(t=>{
        const p=t.profit||0,w=t.won;
        return '<tr><td>'+(t.symbol||'-')+'</td><td>'+(t.direction||'-')+'</td><td>'+(t.module||'-')+'</td><td>'+fmt(t.pips,1)+'</td><td class="'+pnlClass(p)+'">'+(p>=0?'+':'')+fmt(p)+'€</td><td><span style="color:'+(w?'#81c784':'#ef5350')+'">'+(w?'WIN':'LOSS')+'</span></td><td>'+(t.ai_confidence||'-')+'%</td></tr>'
      }).join('');
    }
  }).catch(e=>{
    console.error('Dashboard fetch error:',e);
    document.getElementById('lastUpdate').textContent='❌ Errore connessione al server';
    document.getElementById('lastUpdate').style.color='#ef5350';
  });
}
function saveAllConfig(){
  const cfg={aggressiveness:parseInt(document.getElementById('cfgAggr').value),use_ai:document.getElementById('cfgAI').value==='true',ai_min_conf:parseInt(document.getElementById('cfgMinConf').value),max_consec_loss:parseInt(document.getElementById('cfgMaxCLoss').value),max_daily_loss:parseInt(document.getElementById('cfgMaxDLoss').value),max_daily_profit:parseFloat(document.getElementById('cfgMaxProf').value),rv_max:parseInt(document.getElementById('cfgRVMax').value),adr_max:parseFloat(document.getElementById('cfgADRMax').value),min_rr:parseFloat(document.getElementById('cfgMinRR').value),breakout_on:document.getElementById('cfgBrk').value==='true',reversal_on:document.getElementById('cfgRev').value==='true'};
  fetch(API+'/ea_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)}).then(r=>r.json()).then(d=>{
    const m=document.getElementById('cfgMsg');m.textContent=d.status==='ok'?'✅ '+d.message:'❌ '+(d.message||'errore');m.style.color=d.status==='ok'?'#81c784':'#ef5350';setTimeout(()=>{m.textContent=''},5000);
  }).catch(()=>{document.getElementById('cfgMsg').textContent='❌ Errore connessione'});
}
function retrain(){
  document.getElementById('actionMsg').textContent='⏳ Riaddestramento...';
  fetch(API+'/retrain',{method:'POST'}).then(r=>r.json()).then(d=>{
    document.getElementById('actionMsg').innerHTML=d.status==='trained'?'✅ Trained! Samples: '+d.samples+' | WR: '+d.win_rate+'%':'⚠️ '+(d.error||'Errore');
  }).catch(()=>{document.getElementById('actionMsg').textContent='❌ Errore'});
}
function trainGithub(){
  document.getElementById('actionMsg').textContent='⏳ Importazione da GitHub...';
  fetch(API+'/train_from_github',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>r.json()).then(d=>{
    if(d.status==='error'){
      document.getElementById('actionMsg').innerHTML='❌ Errore: '+(d.message||'unknown')+'<br>💡 '+(d.hint||'');
      return;
    }
    let m='✅ Importati '+(d.new_imported||0)+' trade | Totale: '+(d.total_feedback||0);
    if(d.train_result) m+=' | Train: '+JSON.stringify(d.train_result).substring(0,150);
    else if(d.ready_to_train) m+=' | Pronto per training!';
    document.getElementById('actionMsg').innerHTML=m;
  }).catch(()=>{document.getElementById('actionMsg').textContent='❌ Errore connessione'});
}
function loadAB(){
  fetch(API+'/ab_stats').then(r=>r.json()).then(d=>{
    document.getElementById('abTotal').textContent=d.total||0;
    document.getElementById('abAgree').textContent=d.agreement_same||0;
    document.getElementById('abDisagree').textContent=d.agreement_diff||0;
    document.getElementById('abGptStatus').textContent=d.gpt_enabled?'🟢 Attivo':'🔴 No Key';
    document.getElementById('abModel').textContent=d.gpt_model||'-';
    document.getElementById('actionMsg').innerHTML='✅ A/B aggiornati';
  }).catch(()=>{});
}
refresh();setInterval(refresh,30000);
</script>
</body>
</html>"""


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return Response(DASHBOARD_HTML, mimetype="text/html")


@app.route("/export_logs", methods=["GET"])
def export_logs():
    """Mostra i log A/B e requests per download manuale su GitHub."""
    result = {"ab_results": {"exists": False, "rows": 0}, "requests_log": {"exists": False, "rows": 0}}
    if os.path.exists(AB_RESULTS_PATH):
        try:
            df = pd.read_csv(AB_RESULTS_PATH)
            result["ab_results"] = {"exists": True, "rows": len(df), "columns": list(df.columns)}
        except:
            result["ab_results"] = {"exists": True, "rows": -1, "error": "read failed"}
    if os.path.exists(REQUESTS_PATH):
        try:
            df = pd.read_csv(REQUESTS_PATH)
            result["requests_log"] = {"exists": True, "rows": len(df), "columns": list(df.columns)}
        except:
            result["requests_log"] = {"exists": True, "rows": -1, "error": "read failed"}
    result["hint"] = "Per salvare su GitHub: copia i file da Data/ nel repo. Il restore è automatico al prossimo deploy."
    return jsonify(result)


# ============================================================
#  RIPRISTINO LOG DA GITHUB (post-deploy)
# ============================================================
def restore_logs_from_github():
    """Ripristina ab_results.csv e requests_log.csv da GitHub dopo un deploy."""
    import urllib.request
    restored = []
    for name, path, url in [
        ("ab_results.csv", AB_RESULTS_PATH, GITHUB_AB_URL),
        ("requests_log.csv", REQUESTS_PATH, GITHUB_REQUESTS_URL),
    ]:
        if os.path.exists(path):
            try:
                existing = len(pd.read_csv(path))
                if existing > 0:
                    print(f"[RESTORE] {name}: già presente ({existing} righe), skip")
                    continue
            except:
                pass
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ProfitRadarAI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode("utf-8")
            if content and len(content) > 50:
                with open(path, "w") as f:
                    f.write(content)
                rows = content.count("\n")
                restored.append(f"{name}: {rows} righe")
                print(f"[RESTORE] {name}: scaricato da GitHub ({rows} righe)")
            else:
                print(f"[RESTORE] {name}: file vuoto su GitHub")
        except Exception as e:
            print(f"[RESTORE] {name}: non disponibile ({e})")
    return restored


# ============================================================
#  INIT — All'avvio del server
# ============================================================
ensure_data_dir()
load_model()
restore_logs_from_github()
print(f"[INIT] Profit Radar Pro AI Server v4.0")
print(f"[INIT] Data dir: {DATA_DIR}")
print(f"[INIT] Modello: {'LOADED' if stats['model_is_trained'] else 'REGOLE (nessun modello)'}")
print(f"[INIT] Min feedback per training: {MIN_FEEDBACK_FOR_TRAIN}")
print(f"[INIT] GPT: {GPT_MODEL} | Key: {'SET' if OPENAI_API_KEY else 'NOT SET'}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[START] Server on port {port}")
    ensure_data_dir()
    load_model()
    app.run(host="0.0.0.0", port=port, debug=False)
