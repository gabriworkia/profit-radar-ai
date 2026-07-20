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
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ============================================================
#  LOGGING OPENAI API
# ============================================================
LOG_PATH = os.path.join(os.environ.get("DATA_DIR", "data"), "gpt_api.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
gpt_logger = logging.getLogger("gpt_api")

# ============================================================
#  CONFIGURAZIONE
# ============================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
REV_MODEL_PATH = os.path.join(DATA_DIR, "model_reversal.pkl")  # modello dedicato al Reversal
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
REQUESTS_PATH = os.path.join(DATA_DIR, "requests_log.csv")
EA_CONFIG_PATH = os.path.join(DATA_DIR, "ea_config.json")
EA_STATUS_PATH = os.path.join(DATA_DIR, "ea_status.json")
AB_RESULTS_PATH = os.path.join(DATA_DIR, "ab_results.csv")
AB_OUTCOMES_PATH = os.path.join(DATA_DIR, "ab_outcomes.csv")  # confronto A/B + esito reale del trade
# GitHub URLs per ripristino post-deploy
GITHUB_AB_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/data-backup/Data/ab_results.csv"
GITHUB_REQUESTS_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/data-backup/Data/requests_log.csv"
GITHUB_FEEDBACK_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/data-backup/Data/feedback.csv"
GITHUB_ABOUT_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/data-backup/Data/ab_outcomes.csv"
GITHUB_REVSIG_URL = "https://raw.githubusercontent.com/gabriworkia/profit-radar-ai/data-backup/Data/PRP_ReversalSignals.csv"

MIN_FEEDBACK_FOR_TRAIN = int(os.environ.get("MIN_FEEDBACK_FOR_TRAIN", "50"))
AUTO_BACKUP_EVERY = int(os.environ.get("AUTO_BACKUP_EVERY", "5"))  # backup GitHub ogni N feedback
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5-nano-2025-08-07"

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
            "lgbm_signal": result.get("lgbm_signal", ""),
            "lgbm_conf": result.get("lgbm_conf", 0),
            "hybrid_reasoning": result.get("hybrid_reasoning", ""),
            "hist_consec_color": data.get("hist_consec_color", 0),
            "hist_bars_since_gray": data.get("hist_bars_since_gray", 0),
            "hist_cycle_count": data.get("hist_cycle_count", 0),
            "hist_crossed_zero": data.get("hist_crossed_zero", 0),
            "hist_bar_slope": data.get("hist_bar_slope", 0),
            "hist_pullback_depth": data.get("hist_pullback_depth", 0),
            "hist_seq_encoded": data.get("hist_seq_encoded", 0),
            "hist_bar_ratio": data.get("hist_bar_ratio", 0),
            "hist_color_now": data.get("hist_color_now", 0),
            "hist_color_slope": data.get("hist_color_slope", 0),
            "hist_color_curve": data.get("hist_color_curve", 0),
            "hist_color_r2": data.get("hist_color_r2", 0),
            "hist_color_centroid": data.get("hist_color_centroid", 0.5),
            "hist_color_sum": data.get("hist_color_sum", 0),
        }
        pd.DataFrame([row]).to_csv(REQUESTS_PATH, mode='a',
            header=not os.path.exists(REQUESTS_PATH), index=False)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def log_feedback(fb_data):
    ensure_data_dir()
    try:
        ctx = fb_data.get("context", {}) or {}
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
            # --- Core market features ---
            "rv": fb_data.get("rv", 0),
            "adx": fb_data.get("adx", 0),
            "adr_pct": fb_data.get("adr_pct", 0),
            "hist": fb_data.get("hist", ""),
            "nm": fb_data.get("nm", 0),
            "nm_accel": fb_data.get("nm_accel", 0),
            "nm_dist": fb_data.get("nm_dist", 0),
            "is_compressing": 1 if fb_data.get("is_compressing", False) else 0,
            # --- Extended features (added 2026-06) ---
            "adr_pip": fb_data.get("adr_pip", 0),
            "adr_media": fb_data.get("adr_media", 0),
            "ema_pos": fb_data.get("ema_pos", 0),
            "ema_gap_pct": fb_data.get("ema_gap_pct", 0),
            "rv_prev": fb_data.get("rv_prev", 0),
            "rv_prev2": fb_data.get("rv_prev2", 0),
            "light_streak": fb_data.get("light_streak", 0),
            "was_gray": 1 if fb_data.get("was_gray", False) else 0,
            "hist_flip_bar": fb_data.get("hist_flip_bar", 999),
            "nm_signal": fb_data.get("nm_signal", 0),
            # --- Context object (state of all 28 pairs) ---
            "ctx_total": ctx.get("total", 0),
            "ctx_non_gray": ctx.get("non_gray", 0),
            "ctx_green": ctx.get("green", 0),
            "ctx_red": ctx.get("red", 0),
            "ctx_avg_abs_rv": ctx.get("avg_abs_rv", 0),
            "ctx_extreme_rv": ctx.get("extreme_rv", 0),
            # --- Histogram sequence features ---
            "hist_consec_color": fb_data.get("hist_consec_color", 0),
            "hist_bars_since_gray": fb_data.get("hist_bars_since_gray", 0),
            "hist_cycle_count": fb_data.get("hist_cycle_count", 0),
            "hist_crossed_zero": fb_data.get("hist_crossed_zero", 0),
            "hist_bar_slope": fb_data.get("hist_bar_slope", 0),
            "hist_pullback_depth": fb_data.get("hist_pullback_depth", 0),
            "hist_seq_encoded": fb_data.get("hist_seq_encoded", 0),
            "hist_bar_ratio": fb_data.get("hist_bar_ratio", 1),
            "hist_color_now": fb_data.get("hist_color_now", 0),
            "hist_color_slope": fb_data.get("hist_color_slope", 0),
            "hist_color_curve": fb_data.get("hist_color_curve", 0),
            "hist_color_r2": fb_data.get("hist_color_r2", 0),
            "hist_color_centroid": fb_data.get("hist_color_centroid", 0.5),
            "hist_color_sum": fb_data.get("hist_color_sum", 0),
        }
        # Append in modo robusto: se il CSV esiste con uno schema diverso,
        # uniamo le colonne cosi' i vecchi trade (senza feature nuove) restano
        # con valori vuoti senza rompere il file.
        new_df = pd.DataFrame([row])
        if os.path.exists(FEEDBACK_PATH):
            try:
                old_df = pd.read_csv(FEEDBACK_PATH)
                combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
                combined.to_csv(FEEDBACK_PATH, index=False)
            except Exception:
                # Fallback: append semplice
                new_df.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)
        else:
            new_df.to_csv(FEEDBACK_PATH, index=False)
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")

    # Dopo aver loggato il feedback, prova a collegarlo a un confronto A/B
    try:
        record_ab_outcome(fb_data)
    except Exception as e:
        print(f"[AB-OUTCOME ERROR] {e}")


def record_ab_outcome(fb_data):
    """Collega un trade chiuso (feedback) al confronto A/B GPT-vs-LightGBM fatto
    al momento della decisione, e registra CHI AVEVA RAGIONE in ab_outcomes.csv.

    Cosi' l'analisi 'quando GPT e LightGBM erano discordi, chi ha azzeccato?'
    e' gia' pronta, senza incrociare i file a mano.
    """
    if not os.path.exists(AB_RESULTS_PATH):
        return

    symbol = str(fb_data.get("symbol", "")).replace("+", "").strip().upper()
    direction = str(fb_data.get("direction", "")).strip().upper()
    won = bool(fb_data.get("won", False))
    pips = float(fb_data.get("pips", 0) or 0)
    if not symbol or not direction:
        return

    try:
        ab = pd.read_csv(AB_RESULTS_PATH)
    except Exception:
        return
    if ab.empty:
        return

    # Normalizza per il match
    ab["_sym"] = ab["symbol"].astype(str).str.replace("+", "", regex=False).str.strip().str.upper()
    ab["_dir"] = ab["direction"].astype(str).str.strip().str.upper()

    # Cerca l'ULTIMO confronto A/B per questo symbol+direction (il piu' recente
    # prima della chiusura) che non sia gia' stato risolto.
    already = set()
    if os.path.exists(AB_OUTCOMES_PATH):
        try:
            done = pd.read_csv(AB_OUTCOMES_PATH)
            already = set(done["ab_timestamp"].astype(str).tolist())
        except Exception:
            already = set()

    cand = ab[(ab["_sym"] == symbol) & (ab["_dir"] == direction)]
    cand = cand[~cand["timestamp"].astype(str).isin(already)]
    if cand.empty:
        return

    match = cand.iloc[-1]   # il piu' recente non ancora risolto

    lgbm_sig = str(match.get("lgbm_signal", "")).upper()
    gpt_sig = str(match.get("gpt_signal", "")).upper()
    agreement = str(match.get("agreement", ""))

    # CHI AVEVA RAGIONE?
    #  - LightGBM "voleva entrare" nella direzione del trade -> se WIN aveva ragione
    #  - GPT spesso diceva HOLD (non entrare) -> se LOSS aveva ragione GPT
    lgbm_right = ""
    gpt_right = ""
    if agreement == "DIFF":
        # tipicamente: LGBM=BUY/SELL (entra), GPT=HOLD (non entra)
        if gpt_sig == "HOLD":
            lgbm_right = "SI" if won else "NO"
            gpt_right = "NO" if won else "SI"
        else:
            # entrambi danno una direzione ma diversa
            lgbm_right = "SI" if (lgbm_sig == direction and won) else "NO"
            gpt_right = "SI" if (gpt_sig == direction and won) else "NO"
    else:  # SAME: erano d'accordo
        lgbm_right = "SI" if won else "NO"
        gpt_right = "SI" if won else "NO"

    out_row = {
        "ab_timestamp": match.get("timestamp", ""),
        "feedback_timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "direction": direction,
        "module": fb_data.get("module", ""),
        "lgbm_signal": lgbm_sig,
        "lgbm_conf": match.get("lgbm_conf", ""),
        "gpt_signal": gpt_sig,
        "gpt_conf": match.get("gpt_conf", ""),
        "agreement": agreement,
        "outcome": "WIN" if won else "LOSS",
        "pips": pips,
        "lgbm_right": lgbm_right,
        "gpt_right": gpt_right,
    }
    pd.DataFrame([out_row]).to_csv(
        AB_OUTCOMES_PATH, mode="a",
        header=not os.path.exists(AB_OUTCOMES_PATH), index=False)
    print(f"[AB-OUTCOME] {symbol} {direction} | {agreement} | "
          f"LGBM:{lgbm_sig}({lgbm_right}) GPT:{gpt_sig}({gpt_right}) | {out_row['outcome']}")


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

        # --- Numeric coercion delle colonne base ---
        df["rv"] = pd.to_numeric(df["rv"], errors="coerce").fillna(0)
        df["adx"] = pd.to_numeric(df["adx"], errors="coerce").fillna(0)
        df["adr_pct"] = pd.to_numeric(df["adr_pct"], errors="coerce").fillna(0)

        # --- Feature engineering derivate ---
        df["rv_abs"] = df["rv"].abs()
        df["adr_residual_pct"] = (100 - df["adr_pct"]).clip(lower=0)

        # rv_decel: decelerazione del radar value (|rv_prev| - |rv|).
        # Coerente con il calcolo in /predict. Se manca rv_prev resta 0.
        if "rv_prev" in df.columns:
            rv_prev_num = pd.to_numeric(df["rv_prev"], errors="coerce").fillna(0)
            df["rv_decel"] = rv_prev_num.abs() - df["rv"].abs()
        else:
            df["rv_decel"] = 0.0

        # Set base sempre presente
        feature_cols = ["rv", "adx", "adr_pct", "rv_abs", "adr_residual_pct", "rv_decel"]

        # --- Feature opzionali: aggiunte automaticamente se presenti nel CSV ---
        # Ogni feature qui sotto, se la colonna esiste, viene resa numerica,
        # i valori mancanti riempiti a 0, e aggiunta al set di training.
        # I vecchi trade (senza queste colonne) avranno valore 0 => il modello
        # impara "feature=0 => meno informazione => piu' cautela".
        OPTIONAL_FEATURES = [
            # Neural momentum
            "nm", "nm_accel", "nm_dist", "nm_signal", "is_compressing",
            # ADR / EMA estese
            "adr_pip", "adr_media", "ema_pos", "ema_gap_pct",
            # Sequenza radar value
            "rv_prev", "rv_prev2", "light_streak", "was_gray", "hist_flip_bar",
            # Context (stato delle 28 coppie)
            "ctx_total", "ctx_non_gray", "ctx_green", "ctx_red",
            "ctx_avg_abs_rv", "ctx_extreme_rv",
            # Histogram sequence features
            "hist_consec_color", "hist_bars_since_gray", "hist_cycle_count",
            "hist_crossed_zero", "hist_bar_slope", "hist_pullback_depth",
            "hist_seq_encoded", "hist_bar_ratio",
            # Forma istogramma (colore-pesata)
            "hist_color_now", "hist_color_slope", "hist_color_curve",
            "hist_color_r2", "hist_color_centroid", "hist_color_sum",
        ]
        for feat in OPTIONAL_FEATURES:
            if feat in df.columns:
                # bool/stringhe ("True"/"False") -> numerico
                df[feat] = (
                    df[feat]
                    .replace({True: 1, False: 0, "True": 1, "False": 0,
                              "true": 1, "false": 0})
                )
                df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)
                # Scarta feature costanti (zero varianza) => inutili e rumorose
                if df[feat].nunique() > 1:
                    feature_cols.append(feat)

        df["won"] = df["won"].astype(bool)
        print(f"[TRAIN] Feature usate ({len(feature_cols)}): {feature_cols}")
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
    """
    Sistema ibrido LightGBM + GPT:
    - Se LightGBM confidence >= 60% → usa LightGBM
    - Se LightGBM < 60% ma GPT confidence >= 60% → usa GPT
    - Se entrambi < 50% → HOLD (confidence 0)
    - Se uno è 50-59% e l'altro < 50% → HOLD
    """
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
            # Histogram sequence features
            "hist_consec_color": int(data.get("hist_consec_color", 0)),
            "hist_bars_since_gray": int(data.get("hist_bars_since_gray", 0)),
            "hist_cycle_count": int(data.get("hist_cycle_count", 0)),
            "hist_crossed_zero": int(data.get("hist_crossed_zero", 0)),
            "hist_bar_slope": float(data.get("hist_bar_slope", 0)),
            "hist_pullback_depth": float(data.get("hist_pullback_depth", 0)),
            "hist_seq_encoded": int(data.get("hist_seq_encoded", 0)),
            "hist_bar_ratio": float(data.get("hist_bar_ratio", 1)),
            "hist_color_now": float(data.get("hist_color_now", 0)),
            "hist_color_slope": float(data.get("hist_color_slope", 0)),
            "hist_color_curve": float(data.get("hist_color_curve", 0)),
            "hist_color_r2": float(data.get("hist_color_r2", 0)),
            "hist_color_centroid": float(data.get("hist_color_centroid", 0.5)),
            "hist_color_sum": int(data.get("hist_color_sum", 0)),
        }

        # === STEP 1: Prova LightGBM ===
        lgbm_signal, lgbm_conf = direction, 0
        lgbm_method = "rules_v1"

        if stats["model_is_trained"] and model is not None:
            try:
                features_df = pd.DataFrame([features_row])
                ml_signal, ml_conf = predict_with_model(features_df)
                if ml_conf > 0:
                    lgbm_signal, lgbm_conf = ml_signal, ml_conf
                    lgbm_method = f"lgbm_v{stats['model_version']}"
            except: pass

        if lgbm_conf == 0:
            lgbm_signal, lgbm_conf, _ = rules_based_score(data)
            lgbm_method = "rules_v1"

        # === STEP 2: Decisione ibrida ===
        LGBM_THRESHOLD = 60  # Soglia per usare direttamente LightGBM
        GPT_THRESHOLD = 60   # Soglia per usare GPT come fallback
        HOLD_THRESHOLD = 50  # Sotto questa soglia da entrambi → HOLD

        hybrid_reasoning = ""
        used_gpt = False

        if lgbm_conf >= LGBM_THRESHOLD:
            # LightGBM è abbastanza confidente → usa lui
            signal, confidence, method = lgbm_signal, lgbm_conf, lgbm_method
            hybrid_reasoning = f"LightGBM {lgbm_conf}% >= {LGBM_THRESHOLD}% → uso LightGBM"
        else:
            # LightGBM non è confidente → prova GPT
            gpt_result = call_gpt(data)
            gpt_signal = gpt_result.get("signal", "HOLD").upper()
            gpt_conf = gpt_result.get("confidence", 0)
            gpt_error = gpt_result.get("error", False)

            if not gpt_error and gpt_conf >= GPT_THRESHOLD and gpt_signal in ("BUY", "SELL"):
                # GPT è confidente → usa GPT
                signal = gpt_signal
                confidence = gpt_conf
                method = "gpt_hybrid"
                used_gpt = True
                hybrid_reasoning = f"LightGBM {lgbm_conf}% < {LGBM_THRESHOLD}% → GPT {gpt_signal} {gpt_conf}% >= {GPT_THRESHOLD}% → uso GPT"
            elif lgbm_conf >= HOLD_THRESHOLD and not gpt_error and gpt_conf < HOLD_THRESHOLD:
                # LightGBM ha 50-59%, GPT sotto 50% → usiamo LightGBM ma con cautela
                signal, confidence, method = lgbm_signal, lgbm_conf, lgbm_method
                hybrid_reasoning = f"LightGBM {lgbm_conf}% (50-59%) + GPT {gpt_conf}% < 50% → uso LightGBM con cautela"
            elif not gpt_error and gpt_conf >= HOLD_THRESHOLD and gpt_signal in ("BUY", "SELL"):
                # GPT 50-59% ma non >= 60%, e LightGBM < 50% → HOLD
                signal, confidence, method = "HOLD", 0, "hybrid_hold"
                hybrid_reasoning = f"LightGBM {lgbm_conf}% < 50% + GPT {gpt_conf}% (50-59%) → HOLD"
            else:
                # Entrambi < 50% → HOLD
                signal, confidence, method = "HOLD", 0, "hybrid_hold"
                if gpt_error:
                    hybrid_reasoning = f"LightGBM {lgbm_conf}% < {LGBM_THRESHOLD}% + GPT ERRORE → HOLD"
                else:
                    hybrid_reasoning = f"LightGBM {lgbm_conf}% < {LGBM_THRESHOLD}% + GPT {gpt_conf}% < {GPT_THRESHOLD}% → HOLD"

            # Log A/B per tracciamento
            ensure_data_dir()
            ab_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": data.get("symbol", ""),
                "direction": data.get("direction", ""),
                "module": data.get("module", ""),
                "rv": data.get("rv", 0), "adx": data.get("adx", 0),
                "adr_pct": data.get("adr_pct", 0), "hist": data.get("hist", ""),
                "lgbm_signal": lgbm_signal,
                "lgbm_conf": lgbm_conf,
                "gpt_signal": gpt_signal,
                "gpt_conf": gpt_conf,
                "gpt_reasoning": gpt_result.get("reasoning", ""),
                "agreement": "SAME" if lgbm_signal.upper() == gpt_signal.upper() else "DIFF",
            }
            pd.DataFrame([ab_row]).to_csv(AB_RESULTS_PATH, mode="a",
                header=not os.path.exists(AB_RESULTS_PATH), index=False)

        result = {
            "signal": signal, "confidence": confidence, "method": method,
            "symbol": data.get("symbol", ""),
            "direction_proposed": direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lgbm_signal": lgbm_signal, "lgbm_conf": lgbm_conf,
            "hybrid_reasoning": hybrid_reasoning,
        }

        log_request(data, result)
        print(f"[HYBRID] {data.get('symbol','')} | LGBM: {lgbm_signal} {lgbm_conf}% | Final: {signal} {confidence}% via {method} | {hybrid_reasoning}")
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

        # --- BACKUP AUTOMATICO su GitHub ogni N feedback ---
        # Cosi' i dati sopravvivono ai reset di Render senza intervento manuale.
        try:
            if fb_count > 0 and fb_count % AUTO_BACKUP_EVERY == 0:
                saved = auto_backup_to_github()
                print(f"[AUTO-BACKUP] feedback #{fb_count} -> salvati {saved} file su GitHub")
        except Exception as _e:
            print(f"[AUTO-BACKUP] errore: {_e}")

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
#  MODELLO REVERSAL DEDICATO
# ============================================================

def train_reversal_model(csv_url=None, csv_text=None):
    """Addestra un modello LightGBM DEDICATO al Reversal, separato dallo Standard.
    Legge i segnali da PRP_ReversalSignals.csv (arricchito con tutte le feature).
    La label e' 'Outcome' (WIN/LOSS). Usa solo le righe con esito definito.
    """
    try:
        import io as _io
        import joblib
        import lightgbm as lgb

        # 1) Carica i dati: da URL, da testo, o dal file locale
        if csv_text:
            df = pd.read_csv(_io.StringIO(csv_text), sep=";")
        elif csv_url:
            import urllib.request
            req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                df = pd.read_csv(_io.StringIO(resp.read().decode("utf-8")), sep=";")
        else:
            rev_csv = os.path.join(DATA_DIR, "PRP_ReversalSignals.csv")
            if not os.path.exists(rev_csv):
                return {"error": "Nessun PRP_ReversalSignals.csv disponibile. Fornisci csv_url o caricalo."}
            df = pd.read_csv(rev_csv, sep=";")

        # 2) Tieni solo righe con esito definito (WIN/LOSS), scarta PENDING/TEST
        if "Outcome" not in df.columns:
            return {"error": "Colonna 'Outcome' mancante nel CSV"}
        df = df[df["Outcome"].isin(["WIN", "LOSS"])].copy()
        if len(df) < 20:
            return {"error": f"Servono almeno 20 segnali con esito, attuali: {len(df)}"}

        df["won"] = (df["Outcome"] == "WIN").astype(int)

        # 3) Feature: tutte le colonne numeriche tranne metadati/esito
        exclude = {"Time", "Symbol", "Direction", "Entry", "SL", "TP",
                   "AI_Conf", "PriceAfter", "PipsMove", "Outcome", "won"}
        feature_cols = []
        for c in df.columns:
            if c in exclude:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if df[c].nunique() > 1:   # scarta colonne costanti
                feature_cols.append(c)

        if len(feature_cols) < 3:
            return {"error": f"Troppe poche feature utili: {feature_cols}"}

        X = df[feature_cols].values
        y = df["won"].values
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        if pos < 3 or neg < 3:
            return {"error": f"Classi sbilanciate: WIN={pos}, LOSS={neg}"}

        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 15, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_child_samples": 5, "verbose": -1, "n_jobs": 1, "seed": 42,
        }
        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        rev_model = lgb.train(params, train_data, num_boost_round=100,
                              valid_sets=[train_data], callbacks=[lgb.log_evaluation(0)])

        joblib.dump(rev_model, REV_MODEL_PATH)
        feat_path = REV_MODEL_PATH.replace(".pkl", "_features.json")
        with open(feat_path, "w") as f:
            json.dump(list(feature_cols), f)

        importance = dict(zip(feature_cols, rev_model.feature_importance().tolist()))
        print(f"[REV-TRAIN] Modello Reversal addestrato: {len(df)} segnali, {len(feature_cols)} feature")

        return {
            "status": "trained", "model": "reversal",
            "samples": len(df), "won": pos, "lost": neg,
            "win_rate": round(pos / len(y) * 100, 1),
            "features_used": len(feature_cols),
            "features": importance,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.route("/train_reversal", methods=["POST"])
def train_reversal():
    """Addestra il modello Reversal. Accetta opzionalmente {csv_url:...} per
    importare il CSV dei segnali Reversal (es. da GitHub raw)."""
    data = request.get_json(force=True, silent=True) or {}
    result = train_reversal_model(csv_url=data.get("csv_url"))
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
    # --- Configurazione principale ---
    "aggressiveness" : 1,
    "use_ai" : False,
    "ai_min_conf" : 70,
    "send_feedback" : True,
    "daily_stop_on" : False,
    "max_consec_loss" : 2,
    "loss_weight" : 1.5,
    "max_concurrent" : 10,
    "max_per_pair" : 1,
    # --- Lotto, rischio e fasce ---
    "fixed_lots" : 0.05,
    "max_lots_cap" : 0.05,
    "max_lots_safety" : 0.05,
    "dynamic_lots_on" : False,
    "dynamic_lookback" : 20,
    "friday_lots" : 0.01,
    "afternoon_lots" : 0.01,
    # --- Filtri giorno e direzione ---
    "no_monday_trade" : False,
    "no_buy" : False,
    "symbol_blacklist" : '',
    "hyper_on" : False,
    "hyper_symbols" : 'CHFJPY+,EURCAD+,NZDUSD+',
    # --- TP, RR, Trailing e Break-Even ---
    "tp_percent" : 35,
    "tp_percent_min" : 20,
    "tp_adaptive" : True,
    "max_tp_pips" : 0,
    "min_rr" : 1.0,
    "be_pips" : 15,
    "be_profit" : 5,
    "trailing_on" : True,
    "trail_activate" : 1.5,
    "trail_atr_mult" : 0.5,
    "trail_step_pips" : 5,
    # --- Filtri standard ---
    "rv_max" : 30,
    "adr_max" : 50.0,
    "max_consecutive" : 15,
    "min_ema_gap_pct" : 0.05,
    "rev_min_ema_gap_pct" : 0.1,
    # --- Filtro RX ---
    "rx_required" : False,
    "rx_max_age" : 4,
    "rx_bonus_score" : True,
    # --- Modulo Breakout ---
    "breakout_on" : True,
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
    "reversal_on" : False,
    "dynamic_reversal_on" : True,
    "reversal_observe" : False,
    "rev_lots" : 0.01,
    "reversal_rv" : 65,
    "reversal_rv_max" : 120,
    "reversal_adr" : 90.0,
    "rev_req_decel" : True,
    "rev_min_decel" : 5,
    "rev_req_rx" : True,
    "rev_rx_bonus" : True,
    "rev_req_diverg" : True,
    "rev_diverg_bars" : 8,
    "rev_req_hist_flip" : True,
    "rev_max_hist_age" : 3,
    # --- Orari e sessione ---
    "session_filter_on" : True,
    "session_start_utc" : 9,
    "session_end_utc" : 14,
    "time_offset" : -1,
    "no_night_trade" : True,
    "night_start_h" : 23,
    "night_end_h" : 7,
    "sunday_start_h" : 23,
    "fri_close_profit_h" : 20,
    "fri_close_profit_m" : 0,
    "fri_close_loss_h" : 22,
    "fri_close_loss_m" : 0,
    "fri_force_close_h" : 22,
    "fri_force_close_m" : 30,
    # --- Dati, AI e log ---
    "data_mode" : 1,
    "csv_file" : 'AI_M15_LIGHT.csv',
    "csv_max_age_sec" : 0,
    "radar_indicator" : 'THE_PROFIT_RADAR_PRO_by_ULTIMA_MARKETS_v2_7',
    "export_csv" : True,
    "auto_fallback" : True,
    "fallback_after" : 1,
    "show_export_btn" : True,
    "auto_export" : True,
    "strategy_test" : False,
    "ai_url" : 'https://profit-radar-ai.onrender.com/predict',
    "ai_timeout" : 18000,
    "ai_log" : True,
    "test_trade" : False,
    # --- Tecnici e sicurezza ---
    "magic_number" : 270101,
    "max_slippage" : 3,
    "max_spread" : 30,
    "spread_dyn_mult" : 3.0,
    "atr_mult" : 1.2,
    "atr_period" : 14,
    "fractal_bars" : 50,
    # --- Dashboard grafico MT4 ---
    "dash_x" : 1300,
    "dash_y" : 30,
    "dash_font_size" : 9,
    "dash_color" : 16777215,
    "dash_bg_color" : 2626580,
    "dash_bg" : True,
}

ea_status = {
    "last_update": None, "balance": 0, "equity": 0,
    "open_trades": 0, "daily_pnl": 0, "daily_wins": 0,
    "daily_losses": 0, "consecutive_losses": 0,
    "daily_win_amount": 0, "daily_loss_amount": 0, "loss_weight": 1.5,
    "ai_calls": 0, "ai_confirm": 0, "ai_reject": 0,
    "ai_errors": 0, "ai_missed_trades": 0,
    "warmup_ok": False, "warmup_last": None,
    "data_source": "", "cross_active": 0, "cross_total": 0,
    "daily_stopped": False, "daily_stop_on": True, "account_currency": "EUR", "ea_version": "",
    "peaks": {},
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
        updatable = ["aggressiveness", "use_ai", "ai_min_conf", "send_feedback", 
    "daily_stop_on", "max_consec_loss", "loss_weight", "max_concurrent", 
    "max_per_pair", "fixed_lots", "max_lots_cap", "max_lots_safety", 
    "dynamic_lots_on", "dynamic_lookback", "friday_lots", "afternoon_lots", 
    "no_monday_trade", "no_buy", "symbol_blacklist", "hyper_on", "hyper_symbols", 
    "tp_percent", "tp_percent_min", "tp_adaptive", "max_tp_pips", "min_rr", "be_pips", 
    "be_profit", "trailing_on", "trail_activate", "trail_atr_mult", "trail_step_pips", 
    "rv_max", "adr_max", "max_consecutive", "min_ema_gap_pct", "rev_min_ema_gap_pct", 
    "rx_required", "rx_max_age", "rx_bonus_score", "breakout_on", 
    "breakout_min_light", "breakout_ema_gap_pct", "breakout_max_rv", 
    "breakout_max_adx", "breakout_max_adr", "breakout_min_rr", "breakout_req_rx", 
    "breakout_max_rx_age", "breakout_atr_exp", "breakout_price_ema", 
    "breakout_min_body", "breakout_score_bonus", "reversal_on", "dynamic_reversal_on", "reversal_observe", 
    "rev_lots", "reversal_rv", "reversal_rv_max", "reversal_adr", "rev_req_decel", 
    "rev_min_decel", "rev_req_rx", "rev_rx_bonus", "rev_req_diverg", 
    "rev_diverg_bars", "rev_req_hist_flip", "rev_max_hist_age", "session_filter_on", 
    "session_start_utc", "session_end_utc", "time_offset", "no_night_trade", 
    "night_start_h", "night_end_h", "sunday_start_h", "fri_close_profit_h", 
    "fri_close_profit_m", "fri_close_loss_h", "fri_close_loss_m", "fri_force_close_h", 
    "fri_force_close_m", "data_mode", "csv_file", "csv_max_age_sec", 
    "radar_indicator", "export_csv", "auto_fallback", "fallback_after", 
    "show_export_btn", "auto_export", "strategy_test", "ai_url", "ai_timeout", 
    "ai_log", "test_trade", "magic_number", "max_slippage", "max_spread", 
    "spread_dyn_mult", "atr_mult", "atr_period", "fractal_bars", "dash_x", "dash_y", 
    "dash_font_size", "dash_color", "dash_bg_color", "dash_bg"]
        updated = []
        for key in updatable:
            if key in data:
                old_val = cfg.get(key)
                new_val = data[key]
                bool_keys = {"use_ai", "send_feedback", "daily_stop_on", "dynamic_lots_on", "no_monday_trade", "no_buy", "hyper_on", "tp_adaptive", "trailing_on", "rx_required", "rx_bonus_score", "breakout_on", "breakout_req_rx", "breakout_atr_exp", "breakout_price_ema", "reversal_on", "dynamic_reversal_on", "reversal_observe", "rev_req_decel", "rev_req_rx", "rev_rx_bonus", "rev_req_diverg", "rev_req_hist_flip", "session_filter_on", "no_night_trade", "export_csv", "auto_fallback", "show_export_btn", "auto_export", "strategy_test", "ai_log", "test_trade", "dash_bg"}

                int_keys = {"aggressiveness", "ai_min_conf", "max_consec_loss", "max_concurrent", "max_per_pair", "dynamic_lookback", "tp_percent", "tp_percent_min", "max_tp_pips", "be_pips", "be_profit", "trail_step_pips", "rv_max", "max_consecutive", "rx_max_age", "breakout_min_light", "breakout_max_rv", "breakout_max_rx_age", "breakout_score_bonus", "reversal_rv", "reversal_rv_max", "rev_min_decel", "rev_diverg_bars", "rev_max_hist_age", "session_start_utc", "session_end_utc", "time_offset", "night_start_h", "night_end_h", "sunday_start_h", "fri_close_profit_h", "fri_close_profit_m", "fri_close_loss_h", "fri_close_loss_m", "fri_force_close_h", "fri_force_close_m", "data_mode", "csv_max_age_sec", "fallback_after", "ai_timeout", "magic_number", "max_slippage", "max_spread", "atr_period", "fractal_bars", "dash_x", "dash_y", "dash_font_size", "dash_color", "dash_bg_color"}

                float_keys = {"loss_weight", "fixed_lots", "max_lots_cap", "max_lots_safety", "friday_lots", "afternoon_lots", "min_rr", "trail_activate", "trail_atr_mult", "adr_max", "min_ema_gap_pct", "rev_min_ema_gap_pct", "breakout_ema_gap_pct", "breakout_max_adx", "breakout_max_adr", "breakout_min_rr", "breakout_min_body", "rev_lots", "reversal_adr", "spread_dyn_mult", "atr_mult"}

                str_keys = {"symbol_blacklist", "hyper_symbols", "csv_file", "radar_indicator", "ai_url"}

                if key in bool_keys:

                    new_val = bool(new_val)

                elif key in int_keys:

                    new_val = int(new_val)

                elif key in float_keys:

                    new_val = float(new_val)

                elif key in str_keys:

                    new_val = str(new_val)
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


def get_trade_stats():
    stats = {}
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
                    
                    stats[sym] = {
                        "count": count,
                        "avg_rv": round(avg_rv, 1),
                        "max_rv": round(max_rv, 1),
                        "win_rate": round(win_rate, 1)
                    }
        except Exception as e:
            print("[STATS ERROR]", str(e))
    return stats


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
        "trade_stats": get_trade_stats(),
    }
    # Doppia sicurezza: pulisci tutto da NaN/Inf
    result = sanitize_for_json(result)
    return jsonify(result)


# ============================================================
#  GPT A/B TEST
# ============================================================

GPT_SYSTEM_PROMPT = """Sei un analista forex quantitativo esperto. Valuta trade candidate e rispondi in formato json.

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

Feature avanzate istogramma:
- hist_consec_color: candele consecutive stesso colore (1-3=fresco, 4-8=in corso, 9+=maturo)
- hist_bars_since_gray: candele dall'ultimo GRAY (meno=segnale fresco)
- hist_cycle_count: cicli verde-rosso completati oggi (piu cicoli = movimento frammentato)
- hist_crossed_zero: 1 se l'istogramma ha attraversato lo zero di recente (inversione)
- hist_bar_slope: tendenza altezza barre (positivo=crescente=accelerazione, negativo=decelerazione)
- hist_pullback_depth: profondità ultimo pullback (0=minimo, 100=molto profondo)
- hist_bar_ratio: barra attuale vs media storica (>1.5=accelerazione forte, <0.5=indebolimento)

REGOLE:
- Se i dati sono insufficienti o ambigui, dai confidenza bassa (<50)
- Non avere paura di dire HOLD se il trade non e' chiaro

Rispondi SOLO con un oggetto json: {"signal":"BUY" o "SELL" o "HOLD","confidence":0-100,"reasoning":"motivo in 1 frase"}"""


def call_gpt(data):
    if not OPENAI_API_KEY:
        return {"signal": "HOLD", "confidence": 0, "reasoning": "API key non configurata", "error": True}

    symbol = data.get("symbol", "")
    direction = data.get("direction", "BUY")
    module = data.get("module", "STD")
    gpt_logger.info(f"📤 RICHIESTA GPT | {symbol} {direction} ({module})")

    try:
        import urllib.request
        import urllib.error

        rv = float(data.get("rv", 0))
        adx = float(data.get("adx", 0))
        adr_pct = float(data.get("adr_pct", 0))
        direction = data.get("direction", "BUY")
        module = data.get("module", "STD")
        hist = data.get("hist", "UNKNOWN")
        symbol = data.get("symbol", "")
        nm = float(data.get("nm", 0))

        hist_consec = int(data.get("hist_consec_color", 0))
        hist_since_gray = int(data.get("hist_bars_since_gray", 0))
        hist_cycles = int(data.get("hist_cycle_count", 0))
        hist_zero = int(data.get("hist_crossed_zero", 0))
        hist_slope = float(data.get("hist_bar_slope", 0))
        hist_pullback = float(data.get("hist_pullback_depth", 0))
        hist_bar_ratio = float(data.get("hist_bar_ratio", 1))

        user_msg = f"""Valuta questo trade:
Simbolo: {symbol}
Direzione proposta: {direction}
Modulo: {module}
Radar Value: {rv}
ADX(14): {adx}
ADR%: {adr_pct}%
Histogram: {hist}
Normalized Momentum: {nm}
Barre consecutive: {hist_consec}
Candele da ultimo GRAY: {hist_since_gray}
Cicli oggi: {hist_cycles}
Attraversato zero: {"SI" if hist_zero else "NO"}
Slope barre: {hist_slope:.4f}
Pullback depth: {hist_pullback:.1f}%
Barra/Media ratio: {hist_bar_ratio:.2f}"""

        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            # gpt-5-nano e' un modello di reasoning: "minimal" riduce drasticamente
            # il tempo di "pensiero" interno (per una decisione BUY/SELL non serve
            # ragionare a lungo). Passa da ~7-13s a ~1-2s.
            "reasoning_effort": "minimal",
            # La risposta e' solo {signal, confidence, reasoning}: ~150 token bastano.
            # Il budget include i reasoning tokens, quindi teniamo un margine.
            "max_completion_tokens": 400,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
            method="POST"
        )

        try:
            # Timeout 12s: rete di sicurezza. Deve restare SOTTO il timeout EA (18s)
            # cosi' l'EA non si arrende prima che il server finisca.
            with urllib.request.urlopen(req, timeout=12) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            error_body = he.read().decode("utf-8") if he.fp else ""
            gpt_logger.error(f"❌ ERRORE HTTP {he.code} | {symbol} | {error_body[:300]}")
            return {"signal": "HOLD", "confidence": 0,
                    "reasoning": f"OpenAI HTTP {he.code}: {error_body[:200]}",
                    "model": GPT_MODEL, "error": True}

        content = result["choices"][0]["message"]["content"]
        gpt_response = json.loads(content)
        usage = result.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        signal_out = gpt_response.get("signal", "HOLD").upper()
        confidence_out = min(100, max(0, int(gpt_response.get("confidence", 0))))
        reasoning_out = gpt_response.get("reasoning", "")

        gpt_logger.info(f"📥 RISPOSTA GPT | {symbol} → {signal_out} {confidence_out}% | 💰 Tokens: {total_tokens} | {reasoning_out[:80]}")

        return {
            "signal": signal_out,
            "confidence": confidence_out,
            "reasoning": reasoning_out,
            "model": GPT_MODEL, "error": False
        }
    except Exception as e:
        gpt_logger.error(f"❌ ERRORE GPT | {symbol} | {str(e)}")
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


@app.route("/ab_outcomes", methods=["GET"])
def ab_outcomes():
    """Statistiche 'chi aveva ragione' (LightGBM vs GPT) sui trade gia' chiusi.
    Risponde alla domanda: quando erano DISCORDI, chi ha azzeccato di piu'?"""
    result = {
        "total_resolved": 0,
        "diff": {"count": 0, "lgbm_right": 0, "gpt_right": 0, "lgbm_right_pct": 0},
        "same": {"count": 0, "won": 0, "win_rate": 0},
        "note": "Servono molti dati per conclusioni affidabili (>30 DIFF).",
    }
    if not os.path.exists(AB_OUTCOMES_PATH):
        return jsonify(result)
    try:
        df = pd.read_csv(AB_OUTCOMES_PATH)
        result["total_resolved"] = len(df)

        diff = df[df["agreement"] == "DIFF"]
        if len(diff) > 0:
            lr = int((diff["lgbm_right"] == "SI").sum())
            gr = int((diff["gpt_right"] == "SI").sum())
            result["diff"] = {
                "count": len(diff),
                "lgbm_right": lr,
                "gpt_right": gr,
                "lgbm_right_pct": round(lr / len(diff) * 100, 1),
                "net_pips_if_followed_lgbm": round(float(diff["pips"].sum()), 1),
            }

        same = df[df["agreement"] == "SAME"]
        if len(same) > 0:
            won = int((same["outcome"] == "WIN").sum())
            result["same"] = {
                "count": len(same),
                "won": won,
                "win_rate": round(won / len(same) * 100, 1),
            }
        return jsonify(result)
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


@app.route("/test_gpt", methods=["GET"])
def test_gpt():
    """Testa la connessione OpenAI con il modello configurato e mostra l'errore esatto."""
    import urllib.request, urllib.error

    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY non configurata"})

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "user", "content": "Rispondi solo con json: {\"test\": \"ok\"}"}
        ],
        "max_completion_tokens": 2000,
        "response_format": {"type": "json_object"}
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        return jsonify({
            "status": "ok",
            "model": GPT_MODEL,
            "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model_used": result.get("model", ""),
            "usage": result.get("usage", {}),
        })
    except urllib.error.HTTPError as he:
        error_body = he.read().decode("utf-8") if he.fp else ""
        return jsonify({
            "status": "error",
            "http_code": he.code,
            "model": GPT_MODEL,
            "openai_error": error_body[:1000],
            "reason": "Il modello non esiste o la key non ha accesso"
        })
    except Exception as e:
        return jsonify({"status": "error", "model": GPT_MODEL, "exception": str(e)})


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

/* === TOOLTIP: aiuto a scomparsa === */
.tooltip{position:relative;display:inline-block;margin-left:5px;cursor:help;color:#4fc3f7;font-weight:700}
.tooltip .tooltiptext{visibility:hidden;width:260px;background:#1a1a35;color:#e0e0e0;text-align:left;border-radius:8px;padding:10px;border:1px solid #4fc3f7;position:absolute;z-index:100;bottom:125%;left:50%;margin-left:-130px;opacity:0;transition:opacity .2s;font-size:.85em;line-height:1.4;text-transform:none;box-shadow:0 4px 12px rgba(0,0,0,.5)}
.tooltip .tooltiptext::after{content:'';position:absolute;top:100%;left:50%;margin-left:-5px;border-width:5px;border-style:solid;border-color:#4fc3f7 transparent transparent transparent}
.tooltip:hover .tooltiptext{visibility:visible;opacity:1}

/* === SEZIONI COLLASSABILI (details/summary) === */
details.section {
  transition: all 0.3s ease;
}
details.section summary {
  list-style: none;
  cursor: pointer;
  outline: none;
  user-select: none;
}
details.section summary::-webkit-details-marker {
  display: none;
}
details.section summary h2 {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: 0 !important;
}
details.section summary h2::after {
  content: '▼';
  font-size: 0.75em;
  color: #4fc3f7;
  transition: transform 0.2s ease;
  margin-left: auto;
}
details[open].section summary h2::after {
  transform: rotate(180deg);
}
details.section > :not(summary) {
  margin-top: 14px;
}
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
  <div class="card"><div class="val blue" id="crossActive">-</div><div class="lbl">Cross non-gray</div></div>
  <div class="card"><div class="val" id="dailyWL">-</div><div class="lbl">W / L Oggi</div></div>
</div></div>

<div class="section"><h2>Daily Stop (W/L pesato)</h2>
<div class="row">
  <div class="card"><div class="val green" id="dWin">-</div><div class="lbl">Vinti EUR</div></div>
  <div class="card"><div class="val red" id="dLoss">-</div><div class="lbl">Persi EUR</div></div>
  <div class="card"><div class="val white" id="dConsec">-</div><div class="lbl">Loss di fila</div></div>
  <div class="card"><div class="val" id="dStopState">-</div><div class="lbl">Stato</div></div>
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

<details class="section"><summary><h2>Configurazione principale</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Stile (aggressività)<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto il robot è esigente. 1=Conservativo, 2=Moderato, 3=Aggressivo, 4=Iperconservativo.</span></span></label>
    <input type="number" id="cfgAggressiveness" value="2" min="1" max="4" step="1"></div>
  <div class="cfg-item"><label>AI Attiva<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, l&#39;IA decide se il trade è buono. Se DISATTIVO, il robot decide da solo.</span></span></label>
    <select id="cfgUseAi"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Confidenza minima %<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto l&#39;IA deve essere sicura prima di dire OK. Più alta = meno trade ma più scelti.</span></span></label>
    <input type="number" id="cfgAiMinConf" value="70" min="50" max="95" step="1"></div>
  <div class="cfg-item"><label>Invia Feedback<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot manda i risultati al server per insegnare all&#39;IA. Lascialo ATTIVO.</span></span></label>
    <select id="cfgSendFeedback"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>🛑 Daily Stop<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot smette di tradare quel giorno quando ha perso troppo. Come dire &#39;basta!&#39;.</span></span></label>
    <select id="cfgDailyStopOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Max loss consecutivi<span class="tooltip"> ⓘ<span class="tooltiptext">Dopo quante perdite di fila il robot si ferma quel giorno.</span></span></label>
    <input type="number" id="cfgMaxConsecLoss" value="2" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>Peso perdite (x vincite)<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto contano le perdite rispetto alle vincite. 1.5 = una perdita vale come una vittoria e mezza.</span></span></label>
    <input type="number" id="cfgLossWeight" value="1.5" min="1.0" max="5.0" step="0.1"></div>
  <div class="cfg-item"><label>Max trade aperti<span class="tooltip"> ⓘ<span class="tooltiptext">Quanti trade il robot può tenere aperti contemporaneamente.</span></span></label>
    <input type="number" id="cfgMaxConcurrent" value="10" min="1" max="28" step="1"></div>
  <div class="cfg-item"><label>Max trade per coppia<span class="tooltip"> ⓘ<span class="tooltiptext">Quanti trade può aprire sullo stesso cross contemporaneamente.</span></span></label>
    <input type="number" id="cfgMaxPerPair" value="1" min="1" max="5" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Lotto, rischio e fasce</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Lotto base<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto compra o vende in un trade normale.</span></span></label>
    <input type="number" id="cfgFixedLots" value="0.01" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto max cap<span class="tooltip"> ⓘ<span class="tooltiptext">Il robot non aprirà mai un trade più grande di questo, nemmeno se tutto dice di farlo.</span></span></label>
    <input type="number" id="cfgMaxLotsCap" value="0.05" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto max di sicurezza<span class="tooltip"> ⓘ<span class="tooltiptext">Protezione assoluta contro errori o valori pazzi del lotto.</span></span></label>
    <input type="number" id="cfgMaxLotsSafety" value="0.05" min="0.01" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Lotto dinamico<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot alza o abbassa il lotto da solo in base a quanto sta vincendo ultimamente.</span></span></label>
    <select id="cfgDynamicLotsOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Lotto dyn lookback<span class="tooltip"> ⓘ<span class="tooltiptext">Quanti trade passati guarda il robot per decidere il lotto dinamico.</span></span></label>
    <input type="number" id="cfgDynamicLookback" value="20" min="5" max="200" step="1"></div>
  <div class="cfg-item"><label>🗓️ Lotto Venerdi<span class="tooltip"> ⓘ<span class="tooltiptext">0.00 = venerdì chiuso. 0.01 = trade il venerdì con lotto piccolo (fino alle 22:30 Italia).</span></span></label>
    <input type="number" id="cfgFridayLots" value="0.01" min="0.0" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>🌇 Lotto Pomeriggio<span class="tooltip"> ⓘ<span class="tooltiptext">0.00 = pomeriggio chiuso (fuori dalla sessione principale). 0.01 = trade con lotto piccolo.</span></span></label>
    <input type="number" id="cfgAfternoonLots" value="0.01" min="0.0" max="1.0" step="0.01"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Filtri giorno e direzione</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>🚫 Filtro Lunedi<span class="tooltip"> ⓘ<span class="tooltiptext">ATTIVO = il lunedì il robot sta fermo, nessun trade. DISATTIVO = il lunedì trade normali. Esempio: se metti ATTIVO, lunedì nessun trade.</span></span></label>
    <select id="cfgNoMondayTrade"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>🚫 Filtro BUY<span class="tooltip"> ⓘ<span class="tooltiptext">ATTIVO = il robot vende solo (SELL), nessun acquisto (BUY). DISATTIVO = il robot può anche comprare (BUY). Esempio: se metti ATTIVO, solo SELL.</span></span></label>
    <select id="cfgNoBuy"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Blacklist simboli<span class="tooltip"> ⓘ<span class="tooltiptext">Scrivi qui i simboli che il robot deve evitare, separati da virgola (es. GBPAUD+,AUDJPY+).</span></span></label>
    <input type="text" id="cfgSymbolBlacklist" value="" placeholder=""></div>
  <div class="cfg-item"><label>Iperconservativo ON<span class="tooltip"> ⓘ<span class="tooltiptext">Modalità super-selettiva: solo simboli in whitelist e 6 regole rigide.</span></span></label>
    <select id="cfgHyperOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Whitelist ipercons.<span class="tooltip"> ⓘ<span class="tooltiptext">Simboli permessi quando la modalità iperconservativo è attiva. Separati da virgola.</span></span></label>
    <input type="text" id="cfgHyperSymbols" value="CHFJPY+,EURCAD+,NZDUSD+" placeholder="CHFJPY+,EURCAD+,NZDUSD+"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>TP, RR, Trailing e Break-Even</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>TP % ADR max<span class="tooltip"> ⓘ<span class="tooltiptext">Distanza massima del Take Profit. Più % = TP più lontano e rischioso.</span></span></label>
    <input type="number" id="cfgTpPercent" value="35" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>TP % ADR min<span class="tooltip"> ⓘ<span class="tooltiptext">Distanza minima del Take Profit. Più % = TP più vicino e sicuro.</span></span></label>
    <input type="number" id="cfgTpPercentMin" value="20" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>TP adattivo<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot sceglie il TP tra min e max in base alla forza del trend.</span></span></label>
    <select id="cfgTpAdaptive"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>TP max in pip (0=off)<span class="tooltip"> ⓘ<span class="tooltiptext">Numero massimo di pip per il TP. 0 = usa la % ADR.</span></span></label>
    <input type="number" id="cfgMaxTpPips" value="0" min="0" max="200" step="1"></div>
  <div class="cfg-item"><label>R:R minimo<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto deve essere grande il guadagno possibile rispetto al rischio. 1.5 = guadagno 1.5 volte la perdita.</span></span></label>
    <input type="number" id="cfgMinRr" value="1.0" min="0.5" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Break-Even pip<span class="tooltip"> ⓘ<span class="tooltiptext">Dopo quanti pip di profitto il robot sposta lo stop a protezione del capitale.</span></span></label>
    <input type="number" id="cfgBePips" value="15" min="0" max="100" step="1"></div>
  <div class="cfg-item"><label>Break-Even profitto bloccato<span class="tooltip"> ⓘ<span class="tooltiptext">Quanti pip di profitto blocca quando attiva il break-even.</span></span></label>
    <input type="number" id="cfgBeProfit" value="5" min="0" max="50" step="1"></div>
  <div class="cfg-item"><label>Trailing Stop<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot sposta lo Stop Loss per proteggere i profitti quando il trade va bene.</span></span></label>
    <select id="cfgTrailingOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Trail attiva a R<span class="tooltip"> ⓘ<span class="tooltiptext">Dopo quanto guadagno (in volte il rischio) il trailing stop si attiva.</span></span></label>
    <input type="number" id="cfgTrailActivate" value="1.5" min="0.5" max="5.0" step="0.1"></div>
  <div class="cfg-item"><label>Trail ATR mult<span class="tooltip"> ⓘ<span class="tooltiptext">Distanza del trailing stop dall&#39;EMA, misurata con l&#39;ATR. Più basso = più stretto.</span></span></label>
    <input type="number" id="cfgTrailAtrMult" value="0.5" min="0.1" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Trail step pip<span class="tooltip"> ⓘ<span class="tooltiptext">Il robot muove lo stop solo se migliora di almeno questi pip. Evita troppi spostamenti.</span></span></label>
    <input type="number" id="cfgTrailStepPips" value="5" min="0" max="50" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Filtri standard</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>RV massimo<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto forte deve essere il segnale Radar. Più basso = scarta i segnali troppo deboli.</span></span></label>
    <input type="number" id="cfgRvMax" value="30" min="10" max="100" step="1"></div>
  <div class="cfg-item"><label>ADR% massimo<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto della giornata è già stato fatto. Sopra questa % il mercato è stanco e il robot non entra.</span></span></label>
    <input type="number" id="cfgAdrMax" value="50.0" min="30" max="150" step="1"></div>
  <div class="cfg-item"><label>Max candele consecutive<span class="tooltip"> ⓘ<span class="tooltiptext">Se il trend dura più di N candele, il robot lo considera stanco e non entra.</span></span></label>
    <input type="number" id="cfgMaxConsecutive" value="15" min="5" max="50" step="1"></div>
  <div class="cfg-item"><label>Min gap EMA %<span class="tooltip"> ⓘ<span class="tooltiptext">Distanza minima tra EMA21 e EMA200. Sotto = trend troppo debole.</span></span></label>
    <input type="number" id="cfgMinEmaGapPct" value="0.05" min="0.0" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Min gap EMA % (Reversal)<span class="tooltip"> ⓘ<span class="tooltiptext">Distanza minima EMA per il modulo Reversal. Serve un trend definito da invertire.</span></span></label>
    <input type="number" id="cfgRevMinEmaGapPct" value="0.1" min="0.0" max="1.0" step="0.01"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Filtro RX</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>RX richiesto<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot entra solo se c&#39;è un segnale RX (nuovi massimi/minimi).</span></span></label>
    <select id="cfgRxRequired"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>RX eta max (candele)<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto può essere vecchio il segnale RX per essere ancora valido.</span></span></label>
    <input type="number" id="cfgRxMaxAge" value="4" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>RX bonus punteggio<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il segnale RX aumenta il punteggio interno del trade.</span></span></label>
    <select id="cfgRxBonusScore"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Modulo Breakout</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Breakout attivo<span class="tooltip"> ⓘ<span class="tooltiptext">Modulo che cerca i movimenti che partono dopo il grigio.</span></span></label>
    <select id="cfgBreakoutOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Breakout min candele LIGHT<span class="tooltip"> ⓘ<span class="tooltiptext">Quante candele LIGHT consecutive servono per considerare un breakout.</span></span></label>
    <input type="number" id="cfgBreakoutMinLight" value="2" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>Breakout max gap EMA %<span class="tooltip"> ⓘ<span class="tooltiptext">Gap EMA massimo accettato per un breakout. Sopra = trend troppo forte.</span></span></label>
    <input type="number" id="cfgBreakoutEmaGapPct" value="0.2" min="0.0" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Breakout max RV<span class="tooltip"> ⓘ<span class="tooltiptext">Radar Value massimo per un breakout. Sopra = segnale troppo forte/esaurito.</span></span></label>
    <input type="number" id="cfgBreakoutMaxRv" value="15" min="5" max="50" step="1"></div>
  <div class="cfg-item"><label>Breakout max ADX<span class="tooltip"> ⓘ<span class="tooltiptext">ADX massimo per un breakout. Sopra = trend già troppo maturo.</span></span></label>
    <input type="number" id="cfgBreakoutMaxAdx" value="25" min="10" max="60" step="1"></div>
  <div class="cfg-item"><label>Breakout max ADR%<span class="tooltip"> ⓘ<span class="tooltiptext">ADR% massimo per un breakout. Sopra = mercato stanco.</span></span></label>
    <input type="number" id="cfgBreakoutMaxAdr" value="50" min="30" max="150" step="1"></div>
  <div class="cfg-item"><label>Breakout R:R minimo<span class="tooltip"> ⓘ<span class="tooltiptext">Rapporto rischio/rendimento minimo per il modulo Breakout.</span></span></label>
    <input type="number" id="cfgBreakoutMinRr" value="1.8" min="1.0" max="3.0" step="0.1"></div>
  <div class="cfg-item"><label>Breakout richiede RX<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il breakout richiede anche un segnale RX.</span></span></label>
    <select id="cfgBreakoutReqRx"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Breakout RX eta max<span class="tooltip"> ⓘ<span class="tooltiptext">Eta massima del segnale RX per il breakout.</span></span></label>
    <input type="number" id="cfgBreakoutMaxRxAge" value="2" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>Breakout ATR in espansione<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, richiede che l&#39;ATR sia in espansione.</span></span></label>
    <select id="cfgBreakoutAtrExp"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Breakout prezzo vs EMA<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il prezzo deve essere dalla parte giusta dell&#39;EMA.</span></span></label>
    <select id="cfgBreakoutPriceEma"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Breakout min corpo candela<span class="tooltip"> ⓘ<span class="tooltiptext">Corpo minimo della candela rispetto all&#39;ATR.</span></span></label>
    <input type="number" id="cfgBreakoutMinBody" value="0.3" min="0.1" max="1.0" step="0.05"></div>
  <div class="cfg-item"><label>Breakout bonus punteggio<span class="tooltip"> ⓘ<span class="tooltiptext">Bonus punteggio interno del modulo Breakout (solitamente negativo = più selettivo).</span></span></label>
    <input type="number" id="cfgBreakoutScoreBonus" value="-80" min="-200" max="50" step="5"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Modulo Reversal</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Reversal dinamico (Picchi)<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, l'EA calcola la media dei 4 picchi storici maggiori per ciascun cross invece di usare il valore fisso.</span></span></label>
    <select id="cfgDynamicReversalOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal attivo<span class="tooltip"> ⓘ<span class="tooltiptext">Modulo che cerca i ribaltamenti di trend estremo.</span></span></label>
    <select id="cfgReversalOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal solo osservazione<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il Reversal logga i segnali ma NON apre trade.</span></span></label>
    <select id="cfgReversalObserve"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal lotto<span class="tooltip"> ⓘ<span class="tooltiptext">Lotto dedicato al Reversal. 0 = usa il lotto principale.</span></span></label>
    <input type="number" id="cfgRevLots" value="0.01" min="0.0" max="1.0" step="0.01"></div>
  <div class="cfg-item"><label>Reversal RV minimo<span class="tooltip"> ⓘ<span class="tooltiptext">Radar Value minimo per considerare un trend estremo da invertire.</span></span></label>
    <input type="number" id="cfgReversalRv" value="65" min="30" max="150" step="1"></div>
  <div class="cfg-item"><label>Reversal RV massimo<span class="tooltip"> ⓘ<span class="tooltiptext">Sopra questo RV il robot pensa che il trend sia già invertito o il dato sia anomalo.</span></span></label>
    <input type="number" id="cfgReversalRvMax" value="120" min="50" max="200" step="1"></div>
  <div class="cfg-item"><label>Reversal ADR% minimo<span class="tooltip"> ⓘ<span class="tooltiptext">Il mercato deve aver superato la media giornaliera per un reversal valido.</span></span></label>
    <input type="number" id="cfgReversalAdr" value="90.0" min="50" max="150" step="1"></div>
  <div class="cfg-item"><label>Reversal richiede decelerazione<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, richiede che il trend stia perdendo velocità.</span></span></label>
    <select id="cfgRevReqDecel"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal decelerazione min<span class="tooltip"> ⓘ<span class="tooltiptext">Quanto deve essere la decelerazione del Radar Value.</span></span></label>
    <input type="number" id="cfgRevMinDecel" value="5" min="1" max="50" step="1"></div>
  <div class="cfg-item"><label>Reversal richiede RX<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, richiede un segnale RX (nuovo max/min) per il reversal.</span></span></label>
    <select id="cfgRevReqRx"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal RX bonus<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il segnale RX aumenta il punteggio del reversal.</span></span></label>
    <select id="cfgRevRxBonus"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal richiede divergenza<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, richiede una divergenza per confermare il reversal.</span></span></label>
    <select id="cfgRevReqDiverg"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal candele divergenza<span class="tooltip"> ⓘ<span class="tooltiptext">Finestra di candele in cui cercare la divergenza.</span></span></label>
    <input type="number" id="cfgRevDivergBars" value="8" min="3" max="30" step="1"></div>
  <div class="cfg-item"><label>Reversal richiede flip istogramma<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, richiede che l&#39;istogramma abbia cambiato colore recentemente.</span></span></label>
    <select id="cfgRevReqHistFlip"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal eta max flip<span class="tooltip"> ⓘ<span class="tooltiptext">Quante candele fa può essere avvenuto il cambio colore.</span></span></label>
    <input type="number" id="cfgRevMaxHistAge" value="3" min="1" max="10" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Orari e sessione</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Filtro sessione<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot trade solo nella sessione principale (lotto normale). Fuori = pomeriggio.</span></span></label>
    <select id="cfgSessionFilterOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Sessione inizio UTC<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di inizio della sessione principale (UTC).</span></span></label>
    <input type="number" id="cfgSessionStartUtc" value="9" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Sessione fine UTC<span class="tooltip"> ⓘ<span class="tooltiptext">Ora di fine della sessione principale (UTC). Dopo = pomeriggio.</span></span></label>
    <input type="number" id="cfgSessionEndUtc" value="14" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Fuso orario broker vs Italia<span class="tooltip"> ⓘ<span class="tooltiptext">Differenza di ore tra l&#39;orario del broker e l&#39;orario italiano.</span></span></label>
    <input type="number" id="cfgTimeOffset" value="-1" min="-12" max="12" step="1"></div>
  <div class="cfg-item"><label>Blocco notte<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, il robot non trade di notte (spread alti, poca liquidità).</span></span></label>
    <select id="cfgNoNightTrade"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Notte inizio (Italia)<span class="tooltip"> ⓘ<span class="tooltiptext">Ora italiana di inizio del blocco notturno.</span></span></label>
    <input type="number" id="cfgNightStartH" value="23" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Notte fine (Italia)<span class="tooltip"> ⓘ<span class="tooltiptext">Ora italiana di fine del blocco notturno.</span></span></label>
    <input type="number" id="cfgNightEndH" value="7" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Domenica inizio (Italia)<span class="tooltip"> ⓘ<span class="tooltiptext">Ora italiana in cui la domenica sera il robot può ricominciare a tradare.</span></span></label>
    <input type="number" id="cfgSundayStartH" value="23" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven chiudi profitto ora<span class="tooltip"> ⓘ<span class="tooltiptext">Ora italiana in cui il venerdì chiude i trade in profitto.</span></span></label>
    <input type="number" id="cfgFriCloseProfitH" value="20" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven chiudi profitto min<span class="tooltip"> ⓘ<span class="tooltiptext">Minuti dell&#39;ora in cui chiudere i trade in profitto il venerdì.</span></span></label>
    <input type="number" id="cfgFriCloseProfitM" value="0" min="0" max="59" step="1"></div>
  <div class="cfg-item"><label>Ven chiudi perdita ora<span class="tooltip"> ⓘ<span class="tooltiptext">Ora italiana in cui il venerdì chiude i trade in perdita.</span></span></label>
    <input type="number" id="cfgFriCloseLossH" value="22" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven chiudi perdita min<span class="tooltip"> ⓘ<span class="tooltiptext">Minuti dell&#39;ora in cui chiudere le perdite il venerdì.</span></span></label>
    <input type="number" id="cfgFriCloseLossM" value="0" min="0" max="59" step="1"></div>
  <div class="cfg-item"><label>Ven chiusura forzata ora<span class="tooltip"> ⓘ<span class="tooltiptext">Ora italiana di chiusura forzata totale del venerdì.</span></span></label>
    <input type="number" id="cfgFriForceCloseH" value="22" min="0" max="23" step="1"></div>
  <div class="cfg-item"><label>Ven chiusura forzata min<span class="tooltip"> ⓘ<span class="tooltiptext">Minuti dell&#39;ora di chiusura forzata del venerdì.</span></span></label>
    <input type="number" id="cfgFriForceCloseM" value="30" min="0" max="59" step="1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Dati, AI e log</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Fonte dati<span class="tooltip"> ⓘ<span class="tooltiptext">0=CSV, 1=Auto, 2=CSV forzato. Lascia 1.</span></span></label>
    <input type="number" id="cfgDataMode" value="1" min="0" max="2" step="1"></div>
  <div class="cfg-item"><label>Nome file CSV<span class="tooltip"> ⓘ<span class="tooltiptext">Nome del file CSV con i dati dell&#39;indicatore.</span></span></label>
    <input type="text" id="cfgCsvFile" value="AI_M15_LIGHT.csv"></div>
  <div class="cfg-item"><label>CSV eta max (sec)<span class="tooltip"> ⓘ<span class="tooltiptext">0 = illimitata. Altrimenti scarta il CSV più vecchio di N secondi.</span></span></label>
    <input type="number" id="cfgCsvMaxAgeSec" value="0" min="0" max="3600" step="60"></div>
  <div class="cfg-item"><label>Nome indicatore Radar<span class="tooltip"> ⓘ<span class="tooltiptext">Nome esatto dell&#39;indicatore Radar su MT4.</span></span></label>
    <input type="text" id="cfgRadarIndicator" value="THE_PROFIT_RADAR_PRO_by_ULTIMA_MARKETS_v2_7"></div>
  <div class="cfg-item"><label>Esporta CSV<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, l&#39;EA esporta i dati su CSV.</span></span></label>
    <select id="cfgExportCsv"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Fallback automatico<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, passa al CSV se i dati live mancano.</span></span></label>
    <select id="cfgAutoFallback"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Fallback dopo N tentativi<span class="tooltip"> ⓘ<span class="tooltiptext">Dopo quanti fallimenti dati live usa il CSV.</span></span></label>
    <input type="number" id="cfgFallbackAfter" value="1" min="1" max="10" step="1"></div>
  <div class="cfg-item"><label>Mostra pulsante Export<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, mostra il pulsante Export sul grafico MT4.</span></span></label>
    <select id="cfgShowExportBtn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Export automatico<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, esporta i dati automaticamente ad ogni candela.</span></span></label>
    <select id="cfgAutoExport"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Modalita Strategy Tester<span class="tooltip"> ⓘ<span class="tooltiptext">ATTIVO solo per testare in Strategy Tester.</span></span></label>
    <select id="cfgStrategyTest"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>URL server AI<span class="tooltip"> ⓘ<span class="tooltiptext">Indirizzo del server AI su Render.</span></span></label>
    <input type="text" id="cfgAiUrl" value="https://profit-radar-ai.onrender.com/predict"></div>
  <div class="cfg-item"><label>Timeout AI (ms)<span class="tooltip"> ⓘ<span class="tooltiptext">Quanti millisecondi aspetta la risposta dell&#39;IA prima di arrendersi.</span></span></label>
    <input type="number" id="cfgAiTimeout" value="18000" min="1000" max="60000" step="1000"></div>
  <div class="cfg-item"><label>Log AI dettagliato<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, scrive nel journal di MT4 i dettagli delle chiamate AI.</span></span></label>
    <select id="cfgAiLog"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Apri trade di test<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, all&#39;avvio apre un trade di test. Utile solo per debug.</span></span></label>
    <select id="cfgTestTrade"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Tecnici e sicurezza</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Magic Number<span class="tooltip"> ⓘ<span class="tooltiptext">ID univoco dell&#39;EA. Non cambiarlo a meno che non sai perché.</span></span></label>
    <input type="number" id="cfgMagicNumber" value="270101" min="100000" max="999999" step="1"></div>
  <div class="cfg-item"><label>Max slippage (points)<span class="tooltip"> ⓘ<span class="tooltiptext">Slippage massimo accettato dall&#39;EA.</span></span></label>
    <input type="number" id="cfgMaxSlippage" value="3" min="0" max="50" step="1"></div>
  <div class="cfg-item"><label>Max spread (points)<span class="tooltip"> ⓘ<span class="tooltiptext">Spread massimo assoluto per aprire un trade.</span></span></label>
    <input type="number" id="cfgMaxSpread" value="30" min="5" max="200" step="1"></div>
  <div class="cfg-item"><label>Spread dinamico mult<span class="tooltip"> ⓘ<span class="tooltiptext">Blocca il trade se lo spread è più alto di N volte la media. 0 = disattivato.</span></span></label>
    <input type="number" id="cfgSpreadDynMult" value="3.0" min="0.0" max="10.0" step="0.5"></div>
  <div class="cfg-item"><label>ATR mult per SL<span class="tooltip"> ⓘ<span class="tooltiptext">Stop Loss = ATR x questo valore.</span></span></label>
    <input type="number" id="cfgAtrMult" value="1.2" min="0.5" max="5.0" step="0.1"></div>
  <div class="cfg-item"><label>Periodo ATR<span class="tooltip"> ⓘ<span class="tooltiptext">Periodo dell&#39;indicatore ATR per calcolare SL e trailing.</span></span></label>
    <input type="number" id="cfgAtrPeriod" value="14" min="5" max="50" step="1"></div>
  <div class="cfg-item"><label>Candele fractal SL<span class="tooltip"> ⓘ<span class="tooltiptext">Quante candele indietro guardare per trovare il fractal per lo Stop Loss.</span></span></label>
    <input type="number" id="cfgFractalBars" value="50" min="10" max="200" step="10"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>
<details class="section"><summary><h2>Dashboard grafico MT4</h2></summary>
<div class="config-grid">
  <div class="cfg-item"><label>Dashboard X (pixel)<span class="tooltip"> ⓘ<span class="tooltiptext">Posizione orizzontale della dashboard sul grafico MT4.</span></span></label>
    <input type="number" id="cfgDashX" value="1300" min="0" max="3000" step="10"></div>
  <div class="cfg-item"><label>Dashboard Y (pixel)<span class="tooltip"> ⓘ<span class="tooltiptext">Posizione verticale della dashboard sul grafico MT4.</span></span></label>
    <input type="number" id="cfgDashY" value="30" min="0" max="2000" step="10"></div>
  <div class="cfg-item"><label>Dashboard font size<span class="tooltip"> ⓘ<span class="tooltiptext">Dimensione del testo della dashboard su MT4.</span></span></label>
    <input type="number" id="cfgDashFontSize" value="9" min="6" max="20" step="1"></div>
  <div class="cfg-item"><label>Dashboard colore testo<span class="tooltip"> ⓘ<span class="tooltiptext">Colore del testo della dashboard su MT4.</span></span></label>
    <select id="cfgDashColor"><option value="16777215" selected>Bianco</option><option value="0">Nero</option><option value="255">Rosso</option><option value="65280">Verde lime</option><option value="16711680">Blu</option><option value="65535">Giallo</option><option value="16776960">Ciano</option><option value="16711935">Magenta</option><option value="42495">Arancione</option><option value="9109504">Blu scuro</option><option value="3100495">Grigio ardesia</option></select></div>
  <div class="cfg-item"><label>Dashboard colore sfondo<span class="tooltip"> ⓘ<span class="tooltiptext">Colore dello sfondo della dashboard su MT4.</span></span></label>
    <select id="cfgDashBgColor"><option value="16777215">Bianco</option><option value="0">Nero</option><option value="255">Rosso</option><option value="65280">Verde lime</option><option value="16711680">Blu</option><option value="65535">Giallo</option><option value="16776960">Ciano</option><option value="16711935">Magenta</option><option value="42495">Arancione</option><option value="9109504">Blu scuro</option><option value="3100495">Grigio ardesia</option></select></div>
  <div class="cfg-item"><label>Dashboard sfondo<span class="tooltip"> ⓘ<span class="tooltiptext">Se ATTIVO, mostra lo sfondo della dashboard su MT4.</span></span></label>
    <select id="cfgDashBg"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></details>


<details class="section">
  <summary><h2>📊 Analisi Picchi e Statistiche Cross</h2></summary>
  <div style="font-size: 0.8em; color: #aaa; margin-bottom: 12px; line-height: 1.4;">
    Questa finestra interattiva mostra l'analisi statistica dei picchi di tendenza (Radar Value) e del Win Rate registrato storicamente per ciascuno dei 28 cross forex. Puoi confrontare il picco dinamico calcolato dall'EA con le statistiche reali dei trade passati per regolare al meglio la sensibilità.
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

    // --- Popola Tabella Statistiche e Picchi Dinamici ---
    const statsTable = document.getElementById('statsTable');
    if (statsTable) {
      const stats = d.trade_stats || {};
      const peaks = (d.ea && d.ea.peaks) ? d.ea.peaks : {};
      const unified = {};
      
      Object.keys(stats).forEach(rawSym => {
        const clean = rawSym.replace('+', '').toUpperCase().trim();
        if (!unified[clean]) {
          unified[clean] = { count: 0, win_rate: 0, avg_rv: 0, max_rv: 0, peak: '-' };
        }
        const s = stats[rawSym];
        unified[clean].count = s.count || 0;
        unified[clean].win_rate = s.win_rate != null ? s.win_rate : 0;
        unified[clean].avg_rv = s.avg_rv != null ? s.avg_rv : 0;
        unified[clean].max_rv = s.max_rv != null ? s.max_rv : 0;
      });
      
      Object.keys(peaks).forEach(rawSym => {
        const clean = rawSym.replace('+', '').toUpperCase().trim();
        if (!unified[clean]) {
          unified[clean] = { count: 0, win_rate: 0, avg_rv: 0, max_rv: 0, peak: '-' };
        }
        unified[clean].peak = peaks[rawSym] != null ? peaks[rawSym] : '-';
      });
      
      const sortedSymbols = Object.keys(unified).sort();
      
      if (sortedSymbols.length > 0) {
        statsTable.innerHTML = sortedSymbols.map(sym => {
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
        }).join('');
      } else {
        statsTable.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;padding:12px;">In attesa del primo sync dell&apos;EA...</td></tr>';
      }
    }

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
    if(limit>0){
      let pct=Math.min(100,Math.round(loss/limit*100));
      bar.style.width=pct+'%';
      bar.style.background=pct<50?'#81c784':pct<80?'#ffd54f':'#ef5350';
      document.getElementById('dStopPct').textContent=pct+'%';
      document.getElementById('dStopDetail').textContent='persi '+fmt(loss)+' / soglia '+fmt(limit)+' EUR (peso x'+lw+') | margine '+fmt(limit-loss);
    } else {
      bar.style.width='0%';
      document.getElementById('dStopPct').textContent='-';
      document.getElementById('dStopDetail').textContent='In attesa della 1a vincita (per ora conta solo lo stop loss di fila)';
    }

    document.getElementById('cfgDailyStopOn').value=(cfg.daily_stop_on!==false)?'true':'false';
    document.getElementById('cfgAggressiveness').value=cfg.aggressiveness||2;
    document.getElementById('cfgSendFeedback').value=cfg.send_feedback?'true':'false';
    document.getElementById('cfgUseAi').value=cfg.use_ai?'true':'false';
    document.getElementById('cfgAiMinConf').value=cfg.ai_min_conf||70;
    document.getElementById('cfgMaxConsecLoss').value=cfg.max_consec_loss||2;
    document.getElementById('cfgLossWeight').value=cfg.loss_weight||1.5;
    document.getElementById('cfgRvMax').value=cfg.rv_max||30;
    document.getElementById('cfgAdrMax').value=cfg.adr_max||60;
    document.getElementById('cfgMinRr').value=cfg.min_rr||1.5;
    document.getElementById('cfgTpPercent').value=cfg.tp_percent||35;
    document.getElementById('cfgTpPercentMin').value=cfg.tp_percent_min||20;
    document.getElementById('cfgMaxTpPips').value=cfg.max_tp_pips||0;
    document.getElementById('cfgFixedLots').value=cfg.fixed_lots||0.01;
    document.getElementById('cfgMaxLotsCap').value=cfg.max_lots_cap||0.05;
    document.getElementById('cfgDynamicLotsOn').value=cfg.dynamic_lots_on?'true':'false';
    document.getElementById('cfgDynamicLookback').value=cfg.dynamic_lookback||20;
    document.getElementById('cfgFridayLots').value=(cfg.friday_lots!=null?cfg.friday_lots.toFixed(2):'0.01');
    document.getElementById('cfgAfternoonLots').value=(cfg.afternoon_lots!=null?cfg.afternoon_lots.toFixed(2):'0.01');
    document.getElementById('cfgNoMondayTrade').value=cfg.no_monday_trade?'true':'false';
    document.getElementById('cfgNoBuy').value=cfg.no_buy?'true':'false';
    document.getElementById('cfgSymbolBlacklist').value=cfg.symbol_blacklist||'';
    document.getElementById('cfgTrailingOn').value=cfg.trailing_on?'true':'false';
    document.getElementById('cfgTrailActivate').value=cfg.trail_activate||1.5;
    document.getElementById('cfgTrailAtrMult').value=cfg.trail_atr_mult||0.5;
    document.getElementById('cfgTrailStepPips').value=cfg.trail_step_pips||5;
    document.getElementById('cfgHyperOn').value=cfg.hyper_on?'true':'false';
    document.getElementById('cfgBreakoutOn').value=cfg.breakout_on?'true':'false';
    document.getElementById('cfgReversalOn').value=cfg.reversal_on?'true':'false';
    document.getElementById('cfgDynamicReversalOn').value=cfg.dynamic_reversal_on?'true':'false';
    document.getElementById('cfgMaxConcurrent').value=cfg.max_concurrent||10;
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
function saveAllConfig(btn = null){
  const cfg={
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
    breakout_min_light:parseInt(document.getElementById('cfgBreakoutMinLight').value),
    breakout_ema_gap_pct:parseFloat(document.getElementById('cfgBreakoutEmaGapPct').value),
    breakout_max_rv:parseInt(document.getElementById('cfgBreakoutMaxRv').value),
    breakout_max_adx:parseFloat(document.getElementById('cfgBreakoutMaxAdx').value),
    breakout_max_adr:parseFloat(document.getElementById('cfgBreakoutMaxAdr').value),
    breakout_min_rr:parseFloat(document.getElementById('cfgBreakoutMinRr').value),
    breakout_req_rx:document.getElementById('cfgBreakoutReqRx').value==='true',
    breakout_max_rx_age:parseInt(document.getElementById('cfgBreakoutMaxRxAge').value),
    breakout_atr_exp:document.getElementById('cfgBreakoutAtrExp').value==='true',
    breakout_price_ema:document.getElementById('cfgBreakoutPriceEma').value==='true',
    breakout_min_body:parseFloat(document.getElementById('cfgBreakoutMinBody').value),
    breakout_score_bonus:parseInt(document.getElementById('cfgBreakoutScoreBonus').value),
    reversal_on:document.getElementById('cfgReversalOn').value==='true',
    dynamic_reversal_on:document.getElementById('cfgDynamicReversalOn').value==='true',
    reversal_observe:document.getElementById('cfgReversalObserve').value==='true',
    rev_lots:parseFloat(document.getElementById('cfgRevLots').value),
    reversal_rv:parseInt(document.getElementById('cfgReversalRv').value),
    reversal_rv_max:parseInt(document.getElementById('cfgReversalRvMax').value),
    reversal_adr:parseFloat(document.getElementById('cfgReversalAdr').value),
    rev_req_decel:document.getElementById('cfgRevReqDecel').value==='true',
    rev_min_decel:parseInt(document.getElementById('cfgRevMinDecel').value),
    rev_req_rx:document.getElementById('cfgRevReqRx').value==='true',
    rev_rx_bonus:document.getElementById('cfgRevRxBonus').value==='true',
    rev_req_diverg:document.getElementById('cfgRevReqDiverg').value==='true',
    rev_diverg_bars:parseInt(document.getElementById('cfgRevDivergBars').value),
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
    dash_bg:document.getElementById('cfgDashBg').value==='true',};
  fetch(API+'/ea_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)}).then(r=>r.json()).then(d=>{
    const m_text = d.status==='ok'?'✅ '+d.message:'❌ '+(d.message||'errore');
    const m_color = d.status==='ok'?'#81c784':'#ef5350';
    if(btn) {
      let m = btn.parentNode.querySelector('.cfg-msg');
      if(!m) {
        m = document.createElement('span');
        m.className = 'cfg-msg';
        m.style.fontSize = '0.85em';
        m.style.marginLeft = '12px';
        m.style.alignSelf = 'center';
        btn.parentNode.appendChild(m);
      }
      m.textContent = m_text;
      m.style.color = m_color;
      setTimeout(()=>{m.textContent=''},5000);
    }
    const g=document.getElementById('cfgMsg');
    if(g){g.textContent=m_text;g.style.color=m_color;setTimeout(()=>{g.textContent=''},5000);}
  }).catch(()=>{
    const err_text = '❌ Errore connessione';
    if(btn) {
      let m = btn.parentNode.querySelector('.cfg-msg');
      if(!m) {
        m = document.createElement('span');
        m.className = 'cfg-msg';
        m.style.fontSize = '0.85em';
        m.style.marginLeft = '12px';
        m.style.alignSelf = 'center';
        btn.parentNode.appendChild(m);
      }
      m.textContent = err_text;
      m.style.color = '#ef5350';
      setTimeout(()=>{m.textContent=''},5000);
    }
    const g=document.getElementById('cfgMsg');
    if(g){g.textContent=err_text;g.style.color='#ef5350';setTimeout(()=>{g.textContent=''},5000);}
  });
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


@app.route("/gpt_log", methods=["GET"])
def gpt_log():
    """Mostra le ultime righe del log GPT."""
    lines_count = int(request.args.get("lines", 50))
    if not os.path.exists(LOG_PATH):
        return jsonify({"log": "Nessun log GPT ancora", "lines": 0})
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        last_lines = all_lines[-lines_count:]
        return jsonify({"log": "".join(last_lines), "lines": len(last_lines), "total": len(all_lines)})
    except Exception as e:
        return jsonify({"log": f"Errore lettura log: {e}", "lines": 0})


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """Scarica un file dalla cartella data (ab_results.csv, requests_log.csv, gpt_api.log, feedback.csv)."""
    allowed = ["ab_results.csv", "requests_log.csv", "gpt_api.log", "feedback.csv", "imported_tradelog.csv", "PRP_TradeLog.csv"]
    if filename not in allowed:
        return jsonify({"error": f"File non permesso. Usa uno di: {allowed}"}), 403
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": f"File {filename} non trovato"}), 404
    from flask import send_file
    mime = "text/csv" if filename.endswith(".csv") else "text/plain"
    return send_file(filepath, mimetype=mime, as_attachment=True, download_name=filename)


def _github_put_file(fname, github_token, repo, branch):
    """Carica un singolo file su GitHub. Ritorna dict con esito."""
    import urllib.request, base64
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return {"file": fname, "status": "skipped", "reason": "non esiste"}
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        if not content or len(content) < 10:
            return {"file": fname, "status": "skipped", "reason": "vuoto"}
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        api_url = f"https://api.github.com/repos/{repo}/contents/Data/{fname}"
        sha = None
        try:
            req = urllib.request.Request(api_url, headers={
                "Authorization": f"token {github_token}", "User-Agent": "ProfitRadarAI"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                sha = json.loads(resp.read().decode("utf-8")).get("sha")
        except Exception:
            pass
        payload = {"message": f"auto-save: {fname} ({len(content)} bytes)",
                   "content": encoded, "branch": branch}
        if sha:
            payload["sha"] = sha
        req = urllib.request.Request(api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"token {github_token}",
                     "Content-Type": "application/json", "User-Agent": "ProfitRadarAI"},
            method="PUT")
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        return {"file": fname, "status": "saved", "size": len(content)}
    except Exception as e:
        return {"file": fname, "status": "error", "reason": str(e)}


# File salvati nel backup (CSV: dati grezzi; il modello si riaddestra al boot)
BACKUP_FILES = ["ab_results.csv", "requests_log.csv", "gpt_api.log",
                "feedback.csv", "ab_outcomes.csv", "PRP_ReversalSignals.csv"]


def auto_backup_to_github():
    """Backup automatico (chiamato dal /feedback). Ritorna n. file salvati."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return 0
    repo = "gabriworkia/profit-radar-ai"
    saved = 0
    for fname in BACKUP_FILES:
        r = _github_put_file(fname, token, repo, "data-backup")
        if r.get("status") == "saved":
            saved += 1
    return saved


@app.route("/save_to_github", methods=["POST"])
def save_to_github():
    """Salva i file di log su GitHub branch 'data-backup' (NON triggera' Render deploy)."""
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        return jsonify({"error": "GITHUB_TOKEN non configurata. Aggiungila come env var su Render."}), 200

    repo = "gabriworkia/profit-radar-ai"
    branch = "data-backup"  # Branch separato — NON triggera' Render deploy!
    import urllib.request, urllib.error, base64

    files_to_save = BACKUP_FILES
    results = []

    for fname in files_to_save:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            results.append({"file": fname, "status": "skipped", "reason": "non esiste"})
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            if not content or len(content) < 10:
                results.append({"file": fname, "status": "skipped", "reason": "vuoto"})
                continue

            encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

            # Controlla se il file esiste già su GitHub per prendere lo SHA
            api_url = f"https://api.github.com/repos/{repo}/contents/Data/{fname}"
            sha = None
            try:
                req = urllib.request.Request(api_url, headers={
                    "Authorization": f"token {github_token}",
                    "User-Agent": "ProfitRadarAI"
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    existing = json.loads(resp.read().decode("utf-8"))
                    sha = existing.get("sha")
            except:
                pass  # File non esiste ancora

            # Upload
            payload = {
                "message": f"auto-save: {fname} ({len(content)} bytes)",
                "content": encoded,
                "branch": branch
            }
            if sha:
                payload["sha"] = sha

            req = urllib.request.Request(api_url, 
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"token {github_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "ProfitRadarAI"
                },
                method="PUT"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            results.append({"file": fname, "status": "saved", "size": len(content)})
            gpt_logger.info(f"💾 Salvato {fname} su GitHub ({len(content)} bytes)")
        except Exception as e:
            results.append({"file": fname, "status": "error", "reason": str(e)})

    return jsonify({"results": results, "saved": sum(1 for r in results if r["status"] == "saved")})


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
        ("feedback.csv", FEEDBACK_PATH, GITHUB_FEEDBACK_URL),
        ("ab_outcomes.csv", AB_OUTCOMES_PATH, GITHUB_ABOUT_URL),
        ("PRP_ReversalSignals.csv", os.path.join(DATA_DIR, "PRP_ReversalSignals.csv"), GITHUB_REVSIG_URL),
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
restore_logs_from_github()   # PRIMA ripristina i dati (incl. feedback.csv)
load_model()                 # poi prova a caricare il modello
# Se il modello non c'e' ma abbiamo feedback ripristinati, riaddestra al boot
if not stats.get("model_is_trained") and os.path.exists(FEEDBACK_PATH):
    try:
        if len(pd.read_csv(FEEDBACK_PATH)) >= MIN_FEEDBACK_FOR_TRAIN:
            print("[INIT] Modello assente ma feedback ripristinati -> riaddestro...")
            r = train_model()
            print(f"[INIT] Retrain al boot: {r.get('status', r)}")
    except Exception as _e:
        print(f"[INIT] Retrain al boot fallito: {_e}")
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
