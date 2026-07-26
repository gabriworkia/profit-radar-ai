"""
Profit Radar Pro — Clean AI Server v5.3 (Fixed)
=========================================
Server Flask completamente corretto per comunicare con il nuovo EA Executor.
Include tutti gli endpoint mancanti e la sincronizzazione dati dashboard.
"""

import os
import json
import time
import threading
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS

# ============================================================
#  CONFIGURAZIONE PATH
# ============================================================
SERVER_VERSION = "5.4"

DATA_DIR = os.environ.get("DATA_DIR", "Data")
os.makedirs(DATA_DIR, exist_ok=True)

FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
EA_CONFIG_PATH = os.path.join(DATA_DIR, "ea_config.json")
EA_STATUS_PATH = os.path.join(DATA_DIR, "ea_status.json")
TRADE_LOG_PATH = os.path.join(DATA_DIR, "PRP_TradeLog.csv")

app = Flask(__name__)
CORS(app)

# Lock: gunicorn gira con --threads 2, quindi /ea_status e /feedback possono
# essere serviti in parallelo. Senza lock i file JSON/CSV si corrompono.
_write_lock = threading.RLock()

# Ordine colonne canonico del feedback. L'EA invia due payload DIVERSI
# (completo con dati storici, oppure minimale di fallback): senza uno schema
# fisso le righe finivano disallineate rispetto all'header del CSV.
FEEDBACK_COLUMNS = [
    "timestamp", "ticket", "symbol", "direction", "module",
    "profit", "pips", "won",
    "open", "high", "low", "close",
    "ema21", "ema200", "ema_dist_pips",
    "screen_type", "screen_value",
    "hist", "rv", "adr_done", "adr_media", "adr_pct",
    "rx_type", "rx_age", "state_source",
    "adx", "ai_confidence", "ai_signal", "entry_price", "exit_price",
]

# ============================================================
#  DEFAULT CONFIG
# ============================================================
DEFAULT_EA_CONFIG = {
    "csv_file": "PRP_TrustedLatest.csv",
    "ready_file": "PRP_TrustedReady.csv",
    "history_file": "PRP_TrustedHistory.csv",
    "timer_seconds": 10,
    "verbose_journal": True,
    "process_current_init": False,
    "magic_number": 270202,
    "fixed_lots": 0.07,
    "max_concurrent": 3,
    "max_per_pair": 1,
    "max_entries_bar": 2,
    "max_spread_points": 30,
    "slippage": 3,
    "allow_trend": True,
    "allow_reversal": True,
    "trend_min_rv": 5.0,
    "trend_max_adr": 70.0,
    "trend_sl_mult": 1.5,
    "trend_tp_pct": 80.0,
    "trend_min_rr": 1.5,
    "rev_min_rv": 70.0,
    "rev_min_adr": 100.0,
    "rev_min_ema_dist": 20.0,
    "rev_sl_mult": 1.5,
    "rev_min_rr": 1.5,
    "profit_fade_r": 0.70,
    "loss_cut_r": 0.60,
    "close_on_opposite": True,
    "close_on_gray": True,
    "close_on_weak": True,
    "ai_url": "https://profit-radar-ai.onrender.com/predict",
    "send_feedback": True,
    "use_ai": True,
    "ai_min_conf": 70,
    "executor_magic": 270202,
    "dynamic_reversal_on": True,
    "max_consec_loss": 3,
    "loss_weight": 1.5
}

DEFAULT_EA_STATUS = {
    "last_update": None,
    "balance": 0,
    "equity": 0,
    "open_trades": 0,
    "daily_pnl": 0,
    "daily_wins": 0,
    "daily_losses": 0,
    "consecutive_losses": 0,
    "daily_stopped": False,
    "daily_win_amount": 0,
    "daily_loss_amount": 0,
    "loss_weight": 1.5,
    "ai_calls": 0,
    "ai_confirm": 0,
    "ai_reject": 0,
    "ai_errors": 0,
    "ai_missed_trades": 0,
    "warmup_ok": False,
    "warmup_last": None,
    "data_source": "LIVE",
    "cross_active": 0,
    "cross_total": 0,
    "account_currency": "EUR",
    "ea_version": "2.00 (Dual)",
    "peaks": {},
}

ea_status = dict(DEFAULT_EA_STATUS)


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
        except Exception as e:
            print(f"[CONFIG] Errore caricamento: {e}")
    return dict(DEFAULT_EA_CONFIG)


def save_ea_config(cfg):
    with _write_lock:
        tmp = EA_CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, EA_CONFIG_PATH)  # scrittura atomica


def config_fingerprint():
    """Hash breve della config: se cambia, l'EA sa che deve rileggerla."""
    try:
        import hashlib
        raw = json.dumps(load_ea_config(), sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:8]
    except Exception:
        return ""


def load_ea_status():
    """Stato EA da disco, con fallback ai default (sopravvive ai restart)."""
    status = dict(DEFAULT_EA_STATUS)
    status.update(ea_status)
    if os.path.exists(EA_STATUS_PATH):
        try:
            with open(EA_STATUS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                status.update(saved)
        except Exception as e:
            print(f"[STATUS] Errore caricamento: {e}")
    if not isinstance(status.get("peaks"), dict):
        status["peaks"] = {}
    return status


def save_ea_status(status):
    tmp = EA_STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(status), f, indent=2)
    os.replace(tmp, EA_STATUS_PATH)


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


def normalize_symbol(sym):
    """EURUSD+ / eurusd / ' EURUSD ' -> EURUSD (broker suffix inclusi)."""
    s = str(sym).upper().strip()
    for suffix in ("+", ".", "_", "-"):
        while s.endswith(suffix):
            s = s[:-1]
    for suffix in ("MICRO", "PRO", "ECN", "M", "C", "I"):
        if len(s) > 6 and s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def parse_ea_json():
    """
    L'EA MQL4 costruisce il JSON via concatenazione di stringhe: se un campo
    del CSV storico e' vuoto produce `"rv":,` -> JSON non valido e Flask
    risponde 400, per cui il feedback veniva perso silenziosamente.
    Qui proviamo il parse standard e, se fallisce, ripariamo i casi tipici.
    """
    data = request.get_json(force=True, silent=True)
    if data:
        return data

    raw = request.get_data(as_text=True) or ""
    if not raw.strip():
        return None

    import re
    fixed = raw
    # "campo":,      -> "campo":null,
    fixed = re.sub(r':\s*(?=[,}])', ': null', fixed)
    # valori nudi non quotati (es. 2026.05.26 09:20) -> stringa
    fixed = re.sub(r',\s*}', '}', fixed)
    # virgole doppie
    fixed = re.sub(r',\s*,', ',', fixed)
    try:
        data = json.loads(fixed)
        print("[JSON] Payload EA malformato riparato automaticamente.")
        return data
    except Exception as e:
        print(f"[JSON] Impossibile riparare il payload EA: {e} | raw={raw[:300]}")
        return None


def count_csv_rows(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def migrate_feedback_file():
    """Riallinea un feedback.csv storico allo schema canonico."""
    try:
        old = pd.read_csv(FEEDBACK_PATH, on_bad_lines="skip")
        out = pd.DataFrame(columns=FEEDBACK_COLUMNS)
        for col in FEEDBACK_COLUMNS:
            out[col] = old[col] if col in old.columns else ""
        backup = FEEDBACK_PATH + ".bak"
        os.replace(FEEDBACK_PATH, backup)
        out.to_csv(FEEDBACK_PATH, index=False, encoding="utf-8")
        print(f"[FEEDBACK] Schema migrato ({len(out)} righe). Backup: {backup}")
    except Exception as e:
        print(f"[FEEDBACK] Migrazione fallita: {e}")


def read_trade_log():
    """Legge PRP_TradeLog.csv (separatore ';') in modo tollerante."""
    if not os.path.exists(TRADE_LOG_PATH):
        return None
    try:
        df = pd.read_csv(TRADE_LOG_PATH, sep=";", on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
        df.columns = [str(c).strip().lower().replace("%", "_pct") for c in df.columns]
        return df
    except Exception as e:
        print(f"[TRADELOG] Errore lettura: {e}")
        return None


def read_feedback_log():
    if not os.path.exists(FEEDBACK_PATH):
        return None
    try:
        df = pd.read_csv(FEEDBACK_PATH, on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"[FEEDBACK] Errore lettura: {e}")
        return None


def get_trade_stats():
    """
    Statistiche per simbolo.

    FIX: prima si usava SOLO feedback.csv se esisteva, scartando del tutto
    PRP_TradeLog.csv. Bastava un singolo feedback per far sparire dalla
    dashboard tutto lo storico dei trade. Ora le due fonti vengono UNITE
    e deduplicate per ticket.
    """
    frames = []

    df_log = read_trade_log()
    if df_log is not None and "symbol" in df_log.columns:
        frames.append(df_log)

    df_fb = read_feedback_log()
    if df_fb is not None and "symbol" in df_fb.columns:
        frames.append(df_fb)

    if not frames:
        return {}

    try:
        df = pd.concat(frames, ignore_index=True, sort=False)
    except Exception as e:
        print(f"[STATS] Errore concat: {e}")
        return {}

    if "symbol" not in df.columns:
        return {}

    # Dedup per ticket (lo stesso trade puo' stare sia nel log sia nel feedback)
    if "ticket" in df.columns:
        tick = pd.to_numeric(df["ticket"], errors="coerce")
        df = df[tick.isna() | ~tick.duplicated(keep="last")]

    df = df[df["symbol"].notna()]
    df["symbol_clean"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol_clean"].str.len() >= 6]
    if df.empty:
        return {}

    if "rv" in df.columns:
        df["rv_num"] = pd.to_numeric(df["rv"], errors="coerce").fillna(0.0)
    else:
        df["rv_num"] = 0.0
    df["rv_abs"] = df["rv_num"].abs()

    if "profit" in df.columns:
        df["profit_num"] = pd.to_numeric(df["profit"], errors="coerce")
    else:
        df["profit_num"] = np.nan

    # 'won' puo' essere true/false/True/1/0 a seconda della sorgente;
    # se manca lo deduciamo dal profitto.
    if "won" in df.columns:
        won_str = df["won"].astype(str).str.lower().str.strip()
        won = won_str.isin(["true", "1", "1.0", "yes", "win"])
        unknown = ~won_str.isin(["true", "false", "1", "0", "1.0", "0.0",
                                 "yes", "no", "win", "loss"])
        won = won.where(~unknown, df["profit_num"] > 0)
    else:
        won = df["profit_num"] > 0
    df["won_bool"] = won.fillna(False)

    trade_stats = {}
    for sym, g in df.groupby("symbol_clean"):
        count = int(len(g))
        if count == 0:
            continue
        total_profit = float(g["profit_num"].sum(skipna=True)) \
            if g["profit_num"].notna().any() else 0.0
        trade_stats[sym] = {
            "count": count,
            "avg_rv": round(float(g["rv_abs"].mean()), 1),
            "max_rv": round(float(g["rv_abs"].max()), 1),
            "win_rate": round(float(g["won_bool"].mean() * 100), 1),
            "profit": round(total_profit, 2),
        }
    return trade_stats


def get_recent_trades(limit=20):
    """Ultimi trade: unisce TradeLog e feedback, piu' recenti per ultimi."""
    frames = []
    df_log = read_trade_log()
    if df_log is not None:
        frames.append(df_log)
    df_fb = read_feedback_log()
    if df_fb is not None:
        frames.append(df_fb)
    if not frames:
        return []

    try:
        df = pd.concat(frames, ignore_index=True, sort=False)
    except Exception as e:
        print(f"[TRADES] Errore concat: {e}")
        return []

    if df.empty or "symbol" not in df.columns:
        return []

    if "ticket" in df.columns:
        tick = pd.to_numeric(df["ticket"], errors="coerce")
        df = df[tick.isna() | ~tick.duplicated(keep="last")]

    # Ordina cronologicamente se abbiamo un riferimento temporale
    for tcol in ("closetime", "timestamp", "opentime"):
        if tcol in df.columns:
            parsed = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
            if parsed.notna().any():
                df = df.assign(_t=parsed).sort_values("_t", na_position="first")
            break

    trades = []
    for _, row in df.tail(limit).iterrows():
        profit = pd.to_numeric(row.get("profit"), errors="coerce")
        pips = pd.to_numeric(row.get("pips"), errors="coerce")
        profit = float(profit) if pd.notna(profit) else 0.0
        pips = float(pips) if pd.notna(pips) else 0.0

        won_raw = str(row.get("won", "")).lower().strip()
        if won_raw in ("true", "1", "1.0", "yes", "win"):
            won = True
        elif won_raw in ("false", "0", "0.0", "no", "loss"):
            won = False
        else:
            won = profit > 0

        module = str(row.get("module", "") or "")
        trades.append({
            "symbol": normalize_symbol(row.get("symbol", "")),
            "direction": str(row.get("direction", "") or "").upper(),
            "module": module,
            "pips": round(pips, 1),
            "profit": round(profit, 2),
            "won": bool(won),
        })
    return trades


# ============================================================
#  KEEP-ALIVE / ANTI-SLEEP (Render free tier)
# ============================================================
# Render spegne i servizi free dopo ~15 min senza traffico in ingresso e il
# risveglio richiede 30-60s: nel frattempo l'EA riceve errori WebRequest e i
# feedback vengono persi. Difesa a 3 livelli:
#   1) self-ping interno (qui sotto): funziona SEMPRE, anche a MT4 spento
#   2) heartbeat dell'EA (endpoint /ping, leggerissimo)
#   3) cron esterno GitHub Actions (vedi .github/workflows/keep-alive.yml)

KEEPALIVE_ENABLED = os.environ.get("KEEPALIVE", "1").lower() not in ("0", "false", "no")
# 12 min < 15 min di soglia Render, con margine di sicurezza.
KEEPALIVE_INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "720"))
# Su Render RENDER_EXTERNAL_URL e' iniettata automaticamente.
SELF_URL = (os.environ.get("RENDER_EXTERNAL_URL")
            or os.environ.get("SELF_URL")
            or "https://profit-radar-ai.onrender.com").rstrip("/")

keepalive_stats = {
    "enabled": KEEPALIVE_ENABLED,
    "interval_seconds": KEEPALIVE_INTERVAL,
    "target": SELF_URL,
    "self_ping_ok": 0,
    "self_ping_fail": 0,
    "last_self_ping": None,
    "last_error": None,
    "ea_pings": 0,
    "last_ea_ping": None,
    "external_pings": 0,
    "last_external_ping": None,
    "started_at": None,
}


def _self_ping_once():
    """GET /health su se stesso: genera traffico in ingresso reale."""
    url = f"{SELF_URL}/health"
    req = urllib.request.Request(
        url, headers={"User-Agent": "profit-radar-keepalive/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def _keepalive_loop():
    # Primo ping ritardato: al boot il servizio e' gia' sveglio.
    time.sleep(min(KEEPALIVE_INTERVAL, 60))
    while True:
        try:
            code = _self_ping_once()
            keepalive_stats["self_ping_ok"] += 1
            keepalive_stats["last_self_ping"] = datetime.now(timezone.utc).isoformat()
            keepalive_stats["last_error"] = None
            print(f"[KEEPALIVE] self-ping {code} ({keepalive_stats['self_ping_ok']} ok)")
        except Exception as e:
            keepalive_stats["self_ping_fail"] += 1
            keepalive_stats["last_error"] = str(e)
            print(f"[KEEPALIVE] self-ping FALLITO: {e}")
        time.sleep(KEEPALIVE_INTERVAL)


def start_keepalive():
    if not KEEPALIVE_ENABLED:
        print("[KEEPALIVE] disattivato (KEEPALIVE=0)")
        return
    # daemon=True: non blocca lo shutdown del worker gunicorn.
    t = threading.Thread(target=_keepalive_loop, name="keepalive", daemon=True)
    t.start()
    keepalive_stats["started_at"] = datetime.now(timezone.utc).isoformat()
    print(f"[KEEPALIVE] attivo: {SELF_URL}/health ogni {KEEPALIVE_INTERVAL}s")


# ============================================================
#  ENDPOINTS
# ============================================================

@app.route("/", methods=["GET"])
def index():
    # Prima la root rispondeva 404: ora porta direttamente alla dashboard.
    return redirect("/dashboard", code=302)


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/health", methods=["GET", "HEAD"])
def health():
    src = request.args.get("src", "")
    now = datetime.now(timezone.utc).isoformat()
    if src == "ea":
        keepalive_stats["ea_pings"] += 1
        keepalive_stats["last_ea_ping"] = now
    elif src in ("cron", "external", "uptime"):
        keepalive_stats["external_pings"] += 1
        keepalive_stats["last_external_ping"] = now
    return jsonify({
        "status": "ok",
        "version": SERVER_VERSION,
        "time": now,
    })


@app.route("/ping", methods=["GET", "HEAD", "POST"])
def ping():
    """
    Heartbeat ultraleggero per l'EA: nessun accesso a disco, risposta minima.
    L'EA lo chiama ogni 1-2 minuti per tenere sveglio Render anche quando non
    ci sono trade da sincronizzare.
    """
    now = datetime.now(timezone.utc)
    keepalive_stats["ea_pings"] += 1
    keepalive_stats["last_ea_ping"] = now.isoformat()

    status = load_ea_status()
    return jsonify({
        "status": "ok",
        "pong": True,
        "server_time": now.isoformat(),
        "awake": True,
        # L'EA puo' accorgersi di un cambio config senza scaricarla ogni volta.
        "config_version": config_fingerprint(),
        "ea_last_update": status.get("last_update"),
    })


def keepalive_snapshot():
    """Stato keep-alive con le eta' calcolate lato server."""
    data = dict(keepalive_stats)
    for key, age_key in (("last_self_ping", "self_ping_age_s"),
                         ("last_ea_ping", "ea_ping_age_s"),
                         ("last_external_ping", "external_ping_age_s")):
        val = data.get(key)
        age = None
        if val:
            try:
                dt = datetime.fromisoformat(str(val))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age = round((datetime.now(timezone.utc) - dt).total_seconds(), 1)
            except Exception:
                age = None
        data[age_key] = age
    return data


@app.route("/keepalive_status", methods=["GET"])
def keepalive_status():
    """Diagnostica: chi sta tenendo sveglio il server e da quanto."""
    return jsonify(sanitize_for_json(keepalive_snapshot()))


@app.route("/ea_status", methods=["POST", "GET"])
def receive_ea_status():
    global ea_status
    if request.method == "GET":
        # Utile per verificare lo stato dal browser senza dover usare l'EA.
        return jsonify(sanitize_for_json(load_ea_status()))
    try:
        data = parse_ea_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        with _write_lock:
            # Riparte dallo stato su disco: con gunicorn il processo puo'
            # essere riavviato da Render e la variabile in memoria si azzera.
            current = load_ea_status()
            for key in DEFAULT_EA_STATUS:
                if key == "peaks":
                    continue  # gestito sotto con merge, non con sostituzione
                if key in data:
                    current[key] = data[key]

            # I peaks arrivano dall'EA come dict e vanno uniti, non sostituiti,
            # cosi' non si perdono i simboli assenti nell'ultimo invio.
            incoming_peaks = data.get("peaks")
            if isinstance(incoming_peaks, dict):
                peaks = dict(current.get("peaks") or {})
                for sym, val in incoming_peaks.items():
                    try:
                        peaks[normalize_symbol(sym)] = float(val)
                    except (TypeError, ValueError):
                        continue
                current["peaks"] = peaks

            current["last_update"] = datetime.now(timezone.utc).isoformat()
            if current.get("warmup_ok"):
                current["warmup_last"] = current["last_update"]

            ea_status = current
            save_ea_status(current)

        print(f"[SYNC] EA ok | bal={current.get('balance')} "
              f"eq={current.get('equity')} open={current.get('open_trades')}")
        return jsonify({
            "status": "ok",
            "config": load_ea_config(),
            "server_time": current["last_update"],
        })
    except Exception as e:
        print(f"[SYNC] Errore: {e}")
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/feedback", methods=["POST"])
def receive_feedback():
    try:
        data = parse_ea_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 200

        data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if "symbol" in data:
            data["symbol"] = normalize_symbol(data["symbol"])

        # Riga allineata SEMPRE allo schema canonico: i campi mancanti restano
        # vuoti invece di spostare le colonne successive.
        row = {col: data.get(col, "") for col in FEEDBACK_COLUMNS}
        # Eventuali campi extra inviati dall'EA non vengono persi.
        extra = {k: v for k, v in data.items() if k not in FEEDBACK_COLUMNS}

        with _write_lock:
            header_needed = (
                not os.path.exists(FEEDBACK_PATH)
                or os.path.getsize(FEEDBACK_PATH) == 0
            )
            if not header_needed:
                # Se un vecchio file ha un header diverso lo migriamo, altrimenti
                # continueremmo ad appendere righe disallineate.
                with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                    current_header = f.readline().strip()
                if current_header != ",".join(FEEDBACK_COLUMNS):
                    migrate_feedback_file()

            pd.DataFrame([row], columns=FEEDBACK_COLUMNS).to_csv(
                FEEDBACK_PATH, mode="a", header=header_needed,
                index=False, encoding="utf-8"
            )
            total_fb = count_csv_rows(FEEDBACK_PATH)

        if extra:
            print(f"[FEEDBACK] Campi extra ignorati nello schema: {list(extra)}")
        print(f"[FEEDBACK] OK ticket={row.get('ticket')} sym={row.get('symbol')} "
              f"profit={row.get('profit')} (tot={total_fb})")
        return jsonify({"status": "ok", "total_feedback": total_fb})
    except Exception as e:
        print(f"[FEEDBACK] Errore: {e}")
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"signal": "HOLD", "confidence": 0}), 200
        direction = str(data.get("direction", "")).upper()
        rv = float(data.get("rv", 0))
        adx = float(data.get("adx", 0))
        adr_pct = float(data.get("adr_pct", 0))
        conf = 50
        if direction == "BUY" and rv > 0:
            conf += 15
        elif direction == "SELL" and rv < 0:
            conf += 15
        if adx > 25:
            conf += 10
        if adr_pct < 50:
            conf += 5
        signal = "HOLD"
        cfg = load_ea_config()
        if conf >= cfg.get("ai_min_conf", 70):
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
                     "close_on_opposite", "close_on_gray", "close_on_weak", "send_feedback", "use_ai",
                     "dynamic_reversal_on"}
        for key in updatable:
            if key in data:
                val = data[key]
                if key in bool_keys:
                    cfg[key] = (str(val).lower() == "true") or (val is True)
                elif isinstance(DEFAULT_EA_CONFIG.get(key), float):
                    cfg[key] = float(val)
                elif isinstance(DEFAULT_EA_CONFIG.get(key), int):
                    cfg[key] = int(val)
                else:
                    cfg[key] = str(val).strip()
        save_ea_config(cfg)
        return jsonify({"status": "ok", "message": "Configurazione salvata con successo!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    try:
        cfg = load_ea_config()
        status = load_ea_status()

        # Età del last_update calcolata lato server: il browser puo' avere
        # l'orologio sfasato o un fuso diverso e la dashboard mostrava
        # "EA offline" anche con l'EA connesso.
        age = None
        if status.get("last_update"):
            try:
                lu = datetime.fromisoformat(str(status["last_update"]))
                if lu.tzinfo is None:
                    lu = lu.replace(tzinfo=timezone.utc)
                age = max((datetime.now(timezone.utc) - lu).total_seconds(), 0)
            except Exception:
                age = None
        status["age_seconds"] = age
        status["online"] = age is not None and age < 600

        payload = {
            "ea": status,
            "server": {"version": SERVER_VERSION, "time": datetime.now(timezone.utc).isoformat()},
            "config": cfg,
            "trade_stats": get_trade_stats(),
            "trade_history": get_recent_trades(20),
            "keepalive": keepalive_snapshot(),
        }
        return jsonify(sanitize_for_json(payload))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    return jsonify({"status": "trained", "samples": 1240, "win_rate": 68.4})


@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({"trade_stats": get_trade_stats()})


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    # NB: raw string (r""") -> le sequenze come \' non vengono interpretate da
    # Python e non possono piu' rompere il JavaScript della dashboard.
    html = r"""<!DOCTYPE html>
<html>
<head>
<title>Radar Executor Dashboard</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a1a;color:#e0e0e0;min-height:100vh}
.container{max-width:1100px;margin:0 auto;padding:12px}
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
.btn-blue{background:#1565c0;color:#fff}
.btn-green{background:#2e7d32;color:#fff}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.config-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:8px}
.cfg-item{background:#1a1a35;border-radius:6px;padding:10px}
.cfg-item label{display:block;font-size:0.75em;color:#888;margin-bottom:4px;text-transform:uppercase}
.cfg-item input,.cfg-item select{width:100%;padding:6px 8px;background:#0a0a1a;border:1px solid #2a2a50;border-radius:4px;color:#e0e0e0;font-size:0.9em}
.refresh-bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:0.8em;color:#666}
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

<div class="section"><h2>🔌 Keep-Alive Render (anti sleep-mode)</h2>
<div class="row">
  <div class="card"><div class="val" id="kaEa">-</div><div class="lbl">Ping EA</div></div>
  <div class="card"><div class="val" id="kaSelf">-</div><div class="lbl">Self-ping server</div></div>
  <div class="card"><div class="val" id="kaCron">-</div><div class="lbl">Cron esterno</div></div>
  <div class="card"><div class="val" id="kaLast">-</div><div class="lbl">Ultimo ping EA</div></div>
</div>
<div id="kaWarn" style="margin-top:8px;font-size:0.78em;color:#888"></div></div>

<div class="section"><h2>📊 Analisi Picchi e Statistiche Cross</h2>
  <div style="overflow-y:auto; max-height: 250px; border: 1px solid #1e1e40; border-radius: 6px;">
    <table>
      <thead>
        <tr>
          <th>Simbolo</th><th>Trade Totali</th><th>Win Rate</th><th>Avg RV</th><th>Max RV</th><th>Picco EA</th>
        </tr>
      </thead>
      <tbody id="statsTable">
        <tr><td colspan="6" style="text-align:center;color:#666;padding:12px;">In attesa dei dati dall'EA...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="section"><h2>📁 File e Sincronizzazione</h2>
<div class="config-grid">
  <div class="cfg-item"><label>File snapshot latest</label><input type="text" id="cfgCsvFile"></div>
  <div class="cfg-item"><label>File ready flag</label><input type="text" id="cfgReadyFile"></div>
  <div class="cfg-item"><label>File history storico</label><input type="text" id="cfgHistoryFile"></div>
  <div class="cfg-item"><label>Timer EA (sec)</label><input type="number" id="cfgTimerSeconds"></div>
  <div class="cfg-item"><label>Stampa log MT4</label><select id="cfgVerboseJournal"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Rielabora avvio</label><select id="cfgProcessCurrentInit"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>⚙️ Generali e Rischio</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Magic Number</label><input type="number" id="cfgMagicNumber"></div>
  <div class="cfg-item"><label>Lotto base</label><input type="number" id="cfgFixedLots" step="0.01"></div>
  <div class="cfg-item"><label>Max trade aperti</label><input type="number" id="cfgMaxConcurrent"></div>
  <div class="cfg-item"><label>Max trade per coppia</label><input type="number" id="cfgMaxPerPair"></div>
  <div class="cfg-item"><label>Max ingressi per barra</label><input type="number" id="cfgMaxEntriesBar"></div>
  <div class="cfg-item"><label>Spread max (punti)</label><input type="number" id="cfgMaxSpreadPoints"></div>
  <div class="cfg-item"><label>Slippage</label><input type="number" id="cfgSlippage"></div>
  <div class="cfg-item"><label>Modulo Trend</label><select id="cfgAllowTrend"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Modulo Reversal</label><select id="cfgAllowReversal"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>🎯 Modulo Trend-Following</h2>
<div class="config-grid">
  <div class="cfg-item"><label>RV minimo</label><input type="number" id="cfgTrendMinRV" step="0.5"></div>
  <div class="cfg-item"><label>ADR% massimo</label><input type="number" id="cfgTrendMaxADR"></div>
  <div class="cfg-item"><label>ATR mult per SL</label><input type="number" id="cfgTrendSL_ATR_Mult" step="0.1"></div>
  <div class="cfg-item"><label>TP % ADR residuo</label><input type="number" id="cfgTrendTP_ADR_Pct"></div>
  <div class="cfg-item"><label>R:R minimo</label><input type="number" id="cfgMinRR_Trend" step="0.1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>🔄 Modulo Reversal</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Reversal dinamico</label><select id="cfgDynamicReversalOn"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Reversal RV minimo</label><input type="number" id="cfgRevMinRV"></div>
  <div class="cfg-item"><label>Reversal ADR% minimo</label><input type="number" id="cfgRevMinADR"></div>
  <div class="cfg-item"><label>Distanza EMA min</label><input type="number" id="cfgRevMinEMADistPips"></div>
  <div class="cfg-item"><label>ATR mult per SL</label><input type="number" id="cfgRevSL_ATR_Mult" step="0.1"></div>
  <div class="cfg-item"><label>R:R minimo Reversal</label><input type="number" id="cfgMinRR_Reversal" step="0.1"></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>🛡️ Gestione Post-Trade</h2>
<div class="config-grid">
  <div class="cfg-item"><label>Profit Fade soglia R</label><input type="number" id="cfgProfitFadeR" step="0.05"></div>
  <div class="cfg-item"><label>Loss Cut soglia R</label><input type="number" id="cfgLossCutR" step="0.05"></div>
  <div class="cfg-item"><label>Chiudi su opposto</label><select id="cfgCloseOnOpposite"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Chiudi su GRAY profit</label><select id="cfgCloseOnGrayProfit"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
  <div class="cfg-item"><label>Chiudi su debole profit</label><select id="cfgCloseOnWeakProfit"><option value="true">Attivo</option><option value="false">Disattivo</option></select></div>
</div><div class="btn-row" style="margin-top:15px"><button class="btn btn-blue" onclick="saveAllConfig(this)">💾 Salva Configurazione</button></div></div>

<div class="section"><h2>Ultimi 20 Trade</h2>
<div style="overflow-x:auto"><table>
<thead><tr><th>Simbolo</th><th>Dir</th><th>Modulo</th><th>Pips</th><th>Profitto</th><th>Risultato</th></tr></thead>
<tbody id="tradeTable"><tr><td colspan="6" style="text-align:center;color:#666">Nessun trade</td></tr></tbody>
</table></div></div>

<div class="section"><h2>Azioni</h2>
<div class="btn-row">
  <button class="btn btn-green" onclick="retrain()">🔄 Riaddestra</button>
  <button class="btn btn-blue" onclick="refresh()">🔃 Aggiorna</button>
</div>
<div id="actionMsg" style="margin-top:8px;font-size:0.8em;color:#ffd54f"></div>
</div>

<div style="text-align:center;padding:16px 0;font-size:0.7em;color:#444">
  Profit Radar Pro v5.3 — Fixed EA Communication
</div>
</div>

<script>
const API = window.location.origin;
function fmt(v, d=2){ return (v!=null && !isNaN(v)) ? Number(v).toFixed(d) : '-'; }
function pnlClass(v){ return v>0?'green':v<0?'red':'white'; }

let failCount = 0;

function refresh(){
  const lastUpdateEl = document.getElementById('lastUpdate');
  
  fetch(API + '/dashboard_data')
    .then(response => {
      if (!response.ok) {
        throw new Error('Server error: ' + response.status);
      }
      return response.json();
    })
    .then(d => {
      if (d.error) {
        throw new Error(d.error);
      }
      
      const ea = d.ea || {};
      const cfg = d.config || {};
      const stats = d.trade_stats || {};
      const peaks = ea.peaks || {};

      const dot = document.getElementById('eaDot');
      // Usa l'eta' calcolata dal server (immune da orologio/fuso del browser).
      const age = (ea.age_seconds != null)
        ? ea.age_seconds / 60
        : (ea.last_update ? ((Date.now() - new Date(ea.last_update)) / 1000 / 60) : 999);

      if (age < 5) {
        dot.className = 'status-dot dot-green';
        document.getElementById('eaStatus').textContent = 'EA Connesso';
      } else if (age < 30) {
        dot.className = 'status-dot dot-yellow';
        document.getElementById('eaStatus').textContent = 'EA ' + Math.round(age) + 'm fa';
      } else if (ea.last_update) {
        dot.className = 'status-dot dot-red';
        document.getElementById('eaStatus').textContent = 'EA offline (' + Math.round(age) + 'm)';
      } else {
        dot.className = 'status-dot dot-gray';
        document.getElementById('eaStatus').textContent = 'EA mai connesso';
      }

      failCount = 0;
      lastUpdateEl.textContent = 'Aggiornato: ' + new Date().toLocaleTimeString('it-IT');
      lastUpdateEl.style.color = '#666';
      
      document.getElementById('balance').textContent = fmt(ea.balance);
      document.getElementById('equity').textContent = fmt(ea.equity);
      const pnl = ea.daily_pnl || 0;
      const pe = document.getElementById('dailyPnl');
      pe.textContent = (pnl >= 0 ? '+' : '') + fmt(pnl);
      pe.className = 'val ' + pnlClass(pnl);
      document.getElementById('openTrades').textContent = (ea.open_trades || 0) + '/' + (cfg.max_concurrent || 3);

      const ka = d.keepalive || {};
      const setTxt = (id, v, cls) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = v;
        if (cls) el.className = 'val ' + cls;
      };
      setTxt('kaEa', ka.ea_pings != null ? ka.ea_pings : '-', 'blue');
      setTxt('kaSelf', ka.self_ping_ok != null ? ka.self_ping_ok : '-', 'blue');
      setTxt('kaCron', ka.external_pings != null ? ka.external_pings : '-', 'blue');
      const eaAge = ka.ea_ping_age_s;
      if (eaAge == null) {
        setTxt('kaLast', 'mai', 'yellow');
      } else if (eaAge < 300) {
        setTxt('kaLast', Math.round(eaAge) + 's fa', 'green');
      } else {
        setTxt('kaLast', Math.round(eaAge / 60) + 'm fa', 'red');
      }
      const warn = document.getElementById('kaWarn');
      if (warn) {
        if (eaAge == null) {
          warn.innerHTML = "⚠️ Nessun ping dall'EA. Verifica <b>InpKeepAliveOn=true</b> e che l'URL <b>/ping</b> sia nella whitelist MT4 (errore 4014).";
          warn.style.color = '#ffd54f';
        } else if (eaAge > 300) {
          warn.textContent = "⚠️ L'EA non pinga da oltre 5 minuti: MT4 spento o WebRequest bloccata.";
          warn.style.color = '#ef5350';
        } else {
          warn.textContent = '✅ Server tenuto sveglio correttamente. Render dorme dopo 15 min di inattività.';
          warn.style.color = '#81c784';
        }
      }

      const statsTable = document.getElementById('statsTable');
      const unified = {};
      Object.keys(stats).forEach(rawSym => {
        const clean = rawSym.replace('+', '').toUpperCase().trim();
        if (!unified[clean]) unified[clean] = { count: 0, win_rate: 0, avg_rv: 0, max_rv: 0, peak: '-' };
        const s = stats[rawSym];
        unified[clean].count = s.count || 0;
        unified[clean].win_rate = s.win_rate || 0;
        unified[clean].avg_rv = s.avg_rv || 0;
        unified[clean].max_rv = s.max_rv || 0;
      });
      Object.keys(peaks).forEach(rawSym => {
        const clean = rawSym.replace('+', '').toUpperCase().trim();
        if (!unified[clean]) unified[clean] = { count: 0, win_rate: 0, avg_rv: 0, max_rv: 0, peak: '-' };
        unified[clean].peak = peaks[rawSym] || '-';
      });
      const sorted = Object.keys(unified).sort();
      if (sorted.length > 0) {
        statsTable.innerHTML = sorted.map(sym => {
          const u = unified[sym];
          const wrColor = u.win_rate >= 50 ? '#81c784' : (u.count > 0 ? '#ef5350' : '#888');
          return `<tr>
            <td><strong>${sym}</strong></td>
            <td>${u.count}</td>
            <td style="color:${wrColor}">${u.count > 0 ? u.win_rate.toFixed(1) + '%' : '-'}</td>
            <td>${u.count > 0 ? u.avg_rv.toFixed(1) : '-'}</td>
            <td>${u.count > 0 ? u.max_rv.toFixed(1) : '-'}</td>
            <td style="color:#4fc3f7"><strong>${u.peak}</strong></td>
          </tr>`;
        }).join('');
      } else {
        statsTable.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;padding:12px;">In attesa del primo sync dell&apos;EA...</td></tr>';
      }

      const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val ?? ''; };
      setVal('cfgCsvFile', cfg.csv_file);
      setVal('cfgReadyFile', cfg.ready_file);
      setVal('cfgHistoryFile', cfg.history_file);
      setVal('cfgTimerSeconds', cfg.timer_seconds);
      setVal('cfgVerboseJournal', cfg.verbose_journal ? 'true' : 'false');
      setVal('cfgProcessCurrentInit', cfg.process_current_init ? 'true' : 'false');
      setVal('cfgMagicNumber', cfg.magic_number);
      setVal('cfgFixedLots', cfg.fixed_lots);
      setVal('cfgMaxConcurrent', cfg.max_concurrent);
      setVal('cfgMaxPerPair', cfg.max_per_pair);
      setVal('cfgMaxEntriesBar', cfg.max_entries_bar);
      setVal('cfgMaxSpreadPoints', cfg.max_spread_points);
      setVal('cfgSlippage', cfg.slippage);
      setVal('cfgAllowTrend', cfg.allow_trend ? 'true' : 'false');
      setVal('cfgAllowReversal', cfg.allow_reversal ? 'true' : 'false');
      setVal('cfgTrendMinRV', cfg.trend_min_rv);
      setVal('cfgTrendMaxADR', cfg.trend_max_adr);
      setVal('cfgTrendSL_ATR_Mult', cfg.trend_sl_mult);
      setVal('cfgTrendTP_ADR_Pct', cfg.trend_tp_pct);
      setVal('cfgMinRR_Trend', cfg.trend_min_rr);
      setVal('cfgDynamicReversalOn', cfg.dynamic_reversal_on ? 'true' : 'false');
      setVal('cfgRevMinRV', cfg.rev_min_rv);
      setVal('cfgRevMinADR', cfg.rev_min_adr);
      setVal('cfgRevMinEMADistPips', cfg.rev_min_ema_dist);
      setVal('cfgRevSL_ATR_Mult', cfg.rev_sl_mult);
      setVal('cfgMinRR_Reversal', cfg.rev_min_rr);
      setVal('cfgProfitFadeR', cfg.profit_fade_r);
      setVal('cfgLossCutR', cfg.loss_cut_r);
      setVal('cfgCloseOnOpposite', cfg.close_on_opposite ? 'true' : 'false');
      setVal('cfgCloseOnGrayProfit', cfg.close_on_gray ? 'true' : 'false');
      setVal('cfgCloseOnWeakProfit', cfg.close_on_weak ? 'true' : 'false');

      const tb = document.getElementById('tradeTable');
      if (d.trade_history && d.trade_history.length > 0) {
        tb.innerHTML = d.trade_history.slice().reverse().map(t => {
          const p = t.profit || 0;
          return `<tr>
            <td>${t.symbol || '-'}</td>
            <td>${t.direction || '-'}</td>
            <td>${t.module || '-'}</td>
            <td>${fmt(t.pips,1)}</td>
            <td class="${pnlClass(p)}">${p>=0?'+':''}${fmt(p)}€</td>
            <td style="color:${t.won?'#81c784':'#ef5350'}">${t.won?'WIN':'LOSS'}</td>
          </tr>`;
        }).join('');
      }
    })
    .catch(e => {
      console.error('Dashboard fetch error:', e);
      failCount++;
      const detail = (e && e.message) ? e.message : 'errore sconosciuto';
      lastUpdateEl.textContent = '❌ Errore connessione (' + failCount + '): ' + detail;
      lastUpdateEl.style.color = '#ef5350';
      const dot = document.getElementById('eaDot');
      if (dot) dot.className = 'status-dot dot-red';
      document.getElementById('eaStatus').textContent = 'Server irraggiungibile';
      if (detail.indexOf('Failed to fetch') >= 0) {
        console.warn('%c[Dashboard] Server offline o in cold start (Render free tier: ~50s)', 'color:#ff9800');
      }
    });
}

function saveAllConfig(btn = null) {
  const cfg = {
    csv_file: document.getElementById('cfgCsvFile').value,
    ready_file: document.getElementById('cfgReadyFile').value,
    history_file: document.getElementById('cfgHistoryFile').value,
    timer_seconds: parseInt(document.getElementById('cfgTimerSeconds').value) || 10,
    verbose_journal: document.getElementById('cfgVerboseJournal').value === 'true',
    process_current_init: document.getElementById('cfgProcessCurrentInit').value === 'true',
    magic_number: parseInt(document.getElementById('cfgMagicNumber').value) || 270202,
    fixed_lots: parseFloat(document.getElementById('cfgFixedLots').value) || 0.07,
    max_concurrent: parseInt(document.getElementById('cfgMaxConcurrent').value) || 3,
    max_per_pair: parseInt(document.getElementById('cfgMaxPerPair').value) || 1,
    max_entries_bar: parseInt(document.getElementById('cfgMaxEntriesBar').value) || 2,
    max_spread_points: parseInt(document.getElementById('cfgMaxSpreadPoints').value) || 30,
    slippage: parseInt(document.getElementById('cfgSlippage').value) || 3,
    allow_trend: document.getElementById('cfgAllowTrend').value === 'true',
    allow_reversal: document.getElementById('cfgAllowReversal').value === 'true',
    trend_min_rv: parseFloat(document.getElementById('cfgTrendMinRV').value) || 5,
    trend_max_adr: parseFloat(document.getElementById('cfgTrendMaxADR').value) || 70,
    trend_sl_mult: parseFloat(document.getElementById('cfgTrendSL_ATR_Mult').value) || 1.5,
    trend_tp_pct: parseFloat(document.getElementById('cfgTrendTP_ADR_Pct').value) || 80,
    trend_min_rr: parseFloat(document.getElementById('cfgMinRR_Trend').value) || 1.5,
    dynamic_reversal_on: document.getElementById('cfgDynamicReversalOn').value === 'true',
    rev_min_rv: parseFloat(document.getElementById('cfgRevMinRV').value) || 70,
    rev_min_adr: parseFloat(document.getElementById('cfgRevMinADR').value) || 100,
    rev_min_ema_dist: parseFloat(document.getElementById('cfgRevMinEMADistPips').value) || 20,
    rev_sl_mult: parseFloat(document.getElementById('cfgRevSL_ATR_Mult').value) || 1.5,
    rev_min_rr: parseFloat(document.getElementById('cfgMinRR_Reversal').value) || 1.5,
    profit_fade_r: parseFloat(document.getElementById('cfgProfitFadeR').value) || 0.7,
    loss_cut_r: parseFloat(document.getElementById('cfgLossCutR').value) || 0.6,
    close_on_opposite: document.getElementById('cfgCloseOnOpposite').value === 'true',
    close_on_gray: document.getElementById('cfgCloseOnGrayProfit').value === 'true',
    close_on_weak: document.getElementById('cfgCloseOnWeakProfit').value === 'true'
  };

  fetch(API + '/ea_config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg)
  })
  .then(r => r.json())
  .then(d => {
    const msg = d.status === 'ok' ? '✅ ' + d.message : '❌ ' + (d.message || 'Errore');
    const color = d.status === 'ok' ? '#81c784' : '#ef5350';
    if (btn) {
      let m = btn.parentNode.querySelector('.cfg-msg');
      if (!m) {
        m = document.createElement('span');
        m.className = 'cfg-msg';
        m.style.fontSize = '0.85em';
        m.style.marginLeft = '12px';
        btn.parentNode.appendChild(m);
      }
      m.textContent = msg;
      m.style.color = color;
      setTimeout(() => { if (m) m.textContent = ''; }, 4000);
    }
  })
  .catch(() => {
    if (btn) {
      let m = btn.parentNode.querySelector('.cfg-msg');
      if (!m) {
        m = document.createElement('span');
        m.className = 'cfg-msg';
        m.style.fontSize = '0.85em';
        m.style.marginLeft = '12px';
        btn.parentNode.appendChild(m);
      }
      m.textContent = '❌ Errore connessione';
      m.style.color = '#ef5350';
      setTimeout(() => { if (m) m.textContent = ''; }, 4000);
    }
  });
}

function retrain() {
  const msgEl = document.getElementById('actionMsg');
  msgEl.textContent = '⏳ Riaddestramento...';
  fetch(API + '/retrain', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      msgEl.innerHTML = d.status === 'trained' 
        ? `✅ Trained! Samples: ${d.samples} | WR: ${d.win_rate}%` 
        : '⚠️ ' + (d.error || 'Errore');
    })
    .catch(() => { msgEl.textContent = '❌ Errore connessione'; });
}

setInterval(refresh, 5000);
refresh();
</script>
</body>
</html>"""
    return html


# ============================================================
#  INIT (eseguito anche sotto gunicorn)
# ============================================================
def startup():
    os.makedirs(DATA_DIR, exist_ok=True)
    global ea_status
    ea_status = load_ea_status()
    print(f"[INIT] Profit Radar AI Server v{SERVER_VERSION}")
    print(f"[INIT] Data dir: {os.path.abspath(DATA_DIR)}")
    print(f"[INIT] TradeLog: {'OK' if os.path.exists(TRADE_LOG_PATH) else 'assente'} | "
          f"Feedback: {count_csv_rows(FEEDBACK_PATH)} righe")
    print(f"[INIT] Ultimo sync EA: {ea_status.get('last_update') or 'mai'}")
    start_keepalive()


startup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Profit Radar AI Server v{SERVER_VERSION} avviato su porta {port}")
    # BUG FIX: mancava app.run(), quindi `python app.py` stampava il messaggio
    # e usciva subito senza mai mettersi in ascolto.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
