"""
Test rapido per verificare che il server AI funzioni correttamente.
Eseguire DOPO aver avviato il server con: python app.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_health():
    print("=== TEST /health ===")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ERRORE: {e}")
        return False

def test_ea_status():
    print("\n=== TEST /ea_status (simula EA) ===")
    payload = {
        "balance": 12450.75,
        "equity": 12482.30,
        "open_trades": 2,
        "daily_pnl": 87.5,
        "daily_wins": 3,
        "daily_losses": 1,
        "consecutive_losses": 0,
        "daily_stopped": False,
        "ai_calls": 12,
        "ai_confirm": 8,
        "ai_reject": 4,
        "data_source": "LIVE",
        "cross_active": 18,
        "cross_total": 28,
        "ea_version": "2.01 (Dual)",
        "peaks": {"EURUSD": 18.4, "GBPJPY": -72.1, "AUDUSD": 9.8}
    }
    try:
        r = requests.post(f"{BASE_URL}/ea_status", json=payload, timeout=5)
        print(f"  Status: {r.status_code}")
        data = r.json()
        print(f"  Response: {json.dumps(data, indent=2)}")
        return data.get("status") == "ok"
    except Exception as e:
        print(f"  ERRORE: {e}")
        return False

def test_predict():
    print("\n=== TEST /predict ===")
    payload = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "rv": 14.2,
        "adx": 28.5,
        "adr_pct": 42.0
    }
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        print(f"  Status: {r.status_code}")
        data = r.json()
        print(f"  Signal: {data.get('signal')} | Confidence: {data.get('confidence')}%")
        return data.get("signal") in ["BUY", "SELL", "HOLD"]
    except Exception as e:
        print(f"  ERRORE: {e}")
        return False

def test_feedback():
    print("\n=== TEST /feedback ===")
    payload = {
        "ticket": 999999,
        "symbol": "EURUSD",
        "direction": "BUY",
        "module": "STD",
        "profit": 12.40,
        "pips": 8.5,
        "won": True,
        "rv": 14.2,
        "ai_confidence": 82
    }
    try:
        r = requests.post(f"{BASE_URL}/feedback", json=payload, timeout=5)
        print(f"  Status: {r.status_code}")
        data = r.json()
        print(f"  Response: {json.dumps(data, indent=2)}")
        return data.get("status") == "ok"
    except Exception as e:
        print(f"  ERRORE: {e}")
        return False

def test_dashboard_data():
    print("\n=== TEST /dashboard_data ===")
    try:
        r = requests.get(f"{BASE_URL}/dashboard_data", timeout=5)
        print(f"  Status: {r.status_code}")
        data = r.json()
        print(f"  EA last_update: {data.get('ea', {}).get('last_update')}")
        print(f"  Trade stats keys: {list(data.get('trade_stats', {}).keys())[:5]}...")
        print(f"  Config keys: {len(data.get('config', {}))}")
        return r.status_code == 200 and "ea" in data
    except Exception as e:
        print(f"  ERRORE: {e}")
        return False

def test_ea_config():
    print("\n=== TEST /ea_config GET + POST ===")
    try:
        # GET
        r1 = requests.get(f"{BASE_URL}/ea_config", timeout=5)
        print(f"  GET Status: {r1.status_code}")
        
        # POST
        new_cfg = {"timer_seconds": 15, "ai_min_conf": 75, "allow_trend": True}
        r2 = requests.post(f"{BASE_URL}/ea_config", json=new_cfg, timeout=5)
        print(f"  POST Status: {r2.status_code}")
        print(f"  POST Response: {r2.json()}")
        return r1.status_code == 200 and r2.status_code == 200
    except Exception as e:
        print(f"  ERRORE: {e}")
        return False

if __name__ == "__main__":
    print("Profit Radar Pro AI Server — Test Suite v5.3\n")
    print("=" * 55)
    
    results = []
    
    try:
        results.append(("Health", test_health()))
        results.append(("EA Status (connessione EA)", test_ea_status()))
        results.append(("Predict", test_predict()))
        results.append(("Feedback", test_feedback()))
        results.append(("Dashboard Data", test_dashboard_data()))
        results.append(("EA Config", test_ea_config()))
    except requests.exceptions.ConnectionError:
        print("\n>>> ERRORE CRITICO: Server non raggiungibile!")
        print(">>> Avvia prima il server: python app.py")
        exit(1)
    
    print("\n" + "=" * 55)
    print("RISULTATI:")
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status} - {name}")
    
    passed = sum(1 for _, ok in results if ok)
    print(f"\nTotale: {passed}/{len(results)} superati")
    
    if passed == len(results):
        print("\n🎉 TUTTO FUNZIONA! L'EA può ora comunicare correttamente.")
    else:
        print("\n⚠️ Alcuni test falliti. Controlla i log del server.")