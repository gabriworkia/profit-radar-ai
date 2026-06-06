"""
Test rapido per verificare che il server AI funziona.
Eseguire dopo aver avviato il server con: python app.py
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    print("=== TEST HEALTH ===")
    r = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    return r.status_code == 200

def test_predict_std_buy():
    print("\n=== TEST PREDICT STD BUY ===")
    payload = {
        "symbol": "EURUSD",
        "module": "STD",
        "direction": "BUY",
        "rv": 12.5,
        "adx": 22.1,
        "adr_pct": 35.2,
        "adr_pip": 120,
        "adr_media": 150,
        "ema_pos": 1,
        "hist": "GREEN_LIGHT",
        "close": 1.08500,
        "ema_gap_pct": 0.08,
        "rv_prev": 15.0,
        "rv_prev2": 18.0,
        "light_streak": 2,
        "was_gray": True,
        "hist_flip_bar": 1,
        "context": {
            "total": 28,
            "non_gray": 15,
            "green": 8,
            "red": 7,
            "avg_abs_rv": 14.5,
            "extreme_rv": 2
        }
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Signal: {data['signal']}")
    print(f"  Confidence: {data['confidence']}%")
    print(f"  Method: {data['method']}")
    return data["confidence"] >= 60

def test_predict_sell_extreme():
    print("\n=== TEST PREDICT SELL ESTREMO ===")
    payload = {
        "symbol": "GBPJPY",
        "module": "STD",
        "direction": "SELL",
        "rv": -65.0,
        "adx": 55.0,
        "adr_pct": 85.0,
        "adr_pip": 180,
        "adr_media": 160,
        "ema_pos": -1,
        "hist": "RED_DARK",
        "close": 191.500,
        "ema_gap_pct": 0.45,
        "rv_prev": -55.0,
        "rv_prev2": -45.0,
        "light_streak": 0,
        "was_gray": False,
        "hist_flip_bar": 999,
        "context": {
            "total": 28,
            "non_gray": 20,
            "green": 3,
            "red": 17,
            "avg_abs_rv": 35.0,
            "extreme_rv": 8
        }
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Signal: {data['signal']}")
    print(f"  Confidence: {data['confidence']}%")
    print(f"  Method: {data['method']}")
    return True

def test_predict_breakout():
    print("\n=== TEST PREDICT BREAKOUT ===")
    payload = {
        "symbol": "AUDUSD",
        "module": "BRK",
        "direction": "BUY",
        "rv": 8.0,
        "adx": 18.0,
        "adr_pct": 25.0,
        "adr_pip": 40,
        "adr_media": 120,
        "ema_pos": 1,
        "hist": "GREEN_LIGHT",
        "close": 0.65500,
        "ema_gap_pct": 0.05,
        "rv_prev": 3.0,
        "rv_prev2": 1.0,
        "light_streak": 3,
        "was_gray": True,
        "hist_flip_bar": 1,
        "context": {
            "total": 28,
            "non_gray": 10,
            "green": 6,
            "red": 4,
            "avg_abs_rv": 8.0,
            "extreme_rv": 0
        }
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Signal: {data['signal']}")
    print(f"  Confidence: {data['confidence']}%")
    print(f"  Method: {data['method']}")
    return True

def test_feedback():
    print("\n=== TEST FEEDBACK ===")
    payload = {
        "ticket": 12345,
        "symbol": "EURUSD",
        "direction": "BUY",
        "module": "STD",
        "entry_price": 1.08500,
        "exit_price": 1.08650,
        "profit": 1.50,
        "pips": 15,
        "won": True,
        "ai_confidence": 78,
        "ai_signal": "BUY",
        "rv": 12.5,
        "adx": 22.1,
        "adr_pct": 35.2,
        "hist": "GREEN_LIGHT"
    }
    r = requests.post(f"{BASE_URL}/feedback", json=payload)
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Response: {json.dumps(data, indent=2)}")
    return data.get("logged", False)

def test_stats():
    print("\n=== TEST STATS ===")
    r = requests.get(f"{BASE_URL}/stats")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    return r.status_code == 200


if __name__ == "__main__":
    print("Profit Radar Pro AI Server — Test Suite\n")
    print("=" * 50)
    
    results = []
    
    try:
        results.append(("Health", test_health()))
        results.append(("Predict STD BUY", test_predict_std_buy()))
        results.append(("Predict SELL Extremo", test_predict_sell_extreme()))
        results.append(("Predict Breakout", test_predict_breakout()))
        results.append(("Feedback", test_feedback()))
        results.append(("Stats", test_stats()))
    except requests.exceptions.ConnectionError:
        print("\n>>> ERRORE: Server non raggiungibile!")
        print(">>> Avvia il server prima: python app.py")
        exit(1)
    
    print("\n" + "=" * 50)
    print("RISULTATI:")
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status} - {name}")
    
    passed = sum(1 for _, ok in results if ok)
    print(f"\nTotale: {passed}/{len(results)} superati")
