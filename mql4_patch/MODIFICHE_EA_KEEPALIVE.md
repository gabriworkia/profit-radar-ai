# 🔌 Modifiche EA — Keep-Alive costante per Render

**Obiettivo**: fare in modo che Render **non vada mai in sleep mode**.

Render (piano free) spegne il servizio dopo **~15 minuti** senza traffico in
ingresso. Il risveglio richiede 30–60 secondi: in quella finestra l'EA riceve
errori `WebRequest` e **i feedback vengono persi**.

> Questo risolve anche il punto **5.1 della relazione** ("Feedback reali non
> arrivano" → causa: *"Render in sleep al primo feedback"*).

---

## ⚠️ Perché oggi non basta

Nel `EA_DataCollector` attuale il ping esiste già, ma ha **tre difetti**:

| # | Problema | Conseguenza |
|---|---|---|
| 1 | Il ping è **dentro `if(InpSendFeedback)`** | Se disattivi il feedback, sparisce anche il keep-alive |
| 2 | Intervallo **300s = 5 min**, ma il contatore avanza di `InpTimerSeconds` solo dentro `OnTimer()` | Se il timer è lento o l'EA è occupato, si può sforare i 15 min |
| 3 | Il ping usa `/predict` sostituito in `/health`, con timeout **3000 ms** | Durante un cold start Render impiega 30–60s → il ping fallisce **proprio quando serve** |

Inoltre, a **MT4 chiuso** (weekend, VPS spento) non pinga più nessuno.
Per questo il server ha ora anche un **self-ping interno** e un **cron esterno**:
i tre livelli si coprono a vicenda.

---

## 1️⃣ Modifiche a `EA_DataCollector`

### A. Nuovi input (dopo la riga `extern int InpTimerSeconds = 10;`)

```mql4
//--- KEEP-ALIVE RENDER (anti sleep-mode)
extern bool   InpKeepAliveOn        = true;  // Tiene sveglio Render 24/7
extern int    InpKeepAliveSeconds   = 120;   // Ping ogni 2 minuti
extern int    InpKeepAliveTimeout   = 60000; // 60s: copre il cold start
```

> **Perché 120 secondi**: la soglia Render è 15 minuti. Con un ping ogni 2 minuti
> hai un margine di 7 ping mancati prima di rischiare lo sleep. Il traffico è
> irrisorio (~30 richieste/ora, risposta di poche decine di byte).

### B. Nuove variabili globali (vicino a `g_warmupCounter`)

```mql4
//--- Keep-alive tracking
int      g_kaCounter          = 0;
int      g_kaTotalOK          = 0;
int      g_kaTotalFail        = 0;
datetime g_kaLastOK           = 0;
string   g_kaConfigVersion    = "";
```

### B-bis. Prototipi — accanto a `void SyncWithServer();` (riga ~108)

```mql4
void     KeepAlivePing();
string   JsonGetString(string json, string key);
```

### C. Nuova funzione — inseriscila vicino a `AIWarmupPing()`

```mql4
//--- Ping leggero anti-sleep verso /ping (endpoint dedicato del server)
void KeepAlivePing()
{
   string kaUrl = InpAIUrl;
   StringReplace(kaUrl, "/predict", "/ping");

   char postData[];
   char result[];
   string resultHeaders;
   string headers = "Content-Type: application/json\r\n";

   ResetLastError();
   // Timeout ampio: durante un cold start Render puo' impiegare 30-60s.
   int res = WebRequest("GET", kaUrl, headers, InpKeepAliveTimeout,
                        postData, result, resultHeaders);

   if(res == -1)
   {
      int err = GetLastError();
      g_kaTotalFail++;
      g_warmupOK = false;
      if(InpVerboseJournal)
         Print("[KEEPALIVE] ERRORE ", err, " (", ErrDesc(err), ") su ", kaUrl);
      return;
   }

   g_kaTotalOK++;
   g_kaLastOK = TimeCurrent();
   g_warmupOK = true;

   // Rileva un cambio di configurazione dalla dashboard senza scaricarla ogni volta
   string resp = "";
   int n = ArraySize(result);
   for(int i = 0; i < n; i++) resp += CharToString(result[i]);

   string cv = JsonGetString(resp, "config_version");
   if(cv != "" && g_kaConfigVersion != "" && cv != g_kaConfigVersion)
   {
      Print("[KEEPALIVE] Configurazione cambiata sulla dashboard -> risincronizzo");
      SyncWithServer();   // scarica la config aggiornata
   }
   if(cv != "") g_kaConfigVersion = cv;

   if(InpVerboseJournal)
      Print("[KEEPALIVE] OK (", g_kaTotalOK, " ok / ", g_kaTotalFail, " ko)");
}

//--- Estrae un valore stringa da un JSON semplice (senza parser esterni)
string JsonGetString(string json, string key)
{
   string pat = "\"" + key + "\":\"";
   int p = StringFind(json, pat);
   if(p < 0) return "";
   p += StringLen(pat);
   int e = StringFind(json, "\"", p);
   if(e < 0) return "";
   return StringSubstr(json, p, e - p);
}
```

### D. Modifica di `OnTimer()` — **la parte più importante**

Il keep-alive va **FUORI** da `if(InpSendFeedback)`, altrimenti si spegne
insieme al feedback.

```mql4
void OnTimer()
{
   CaptureSnapshot(false);

   //--- KEEP-ALIVE: indipendente dal feedback, gira SEMPRE.
   if(InpKeepAliveOn)
   {
      g_kaCounter += InpTimerSeconds;
      if(g_kaCounter >= InpKeepAliveSeconds)
      {
         g_kaCounter = 0;
         KeepAlivePing();
      }
   }

   if(InpSendFeedback)
   {
      CheckClosedTradesFeedback();
      RecoverMissedFeedback();

      g_warmupCounter += InpTimerSeconds;
      if(g_warmupCounter >= g_warmupInterval)
      {
         g_warmupCounter = 0;
         AIWarmupPing();
      }

      g_syncCounter += InpTimerSeconds;
      if(g_syncCounter >= g_syncInterval)
      {
         g_syncCounter = 0;
         SyncWithServer();
      }
   }
}
```

### E. Ping immediato all'avvio — in `OnInit()`, prima di `return(INIT_SUCCEEDED);`

```mql4
   //--- Sveglia subito Render all'avvio dell'EA (puo' essere in sleep)
   if(InpKeepAliveOn)
   {
      Print("[KEEPALIVE] Risveglio iniziale del server...");
      KeepAlivePing();
   }
```

---

## 2️⃣ Whitelist MT4 — passaggio obbligatorio

`Strumenti → Opzioni → Expert Advisors → Allow WebRequest for listed URL`

Aggiungi **anche** il nuovo endpoint (gli altri della relazione restano):

```text
https://profit-radar-ai.onrender.com/ping
https://profit-radar-ai.onrender.com/health
https://profit-radar-ai.onrender.com/predict
https://profit-radar-ai.onrender.com/feedback
https://profit-radar-ai.onrender.com/ea_status
https://profit-radar-ai.onrender.com/ea_config
```

> ⚠️ Se manca `/ping` in whitelist ottieni **errore 4014** e il keep-alive non parte.
> Dopo la modifica: **rimuovi e riattacca l'EA dal grafico** (§7.1 della relazione).

---

## 3️⃣ Nota sull'`ProfitRadarPro_EA_Executor`

**Non serve modificarlo.** Verificato: l'Executor **non contiene nessuna
`WebRequest`** — a parlare col server è solo il `EA_DataCollector`, che fa da
"ponte web". Tenere il keep-alive in un solo EA evita ping doppi.

Se in futuro l'Executor girasse **senza** il Collector, allora andrà copiato lì
lo stesso blocco (input + `KeepAlivePing()` + chiamata in `OnTimer()`).

---

## 4️⃣ Come verificare che funzioni

**Dal browser** — apri:

```text
https://profit-radar-ai.onrender.com/keepalive_status
```

Devi vedere i contatori salire:

```json
{
  "enabled": true,
  "self_ping_ok": 12,        // ping interni del server
  "ea_pings": 47,            // ping ricevuti dall'EA  <-- deve crescere
  "ea_ping_age_s": 38.2,     // secondi dall'ultimo ping EA (< 180 = OK)
  "external_pings": 6        // ping del cron GitHub Actions
}
```

**Dal journal MT4** — ogni 2 minuti:

```text
[KEEPALIVE] OK (23 ok / 0 ko)
```

Se vedi `ERRORE 4014` → manca l'URL `/ping` in whitelist.

---

## 5️⃣ Riepilogo dei 3 livelli di protezione

| Livello | Chi | Frequenza | Copre |
|---|---|---|---|
| 1. Self-ping interno | Server (già attivo) | 12 min | Sempre, anche a MT4 chiuso |
| 2. Heartbeat EA | `EA_DataCollector` | 2 min | Quando MT4 è acceso |
| 3. Cron esterno | GitHub Actions | 10 min | Anche a servizio già spento |

Il livello 1 da solo non basta: se Render **è già spento**, il processo non
esiste e non può auto-pingarsi. Serve una sveglia dall'esterno (livelli 2 e 3).
