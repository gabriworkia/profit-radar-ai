# ⏰ Cron esterno — terzo livello anti-sleep

Il file `keep-alive.yml` è un workflow GitHub Actions **da attivare a mano**.

## Perché non è già attivo

Il push automatico dei workflow è bloccato: l'app GitHub usata da questa
sessione non ha il permesso `workflows`. Va quindi copiato da te (operazione
di 1 minuto, tutta da browser).

## Come attivarlo

1. Vai sul repo GitHub → **Add file** → **Create new file**
2. Nome file (esatto):
   ```
   .github/workflows/keep-alive.yml
   ```
3. Incolla il contenuto di `keepalive_cron/keep-alive.yml`
4. **Commit** sul branch `main`
5. Tab **Actions** → seleziona *Keep Render Alive* → **Run workflow** per provarlo subito

## Perché serve, se il server si auto-pinga già

Il self-ping interno gira **dentro** il processo del server: se Render **è già
spento**, quel processo non esiste e non può risvegliarsi da solo. Il cron è
l'unica sveglia che arriva **dall'esterno** quando tutto il resto è fermo
(weekend, VPS spento, MT4 chiuso).

## Verifica

Dopo qualche run, apri:

```
https://profit-radar-ai.onrender.com/keepalive_status
```

Il contatore `external_pings` deve salire.

> **Nota**: il cron di GitHub Actions non è puntuale al minuto e in caso di
> carico può slittare di parecchi minuti. Per questo l'intervallo è 10 minuti
> (non 14) e ogni run fa 3 ping distanziati.
