# Verifica dell'obiettivo del video

Il progetto accompagna il video «Costruiamo l'algoritmo di Airbnb». Questo file confronta il
repository con l'obiettivo scritto nel catalogo di Neuroespresso, dice se il progetto gira da
zero e se i numeri tornano, e elenca le lacune corrette e quelle lasciate.

Metodo: clone pulito di `main` (commit `74cb950`) da GitHub, in una cartella temporanea, su macOS
con Python 3.12.13. Dipendenze installate come da README, senza versioni fissate:
scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.5.4, pandas 3.1.1. I modelli e i dati non sono stati
toccati: dove un numero non coincide con l'obiettivo, lo scarto è riportato così com'è.

## Esito in breve

- Il progetto gira da zero: installazione, test e `airbnb run-arc` passano.
- La progressione albero → foresta → boosting c'è e ogni passo abbassa l'errore, ma l'errore di
  arrivo è **44,85 €**, non 43 €, e quello di partenza è **87,13 €**, non 85 €.
- I numeri del README non coincidevano del tutto con quelli ricalcolati (Parti 2–6, fino a 0,5 €).
  Sono stati riallineati; il resto coincideva.
- Il repository non ha segreti, file `.env` né percorsi personali, ed è pubblico.

## 1. Confronto con l'obiettivo

| Voce dell'obiettivo | Cosa c'è nel repository | Esito | Peso per il video / cosa servirebbe |
|---|---|---|---|
| **Prodotto**: il prezzo a notte di un alloggio, previsto da quello che chiedono gli annunci simili | `airbnb simulate`: scegli un luogo di Roma, tipo di stanza e dimensioni; stampa il prezzo suggerito, la fascia in cui cade l'80% degli annunci simili e i cinque annunci reali più vicini. L'esempio del README (Trastevere, 199 €, fascia 138–292 €) esce identico dal comando e da `simulatore.json`. Il README dice onestamente che non è lo Smart Pricing di Airbnb. | ok | Nessuno. |
| **Progressione**: albero (CART) → random forest (bagging) → gradient boosting; ogni passo abbassa l'errore | Parte 1 CART, 2 potatura, 3 bagging, 4 random forest, 5 errore out-of-bag, 6 importanza per permutazione, 7 gradient boosting con XGBoost. Tabella dei passi sotto: l'errore scende a ogni passo. | ok | Un passo è quasi piatto: foresta → boosting con perdita quadratica guadagna solo 0,08 €. Il calo visibile (−1,73 €) viene dal passare alla perdita assoluta, non dal boosting in sé. Nel video conviene dirlo, o presentare il boosting direttamente con la perdita assoluta. |
| **Numero di parti** | Il testo del compito parla di otto parti; il repository ne ha sette. La Parte 8 («no free lunch», con kNN, OLS e Lasso) è stata tolta di proposito nel commit `74cb950` (vedi `CHANGELOG.md`, sezione Removed) per allineare l'arco a CART → foresta → boosting. | differenza | Nessuno: l'obiettivo non cita la Parte 8. Se il video la volesse, andrebbe ripristinata dallo stato precedente (`git show 76d33a0:airbnb/modeling/arc.py`): lavoro non piccolo, non fatto. |
| **Dati e licenza**: Inside Airbnb, Roma, CC BY 4.0 citata | Roma, scraping del 2026-06-20, 31.870 annunci. La licenza è citata nel README (badge, sezione *The data*, sezione *License*) e in `data/sources/SOURCE.md`, con il testo di attribuzione. Il file grezzo, che contiene nomi di host e descrizioni, non è committato; `airbnb fetch-data` lo scarica e ricostruisce lo snapshot (stesso numero di righe e stessi valori; vedi sotto). | ok | Nessuno. |
| **Dati: Milano «se è in programma»** | Nel repository non c'è nulla su Milano (configurazione, luoghi e URL sono di Roma). | manca (solo se Milano è in programma) | Se Milano entra nel video servono: un secondo snapshot, un file di luoghi, la città in `domain/config.py` e di nuovo tutto l'arco. È lavoro grande, non fatto. Se non è in programma, non conta. |
| **Numero**: l'errore medio assoluto (MAE) scende da circa 85 € a circa 43 € | Partenza (rispondere sempre la media): **87,13 €**. Arrivo (boosting, perdita assoluta): **44,85 €**. Il calo è 42,28 € contro i 42 € dell'obiettivo. | differenza | Partenza +2,13 € (+2,5%) e arrivo +1,85 € (+4,3%) rispetto all'obiettivo. Per il video: dire «da circa 87 a circa 45 euro», oppure «un calo di circa 42 euro». «85 → 43» non è quello che il codice calcola. Per ottenerlo servirebbero un altro modello o un altro split, cosa che il compito vieta. Nota: «85» corrisponde solo alla media di partenza; il primo albero (CART, profondità 3) parte da 61,14 €. |
| **Numeri ricalcolabili con un comando** | `airbnb run-arc` ricalcola tutta la tabella (circa 85 s sul computer usato) ed è deterministico sullo stesso ambiente. I file per il video escono da un secondo comando, `python scripts/export_fixtures.py`, che legge l'output dell'arco e salta, dicendo il perché, i file di cui non trova un'esecuzione conforme. | differenza | Basso. I comandi sono due, non uno, e le dipendenze non sono fissate: le Parti 2–6 (alberi di scikit-learn) cambiano fino a circa 0,5 € tra versioni (vedi sezione 3). Per il video: eseguire i due comandi sullo stesso ambiente e citare i numeri dei file esportati. Servirebbe un file di versioni fissate, o un sottocomando unico `airbnb export-fixtures`: scelte di progetto, non fatte. |
| **Repository pubblico, senza segreti** (vincolo) | Repository pubblico. Nessun file `.env` tracciato; nessun segreto né percorso personale nei file committati (cercati `/Users/`, `/home/`, nome utente, `secret`, `token`, `password`, `api_key`, `.env`). | ok | Nessuno. |

### La progressione, passo per passo (MAE sul test, 6.374 annunci)

| Passo | Modello | MAE | Variazione |
|---|---|---:|---:|
| 0 | risponde sempre la media (200,52 €) | 87,13 € | |
| 1 | CART, profondità 3 (8 foglie) | 61,14 € | −25,99 € |
| 2 | CART potato (132 foglie) | 54,14 € | −7,00 € |
| 3 | bagging, 200 alberi | 47,78 € | −6,36 € |
| 4 | random forest, 11 colonne su 33 per taglio | 46,66 € | −1,12 € |
| 7 | boosting, perdita quadratica (564 round) | 46,58 € | −0,08 € |
| 7 | boosting, perdita assoluta (988 round) | 44,85 € | −1,73 € |

## 2. Il progetto gira da zero

Tutto su un clone pulito, nell'ordine:

| Comando | Esito |
|---|---|
| `git clone https://github.com/SilvioBaratto/airbnb-price-prediction` | passato |
| `python3.12 -m venv .venv && source .venv/bin/activate` | passato |
| `pip install -e ".[dev]"` | **passato** (14 s) |
| `pytest -q` | **passato**: 121 test, 64 s, 1 avviso (`Pandas4Warning` in `parse_bathrooms`, corretto: vedi sezione 4) |
| `ruff check .` e `ruff format --check .` | passati |
| `airbnb run-arc --charts` | **passato fino in fondo** (uscita 0, 85 s), scrive le curve, i grafici, `arc_summary.csv` e `arc_run.json` |
| `python scripts/export_fixtures.py` | passato, 11 file in `output/fixtures/airbnb_prezzi/` (28 s) |
| `airbnb train` e `airbnb simulate --model models/price_model.joblib` | passati; il modello salvato pesa 2,7 MB; il prezzo suggerito per Trastevere esce identico al README |
| `airbnb fetch-data` | passato (scarica 22 MB da Inside Airbnb): 37.084 annunci, 2.760 senza prezzo, 2.072 con soggiorno lungo e 382 fuori fascia scartati, 31.870 tenuti, come in `SOURCE.md` |

I dati di Roma si caricano e l'installazione riesce: nessuna domanda al proprietario necessaria.

Sul ramo di questa verifica (dopo le correzioni): `pytest -q` dà 122 passed, senza avvisi;
`ruff` pulito. La CI di `main` è verde (ultima esecuzione: 1 m 7 s).

Due osservazioni dalla prova su pulito:

- Una seconda esecuzione di `run-arc` sullo stesso ambiente ha dato un `arc_summary.csv` identico
  byte per byte.
- Lo snapshot ricostruito da `fetch-data` ha le stesse 31.870 righe e gli stessi valori del file
  committato, ma 2.277 righe differiscono per l'ultima cifra di latitudine e longitudine (fino a
  7·10⁻¹⁵), per la lettura dei decimali di pandas 3.1.1. Il file è quindi diverso byte per byte e
  `snapshot_sha256` in `arc_run.json` cambierebbe, ma nessun numero del modello è stato verificato
  con lo snapshot ricostruito: l'arco è stato eseguito sullo snapshot committato.

## 3. I numeri tornano

### README contro `airbnb run-arc`

Prima delle correzioni, questi numeri del README differivano da quelli ricalcolati:

| Numero | README prima | Ricalcolato |
|---|---:|---:|
| Parte 2, foglie dell'albero pieno | 24.840 | 24.846 |
| Parte 2, MAE test dell'albero pieno | 66,68 € | 67,19 € |
| Parte 2, MAE test dell'albero potato | 54,40 € | 54,14 € |
| Parte 3, albero singolo | 68,08 € | 68,07 € |
| Parte 3, bagging | 47,80 € | 47,78 € |
| Parte 4, ρ della foresta | 0,047 | 0,046 |
| Parte 4, MAE test della foresta | 46,65 € | 46,66 € |
| Parte 5, MAE out-of-bag | 47,32 € | 47,23 € |
| Parte 5, MAE test della foresta | 46,65 € | 46,66 € |
| Parte 6, `km_to_center` | +20,97 € | +21,06 € |
| Parte 6, `bathroom_shared` | +0,21 € | +0,22 € |
| Peso del modello salvato | ~1 MB | 2,7 MB |
| Durata dei test | ~40 s | 64 s |

Ora il README riporta i valori ricalcolati (commit elencati in sezione 4). Tutto il resto
coincideva già: media 200,52 € e MAE 87,13 €, primo taglio `bathrooms ≤ 1,75`, MAE 61,14 € del
primo albero, 132 foglie dopo la potatura, il 63,2% di righe distinte per campione bootstrap, il
36,8% di alberi che non vedono un annuncio, ρ del bagging 0,083 e i margini ±0,003 e ±0,001,
slice di 5.099 righe, 11 colonne su 33, 564 round e validazione 48,51 → 49,07, MAE 46,58 € →
44,85 €, 31.870 annunci, 6.374 di test, 23 luoghi e il prezzo suggerito per Trastevere.

**Causa dello scarto**: i numeri delle Parti 2–6 vengono dagli alberi di scikit-learn e cambiano
con la versione delle librerie, che il progetto non fissa. Lo conferma una seconda esecuzione con
scikit-learn 1.7.2, NumPy 2.3.5 e pandas 2.3.3: le Parti 1 e 7 sono identiche, le Parti 2–6 si
spostano fino a circa 0,5 € (albero potato 54,30 €, foresta 46,59 €). Nemmeno questa seconda
esecuzione coincide con i numeri che il README aveva: quelli venivano da un ambiente che non
conosciamo. Il README ora dice con quali versioni sono stati calcolati.

### File esportati contro `airbnb run-arc`

`arc_metrics.json` coincide con la tabella stampata da `run-arc` e con `arc_summary.csv`, riga per
riga: le 9 righe (parte, modello, MAE di train e test, R², nota) e `nTrain` 25.496, `nTest` 6.374,
snapshot `664d49d2aa08`, seme 20260907. Oltre a quelli del README, contiene tre numeri del
simulatore che il README non cita: MAE sul test 45,22 €, copertura della fascia 79,9% (il README
dice «80%»), larghezza mediana della fascia 113 €.

Controllati anche contro l'output di `run-arc`, senza differenze: `potatura.json` (24.846 foglie,
α 10,21, 132 foglie, 54,60 € in validazione incrociata, 54,14 € sul test), `bagging.json` (63,2%,
albero singolo 68,07 €, bagging 47,78 €), `foresta.json` (ρ 0,083 e 0,046, σ² 8.058 e 8.028,
47,78 € e 46,66 €), `oob.json` (102 alberi su 300 non vedono l'annuncio, 36,8%, OOB 47,23 €, test
46,66 €), `importanza.json` (OOB 47,23 €, `km_to_center` +21,06 €, `bathroom_shared` +0,22 €) e
`simulatore.json` (199 €, fascia 138–292 €, i cinque annunci simili del README, uno per uno).
`albero.json`, `boosting.json`, `mappa.json` e `repo.json` sono stati generati senza errori; i loro
dettagli (nodi dell'albero, residui, punti sulla mappa) non vengono stampati da `run-arc` e non
sono stati confrontati uno per uno.

## 4. Lacune

### Corrette (un commit ciascuna, nessuna tocca modelli o dati)

| Commit | Lacuna |
|---|---|
| `4a6abd5` | I numeri delle Parti 2–6 nel README (e il ρ 0,047 del `CHANGELOG.md`) non coincidevano con quelli ricalcolati; riallineati, con la nota sulle versioni. |
| `d766ab4` | Il README e `CONTRIBUTING.md` dicevano «~40 s» per i test; misurato 64 s. |
| `d997309` | `parse_bathrooms` riempiva una colonna float64 con una serie `Float64` nullable: avviso `Pandas4Warning` oggi, errore con pandas 4, che avrebbe rotto `airbnb fetch-data`. Corretto con un cast; lo snapshot ricostruito dal file grezzo è identico byte per byte prima e dopo, e il test passa con gli avvisi trattati come errori. |
| `1ff6a0a` | `airbnb/__main__.py` non era coperto da nessun test (0%) pur essendo promesso dal README (`python -m airbnb <sottocomando>`). Aggiunto il test. |
| `5f12b40` | Il README dava il modello salvato a «~1 MB»; misurato 2,7 MB. |

### Lasciate, con il motivo

| Lacuna | Motivo |
|---|---|
| MAE di partenza 87,13 € e di arrivo 44,85 € invece di 85 € e 43 € | Il compito vieta di cambiare modelli o dati per far tornare un numero. Scarto riportato nella sezione 1. |
| Milano | Non è nel repository e non risulta in programma; servirebbe un secondo snapshot e l'intero arco da rifare. |
| Parte 8 (no free lunch) | Tolta di proposito; ripristinarla non è un lavoro piccolo. |
| Dipendenze non fissate, quindi le Parti 2–6 si spostano di circa 0,5 € tra versioni | Fissare le versioni (file di vincoli o lock) cambia come si installa il progetto e va deciso dal proprietario. Il README ora dichiara l'ambiente dei numeri. |
| Due comandi (`run-arc`, poi `export_fixtures.py`) invece di uno | Il generatore sta in `scripts/`, fuori dal pacchetto; un sottocomando `airbnb export-fixtures` romperebbe la regola di dipendenza che `tests/test_layering.py` fa rispettare, o richiederebbe di spostare il codice. È una scelta di architettura. |
| Lo snapshot ricostruito da `fetch-data` differisce dal file committato nell'ultima cifra decimale delle coordinate | Dipende dalla versione di pandas. Riscrivere lo snapshot cambierebbe l'impronta `snapshot_sha256` e andrebbe deciso dal proprietario; i valori restano uguali. |
| `pyright` non eseguito | Nella CI è informativo (`continue-on-error`) e il README tollera una linea di base di avvisi sugli stub di pandas e NumPy; non è stato verificato. |
