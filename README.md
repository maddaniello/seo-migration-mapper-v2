# 🔗 URL Migration Mapping Tool v2.0

Un tool avanzato per il mapping automatico di URL durante le migrazioni di siti web, sviluppato con Streamlit e ottimizzato per file di grandi dimensioni (fino a 1GB).

## 🚀 Nuove Caratteristiche v2.0

### 📊 Gestione File Giganti
- **File fino a 1GB**: Supporto nativo per file CSV/Excel di grandi dimensioni
- **Elaborazione Batch**: Processamento automatico in chunks per ottimizzare memoria
- **290K+ righe**: Testato con successo su dataset con centinaia di migliaia di righe
- **Monitoraggio Memoria**: Controllo real-time dell'utilizzo RAM con auto-ottimizzazione

### ⚡ Performance Ottimizzate
- **Batch Size Dinamico**: Calcolo automatico della dimensione ottimale del batch
- **Garbage Collection**: Pulizia automatica della memoria durante l'elaborazione
- **Progress Tracking**: Barre di progresso dettagliate per ogni fase
- **Velocità Adattiva**: Algoritmi che si adattano alle dimensioni del dataset

### 🧠 AI Enhancement Intelligente
- **AI Scalabile**: Limita automaticamente le chiamate AI per dataset grandi
- **Costi Controllati**: Configura il numero massimo di miglioramenti AI
- **Qualità Bilanciata**: Applica AI solo dove necessario (similarità < 0.7)

## 📋 Caratteristiche Complete

- **Supporto multi-formato**: CSV, Excel (.xlsx, .xls) fino a 1GB
- **Matching intelligente**: PolyFuzz con TF-IDF per accuratezza massima
- **Colonne personalizzabili**: Scegli liberamente i campi per il matching
- **AI Enhancement**: Migliora match difficili con OpenAI GPT
- **Analisi qualità**: Statistiche e visualizzazioni dettagliate
- **Export ottimizzato**: Download rapido anche per file grandi
- **Gestione errori avanzata**: URL non-redirectable automaticamente separate

## 🔧 Requisiti Sistema

### Hardware Raccomandato:
- **RAM**: 8GB minimo, 16GB+ per file > 100K righe
- **CPU**: Multi-core per elaborazione parallela
- **Storage**: 2x dimensione file per elaborazione temporanea

### Software:
- Python 3.8+
- Streamlit 1.28+
- Dipendenze: vedi `requirements.txt`

## 📊 Performance per Dimensioni File

| Dimensioni Dataset | RAM Consigliata | Tempo Elaborazione | Batch Size |
|-------------------|------------------|-------------------|------------|
| < 10K righe      | 4GB             | 1-3 minuti        | Standard   |
| 10K-50K righe    | 8GB             | 5-10 minuti       | 5K         |
| 50K-200K righe   | 16GB            | 10-30 minuti      | 10K        |
| 200K+ righe      | 32GB+           | 30+ minuti        | Auto       |

## 🛠️ Installazione Rapida

```bash
# Clona il repository
git clone https://github.com/tuo-username/url-migration-tool-v2.git
cd url-migration-tool-v2

# Installa dipendenze
pip install -r requirements.txt

# Avvia l'applicazione
streamlit run app.py
```

## 🐳 Installazione Docker (Raccomandato per File Grandi)

```bash
# Build dell'immagine
docker build -t url-migration-tool .

# Avvia il container con memoria estesa
docker run -p 8501:8501 --memory=8g url-migration-tool
```

## ⚙️ Configurazione Avanzata

### Impostazioni Performance
- **Batch Size**: Auto-calcolato o personalizzabile (1K-20K)
- **Limite Memoria**: Configurabile 70%-95% (default 85%)
- **AI Calls Limit**: Massimo chiamate OpenAI per controllo costi

### Variabili Ambiente
```bash
# Ottimizzazioni Python per file grandi
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Limite memoria pandas
export PANDAS_MAX_MEMORY=8GB
```

## 🎯 Workflow Ottimizzato

### 1. Preparazione File
- **Formato consigliato**: CSV (più veloce di Excel)
- **Encoding**: UTF-8 per compatibilità massima
- **Pulizia**: Rimuovi colonne non necessarie prima del caricamento

### 2. Configurazione Intelligente
- **File < 50K**: Usa tutte le colonne desiderate
- **File 50K-200K**: Limita a 3-4 colonne chiave
- **File > 200K**: Solo Address + 1-2 colonne essenziali

### 3. Elaborazione Monitorata
- **Memory Tracking**: Osserva l'utilizzo RAM in tempo reale
- **Progress Bars**: Monitora il progresso per ogni fase
- **Auto-Optimization**: Il sistema si adatta automaticamente

### 4. Output Ottimizzato
- **Preview Limitato**: Solo prime 1000 righe per file grandi
- **Download Chunks**: File grandi scaricabili in parti
- **Format Detection**: Dimensione stimata prima del download

## 🔍 Troubleshooting File Grandi

### Problemi Comuni:

**"Memory Error"**
```
Soluzioni:
✅ Riduci numero colonne matching
✅ Aumenta limite memoria nell'app
✅ Usa Docker con più RAM
✅ Dividi file in parti più piccole
```

**"Elaborazione Lenta"**
```
Ottimizzazioni:
✅ Disabilita AI per dataset > 200K righe
✅ Usa formato CSV invece di Excel
✅ Chiudi altre applicazioni
✅ Aumenta batch size se hai RAM
```

**"File Upload Failed"**
```
Verifiche:
✅ File < 1GB effettivo
✅ Formato supportato (CSV/XLSX/XLS)
✅ Connessione internet stabile
✅ Browser moderno
```

### Monitor Sistema:
- **Task Manager**: Monitora RAM e CPU
- **Browser DevTools**: Controlla errori console
- **App Logs**: Osserva messaggi di stato

## 📈 Ottimizzazioni Specifiche per Dimensioni

### File 290K+ righe (Caso Reale):
```python
# Configurazione ottimale testata
batch_size = 10000
memory_limit = 80%
ai_limit = 50  # Solo per casi critici
matching_columns = ["Address", "Title 1"]  # Massimo 2-3
preview_disabled = True
```

### Strategie per File Enormi (500K+):
1. **Pre-processing**: Filtra righe irrilevanti
2. **Sampling**: Testa su campione prima dell'elaborazione completa
3. **Parallel Processing**: Usa più core CPU
4. **Chunked Export**: Salva risultati in parti

## 🔬 Algoritmo Batch Ottimizzato

### 1. **Chunked Loading**
```
File → Chunks 10K → Memory Buffer → Processing
```

### 2. **Adaptive Matching**
```
PolyFuzz TF-IDF → Batch Results → Incremental Merge
```

### 3. **Memory Management**
```
Monitor RAM → Auto-adjust Batch → Cleanup → Continue
```

### 4. **Smart AI Enhancement**
```
Low Similarity Filter → Limited AI Calls → Quality Boost
```

## 🎯 Best Practices

### Preparazione Dati:
- **Pulisci URL**: Rimuovi parametri non necessari
- **Standardizza Format**: Consistent case e struttura
- **Valida Columns**: Verifica colonne richieste prima

### Configurazione Ottimale:
- **Memoria**: Usa 80% RAM disponibile massimo
- **Colonne**: Priorità: Address > Title > H1 > Altri
- **AI**: Solo per progetti critici con budget

### Post-Processing:
- **Review Manual**: Controlla match < 0.5 similarità
- **Backup Results**: Salva copie multiple risultati
- **Validate Sample**: Testa su campione prima implementazione


---

**Sviluppato da Daniele Pisciottano 🦕**
