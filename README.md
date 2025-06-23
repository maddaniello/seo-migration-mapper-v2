# 🔗 URL Migration Mapping Tool

Un tool avanzato per il mapping automatico di URL durante le migrazioni di siti web, sviluppato con Streamlit e potenziato dall'AI.

## 🚀 Caratteristiche

- **Supporto multi-formato**: Carica file CSV e Excel (.xlsx, .xls)
- **Matching intelligente**: Utilizza PolyFuzz con TF-IDF per trovare corrispondenze accurate
- **Colonne personalizzabili**: Scegli su quali campi basare il matching (URL, Title, H1, etc.)
- **AI Enhancement**: Migliora automaticamente i match con bassa similarità usando OpenAI GPT
- **Analisi qualità**: Visualizzazioni e statistiche sui risultati del matching
- **Export completo**: Scarica i risultati in formato CSV
- **Gestione errori**: Identifica e separa URL non-redirectable (3xx, 5xx)

## 📋 Requisiti

### Colonne obbligatorie nei file:

**File PRE-migrazione:**
- `Address` - URL del sito live
- `Status Code` - Codice di stato HTTP

**File POST-migrazione:**
- `Address` - URL del sito staging
- `Status Code` - Codice di stato HTTP  
- `Indexability` - Stato di indicizzabilità

### Colonne opzionali (consigliate):
- `Title 1` - Title tag della pagina
- `H1-1` - Primo heading H1
- Qualsiasi altra colonna presente in entrambi i file

## 🛠️ Installazione

1. **Clona il repository:**
```bash
git clone https://github.com/tuo-username/url-migration-tool.git
cd url-migration-tool
```

2. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

3. **Avvia l'applicazione:**
```bash
streamlit run app.py
```

## 🔧 Configurazione OpenAI (Opzionale)

Per abilitare il miglioramento AI:

1. Ottieni una API key da [OpenAI](https://platform.openai.com/api-keys)
2. Inserisci la chiave nell'interfaccia Streamlit
3. Abilita l'opzione "Miglioramento AI"

Il sistema userà GPT-3.5-turbo per migliorare i match con similarità < 0.7.

## 📖 Come usare

### 1. Caricamento File
- Carica il file PRE-migrazione (sito live)
- Carica il file POST-migrazione (sito staging)
- Supporta formati CSV e Excel

### 2. Selezione Colonne
- Scegli le colonne per il matching
- Più colonne = risultati più accurati
- Address è sempre incluso di default

### 3. Elaborazione
- Clicca "Avvia Mapping"
- Il sistema processerà i dati automaticamente
- Visualizza i risultati in tempo reale

### 4. Analisi Risultati
- Controlla le statistiche di qualità
- Visualizza la distribuzione delle similarità
- Identifica match che richiedono revisione manuale

### 5. Download
- Scarica il mapping completo
- Scarica le URL non-redirectable
- Formati CSV pronti per l'implementazione

## 📊 Output

### File principali:
1. **Mapping completo** (`auto-migration-mapped-all-output.csv`)
   - URL sorgente e destinazione
   - Punteggi di similarità
   - Testi di matching
   - Indicatori di qualità

2. **URL non-redirectable** (`auto-migration-non-redirectable-urls.csv`)
   - URL con status code 3xx e 5xx
   - Non possono essere redirette automaticamente

### Colonne output:
- `URL - Source`: URL originale
- `Best Matching URL`: Migliore corrispondenza trovata
- `Best Match On`: Campo usato per il match (URL/Title/H1)
- `Highest Match Similarity`: Punteggio di similarità (0-1)
- `Second Highest Match`: Seconda migliore opzione
- `Double Matched?`: Indica se i primi due match sono identici

## 🎯 Algoritmo di Matching

### 1. Preprocessing
- Rimozione duplicati
- Separazione URL non-redirectable (3xx, 5xx)
- Gestione valori mancanti (NaN → URL per 404)

### 2. Matching PolyFuzz
- Utilizza TF-IDF per calcolare similarità testuale
- Elabora ogni colonna selezionata separatamente
- Genera punteggi di similarità (0-1)

### 3. AI Enhancement (Opzionale)
- Identifica match con similarità < 0.7
- Usa OpenAI GPT per analisi semantica
- Considera contesto e intento della pagina
- Sostituisce match di bassa qualità

### 4. Ranking e Selezione
- Calcola migliore, secondo e peggiore match
- Seleziona URL finale basato su punteggio più alto
- Identifica potenziali conflitti (double match)

## 📈 Interpretazione Qualità

### Punteggi di Similarità:
- **≥ 0.8**: 🟢 Alta qualità - Match eccellente
- **0.5-0.8**: 🟡 Media qualità - Controllare manualmente
- **< 0.5**: 🔴 Bassa qualità - Revisione necessaria

### Best Practices:
- Usa almeno 2-3 colonne per il matching
- Abilita AI per progetti critici
- Rivedi manualmente match < 0.5
- Verifica double match per conflitti

## 🔍 Troubleshooting

### Errori comuni:

**"Colonne obbligatorie mancanti"**
- Verifica che i file contengano Address e Status Code
- Controlla l'ortografia esatta dei nomi colonna

**"Errore caricamento file"**
- Verifica formato file (CSV/Excel)
- Controlla encoding (UTF-8 consigliato)
- Assicurati che il file non sia corrotto

**"API OpenAI non funziona"**
- Verifica la validità della API key
- Controlla i crediti disponibili
- Verifica la connessione internet

### Performance:
- File grandi (>10k righe): disabilita anteprima
- Molte colonne: tempo elaborazione aumenta
- AI enabled: elaborazione più lenta ma più accurata


## 🔄 Versioni

### v1.0.0
- Versione iniziale con matching base
- Supporto CSV e Excel
- Interfaccia Streamlit

### v1.1.0 (Corrente)
- Aggiunto supporto AI con OpenAI
- Colonne di matching personalizzabili
- Analisi qualità migliorata
- Visualizzazioni statistiche

---

Sviluppato da Daniele Pisciottano 🦕
