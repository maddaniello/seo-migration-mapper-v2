# 🔗 URL Migration Mapper

Un'applicazione Streamlit avanzata per il mapping automatico di URL durante le migrazioni di siti web, con supporto per AI enhancement e processing di file di grandi dimensioni.

## ✨ Funzionalità

- **🔄 Matching Intelligente**: Confronta URL basandosi su Address, Title, H1 e colonne personalizzabili
- **🤖 AI Enhancement**: Utilizza OpenAI per migliorare il matching delle URL difficili da mappare
- **📊 Gestione File Grandi**: Processing ottimizzato in batch per file di grandi dimensioni
- **🎯 Colonne Personalizzabili**: Seleziona colonne aggiuntive per il matching
- **📈 Dashboard Interattiva**: Visualizzazioni e metriche in tempo reale
- **💾 Export Risultati**: Download dei risultati in formato CSV

## 🚀 Installazione e Utilizzo

### Prerequisiti

- Python 3.8+
- pip

### Installazione

1. Clona il repository:
```bash
git clone https://github.com/tuo-username/url-migration-mapper.git
cd url-migration-mapper
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Avvia l'applicazione:
```bash
streamlit run app.py
```

4. Apri il browser all'indirizzo `http://localhost:8501`

## 📋 Preparazione dei File

### Formato dei File CSV

I file CSV devono contenere almeno queste colonne:

**File Live (Pre-migrazione):**
- `Address` - URL della pagina
- `Status Code` - Codice di stato HTTP
- `Title 1` - Titolo della pagina
- `H1-1` - Primo heading H1

**File Staging (Post-migrazione):**
- `Address` - URL della pagina
- `Status Code` - Codice di stato HTTP
- `Title 1` - Titolo della pagina
- `H1-1` - Primo heading H1
- `Indexability` (opzionale) - Stato di indicizzabilità

### Esempio di Struttura

```csv
Address,Status Code,Title 1,H1-1
https://example.com/page1,200,Pagina 1,Heading 1
https://example.com/page2,404,Pagina 2,Heading 2
```

## ⚙️ Configurazione

### Parametri Principali

- **Dimensione Chunk**: Numero di URL processate per batch (default: 5000)
- **Soglia Similarità**: Valore minimo di similarità per considerare un match valido (default: 0.3)

### AI Enhancement

Per utilizzare l'AI enhancement:

1. Ottieni una API key da [OpenAI](https://platform.openai.com/api-keys)
2. Inserisci la chiave nella sidebar dell'applicazione
3. Abilita l'opzione "AI Enhancement"

⚠️ **Nota**: L'AI enhancement ha un costo associato basato sull'utilizzo dell'API OpenAI.

## 🔧 Funzionalità Avanzate

### Processing di File Grandi

L'applicazione gestisce automaticamente file di grandi dimensioni:
- Processing in batch per evitare problemi di memoria
- Ottimizzazione dei tipi di dati
- Garbage collection automatica
- Progress bar per monitorare l'avanzamento

### Colonne Personalizzabili

Puoi selezionare colonne aggiuntive per il matching oltre a quelle standard:
- Meta Description
- Canonical URL
- Content Type
- Qualsiasi altra colonna presente in entrambi i file

### Algoritmi di Matching

1. **TF-IDF Vectorization**: Per il matching semantico del testo
2. **Cosine Similarity**: Per calcolare la similarità tra vettori
3. **AI Enhancement**: Analisi semantica avanzata tramite GPT per casi complessi

## 📊 Output e Risultati

### File di Output

1. **migration-mapped-results.csv**: Contiene tutti i mapping trovati con:
   - URL sorgente e destinazione
   - Tipo di match (URL, Title, H1, AI)
   - Score di similarità
   - Testo sorgente e destinazione per verifica

2. **non-redirectable-urls.csv**: URL con status code 3xx e 5xx che non possono essere redirezionate

### Metriche Disponibili

- **Tasso di Match**: Percentuale di URL matchate con successo
- **Distribuzione Similarità**: Istogramma dei punteggi di similarità
- **Match per Tipo**: Distribuzione dei match per tipo di campo
- **Performance**: Tempo di elaborazione e statistiche

## 🔍 Interpretazione dei Risultati

### Score di Similarità

- **0.8-1.0**: Match molto accurato, alta confidenza
- **0.6-0.8**: Match buono, verifica consigliata
- **0.3-0.6**: Match moderato, verifica necessaria
- **0.0-0.3**: Match scartato automaticamente

### Tipi di Match

- **URL**: Match basato sulla struttura dell'URL
- **Page Title**: Match basato sul titolo della pagina
- **H1 Heading**: Match basato sul primo heading H1
- **AI Enhanced**: Match migliorato tramite analisi AI
- **Custom Column**: Match basato su colonne personalizzate

## 🚀 Deployment

### Streamlit Cloud

1. Fork questo repository
2. Vai su [Streamlit Cloud](https://streamlit.io/cloud)
3. Connetti il tuo repository GitHub
4. Deploy l'applicazione

### Heroku

1. Crea un'app Heroku
2. Aggiungi i file necessari per Heroku:

```bash
# Procfile
web: sh setup.sh && streamlit run app.py

# setup.sh
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml
```

3. Deploy tramite Git

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contribuzioni

Le contribuzioni sono benvenute! Per contribuire:

1. Fork il progetto
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push sul branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📝 Changelog

### v2.0.0 (Corrente)
- ✨ Aggiunto AI enhancement con OpenAI
- 🚀 Supporto per file di grandi dimensioni
- 🎯 Colonne personalizzabili per matching
- 📊 Dashboard interattiva con visualizzazioni
- 💾 Export migliorato dei risultati

### v1.0.0
- 🔄 Matching base con TF-IDF
- 📁 Supporto file CSV
- 📈 Metriche di base

## ⚠️ Limitazioni e Considerazioni

### Performance
- File molto grandi (>100k righe) possono richiedere diversi minuti
- L'AI enhancement è limitato a 50 URL per sessione per controllare i costi
- Memoria richiesta: ~100MB per 10k righe

### Costi
- L'applicazione base è gratuita
- L'AI enhancement comporta costi per l'API OpenAI (~$0.001 per URL processata)

### Precisione
- Il matching automatico ha un'accuratezza del ~85-90%
- Si consiglia sempre una revisione manuale dei risultati
- L'AI enhancement può migliorare l'accuratezza fino al ~95%

## 🛠️ Troubleshooting

### Problemi Comuni

**Errore di memoria con file grandi:**
- Riduci la dimensione del chunk nelle impostazioni
- Processa il file in più sessioni separate

**Match scarsi:**
- Abbassa la soglia di similarità
- Abilita l'AI enhancement
- Verifica la qualità dei dati di input

**Errori OpenAI:**
- Verifica che la API key sia valida
- Controlla i crediti disponibili nel tuo account OpenAI
- Assicurati di avere accesso all'API GPT-3.5-turbo
