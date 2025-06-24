import streamlit as st
import pandas as pd
import numpy as np
import time
import gc
import io
import zipfile
from typing import List, Dict, Tuple, Optional
import openai
from polyfuzz import PolyFuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64

# Configurazione pagina
st.set_page_config(
    page_title="URL Migration Mapper",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class URLMigrationMapper:
    def __init__(self):
        self.chunk_size = 5000
        self.min_similarity_threshold = 0.3
        self.openai_client = None
        self.progress_bar = None
        self.status_text = None
        
    def initialize_openai(self, api_key: str) -> bool:
        """Inizializza il client OpenAI"""
        try:
            openai.api_key = api_key
            self.openai_client = openai
            # Test della connessione
            openai.models.list()
            return True
        except Exception as e:
            st.error(f"Errore nell'inizializzare OpenAI: {str(e)}")
            return False
    
    def detect_encoding(self, file_content: bytes) -> str:
        """Rileva automaticamente l'encoding del file"""
        import chardet
        
        # Usa chardet per rilevare l'encoding
        result = chardet.detect(file_content)
        detected_encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        
        # Se la confidenza è bassa, prova encoding comuni
        if confidence < 0.7:
            common_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in common_encodings:
                try:
                    file_content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
        
        return detected_encoding
    
    def validate_csv_structure(self, df: pd.DataFrame, file_type: str) -> Tuple[bool, List[str]]:
        """Valida la struttura del CSV caricato"""
        required_columns = ['Address', 'Status Code', 'Title 1', 'H1-1']
        missing_columns = []
        warnings = []
        
        # Controlla colonne obbligatorie
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        # Controlli specifici per tipo di file
        if file_type.lower() == 'staging' and 'Indexability' not in df.columns:
            warnings.append("Colonna 'Indexability' non trovata (opzionale)")
        
        # Controlli sulla qualità dei dati
        if 'Address' in df.columns:
            empty_addresses = df['Address'].isna().sum()
            if empty_addresses > 0:
                warnings.append(f"Trovate {empty_addresses} righe con Address vuoto")
        
        if 'Status Code' in df.columns:
            try:
                pd.to_numeric(df['Status Code'], errors='coerce')
            except:
                warnings.append("Alcuni Status Code non sono numerici")
        
        # Controllo duplicati
        if 'Address' in df.columns:
            duplicates = df['Address'].duplicated().sum()
            if duplicates > 0:
                warnings.append(f"Trovati {duplicates} URL duplicati (verranno rimossi)")
        
        is_valid = len(missing_columns) == 0
        
        return is_valid, missing_columns, warnings

    def load_csv_file(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Carica un file CSV gestendo diversi encoding e parametri"""
        try:
            # Leggi il contenuto del file come bytes
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Rileva l'encoding
            encoding = self.detect_encoding(file_content)
            st.info(f"📄 Encoding rilevato per {filename}: {encoding}")
            
            # Lista di encoding da provare in ordine di priorità
            encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            # Rimuovi duplicati mantenendo l'ordine
            encodings_to_try = list(dict.fromkeys(encodings_to_try))
            
            df = None
            last_error = None
            
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    
                    # Parametri ottimizzati per file grandi
                    df = pd.read_csv(
                        uploaded_file,
                        encoding=enc,
                        engine='c',  # Usa engine C invece di python
                        low_memory=False,
                        dtype='object',  # Carica tutto come object inizialmente
                        na_values=['', 'N/A', 'NULL', 'null', 'NaN'],
                        keep_default_na=True
                    )
                    
                    st.success(f"✅ File {filename} caricato con encoding: {enc}")
                    return df
                    
                except UnicodeDecodeError as e:
                    last_error = f"Errore Unicode con {enc}: {str(e)}"
                    continue
                except Exception as e:
                    last_error = f"Errore generico con {enc}: {str(e)}"
                    continue
            
            # Se arriviamo qui, nessun encoding ha funzionato
            raise Exception(f"Impossibile caricare il file con nessuno degli encoding testati. Ultimo errore: {last_error}")
            
        except Exception as e:
            st.error(f"❌ Errore nel caricamento del file {filename}")
            st.error(f"Dettaglio errore: {str(e)}")
            
            # Mostra suggerimenti per la risoluzione
            st.markdown("""
            **🔧 Possibili soluzioni:**
            1. **Controlla l'encoding**: Apri il file in un editor di testo e salvalo come UTF-8
            2. **Verifica il formato**: Assicurati che sia un file CSV valido
            3. **Caratteri speciali**: Rimuovi caratteri speciali non standard dal file
            4. **Dimensione file**: Se il file è molto grande (>100MB), prova a dividerlo
            """)
            
            return None

    def get_file_info(self, df: pd.DataFrame) -> Dict:
        """Ottiene informazioni sul file caricato"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "column_names": list(df.columns)
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Preprocessa il dataframe con ottimizzazioni per la memoria"""
        # Converti i tipi di dati per ottimizzare la memoria
        for col in df.columns:
            if col in ['Address', 'Title 1', 'H1-1'] + required_columns:
                df[col] = df[col].astype('string')
        
        # Gestisci Status Code
        if 'Status Code' in df.columns:
            df['Status Code'] = pd.to_numeric(df['Status Code'], errors='coerce').fillna(0).astype('int16')
        
        # Rimuovi duplicati
        if 'Address' in df.columns:
            df.drop_duplicates(subset="Address", inplace=True)
        
        return df
    
    def ai_enhanced_matching(self, unmatched_sources: List[str], targets: List[str], 
                           context: str = "") -> Dict[str, str]:
        """Usa OpenAI per migliorare il matching delle URL non matchate"""
        if not self.openai_client or len(unmatched_sources) == 0:
            return {}
        
        # Limita il numero di URL da processare con AI per evitare costi eccessivi
        max_ai_matches = min(50, len(unmatched_sources))
        ai_matches = {}
        
        try:
            for i, source_url in enumerate(unmatched_sources[:max_ai_matches]):
                if self.progress_bar:
                    self.progress_bar.progress((i + 1) / max_ai_matches)
                
                prompt = f"""
                Analizza questa URL sorgente e trova la migliore corrispondenza nella lista target.
                
                URL Sorgente: {source_url}
                
                Context: {context}
                
                URL Target disponibili: {targets[:20]}  # Limita per il prompt
                
                Rispondi solo con l'URL target più simile o "NO_MATCH" se nessuna è appropriata.
                Considera:
                - Struttura dell'URL
                - Parole chiave nel path
                - Semantica del contenuto
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                
                suggested_match = response.choices[0].message.content.strip()
                if suggested_match != "NO_MATCH" and suggested_match in targets:
                    ai_matches[source_url] = suggested_match
                
                time.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            st.warning(f"Errore durante il matching AI: {str(e)}")
        
        return ai_matches
    
    def chunked_polyfuzz_matching(self, source_list: List[str], target_list: List[str], 
                                 match_type: str = "URL") -> pd.DataFrame:
        """Esegue il matching PolyFuzz in chunk ottimizzati"""
        if self.status_text:
            self.status_text.text(f"🔄 Processando {match_type} matching...")
        
        all_matches = []
        total_chunks = (len(source_list) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(source_list), self.chunk_size):
            chunk_num = (i // self.chunk_size) + 1
            
            if self.progress_bar:
                self.progress_bar.progress(chunk_num / total_chunks)
            
            chunk_source = source_list[i:i + self.chunk_size]
            
            # Usa TF-IDF ottimizzato
            model = PolyFuzz("TF-IDF")
            model.match(chunk_source, target_list)
            matches = model.get_matches()
            
            # Filtra per soglia minima
            matches = matches[matches['Similarity'] >= self.min_similarity_threshold]
            all_matches.append(matches)
            
            # Pulizia memoria
            del model
            gc.collect()
        
        final_matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
        
        if self.status_text:
            self.status_text.text(f"✅ {match_type} matching completato: {len(final_matches)} match trovati")
        
        return final_matches
    
    def process_migration_mapping(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                                extra_columns: List[str] = None, use_ai: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Processo principale di mapping delle URL"""
        
        start_time = time.time()
        extra_columns = extra_columns or []
        
        # Informazioni sui file
        live_info = self.get_file_info(df_live)
        staging_info = self.get_file_info(df_staging)
        
        st.info(f"""
        📊 **Informazioni sui file:**
        - **Live**: {live_info['rows']:,} righe, {live_info['memory_usage']:.1f} MB
        - **Staging**: {staging_info['rows']:,} righe, {staging_info['memory_usage']:.1f} MB
        """)
        
        # Preprocessing
        required_cols = ['Address', 'Status Code', 'Title 1', 'H1-1'] + extra_columns
        
        # Filtra solo le colonne disponibili
        available_live_cols = [col for col in required_cols if col in df_live.columns]
        available_staging_cols = [col for col in required_cols if col in df_staging.columns]
        
        if 'Indexability' in df_staging.columns:
            available_staging_cols.append('Indexability')
        
        df_live = df_live[available_live_cols].copy()
        df_staging = df_staging[available_staging_cols].copy()
        
        # Preprocess
        df_live = self.preprocess_dataframe(df_live, extra_columns)
        df_staging = self.preprocess_dataframe(df_staging, extra_columns)
        
        # Gestione status codes
        df_3xx_5xx = pd.DataFrame()
        if 'Status Code' in df_live.columns:
            df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)]
            df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)]
            df_3xx_5xx = pd.concat([df_3xx, df_5xx])
            
            # Mantieni solo 2xx e 4xx
            df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)]
            df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)]
            df_live = pd.concat([df_live_200, df_live_400])
        
        # Gestione valori mancanti
        for col in ['Title 1', 'H1-1'] + extra_columns:
            if col in df_live.columns:
                df_live[col] = df_live[col].fillna(df_live.get('Address', ''))
            if col in df_staging.columns:
                df_staging[col] = df_staging[col].fillna(df_staging.get('Address', ''))
        
        # Setup progress tracking
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
        # Matching per colonne base
        matches = {}
        columns_to_match = ['Address', 'Title 1', 'H1-1'] + extra_columns
        
        for col in columns_to_match:
            if col in df_live.columns and col in df_staging.columns:
                self.status_text.text(f"🔄 Matching su: {col}")
                matches[col] = self.chunked_polyfuzz_matching(
                    list(df_live[col].dropna()), 
                    list(df_staging[col].dropna()), 
                    match_type=col
                )
        
        # Preparazione per merge
        staging_lookup = {}
        for col in columns_to_match:
            if col in df_staging.columns:
                staging_lookup[col] = df_staging[[col, 'Address']].drop_duplicates(col)
        
        # Costruzione risultato finale
        df_final = df_live.copy()
        
        # Merge dei match results
        for col, match_df in matches.items():
            if not match_df.empty and col in staging_lookup:
                # Rename columns per il merge
                match_df_renamed = match_df.rename(columns={
                    'Similarity': f'{col}_Similarity',
                    'From': f'From_{col}',
                    'To': f'To_{col}'
                })
                
                # Merge con lookup
                merged = pd.merge(
                    match_df_renamed, 
                    staging_lookup[col], 
                    left_on=f'To_{col}', 
                    right_on=col, 
                    how='inner'
                )
                
                # Merge con df_final
                df_final = pd.merge(
                    df_final, 
                    merged, 
                    left_on=col, 
                    right_on=f'From_{col}', 
                    how='left'
                )
        
        # Calcolo best match
        similarity_cols = [f'{col}_Similarity' for col in columns_to_match if f'{col}_Similarity' in df_final.columns]
        
        if similarity_cols:
            df_final[similarity_cols] = df_final[similarity_cols].fillna(0)
            df_final['Best_Match_On'] = df_final[similarity_cols].idxmax(axis=1)
            df_final['Highest_Match_Similarity'] = df_final[similarity_cols].max(axis=1)
            
            # Determina Best Matching URL
            for col in columns_to_match:
                sim_col = f'{col}_Similarity'
                if sim_col in df_final.columns:
                    mask = df_final['Best_Match_On'] == sim_col
                    df_final.loc[mask, 'Best_Matching_URL'] = df_final.loc[mask, 'Address_y']
        
        # AI Enhancement per URL non matchate (se abilitato)
        if use_ai and self.openai_client:
            unmatched_mask = df_final['Highest_Match_Similarity'].fillna(0) < self.min_similarity_threshold
            unmatched_urls = df_final[unmatched_mask]['Address'].tolist()
            
            if unmatched_urls:
                self.status_text.text("🤖 Miglioramento matching con AI...")
                ai_matches = self.ai_enhanced_matching(
                    unmatched_urls[:50],  # Limita per costi
                    df_staging['Address'].tolist()
                )
                
                # Applica i match AI
                for source_url, target_url in ai_matches.items():
                    mask = df_final['Address'] == source_url
                    df_final.loc[mask, 'Best_Matching_URL'] = target_url
                    df_final.loc[mask, 'Highest_Match_Similarity'] = 0.95  # High confidence AI match
                    df_final.loc[mask, 'Best_Match_On'] = 'AI_Enhanced'
        
        # Pulizia finale
        self.progress_bar.progress(1.0)
        self.status_text.text("✅ Elaborazione completata!")
        
        # Statistiche finali
        processing_time = time.time() - start_time
        st.success(f"""
        🎉 **Elaborazione completata!**
        - ⏱️ Tempo: {processing_time:.1f} secondi
        - 📊 URL processati: {len(df_final):,}
        - 🎯 Match trovati: {len(df_final[df_final['Highest_Match_Similarity'] > self.min_similarity_threshold]):,}
        """)
        
        return df_final, df_3xx_5xx

def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """Crea un link per il download del CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def main():
    st.markdown('<h1 class="main-header">🔗 URL Migration Mapper</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Strumento avanzato per il mapping automatico di URL durante le migrazioni di siti web.**
    
    Funzionalità:
    - 🔄 Matching intelligente basato su URL, Title e H1
    - 🤖 Enhancement con AI (OpenAI)
    - 📊 Supporto per file di grandi dimensioni
    - 🎯 Colonne personalizzabili per matching aggiuntivo
    - 📈 Dashboard con metriche e visualizzazioni
    """)
    
    # Sidebar per configurazioni
    st.sidebar.header("⚙️ Configurazioni")
    
    # Configurazioni principali
    chunk_size = st.sidebar.slider("Dimensione Chunk", 1000, 10000, 5000, 500)
    min_similarity = st.sidebar.slider("Soglia Similarità Minima", 0.1, 0.9, 0.3, 0.05)
    
    # Configurazione OpenAI
    st.sidebar.subheader("🤖 AI Enhancement")
    use_ai = st.sidebar.checkbox("Abilita AI Enhancement")
    openai_api_key = ""
    
    if use_ai:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.warning("Inserisci la tua API Key OpenAI per usare l'AI Enhancement")
    
    # Inizializza il mapper
    mapper = URLMigrationMapper()
    mapper.chunk_size = chunk_size
    mapper.min_similarity_threshold = min_similarity
    
    if use_ai and openai_api_key:
        if not mapper.initialize_openai(openai_api_key):
            use_ai = False
    
    # Upload dei file
    st.header("📁 Upload File")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File Live (Pre-migrazione)")
        live_file = st.file_uploader("Carica file CSV Live", type=['csv'], key="live")
    
    with col2:
        st.subheader("File Staging (Post-migrazione)")
        staging_file = st.file_uploader("Carica file CSV Staging", type=['csv'], key="staging")
    
    if live_file and staging_file:
        try:
            # Caricamento file con gestione encoding migliorata
            with st.spinner("Caricamento file..."):
                df_live = mapper.load_csv_file(live_file, "Live")
                df_staging = mapper.load_csv_file(staging_file, "Staging")
            
            # Controlla se il caricamento è riuscito
            if df_live is None or df_staging is None:
                st.error("❌ Impossibile procedere: uno o entrambi i file non sono stati caricati correttamente.")
                return
            
            # Validazione struttura file
            live_valid, live_missing, live_warnings = mapper.validate_csv_structure(df_live, "Live")
            staging_valid, staging_missing, staging_warnings = mapper.validate_csv_structure(df_staging, "Staging")
            
            # Mostra risultati validazione
            col1, col2 = st.columns(2)
            
            with col1:
                if live_valid:
                    st.success("✅ Struttura file Live: OK")
                else:
                    st.error(f"❌ File Live - Colonne mancanti: {', '.join(live_missing)}")
                
                if live_warnings:
                    for warning in live_warnings:
                        st.warning(f"⚠️ {warning}")
            
            with col2:
                if staging_valid:
                    st.success("✅ Struttura file Staging: OK")
                else:
                    st.error(f"❌ File Staging - Colonne mancanti: {', '.join(staging_missing)}")
                
                if staging_warnings:
                    for warning in staging_warnings:
                        st.warning(f"⚠️ {warning}")
            
            # Blocca l'elaborazione se i file non sono validi
            if not (live_valid and staging_valid):
                st.error("❌ Impossibile procedere: i file non hanno la struttura richiesta.")
                st.info("""
                **Colonne richieste:**
                - Address (URL della pagina)
                - Status Code (Codice di stato HTTP)
                - Title 1 (Titolo della pagina)
                - H1-1 (Primo heading H1)
                """)
                return
            
            st.success("✅ Entrambi i file caricati e validati con successo!")
            
            # Mostra info sui file
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Righe Live", f"{len(df_live):,}")
                st.metric("Colonne Live", len(df_live.columns))
            
            with col2:
                st.metric("Righe Staging", f"{len(df_staging):,}")
                st.metric("Colonne Staging", len(df_staging.columns))
            
            # Selezione colonne extra
            st.header("🎯 Colonne per Matching")
            
            available_live_cols = [col for col in df_live.columns 
                                 if col not in ['Address', 'Status Code', 'Title 1', 'H1-1']]
            available_staging_cols = [col for col in df_staging.columns 
                                    if col not in ['Address', 'Status Code', 'Title 1', 'H1-1', 'Indexability']]
            
            common_extra_cols = list(set(available_live_cols) & set(available_staging_cols))
            
            if common_extra_cols:
                extra_columns = st.multiselect(
                    "Seleziona colonne aggiuntive per il matching:",
                    common_extra_cols,
                    help="Queste colonne saranno utilizzate insieme a URL, Title e H1 per il matching"
                )
            else:
                extra_columns = []
                st.info("Nessuna colonna aggiuntiva comune trovata tra i due file.")
            
            # Pulsante per avviare l'elaborazione
            if st.button("🚀 Avvia Elaborazione", type="primary"):
                
                # Controllo dimensioni file
                total_rows = len(df_live) + len(df_staging)
                if total_rows > 50000:
                    st.warning(f"⚠️ File di grandi dimensioni ({total_rows:,} righe totali). L'elaborazione potrebbe richiedere diversi minuti.")
                
                # Elaborazione
                with st.spinner("Elaborazione in corso..."):
                    df_final, df_non_redirectable = mapper.process_migration_mapping(
                        df_live, df_staging, extra_columns, use_ai
                    )
                
                # Risultati
                st.header("📊 Risultati")
                
                # Metriche principali
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("URL Processati", f"{len(df_final):,}")
                
                with col2:
                    matched_count = len(df_final[df_final['Highest_Match_Similarity'] > min_similarity])
                    st.metric("URL Matchati", f"{matched_count:,}")
                
                with col3:
                    match_rate = (matched_count / len(df_final) * 100) if len(df_final) > 0 else 0
                    st.metric("Tasso di Match", f"{match_rate:.1f}%")
                
                with col4:
                    st.metric("Non Redirectable", f"{len(df_non_redirectable):,}")
                
                # Grafici
                if len(df_final) > 0:
                    st.subheader("📈 Analisi dei Risultati")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribuzione similarità
                        fig_hist = px.histogram(
                            df_final, 
                            x='Highest_Match_Similarity',
                            nbins=20,
                            title="Distribuzione Similarità Match",
                            labels={'Highest_Match_Similarity': 'Similarità', 'count': 'Conteggio'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Match per tipo
                        if 'Best_Match_On' in df_final.columns:
                            match_counts = df_final['Best_Match_On'].value_counts()
                            fig_pie = px.pie(
                                values=match_counts.values,
                                names=match_counts.index,
                                title="Match per Tipo di Campo"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                
                # Preview risultati
                st.subheader("👀 Preview Risultati")
                
                # Filtri per preview
                col1, col2 = st.columns(2)
                
                with col1:
                    min_sim_filter = st.slider("Similarità minima per preview", 0.0, 1.0, min_similarity)
                
                with col2:
                    max_rows_preview = st.selectbox("Righe da mostrare", [10, 25, 50, 100], index=1)
                
                # Applica filtri
                filtered_df = df_final[df_final['Highest_Match_Similarity'] >= min_sim_filter].head(max_rows_preview)
                
                if len(filtered_df) > 0:
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.info("Nessun risultato con la soglia di similarità selezionata.")
                
                # Download
                st.header("💾 Download Risultati")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(df_final) > 0:
                        st.markdown(
                            create_download_link(
                                df_final, 
                                f"migration-mapped-results-{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "📥 Scarica Risultati Mapping"
                            ), 
                            unsafe_allow_html=True
                        )
                
                with col2:
                    if len(df_non_redirectable) > 0:
                        st.markdown(
                            create_download_link(
                                df_non_redirectable, 
                                f"non-redirectable-urls-{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "📥 Scarica URL Non Redirectable"
                            ), 
                            unsafe_allow_html=True
                        )
                
                # Salva in session state per mantenere i risultati
                st.session_state['df_final'] = df_final
                st.session_state['df_non_redirectable'] = df_non_redirectable
                
        except Exception as e:
            st.error(f"Errore durante l'elaborazione: {str(e)}")
            st.error("Assicurati che i file CSV abbiano le colonne richieste: Address, Status Code, Title 1, H1-1")
    
    else:
        st.info("👆 Carica entrambi i file CSV per iniziare l'elaborazione.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔗 URL Migration Mapper | Sviluppato per ottimizzare le migrazioni SEO</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
