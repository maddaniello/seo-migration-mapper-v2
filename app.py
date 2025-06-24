import streamlit as st
import pandas as pd
import numpy as np
import time
import gc
from typing import List, Dict, Tuple, Optional
import openai
from polyfuzz import PolyFuzz
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import chardet

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
</style>
""", unsafe_allow_html=True)

class URLMigrationMapper:
    def __init__(self):
        self.chunk_size = 5000
        self.min_similarity_threshold = 0.3
        self.openai_client = None
        
    def initialize_openai(self, api_key: str) -> bool:
        """Inizializza il client OpenAI"""
        try:
            openai.api_key = api_key
            self.openai_client = openai
            openai.models.list()
            return True
        except Exception as e:
            st.error(f"Errore nell'inizializzare OpenAI: {str(e)}")
            return False
    
    def detect_encoding(self, file_content: bytes) -> str:
        """Rileva automaticamente l'encoding del file"""
        result = chardet.detect(file_content)
        detected_encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        
        if confidence < 0.7:
            common_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in common_encodings:
                try:
                    file_content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
        
        return detected_encoding
    
    def load_csv_file(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Carica un file CSV gestendo diversi encoding"""
        try:
            # Leggi il contenuto del file come bytes
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Rileva l'encoding
            encoding = self.detect_encoding(file_content)
            st.info(f"📄 Encoding rilevato per {filename}: {encoding}")
            
            # Lista di encoding da provare
            encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            encodings_to_try = list(dict.fromkeys(encodings_to_try))
            
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, dtype='object')
                    st.success(f"✅ File {filename} caricato con encoding: {enc}")
                    return df
                except Exception:
                    continue
            
            # Fallback finale
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1', dtype='object')
            st.warning(f"⚠️ File {filename} caricato con fallback latin-1")
            return df
            
        except Exception as e:
            st.error(f"❌ Errore nel caricamento del file {filename}: {str(e)}")
            return None
    
    def chunked_polyfuzz_matching(self, source_list: List[str], target_list: List[str], 
                                 match_type: str = "URL") -> pd.DataFrame:
        """Esegue il matching PolyFuzz in chunk"""
        st.info(f"🔄 Processando {match_type} matching...")
        
        all_matches = []
        total_chunks = (len(source_list) + self.chunk_size - 1) // self.chunk_size
        
        progress_bar = st.progress(0)
        
        for i in range(0, len(source_list), self.chunk_size):
            chunk_num = (i // self.chunk_size) + 1
            progress_bar.progress(chunk_num / total_chunks)
            
            chunk_source = source_list[i:i + self.chunk_size]
            
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
        st.success(f"✅ {match_type} matching completato: {len(final_matches)} match trovati")
        
        return final_matches
    
    def ai_enhanced_matching(self, unmatched_sources: List[str], targets: List[str]) -> Dict[str, str]:
        """Usa OpenAI per migliorare il matching delle URL non matchate"""
        if not self.openai_client or len(unmatched_sources) == 0:
            return {}
        
        max_ai_matches = min(20, len(unmatched_sources))
        ai_matches = {}
        
        try:
            progress_bar = st.progress(0)
            
            for i, source_url in enumerate(unmatched_sources[:max_ai_matches]):
                progress_bar.progress((i + 1) / max_ai_matches)
                
                prompt = f"""
                Trova la migliore corrispondenza per questa URL sorgente nella lista target.
                
                URL Sorgente: {source_url}
                
                URL Target (prime 20): {targets[:20]}
                
                Rispondi solo con l'URL target più simile o "NO_MATCH".
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
                
                time.sleep(0.1)
                
        except Exception as e:
            st.warning(f"Errore durante il matching AI: {str(e)}")
        
        return ai_matches
    
    def process_migration_mapping(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                                extra_columns: List[str] = None, use_ai: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Processo principale con output completo"""
        
        start_time = time.time()
        extra_columns = extra_columns or []
        
        st.info(f"""
        📊 **Informazioni sui file:**
        - **Live**: {len(df_live):,} righe, {len(df_live.columns)} colonne
        - **Staging**: {len(df_staging):,} righe, {len(df_staging.columns)} colonne
        - **Colonne extra per matching**: {extra_columns if extra_columns else 'Nessuna'}
        """)
        
        # Preprocessing
        print(f"Inizio elaborazione alle {time.strftime('%H:%M:%S')}")
        
        # Converti Status Code
        df_live['Status Code'] = pd.to_numeric(df_live['Status Code'], errors='coerce').fillna(0).astype('int16')
        df_staging['Status Code'] = pd.to_numeric(df_staging['Status Code'], errors='coerce').fillna(0).astype('int16')
        
        # Rimuovi duplicati
        df_live.drop_duplicates(subset="Address", inplace=True)
        df_staging.drop_duplicates(subset="Address", inplace=True)
        
        print(f"Dopo rimozione duplicati: Live={len(df_live)} righe, Staging={len(df_staging)} righe")
        
        # Gestione status codes
        df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)]
        df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)]
        df_3xx_5xx = pd.concat([df_3xx, df_5xx])
        
        # Mantieni 2xx e 4xx
        df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)]
        df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)]
        df_live = pd.concat([df_live_200, df_live_400])
        
        print(f"URL processabili: {len(df_live)} righe")
        
        # Gestione valori mancanti per tutte le colonne
        columns_to_fill = ['Title 1', 'H1-1'] + extra_columns
        for col in columns_to_fill:
            if col in df_live.columns:
                df_live[col] = df_live[col].fillna(df_live["Address"])
            if col in df_staging.columns:
                df_staging[col] = df_staging[col].fillna(df_staging["Address"])
        
        # MATCHING con colonne personalizzabili
        matching_columns = ['Address', 'Title 1', 'H1-1'] + extra_columns
        match_results = {}
        
        for col in matching_columns:
            if col in df_live.columns and col in df_staging.columns:
                print(f"Inizio matching per: {col}")
                match_results[col] = self.chunked_polyfuzz_matching(
                    list(df_live[col].dropna()), 
                    list(df_staging[col].dropna()), 
                    match_type=col
                )
        
        # Rinomina colonne per ogni match result
        df_pf_url = match_results.get('Address', pd.DataFrame())
        df_pf_title = match_results.get('Title 1', pd.DataFrame())
        df_pf_h1 = match_results.get('H1-1', pd.DataFrame())
        
        # Rename come nel codice originale
        if not df_pf_url.empty:
            df_pf_url.rename(columns={"Similarity": "URL Similarity", "From": "From (Address)", "To": "To Address"}, inplace=True)
        if not df_pf_title.empty:
            df_pf_title.rename(columns={"Similarity": "Title Similarity", "From": "From (Title)", "To": "To Title"}, inplace=True)
        if not df_pf_h1.empty:
            df_pf_h1.rename(columns={"Similarity": "H1 Similarity", "From": "From (H1)", "To": "To H1"}, inplace=True)
        
        # Gestione colonne extra
        extra_match_dfs = {}
        for col in extra_columns:
            if col in match_results and not match_results[col].empty:
                extra_df = match_results[col].copy()
                extra_df.rename(columns={
                    "Similarity": f"{col} Similarity", 
                    "From": f"From ({col})", 
                    "To": f"To {col}"
                }, inplace=True)
                extra_match_dfs[col] = extra_df
        
        print("Preparazione merge dei risultati...")
        
        # Preparazione lookup tables
        lookup_tables = {
            'Title 1': df_staging[['Title 1', 'Address']].drop_duplicates('Title 1') if 'Title 1' in df_staging.columns else pd.DataFrame(),
            'H1-1': df_staging[['H1-1', 'Address']].drop_duplicates('H1-1') if 'H1-1' in df_staging.columns else pd.DataFrame()
        }
        
        # Lookup per colonne extra
        for col in extra_columns:
            if col in df_staging.columns:
                lookup_tables[col] = df_staging[[col, 'Address']].drop_duplicates(col)
        
        # Merge base con URL - SEMPLIFICATO
        print("Merge dei dati di matching...")
        
        if not df_pf_url.empty:
            df_final = pd.merge(df_live, df_pf_url, left_on="Address", right_on="From (Address)", how="left")
        else:
            df_final = df_live.copy()
            df_final['URL Similarity'] = 0
            df_final['From (Address)'] = df_final['Address']
            df_final['To Address'] = ''
        
        # Merge Title - SEMPLIFICATO
        if not df_pf_title.empty and not lookup_tables['Title 1'].empty:
            df_pf_title_merge = pd.merge(df_pf_title, lookup_tables['Title 1'], left_on="To Title", right_on="Title 1", how="inner")
            # Rinomina per evitare conflitti
            df_pf_title_merge = df_pf_title_merge.rename(columns={'Address': 'Title_Match_URL'})
            df_final = pd.merge(df_final, df_pf_title_merge[['From (Title)', 'Title Similarity', 'To Title', 'Title_Match_URL']], 
                              left_on='Title 1', right_on='From (Title)', how='left')
        else:
            df_final['Title Similarity'] = 0
            df_final['From (Title)'] = df_final.get('Title 1', '')
            df_final['To Title'] = ''
            df_final['Title_Match_URL'] = ''
        
        # Merge H1 - SEMPLIFICATO
        if not df_pf_h1.empty and not lookup_tables['H1-1'].empty:
            df_pf_h1_merge = pd.merge(df_pf_h1, lookup_tables['H1-1'], left_on="To H1", right_on="H1-1", how="inner")
            # Rinomina per evitare conflitti
            df_pf_h1_merge = df_pf_h1_merge.rename(columns={'Address': 'H1_Match_URL'})
            df_final = pd.merge(df_final, df_pf_h1_merge[['From (H1)', 'H1 Similarity', 'To H1', 'H1_Match_URL']], 
                              left_on='H1-1', right_on='From (H1)', how='left')
        else:
            df_final['H1 Similarity'] = 0
            df_final['From (H1)'] = df_final.get('H1-1', '')
            df_final['To H1'] = ''
            df_final['H1_Match_URL'] = ''
        
        # Merge colonne extra - SEMPLIFICATO
        extra_match_urls = {}
        for col in extra_columns:
            if col in extra_match_dfs and col in lookup_tables:
                extra_merge = pd.merge(extra_match_dfs[col], lookup_tables[col], left_on=f"To {col}", right_on=col, how="inner")
                # Rinomina per evitare conflitti
                extra_merge = extra_merge.rename(columns={'Address': f'{col}_Match_URL'})
                df_final = pd.merge(df_final, extra_merge[[f'From ({col})', f'{col} Similarity', f'To {col}', f'{col}_Match_URL']], 
                                  left_on=col, right_on=f'From ({col})', how='left')
                extra_match_urls[col] = f'{col}_Match_URL'
            else:
                df_final[f'{col} Similarity'] = 0
                df_final[f'From ({col})'] = df_final.get(col, '')
                df_final[f'To {col}'] = ''
                df_final[f'{col}_Match_URL'] = ''
                extra_match_urls[col] = f'{col}_Match_URL'
        
        # Rinomina colonne per output finale - SEMPLIFICATO
        df_final = df_final.rename(columns={
            'Address': 'URL - Source',
            'To Address': 'URL - URL Match',
            'Title_Match_URL': 'URL - Title Match',
            'H1_Match_URL': 'URL - H1 Match'
        })
        
        # Rinomina colonne extra
        for col in extra_columns:
            if f'{col}_Match_URL' in df_final.columns:
                df_final = df_final.rename(columns={f'{col}_Match_URL': f'URL - {col} Match'})
        
        print("Calcolo dei match migliori...")
        
        # Calcolo best match con colonne personalizzabili
        similarity_cols = ["URL Similarity", "Title Similarity", "H1 Similarity"]
        for col in extra_columns:
            similarity_cols.append(f"{col} Similarity")
        
        # Assicurati che tutte le colonne esistano
        for col in similarity_cols:
            if col not in df_final.columns:
                df_final[col] = 0
        
        df_final[similarity_cols] = df_final[similarity_cols].fillna(0)
        
        # Get the max value across all similarity columns
        df_final['Best Match On'] = df_final[similarity_cols].idxmax(axis=1)
        
        # Calcolo Highest Match Similarity e Best Matching URL - SEMPLIFICATO
        for col in similarity_cols:
            mask = df_final['Best Match On'] == col
            df_final.loc[mask, 'Highest Match Similarity'] = df_final.loc[mask, col]
            
            # Determina URL corrispondente
            if col == "Title Similarity":
                df_final.loc[mask, 'Best Matching URL'] = df_final.loc[mask, 'URL - Title Match']
                df_final.loc[mask, 'Highest Match Source Text'] = df_final.loc[mask, 'From (Title)']
                df_final.loc[mask, 'Highest Match Destination Text'] = df_final.loc[mask, 'To Title']
            elif col == "H1 Similarity":
                df_final.loc[mask, 'Best Matching URL'] = df_final.loc[mask, 'URL - H1 Match']
                df_final.loc[mask, 'Highest Match Source Text'] = df_final.loc[mask, 'From (H1)']
                df_final.loc[mask, 'Highest Match Destination Text'] = df_final.loc[mask, 'To H1']
            elif col == "URL Similarity":
                df_final.loc[mask, 'Best Matching URL'] = df_final.loc[mask, 'URL - URL Match']
                df_final.loc[mask, 'Highest Match Source Text'] = df_final.loc[mask, 'URL - Source']
                df_final.loc[mask, 'Highest Match Destination Text'] = df_final.loc[mask, 'URL - URL Match']
            else:
                # Colonne extra
                col_name = col.replace(' Similarity', '')
                df_final.loc[mask, 'Best Matching URL'] = df_final.loc[mask, f'URL - {col_name} Match']
                df_final.loc[mask, 'Highest Match Source Text'] = df_final.loc[mask, f'From ({col_name})']
                df_final.loc[mask, 'Highest Match Destination Text'] = df_final.loc[mask, f'To {col_name}']
        
        # Rimuovi duplicati
        df_final.drop_duplicates(subset="URL - Source", inplace=True)
        
        # Calcolo SECONDO match migliore
        print("Calcolo dei match secondari...")
        df_final['Lowest Match On'] = df_final[similarity_cols].idxmin(axis=1)
        
        # Calcolo match intermedio
        df_final['Middle Match On'] = "URL Similarity Title Similarity H1 Similarity"
        for col in extra_columns:
            df_final['Middle Match On'] = df_final['Middle Match On'] + f" {col} Similarity"
        
        df_final['Middle Match On'] = df_final.apply(lambda x: x['Middle Match On'].replace(x['Best Match On'], ''), 1)
        df_final['Middle Match On'] = df_final.apply(lambda x: x['Middle Match On'].replace(x['Lowest Match On'], ''), 1)
        df_final['Middle Match On'] = df_final['Middle Match On'].str.strip()
        
        # Assegna Second Highest Match URL e Similarity
        for col in similarity_cols:
            mask = df_final['Middle Match On'] == col
            df_final.loc[mask, 'Second Highest Match Similarity'] = df_final.loc[mask, col]
            
            if col == "Title Similarity":
                df_final.loc[mask, 'Second Highest Match'] = df_final.loc[mask, 'URL - Title Match']
                df_final.loc[mask, 'Second Match Source Text'] = df_final.loc[mask, 'From (Title)']
                df_final.loc[mask, 'Second Match Destination Text'] = df_final.loc[mask, 'To Title']
            elif col == "H1 Similarity":
                df_final.loc[mask, 'Second Highest Match'] = df_final.loc[mask, 'URL - H1 Match']
                df_final.loc[mask, 'Second Match Source Text'] = df_final.loc[mask, 'From (H1)']
                df_final.loc[mask, 'Second Match Destination Text'] = df_final.loc[mask, 'To H1']
            elif col == "URL Similarity":
                df_final.loc[mask, 'Second Highest Match'] = df_final.loc[mask, 'URL - URL Match']
                df_final.loc[mask, 'Second Match Source Text'] = df_final.loc[mask, 'URL - Source']
                df_final.loc[mask, 'Second Match Destination Text'] = df_final.loc[mask, 'URL - URL Match']
            else:
                # Colonne extra
                col_name = col.replace(' Similarity', '')
                df_final.loc[mask, 'Second Highest Match'] = df_final.loc[mask, f'URL - {col_name} Match']
                df_final.loc[mask, 'Second Match Source Text'] = df_final.loc[mask, f'From ({col_name})']
                df_final.loc[mask, 'Second Match Destination Text'] = df_final.loc[mask, f'To {col_name}']
        
        # Rinomina Second Match On
        df_final.rename(columns={"Middle Match On": "Second Match On"}, inplace=True)
        
        # Check if both url recommendations are the same
        df_final["Double Matched?"] = df_final['Best Matching URL'].str.lower() == df_final['Second Highest Match'].str.lower()
        
        # Rinomina Best Match On per output finale
        df_final['Best Match On'] = df_final['Best Match On'].str.replace("Title Similarity", "Page Title")
        df_final['Best Match On'] = df_final['Best Match On'].str.replace("H1 Similarity", "H1 Heading")
        df_final['Best Match On'] = df_final['Best Match On'].str.replace("URL Similarity", "URL")
        for col in extra_columns:
            df_final['Best Match On'] = df_final['Best Match On'].str.replace(f"{col} Similarity", col)
        
        # Rinomina Second Match On per output finale
        df_final['Second Match On'] = df_final['Second Match On'].str.replace("Title Similarity", "Page Title")
        df_final['Second Match On'] = df_final['Second Match On'].str.replace("H1 Similarity", "H1 Heading")
        df_final['Second Match On'] = df_final['Second Match On'].str.replace("URL Similarity", "URL")
        for col in extra_columns:
            df_final['Second Match On'] = df_final['Second Match On'].str.replace(f"{col} Similarity", col)
        
        # AI Enhancement se abilitato
        if use_ai and self.openai_client:
            print("🤖 AI Enhancement in corso...")
            unmatched_mask = df_final['Highest Match Similarity'].fillna(0) < self.min_similarity_threshold
            unmatched_urls = df_final[unmatched_mask]['URL - Source'].tolist()
            
            if unmatched_urls:
                ai_matches = self.ai_enhanced_matching(
                    unmatched_urls[:20],
                    df_staging['Address'].tolist()
                )
                
                for source_url, target_url in ai_matches.items():
                    mask = df_final['URL - Source'] == source_url
                    df_final.loc[mask, 'Best Matching URL'] = target_url
                    df_final.loc[mask, 'Highest Match Similarity'] = 0.95
                    df_final.loc[mask, 'Best Match On'] = 'AI Enhanced'
                    df_final.loc[mask, 'Highest Match Source Text'] = source_url
                    df_final.loc[mask, 'Highest Match Destination Text'] = target_url
        
        # Set delle colonne finali
        final_columns = [
            "URL - Source",
            "Status Code", 
            "Best Matching URL",
            "Best Match On",
            "Highest Match Similarity",
            "Highest Match Source Text",
            "Highest Match Destination Text",
            "Second Highest Match",
            "Second Match On",
            "Second Highest Match Similarity",
            "Second Match Source Text",
            "Second Match Destination Text",
            "Double Matched?",
        ]
        
        # Mantieni solo le colonne che esistono
        existing_cols = [col for col in final_columns if col in df_final.columns]
        df_final = df_final[existing_cols]
        
        # Ordinamento finale
        df_final.sort_values(["Highest Match Similarity", "Double Matched?"], ascending=[False, False], inplace=True)
        
        # Statistiche finali
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 ELABORAZIONE COMPLETATA! 🎉")
        print(f"⏱️  Tempo totale: {total_time:.2f} secondi ({total_time/60:.1f} minuti)")
        print(f"📊 URL processati: {len(df_final)}")
        
        st.success(f"""
        🎉 **Elaborazione completata!**
        - ⏱️ Tempo: {total_time:.1f} secondi
        - 📊 URL processati: {len(df_final):,}
        - 🎯 Match trovati: {len(df_final[df_final['Highest Match Similarity'] > self.min_similarity_threshold]):,}
        - 📋 Colonne output: {len(existing_cols)}
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
    - 🎯 Colonne personalizzabili per matching aggiuntivo
    - 🤖 Enhancement con AI (OpenAI) - opzionale
    - 📊 Supporto per file di grandi dimensioni
    - 📈 Dashboard con metriche e visualizzazioni
    - 📋 Output completo con Second Match e Double Matched
    """)
    
    # Sidebar per configurazioni
    st.sidebar.header("⚙️ Configurazioni")
    
    chunk_size = st.sidebar.slider("Dimensione Chunk", 1000, 10000, 5000, 500)
    min_similarity = st.sidebar.slider("Soglia Similarità Minima", 0.1, 0.9, 0.3, 0.05)
    
    # Configurazione OpenAI
    st.sidebar.subheader("🤖 AI Enhancement")
    use_ai = st.sidebar.checkbox("Abilita AI Enhancement")
    openai_api_key = ""
    
    if use_ai:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.warning("Inserisci la tua API Key OpenAI")
    
    # Inizializza il mapper
    mapper = URLMigrationMapper()
    mapper.chunk_size = chunk_size
    mapper.min_similarity_threshold = min_similarity
    
    if use_ai and openai_api_key:
        mapper.initialize_openai(openai_api_key)
    
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
            # Caricamento file
            with st.spinner("Caricamento file..."):
                df_live = mapper.load_csv_file(live_file, "Live")
                df_staging = mapper.load_csv_file(staging_file, "Staging")
            
            if df_live is None or df_staging is None:
                st.error("❌ Errore nel caricamento dei file")
                return
            
            # Controllo colonne richieste
            required_cols = ['Address', 'Status Code', 'Title 1', 'H1-1']
            live_missing = [col for col in required_cols if col not in df_live.columns]
            staging_missing = [col for col in required_cols if col not in df_staging.columns]
            
            if live_missing or staging_missing:
                st.error(f"❌ Colonne mancanti - Live: {live_missing}, Staging: {staging_missing}")
                st.info("Colonne richieste: Address, Status Code, Title 1, H1-1")
                return
            
            st.success("✅ File caricati e validati con successo!")
            
            # Mostra info sui file
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Righe Live", f"{len(df_live):,}")
                st.metric("Colonne Live", len(df_live.columns))
            with col2:
                st.metric("Righe Staging", f"{len(df_staging):,}")
                st.metric("Colonne Staging", len(df_staging.columns))
            
            # Selezione colonne extra per matching
            st.header("🎯 Colonne Aggiuntive per Matching")
            
            # Trova colonne comuni (escluse quelle base)
            base_columns = ['Address', 'Status Code', 'Title 1', 'H1-1']
            available_live_cols = [col for col in df_live.columns if col not in base_columns]
            available_staging_cols = [col for col in df_staging.columns if col not in base_columns]
            common_extra_cols = list(set(available_live_cols) & set(available_staging_cols))
            
            # Suggerimenti di colonne utili per SEO
            suggested_cols = []
            for col in ['Meta Description 1', 'Canonical Link Element 1', 'Content Type', 'H2-1', 'Meta Keywords 1']:
                if col in common_extra_cols:
                    suggested_cols.append(col)
            
            if common_extra_cols:
                st.info(f"**Colonne disponibili per matching:** {len(common_extra_cols)}")
                
                if suggested_cols:
                    st.markdown(f"**💡 Colonne consigliate per SEO:** {', '.join(suggested_cols)}")
                
                # Multiselect per colonne extra
                extra_columns = st.multiselect(
                    "Seleziona colonne aggiuntive per il matching:",
                    options=sorted(common_extra_cols),
                    default=suggested_cols[:3] if suggested_cols else [],
                    help="""
                    Queste colonne saranno utilizzate insieme a URL, Title e H1 per il matching.
                    Più colonne aggiungi, più preciso sarà il matching ma più lenta l'elaborazione.
                    """
                )
                
                if extra_columns:
                    st.success(f"✅ Colonne selezionate: {', '.join(extra_columns)}")
                else:
                    st.info("ℹ️ Nessuna colonna aggiuntiva selezionata. Verrà usato solo URL, Title e H1.")
                    
            else:
                extra_columns = []
                st.warning("⚠️ Nessuna colonna aggiuntiva comune trovata tra i due file.")
                st.info("Verrà usato solo il matching base su URL, Title e H1.")
            
            # Pulsante elaborazione
            if st.button("🚀 Avvia Elaborazione", type="primary"):
                
                if len(df_live) + len(df_staging) > 50000:
                    st.warning("⚠️ File di grandi dimensioni. L'elaborazione potrebbe richiedere diversi minuti.")
                
                # Elaborazione
                with st.spinner("Elaborazione in corso..."):
                    df_final, df_non_redirectable = mapper.process_migration_mapping(
                        df_live, df_staging, extra_columns, use_ai
                    )
                
                # Risultati
                st.header("📊 Risultati")
                
                # Metriche
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("URL Processati", f"{len(df_final):,}")
                with col2:
                    matched_count = len(df_final[df_final['Highest Match Similarity'] > min_similarity])
                    st.metric("URL Matchati", f"{matched_count:,}")
                with col3:
                    match_rate = (matched_count / len(df_final) * 100) if len(df_final) > 0 else 0
                    st.metric("Tasso di Match", f"{match_rate:.1f}%")
                with col4:
                    st.metric("Non Redirectable", f"{len(df_non_redirectable):,}")
                
                # Grafici
                if len(df_final) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_hist = px.histogram(
                            df_final, 
                            x='Highest Match Similarity',
                            nbins=20,
                            title="Distribuzione Similarità"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        if 'Best Match On' in df_final.columns:
                            match_counts = df_final['Best Match On'].value_counts()
                            fig_pie = px.pie(
                                values=match_counts.values,
                                names=match_counts.index,
                                title="Match per Tipo"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                
                # Preview risultati con filtri
                st.subheader("👀 Preview Risultati")
                
                # Filtri
                col1, col2 = st.columns(2)
                with col1:
                    min_sim_filter = st.slider("Similarità minima per preview", 0.0, 1.0, min_similarity)
                with col2:
                    max_rows_preview = st.selectbox("Righe da mostrare", [10, 25, 50, 100], index=1)
                
                # Applica filtri
                filtered_df = df_final[df_final['Highest Match Similarity'] >= min_sim_filter].head(max_rows_preview)
                
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
                                f"migration-results-{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "📥 Scarica Risultati Mapping"
                            ), 
                            unsafe_allow_html=True
                        )
                
                with col2:
                    if len(df_non_redirectable) > 0:
                        st.markdown(
                            create_download_link(
                                df_non_redirectable, 
                                f"non-redirectable-{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "📥 Scarica URL Non Redirectable"
                            ), 
                            unsafe_allow_html=True
                        )
                
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
        <p>💡 <strong>Funzionalità principali:</strong> Matching intelligente, Colonne personalizzabili, AI Enhancement, Output completo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
