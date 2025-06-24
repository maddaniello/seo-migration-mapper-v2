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
        """Esegue il matching PolyFuzz in chunk - STRUTTURA ORIGINALE"""
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
        
        max_ai_matches = min(20, len(unmatched_sources))  # Ridotto per costi
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
        """Processo principale - STRUTTURA ORIGINALE SEMPLIFICATA"""
        
        start_time = time.time()
        extra_columns = extra_columns or []
        
        st.info(f"""
        📊 **Informazioni sui file:**
        - **Live**: {len(df_live):,} righe, {len(df_live.columns)} colonne
        - **Staging**: {len(df_staging):,} righe, {len(df_staging.columns)} colonne
        """)
        
        # Preprocessing SEMPLIFICATO - ESATTAMENTE come nel codice originale
        print(f"Inizio elaborazione alle {time.strftime('%H:%M:%S')}")
        
        # Converti Status Code
        df_live['Status Code'] = pd.to_numeric(df_live['Status Code'], errors='coerce').fillna(0).astype('int16')
        df_staging['Status Code'] = pd.to_numeric(df_staging['Status Code'], errors='coerce').fillna(0).astype('int16')
        
        # Rimuovi duplicati
        df_live.drop_duplicates(subset="Address", inplace=True)
        df_staging.drop_duplicates(subset="Address", inplace=True)
        
        print(f"Dopo rimozione duplicati: Live={len(df_live)} righe, Staging={len(df_staging)} righe")
        
        # Gestione status codes ESATTAMENTE come nel codice originale
        df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)]
        df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)]
        df_3xx_5xx = pd.concat([df_3xx, df_5xx])
        
        # Mantieni 2xx e 4xx
        df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)]
        df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)]
        df_live = pd.concat([df_live_200, df_live_400])
        
        print(f"URL processabili: {len(df_live)} righe")
        
        # Gestione valori mancanti ESATTAMENTE come nel codice originale
        df_live["Title 1"] = df_live["Title 1"].fillna(df_live["Address"])
        df_live["H1-1"] = df_live["H1-1"].fillna(df_live["Address"])
        df_staging["Title 1"] = df_staging["Title 1"].fillna(df_staging["Address"])
        df_staging["H1-1"] = df_staging["H1-1"].fillna(df_staging["Address"])
        
        # MATCHING ESATTAMENTE come nel codice originale
        print("Inizio matching degli URL...")
        df_pf_url = self.chunked_polyfuzz_matching(
            list(df_live["Address"]), 
            list(df_staging["Address"]), 
            match_type="URL"
        )
        
        print("Inizio matching dei titoli...")
        df_pf_title = self.chunked_polyfuzz_matching(
            list(df_live["Title 1"]), 
            list(df_staging["Title 1"]), 
            match_type="Title"
        )
        
        print("Inizio matching degli H1...")
        df_pf_h1 = self.chunked_polyfuzz_matching(
            list(df_live["H1-1"]), 
            list(df_staging["H1-1"]), 
            match_type="H1"
        )
        
        # Rinomina colonne ESATTAMENTE come nel codice originale
        df_pf_url.rename(columns={"Similarity": "URL Similarity", "From": "From (Address)", "To": "To Address"}, inplace=True)
        df_pf_title.rename(columns={"Similarity": "Title Similarity", "From": "From (Title)", "To": "To Title"}, inplace=True)
        df_pf_h1.rename(columns={"Similarity": "H1 Similarity", "From": "From (H1)", "To": "To H1"}, inplace=True)
        
        print("Preparazione merge dei risultati...")
        
        # Preparazione merge ESATTAMENTE come nel codice originale
        df_new_title = df_staging[['Title 1', 'Address']].drop_duplicates('Title 1')
        df_new_h1 = df_staging[['H1-1', 'Address']].drop_duplicates('H1-1')
        
        # Merge ESATTAMENTE come nel codice originale
        print("Merge dei dati di matching...")
        df_pf_title_merge = pd.merge(df_pf_title, df_new_title, left_on="To Title", right_on="Title 1", how="inner")
        df_pf_h1_merge = pd.merge(df_pf_h1, df_new_h1, left_on="To H1", right_on="H1-1", how="inner")
        
        # Costruzione finale ESATTAMENTE come nel codice originale
        print("Creazione dataset finale...")
        df_final = pd.merge(df_live, df_pf_url, left_on="Address", right_on="From (Address)", how="inner")
        df_final = df_final.merge(df_pf_title_merge.drop_duplicates('Title 1'), how='left', left_on='Title 1', right_on="From (Title)")
        df_final = df_final.merge(df_pf_h1_merge.drop_duplicates('H1-1'), how='left', left_on='H1-1', right_on="From (H1)")
        
        # Rinomina colonne ESATTAMENTE come nel codice originale
        df_final.rename(
            columns={
                "Address_x": "URL - Source",
                "To Address": "URL - URL Match",
                "Address_y": "URL - Title Match",
                "Address": "URL - H1 Match",
            },
            inplace=True,
        )
        
        print("Calcolo dei match migliori...")
        
        # Calcolo best match ESATTAMENTE come nel codice originale
        similarity_cols = ["URL Similarity", "Title Similarity", "H1 Similarity"]
        df_final[similarity_cols] = df_final[similarity_cols].fillna(0)
        
        df_final['Best Match On'] = df_final[similarity_cols].idxmax(axis=1)
        
        # Best match logic ESATTAMENTE come nel codice originale
        df_final.loc[df_final['Best Match On'] == "Title Similarity", 'Highest Match Similarity'] = df_final['Title Similarity']
        df_final.loc[df_final['Best Match On'] == "Title Similarity", 'Best Matching URL'] = df_final['URL - Title Match']
        df_final.loc[df_final['Best Match On'] == "H1 Similarity", 'Highest Match Similarity'] = df_final['H1 Similarity']
        df_final.loc[df_final['Best Match On'] == "H1 Similarity", 'Best Matching URL'] = df_final['URL - H1 Match']
        df_final.loc[df_final['Best Match On'] == "URL Similarity", 'Highest Match Similarity'] = df_final['URL Similarity']
        df_final.loc[df_final['Best Match On'] == "URL Similarity", 'Best Matching URL'] = df_final['URL - URL Match']
        
        df_final.drop_duplicates(subset="URL - Source", inplace=True)
        
        # AI Enhancement se abilitato
        if use_ai and self.openai_client:
            print("🤖 AI Enhancement in corso...")
            unmatched_mask = df_final['Highest Match Similarity'].fillna(0) < self.min_similarity_threshold
            unmatched_urls = df_final[unmatched_mask]['URL - Source'].tolist()
            
            if unmatched_urls:
                ai_matches = self.ai_enhanced_matching(
                    unmatched_urls[:20],  # Limita per costi
                    df_staging['Address'].tolist()
                )
                
                for source_url, target_url in ai_matches.items():
                    mask = df_final['URL - Source'] == source_url
                    df_final.loc[mask, 'Best Matching URL'] = target_url
                    df_final.loc[mask, 'Highest Match Similarity'] = 0.95
                    df_final.loc[mask, 'Best Match On'] = 'AI Enhanced'
        
        # Ordinamento finale
        df_final.sort_values("Highest Match Similarity", ascending=False, inplace=True)
        
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
    - 🤖 Enhancement con AI (OpenAI) - opzionale
    - 📊 Supporto per file di grandi dimensioni
    - 📈 Dashboard con metriche e visualizzazioni
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
            
            # Pulsante elaborazione
            if st.button("🚀 Avvia Elaborazione", type="primary"):
                
                if len(df_live) + len(df_staging) > 50000:
                    st.warning("⚠️ File di grandi dimensioni. L'elaborazione potrebbe richiedere diversi minuti.")
                
                # Elaborazione
                with st.spinner("Elaborazione in corso..."):
                    df_final, df_non_redirectable = mapper.process_migration_mapping(
                        df_live, df_staging, [], use_ai
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
                
                # Preview risultati
                st.subheader("👀 Preview Risultati")
                preview_df = df_final.head(25)
                st.dataframe(preview_df, use_container_width=True)
                
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

if __name__ == "__main__":
    main()
