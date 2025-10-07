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
import re
from urllib.parse import urlparse

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

class LanguageDetector:
    """Classe per il rilevamento automatico della lingua dalle URL"""
    
    # Pattern comuni per le lingue
    LANGUAGE_PATTERNS = {
        'subdomain': r'^([a-z]{2})\..*',  # es: it.example.com
        'subfolder': r'/([a-z]{2})/',      # es: /it/page
        'subfolder_start': r'^/([a-z]{2})/',  # es: /it/page (all'inizio)
        'param': r'[?&]lang=([a-z]{2})',   # es: ?lang=it
        'country_subfolder': r'/([a-z]{2}-[a-z]{2})/',  # es: /en-us/
    }
    
    # Codici lingua comuni (ISO 639-1)
    COMMON_LANGUAGES = {
        'it', 'en', 'es', 'fr', 'de', 'pt', 'nl', 'ru', 'zh', 'ja',
        'ar', 'pl', 'tr', 'sv', 'da', 'no', 'fi', 'cs', 'ro', 'hu',
        'el', 'he', 'hi', 'th', 'ko', 'id', 'vi', 'uk', 'ca', 'bg'
    }
    
    @classmethod
    def detect_language_from_url(cls, url: str, default_language: Optional[str] = None) -> Optional[str]:
        """
        Rileva la lingua dall'URL usando vari pattern.
        Ritorna il codice lingua (es: 'it', 'en') o la lingua di default se specificata.
        
        Args:
            url: URL da analizzare
            default_language: Lingua da assegnare se nessun pattern viene trovato (es: 'it' per lingua principale)
        """
        if pd.isna(url) or not url:
            return default_language
        
        url = str(url).lower()
        parsed = urlparse(url)
        
        # 1. Controlla sottodominio (es: it.example.com)
        subdomain_match = re.match(cls.LANGUAGE_PATTERNS['subdomain'], parsed.netloc)
        if subdomain_match:
            lang = subdomain_match.group(1)
            if lang in cls.COMMON_LANGUAGES:
                return lang
        
        # 2. Controlla sottocartella all'inizio (es: /it/page)
        path = parsed.path
        subfolder_match = re.match(cls.LANGUAGE_PATTERNS['subfolder_start'], path)
        if subfolder_match:
            lang = subfolder_match.group(1)
            if lang in cls.COMMON_LANGUAGES:
                return lang
        
        # 3. Controlla sottocartella con country code (es: /en-us/)
        country_match = re.search(cls.LANGUAGE_PATTERNS['country_subfolder'], path)
        if country_match:
            lang_country = country_match.group(1)
            lang = lang_country.split('-')[0]
            if lang in cls.COMMON_LANGUAGES:
                return lang
        
        # 4. Controlla parametri URL (es: ?lang=it)
        param_match = re.search(cls.LANGUAGE_PATTERNS['param'], url)
        if param_match:
            lang = param_match.group(1)
            if lang in cls.COMMON_LANGUAGES:
                return lang
        
        # Se nessun pattern trovato, usa la lingua di default (se specificata)
        return default_language
    
    @classmethod
    def normalize_language_code(cls, lang_code: str) -> str:
        """
        Normalizza i codici lingua per gestire varianti simili.
        Esempi:
        - 'it-IT' -> 'it'
        - 'IT' -> 'it'
        - 'en-US' -> 'en'
        - 'en_GB' -> 'en'
        """
        if pd.isna(lang_code) or not lang_code:
            return 'unknown'
        
        lang_code = str(lang_code).lower().strip()
        
        # Rimuovi spazi extra
        lang_code = lang_code.replace(' ', '')
        
        # Gestisci formati con separatori (it-IT, en_US, it-it, etc.)
        if '-' in lang_code:
            # Prendi solo la prima parte (it-IT -> it)
            lang_code = lang_code.split('-')[0]
        elif '_' in lang_code:
            # Prendi solo la prima parte (en_US -> en)
            lang_code = lang_code.split('_')[0]
        
        # Verifica che sia un codice lingua valido
        if lang_code in cls.COMMON_LANGUAGES:
            return lang_code
        
        # Se non è valido, ritorna unknown
        if len(lang_code) != 2:
            return 'unknown'
        
        return lang_code
    
    @classmethod
    def add_language_column(cls, df: pd.DataFrame, url_column: str = 'Address', 
                           language_column: str = 'language', 
                           default_language: Optional[str] = None) -> pd.DataFrame:
        """
        Aggiunge o aggiorna la colonna lingua nel dataframe.
        Se la colonna lingua esiste già, la usa; altrimenti la rileva dall'URL.
        Normalizza automaticamente i codici lingua (it-IT -> it, IT -> it, etc.)
        
        Args:
            df: DataFrame da processare
            url_column: Nome della colonna con gli URL
            language_column: Nome della colonna lingua
            default_language: Lingua da assegnare alle URL senza pattern (es: 'it' per sito.com/page)
        """
        df = df.copy()
        
        # Se la colonna lingua esiste già e ha valori, usala
        if language_column in df.columns:
            # Normalizza i valori esistenti
            df[language_column] = df[language_column].apply(cls.normalize_language_code)
            
            # Rileva solo per righe senza lingua o con 'unknown'
            mask_missing = df[language_column].isna() | (df[language_column] == '') | (df[language_column] == 'unknown')
            if mask_missing.any():
                detected_langs = df.loc[mask_missing, url_column].apply(
                    lambda x: cls.detect_language_from_url(x, default_language)
                )
                # Normalizza anche le lingue rilevate
                df.loc[mask_missing, language_column] = detected_langs.apply(
                    lambda x: cls.normalize_language_code(x) if x else 'unknown'
                )
        else:
            # Crea la colonna ex-novo rilevando dall'URL
            df[language_column] = df[url_column].apply(
                lambda x: cls.detect_language_from_url(x, default_language)
            )
            # Normalizza
            df[language_column] = df[language_column].apply(
                lambda x: cls.normalize_language_code(x) if x else 'unknown'
            )
        
        # Riempi i valori mancanti con 'unknown'
        df[language_column] = df[language_column].fillna('unknown')
        
        return df


class URLMigrationMapper:
    def __init__(self):
        self.chunk_size = 5000
        self.min_similarity_threshold = 0.3
        self.openai_client = None
        self.language_detector = LanguageDetector()
        
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
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            encoding = self.detect_encoding(file_content)
            st.info(f"📄 Encoding rilevato per {filename}: {encoding}")
            
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
            
            matches = matches[matches['Similarity'] >= self.min_similarity_threshold]
            all_matches.append(matches)
            
            del model
            gc.collect()
        
        final_matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
        st.success(f"✅ {match_type} matching completato: {len(final_matches)} match trovati")
        
        return final_matches
    
    def process_migration_mapping(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                                extra_columns: List[str] = None, use_ai: bool = False,
                                use_language_matching: bool = True,
                                default_lang_live: Optional[str] = None,
                                default_lang_staging: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Processo principale con supporto multi-lingua"""
        
        start_time = time.time()
        extra_columns = extra_columns or []
        
        # GESTIONE LINGUE
        if use_language_matching:
            st.info("🌍 Rilevamento lingue in corso...")
            
            # Aggiungi colonna lingua con default language
            df_live = self.language_detector.add_language_column(
                df_live, 
                default_language=default_lang_live
            )
            df_staging = self.language_detector.add_language_column(
                df_staging,
                default_language=default_lang_staging
            )
            
            # Mostra statistiche lingue
            live_langs = df_live['language'].value_counts()
            staging_langs = df_staging['language'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Lingue in Live:**")
                st.write(live_langs.to_dict())
            with col2:
                st.write("**Lingue in Staging:**")
                st.write(staging_langs.to_dict())
        
        st.info(f"""
        📊 **Informazioni sui file:**
        - **Live**: {len(df_live):,} righe, {len(df_live.columns)} colonne
        - **Staging**: {len(df_staging):,} righe, {len(df_staging.columns)} colonne
        - **Colonne extra per matching**: {extra_columns if extra_columns else 'Nessuna'}
        - **Matching multi-lingua**: {'Attivo' if use_language_matching else 'Disattivato'}
        """)
        
        print(f"Inizio elaborazione alle {time.strftime('%H:%M:%S')}")
        
        # Converti Status Code
        df_live['Status Code'] = pd.to_numeric(df_live['Status Code'], errors='coerce').fillna(0).astype('int16')
        df_staging['Status Code'] = pd.to_numeric(df_staging['Status Code'], errors='coerce').fillna(0).astype('int16')
        
        # Rimuovi duplicati
        df_live.drop_duplicates(subset="Address", inplace=True)
        df_staging.drop_duplicates(subset="Address", inplace=True)
        
        # Gestione status codes
        df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)]
        df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)]
        df_3xx_5xx = pd.concat([df_3xx, df_5xx])
        
        df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)]
        df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)]
        df_live = pd.concat([df_live_200, df_live_400])
        
        # Gestione valori mancanti
        columns_to_fill = ['Title 1', 'H1-1'] + extra_columns
        for col in columns_to_fill:
            if col in df_live.columns:
                df_live[col] = df_live[col].fillna(df_live["Address"])
            if col in df_staging.columns:
                df_staging[col] = df_staging[col].fillna(df_staging["Address"])
        
        # MATCHING PER LINGUA
        matching_columns = ['Address', 'Title 1', 'H1-1'] + extra_columns
        match_results = {}
        
        if use_language_matching and 'language' in df_live.columns and 'language' in df_staging.columns:
            # Matching separato per ogni lingua
            languages = set(df_live['language'].unique()) | set(df_staging['language'].unique())
            languages.discard('unknown')  # Processeremo 'unknown' separatamente
            
            st.info(f"🔄 Matching per {len(languages)} lingue: {', '.join(sorted(languages))}")
            
            all_lang_matches = {col: [] for col in matching_columns}
            
            # Matching per ogni lingua
            for lang in sorted(languages):
                st.write(f"**Processando lingua: {lang.upper()}**")
                
                df_live_lang = df_live[df_live['language'] == lang]
                df_staging_lang = df_staging[df_staging['language'] == lang]
                
                if len(df_live_lang) == 0 or len(df_staging_lang) == 0:
                    st.warning(f"⚠️ Lingua {lang}: nessuna URL da matchare")
                    continue
                
                st.info(f"📊 {lang}: {len(df_live_lang)} URL live → {len(df_staging_lang)} URL staging")
                
                for col in matching_columns:
                    if col in df_live_lang.columns and col in df_staging_lang.columns:
                        matches = self.chunked_polyfuzz_matching(
                            list(df_live_lang[col].dropna()), 
                            list(df_staging_lang[col].dropna()), 
                            match_type=f"{col} ({lang})"
                        )
                        if not matches.empty:
                            all_lang_matches[col].append(matches)
            
            # Gestione URL unknown (senza lingua rilevata)
            df_live_unknown = df_live[df_live['language'] == 'unknown']
            df_staging_unknown = df_staging[df_staging['language'] == 'unknown']
            
            if len(df_live_unknown) > 0 and len(df_staging_unknown) > 0:
                st.write("**Processando URL senza lingua rilevata**")
                st.info(f"📊 Unknown: {len(df_live_unknown)} URL live → {len(df_staging_unknown)} URL staging")
                
                for col in matching_columns:
                    if col in df_live_unknown.columns and col in df_staging_unknown.columns:
                        matches = self.chunked_polyfuzz_matching(
                            list(df_live_unknown[col].dropna()), 
                            list(df_staging_unknown[col].dropna()), 
                            match_type=f"{col} (unknown)"
                        )
                        if not matches.empty:
                            all_lang_matches[col].append(matches)
            
            # Combina tutti i match per lingua
            for col in matching_columns:
                if all_lang_matches[col]:
                    match_results[col] = pd.concat(all_lang_matches[col], ignore_index=True)
                else:
                    match_results[col] = pd.DataFrame()
        
        else:
            # Matching standard senza distinzione di lingua
            for col in matching_columns:
                if col in df_live.columns and col in df_staging.columns:
                    match_results[col] = self.chunked_polyfuzz_matching(
                        list(df_live[col].dropna()), 
                        list(df_staging[col].dropna()), 
                        match_type=col
                    )
        
        # Resto del codice rimane identico...
        # [Continua con il processing dei match come nel codice originale]
        
        # Rinomina colonne per ogni match result
        df_pf_url = match_results.get('Address', pd.DataFrame())
        df_pf_title = match_results.get('Title 1', pd.DataFrame())
        df_pf_h1 = match_results.get('H1-1', pd.DataFrame())
        
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
        
        # Preparazione lookup tables
        lookup_tables = {
            'Title 1': df_staging[['Title 1', 'Address']].drop_duplicates('Title 1') if 'Title 1' in df_staging.columns else pd.DataFrame(),
            'H1-1': df_staging[['H1-1', 'Address']].drop_duplicates('H1-1') if 'H1-1' in df_staging.columns else pd.DataFrame()
        }
        
        for col in extra_columns:
            if col in df_staging.columns:
                lookup_tables[col] = df_staging[[col, 'Address']].drop_duplicates(col)
        
        # Merge base con URL
        if not df_pf_url.empty:
            df_final = pd.merge(df_live, df_pf_url, left_on="Address", right_on="From (Address)", how="left")
        else:
            df_final = df_live.copy()
            df_final['URL Similarity'] = 0
            df_final['From (Address)'] = df_final['Address']
            df_final['To Address'] = ''
        
        # Merge Title
        if not df_pf_title.empty and not lookup_tables['Title 1'].empty:
            df_pf_title_merge = pd.merge(df_pf_title, lookup_tables['Title 1'], left_on="To Title", right_on="Title 1", how="inner")
            df_pf_title_merge = df_pf_title_merge.rename(columns={'Address': 'Title_Match_URL'})
            df_final = pd.merge(df_final, df_pf_title_merge[['From (Title)', 'Title Similarity', 'To Title', 'Title_Match_URL']], 
                              left_on='Title 1', right_on='From (Title)', how='left')
        else:
            df_final['Title Similarity'] = 0
            df_final['From (Title)'] = df_final.get('Title 1', '')
            df_final['To Title'] = ''
            df_final['Title_Match_URL'] = ''
        
        # Merge H1
        if not df_pf_h1.empty and not lookup_tables['H1-1'].empty:
            df_pf_h1_merge = pd.merge(df_pf_h1, lookup_tables['H1-1'], left_on="To H1", right_on="H1-1", how="inner")
            df_pf_h1_merge = df_pf_h1_merge.rename(columns={'Address': 'H1_Match_URL'})
            df_final = pd.merge(df_final, df_pf_h1_merge[['From (H1)', 'H1 Similarity', 'To H1', 'H1_Match_URL']], 
                              left_on='H1-1', right_on='From (H1)', how='left')
        else:
            df_final['H1 Similarity'] = 0
            df_final['From (H1)'] = df_final.get('H1-1', '')
            df_final['To H1'] = ''
            df_final['H1_Match_URL'] = ''
        
        # Merge colonne extra
        extra_match_urls = {}
        for col in extra_columns:
            if col in extra_match_dfs and col in lookup_tables:
                extra_merge = pd.merge(extra_match_dfs[col], lookup_tables[col], left_on=f"To {col}", right_on=col, how="inner")
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
        
        # Rinomina colonne per output finale
        df_final = df_final.rename(columns={
            'Address': 'URL - Source',
            'To Address': 'URL - URL Match',
            'Title_Match_URL': 'URL - Title Match',
            'H1_Match_URL': 'URL - H1 Match'
        })
        
        for col in extra_columns:
            if f'{col}_Match_URL' in df_final.columns:
                df_final = df_final.rename(columns={f'{col}_Match_URL': f'URL - {col} Match'})
        
        # Calcolo best match
        similarity_cols = ["URL Similarity", "Title Similarity", "H1 Similarity"]
        for col in extra_columns:
            similarity_cols.append(f"{col} Similarity")
        
        for col in similarity_cols:
            if col not in df_final.columns:
                df_final[col] = 0
        
        df_final[similarity_cols] = df_final[similarity_cols].fillna(0)
        df_final['Best Match On'] = df_final[similarity_cols].idxmax(axis=1)
        
        # Calcolo Highest Match Similarity e Best Matching URL
        for col in similarity_cols:
            mask = df_final['Best Match On'] == col
            df_final.loc[mask, 'Highest Match Similarity'] = df_final.loc[mask, col]
            
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
                col_name = col.replace(' Similarity', '')
                df_final.loc[mask, 'Best Matching URL'] = df_final.loc[mask, f'URL - {col_name} Match']
                df_final.loc[mask, 'Highest Match Source Text'] = df_final.loc[mask, f'From ({col_name})']
                df_final.loc[mask, 'Highest Match Destination Text'] = df_final.loc[mask, f'To {col_name}']
        
        # VALIDAZIONE LINGUA: rimuovi match cross-lingua se language matching è attivo
        if use_language_matching and 'language' in df_final.columns:
            print("Validazione match per lingua...")
            st.info("🔍 Validazione match per lingua in corso...")
            
            # Crea un dizionario per lookup veloce delle lingue in staging
            staging_lang_lookup = df_staging.set_index('Address')['language'].to_dict()
            
            # Per ogni URL matchata, verifica la lingua
            def validate_language_match(row):
                try:
                    source_lang = row.get('language', 'unknown')
                    matched_url = row.get('Best Matching URL', '')
                    
                    if pd.isna(matched_url) or matched_url == '':
                        return matched_url
                    
                    # Trova la lingua dell'URL di destinazione usando il dizionario
                    target_lang = staging_lang_lookup.get(matched_url, 'unknown')
                    
                    # Se le lingue sono diverse, invalida il match
                    if source_lang != target_lang:
                        return ''  # Rimuovi il match
                    
                    return matched_url
                except Exception as e:
                    print(f"Errore validazione riga: {e}")
                    return ''
            
            # Applica validazione
            df_final['Best Matching URL'] = df_final.apply(validate_language_match, axis=1)
            
            # Azzera similarity per match invalidati
            invalid_mask = (df_final['Best Matching URL'] == '') | (df_final['Best Matching URL'].isna())
            df_final.loc[invalid_mask, 'Highest Match Similarity'] = 0
            df_final.loc[invalid_mask, 'Highest Match Source Text'] = ''
            df_final.loc[invalid_mask, 'Highest Match Destination Text'] = ''
            
            # Conta match invalidati
            invalidated_count = invalid_mask.sum()
            if invalidated_count > 0:
                st.warning(f"⚠️ {invalidated_count} match cross-lingua invalidati")
                print(f"Match cross-lingua invalidati: {invalidated_count}")
        
        df_final.drop_duplicates(subset="URL - Source", inplace=True)
        
        # SEPARA URL NON MATCHATE (per inserirle in non-redirectable se language matching attivo)
        if use_language_matching:
            print("Separazione URL non matchate...")
            st.info("📋 Separazione URL non matchate...")
            
            # URL con similarity troppo bassa o senza match vanno in non-redirectable
            no_match_mask = (
                (df_final['Highest Match Similarity'].fillna(0) < self.min_similarity_threshold) |
                (df_final['Best Matching URL'].isna()) |
                (df_final['Best Matching URL'] == '')
            )
            
            df_no_match = df_final[no_match_mask].copy()
            
            if len(df_no_match) > 0:
                # Prepara dataframe per non redirectable
                df_no_match_export = df_no_match[['URL - Source', 'Status Code']].copy()
                df_no_match_export = df_no_match_export.rename(columns={'URL - Source': 'Address'})
                
                # Aggiungi alla lista di non redirectable
                df_3xx_5xx = pd.concat([df_3xx_5xx, df_no_match_export], ignore_index=True)
                
                # Rimuovi duplicati
                df_3xx_5xx.drop_duplicates(subset='Address', inplace=True)
                
                st.info(f"📋 {len(df_no_match)} URL senza match valido nella stessa lingua → spostati in 'non redirectable'")
        
        # Calcolo SECONDO match migliore
        df_final['Lowest Match On'] = df_final[similarity_cols].idxmin(axis=1)
        df_final['Middle Match On'] = "URL Similarity Title Similarity H1 Similarity"
        for col in extra_columns:
            df_final['Middle Match On'] = df_final['Middle Match On'] + f" {col} Similarity"
        
        df_final['Middle Match On'] = df_final.apply(lambda x: x['Middle Match On'].replace(x['Best Match On'], ''), 1)
        df_final['Middle Match On'] = df_final.apply(lambda x: x['Middle Match On'].replace(x['Lowest Match On'], ''), 1)
        df_final['Middle Match On'] = df_final['Middle Match On'].str.strip()
        
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
                col_name = col.replace(' Similarity', '')
                df_final.loc[mask, 'Second Highest Match'] = df_final.loc[mask, f'URL - {col_name} Match']
                df_final.loc[mask, 'Second Match Source Text'] = df_final.loc[mask, f'From ({col_name})']
                df_final.loc[mask, 'Second Match Destination Text'] = df_final.loc[mask, f'To {col_name}']
        
        df_final.rename(columns={"Middle Match On": "Second Match On"}, inplace=True)
        df_final["Double Matched?"] = df_final['Best Matching URL'].str.lower() == df_final['Second Highest Match'].str.lower()
        
        # Rinomina Best Match On per output finale
        df_final['Best Match On'] = df_final['Best Match On'].str.replace("Title Similarity", "Page Title")
        df_final['Best Match On'] = df_final['Best Match On'].str.replace("H1 Similarity", "H1 Heading")
        df_final['Best Match On'] = df_final['Best Match On'].str.replace("URL Similarity", "URL")
        for col in extra_columns:
            df_final['Best Match On'] = df_final['Best Match On'].str.replace(f"{col} Similarity", col)
        
        df_final['Second Match On'] = df_final['Second Match On'].str.replace("Title Similarity", "Page Title")
        df_final['Second Match On'] = df_final['Second Match On'].str.replace("H1 Similarity", "H1 Heading")
        df_final['Second Match On'] = df_final['Second Match On'].str.replace("URL Similarity", "URL")
        for col in extra_columns:
            df_final['Second Match On'] = df_final['Second Match On'].str.replace(f"{col} Similarity", col)
        
        # Set delle colonne finali
        final_columns = [
            "URL - Source",
            "Status Code",
        ]
        
        # Aggiungi colonna lingua se presente
        if 'language' in df_final.columns:
            final_columns.append("language")
        
        final_columns.extend([
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
        ])
        
        existing_cols = [col for col in final_columns if col in df_final.columns]
        df_final = df_final[existing_cols]
        
        df_final.sort_values(["Highest Match Similarity", "Double Matched?"], ascending=[False, False], inplace=True)
        
        # Statistiche finali
        end_time = time.time()
        total_time = end_time - start_time
        
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
    - 🌍 **Supporto multi-lingua** con rilevamento automatico
    - 🎯 Colonne personalizzabili per matching aggiuntivo
    - 🤖 Enhancement con AI (OpenAI) - opzionale
    - 📊 Supporto per file di grandi dimensioni
    """)
    
    # Sidebar per configurazioni
    st.sidebar.header("⚙️ Configurazioni")
    
    chunk_size = st.sidebar.slider("Dimensione Chunk", 1000, 10000, 5000, 500)
    min_similarity = st.sidebar.slider("Soglia Similarità Minima", 0.1, 0.9, 0.3, 0.05)
    
    # NUOVA SEZIONE: Configurazione Multi-Lingua
    st.sidebar.subheader("🌍 Gestione Multi-Lingua")
    use_language_matching = st.sidebar.checkbox(
        "Abilita Matching per Lingua",
        value=True,
        help="Accoppia URL solo della stessa lingua. Se i CSV hanno una colonna 'language', verrà usata; altrimenti la lingua sarà rilevata automaticamente dall'URL."
    )
    
    default_lang_live = None
    default_lang_staging = None
    
    if use_language_matching:
        st.sidebar.info("""
        **Come funziona:**
        
        ✅ **Usa colonna esistente** se presente nel CSV
        
        🔍 **Rileva automaticamente** la lingua da:
        - Sottodomini: `it.example.com`
        - Sottocartelle: `/it/page`
        - Parametri: `?lang=it`
        - Country codes: `/en-us/`
        
        🎯 **Match separati per lingua**
        - `/en/about` ↔ `/en/about-us` ✅
        - `/it/chi-siamo` ↔ `/it/about` ❌
        """)
        
        # Configurazione lingua di default
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ Lingua Principale (senza pattern)**")
        st.sidebar.caption("Se il tuo sito non usa sottocartelle/sottodomini per la lingua principale:")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            use_default_live = st.checkbox(
                "Live ha lingua default",
                value=False,
                help="Es: sito.com/pagina è in italiano, mentre sito.com/en/page è in inglese"
            )
            
            if use_default_live:
                default_lang_live = st.text_input(
                    "Codice Live",
                    value="it",
                    max_chars=2,
                    help="Codice ISO (es: it, en, fr, de)"
                ).lower().strip()
                
                if len(default_lang_live) != 2:
                    st.warning("⚠️ Usa 2 lettere (es: it)")
                    default_lang_live = None
        
        with col2:
            use_default_staging = st.checkbox(
                "Staging ha lingua default",
                value=False,
                help="Es: sito.com/page è in inglese, mentre sito.com/it/pagina è in italiano"
            )
            
            if use_default_staging:
                default_lang_staging = st.text_input(
                    "Codice Staging",
                    value="en",
                    max_chars=2,
                    help="Codice ISO (es: it, en, fr, de)"
                ).lower().strip()
                
                if len(default_lang_staging) != 2:
                    st.warning("⚠️ Usa 2 lettere (es: en)")
                    default_lang_staging = None
        
        # Esempi pratici
        if use_default_live or use_default_staging:
            st.sidebar.markdown("**💡 Esempi:**")
            
            if use_default_live and default_lang_live:
                st.sidebar.success(f"✅ Live: `sito.com/page` → `{default_lang_live}`")
                st.sidebar.info(f"ℹ️ Live: `sito.com/en/page` → `en` (rilevato)")
            
            if use_default_staging and default_lang_staging:
                st.sidebar.success(f"✅ Staging: `sito.com/page` → `{default_lang_staging}`")
                st.sidebar.info(f"ℹ️ Staging: `sito.com/it/page` → `it` (rilevato)")
    
    # Configurazione OpenAI
    st.sidebar.subheader("🤖 AI Enhancement")
    use_ai = st.sidebar.checkbox("Abilita AI Enhancement")
    
    if use_ai:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.warning("Inserisci la tua API Key OpenAI")
    else:
        openai_api_key = ""
    
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
            
            # Selezione colonne extra
            st.header("🎯 Colonne Aggiuntive per Matching")
            
            base_columns = ['Address', 'Status Code', 'Title 1', 'H1-1', 'language']
            available_live_cols = [col for col in df_live.columns if col not in base_columns]
            available_staging_cols = [col for col in df_staging.columns if col not in base_columns]
            common_extra_cols = list(set(available_live_cols) & set(available_staging_cols))
            
            if common_extra_cols:
                extra_columns = st.multiselect(
                    "Seleziona colonne aggiuntive:",
                    options=sorted(common_extra_cols),
                    default=[]
                )
            else:
                extra_columns = []
                st.info("ℹ️ Nessuna colonna aggiuntiva comune trovata.")
            
            # Pulsante elaborazione
            if st.button("🚀 Avvia Elaborazione", type="primary"):
                with st.spinner("Elaborazione in corso..."):
                    df_final, df_non_redirectable = mapper.process_migration_mapping(
                        df_live, df_staging, extra_columns, use_ai, use_language_matching,
                        default_lang_live, default_lang_staging
                    )
                
                # Risultati
                st.header("📊 Risultati")
                
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
                
                # Statistiche per lingua
                if 'language' in df_final.columns:
                    st.subheader("🌍 Statistiche per Lingua")
                    lang_stats = df_final.groupby('language').agg({
                        'URL - Source': 'count',
                        'Highest Match Similarity': lambda x: (x > min_similarity).sum()
                    }).reset_index()
                    lang_stats.columns = ['Lingua', 'Totale URL', 'URL Matchati']
                    lang_stats['Match Rate %'] = (lang_stats['URL Matchati'] / lang_stats['Totale URL'] * 100).round(1)
                    st.dataframe(lang_stats, use_container_width=True)
                
                # Preview risultati
                st.subheader("👀 Preview Risultati")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_sim_filter = st.slider("Similarità minima", 0.0, 1.0, min_similarity)
                with col2:
                    max_rows_preview = st.selectbox("Righe", [10, 25, 50, 100], index=1)
                
                filtered_df = df_final[df_final['Highest Match Similarity'] >= min_sim_filter].head(max_rows_preview)
                
                if len(filtered_df) > 0:
                    st.dataframe(filtered_df, use_container_width=True)
                
                # Download
                st.header("💾 Download Risultati")
                
                col1, col2 = st.columns(2)
                with col1:
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
            st.error(f"Errore: {str(e)}")
    
    else:
        st.info("👆 Carica entrambi i file CSV per iniziare.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>SEO URL Matcher v2.0 - Multi-Language Support 🌍 - by Daniele Pisciottano & Claude</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
