import streamlit as st
import pandas as pd
import numpy as np
from polyfuzz import PolyFuzz
import time
import io
import gc
from typing import List, Dict, Tuple, Optional, Union
import openai
import json
import re
from urllib.parse import urlparse
import openpyxl
import psutil
import os
import math

# Configure page
st.set_page_config(
    page_title="URL Migration Mapping Tool",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class URLMigrationMapper:
    def __init__(self):
        self.df_live = None
        self.df_staging = None
        self.openai_client = None
        self.required_columns = {
            'pre': ['Address', 'Status Code'],
            'post': ['Address', 'Status Code', 'Indexability']
        }
        self.optional_columns = ['Title 1', 'H1-1']
        self.batch_size = 5000
        self.max_memory_usage = 85
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def calculate_optimal_batch_size(self, total_rows: int, available_memory_gb: float = None) -> int:
        """Calculate optimal batch size based on data size and available memory"""
        if available_memory_gb is None:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        memory_per_row_mb = 0.001
        max_rows_in_memory = int((available_memory_gb * 1024 * 0.5) / memory_per_row_mb)
        
        if total_rows <= max_rows_in_memory:
            return total_rows
        
        num_batches = math.ceil(total_rows / max_rows_in_memory)
        optimal_batch_size = math.ceil(total_rows / num_batches)
        optimal_batch_size = max(1000, min(optimal_batch_size, 10000))
        
        return optimal_batch_size
    
    def setup_openai(self, api_key: str) -> bool:
        """Setup OpenAI client"""
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.openai_client.models.list()
            return True
        except Exception as e:
            st.error(f"Errore nella configurazione OpenAI: {str(e)}")
            return False
    
    def load_file_chunked(self, uploaded_file, chunk_size: int = 10000) -> pd.DataFrame:
        """Load large files in chunks to avoid memory issues"""
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 Caricamento file: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            if uploaded_file.name.endswith('.csv'):
                # For CSV files with flexible column handling
                csv_params = {
                    'on_bad_lines': 'warn',  # Handle bad lines gracefully
                    'sep': ',',              # Default separator
                    'quotechar': '"',        # Handle quoted fields
                    'skipinitialspace': True, # Skip spaces after delimiter
                    'low_memory': False,     # Read entire file for consistency
                    'encoding': 'utf-8'      # Default encoding
                }
                
                # Try different encodings if UTF-8 fails
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                if file_size_mb > 100:  # Large file handling
                    st.info("📊 File grande rilevato - utilizzo caricamento chunk ottimizzato")
                    
                    chunks = []
                    df_loaded = False
                    
                    for encoding in encodings_to_try:
                        try:
                            csv_params['encoding'] = encoding
                            chunk_iterator = pd.read_csv(uploaded_file, 
                                                       chunksize=chunk_size, 
                                                       **csv_params)
                            
                            total_chunks = math.ceil(file_size_mb / 10)
                            progress_bar = st.progress(0)
                            
                            for i, chunk in enumerate(chunk_iterator):
                                chunks.append(chunk)
                                progress_bar.progress(min((i + 1) / total_chunks, 1.0))
                                
                                if self.get_memory_usage() > self.max_memory_usage:
                                    st.warning("⚠️ Utilizzo memoria alto, consolidando chunks...")
                                    break
                            
                            df = pd.concat(chunks, ignore_index=True, sort=False)
                            del chunks
                            gc.collect()
                            df_loaded = True
                            st.success(f"✅ File caricato con encoding: {encoding}")
                            break
                            
                        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                            if chunks:
                                del chunks
                                chunks = []
                            st.warning(f"⚠️ Tentativo con encoding {encoding} fallito: {str(e)}")
                            continue
                    
                    if not df_loaded:
                        raise ValueError("Impossibile caricare il file con nessuno degli encoding testati")
                        
                else:  # Smaller files
                    df_loaded = False
                    
                    for encoding in encodings_to_try:
                        try:
                            csv_params['encoding'] = encoding
                            df = pd.read_csv(uploaded_file, **csv_params)
                            df_loaded = True
                            st.success(f"✅ File caricato con encoding: {encoding}")
                            break
                            
                        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                            st.warning(f"⚠️ Tentativo con encoding {encoding} fallito: {str(e)}")
                            continue
                    
                    if not df_loaded:
                        raise ValueError("Impossibile caricare il file con nessuno degli encoding testati")
                    
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                # Excel files
                if file_size_mb > 50:
                    st.warning("⚠️ File Excel grande rilevato. Il caricamento potrebbe richiedere tempo...")
                
                # Excel files are generally more consistent, but we can still handle errors
                try:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception as e:
                    st.warning(f"⚠️ Tentativo con openpyxl fallito: {str(e)}")
                    try:
                        df = pd.read_excel(uploaded_file, engine='xlrd')
                    except Exception as e2:
                        raise ValueError(f"Impossibile caricare il file Excel: {str(e2)}")
            else:
                raise ValueError("Formato file non supportato")
            
            # Post-processing cleanup
            st.info(f"🔍 Analisi file caricato...")
            
            # Handle inconsistent columns
            original_columns = len(df.columns)
            
            # Remove completely empty columns
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                df.drop(columns=empty_cols, inplace=True)
                st.info(f"🗑️ Rimosse {len(empty_cols)} colonne completamente vuote")
            
            # Remove columns with no name (often caused by parsing issues)
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                # Only remove if they're mostly empty
                cols_to_remove = []
                for col in unnamed_cols:
                    if df[col].notna().sum() / len(df) < 0.01:  # Less than 1% non-null
                        cols_to_remove.append(col)
                
                if cols_to_remove:
                    df.drop(columns=cols_to_remove, inplace=True)
                    st.info(f"🗑️ Rimosse {len(cols_to_remove)} colonne 'Unnamed' vuote")
            
            # Clean column names
            df.columns = df.columns.astype(str)
            df.columns = [col.strip() for col in df.columns]  # Remove whitespace
            
            # Report final stats
            final_columns = len(df.columns)
            if final_columns != original_columns:
                st.info(f"📊 Colonne ottimizzate: {original_columns} → {final_columns}")
            
            st.success(f"✅ File caricato: {len(df):,} righe, {len(df.columns)} colonne")
            
            # Show a sample of columns for verification
            if len(df.columns) > 10:
                st.info(f"🔍 Prime 10 colonne: {list(df.columns[:10])}")
            else:
                st.info(f"🔍 Colonne trovate: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            st.error(f"Errore nel caricamento del file {uploaded_file.name}: {str(e)}")
            
            # Provide helpful suggestions
            st.markdown("""
            **💡 Suggerimenti per risolvere problemi di caricamento:**
            
            - **File CSV corrotto**: Apri in Excel e salva di nuovo come CSV UTF-8
            - **Caratteri speciali**: Verifica encoding del file (UTF-8 raccomandato)
            - **Colonne inconsistenti**: Verifica che tutte le righe abbiano lo stesso numero di colonne
            - **File troppo grande**: Prova a dividere il file in parti più piccole
            - **Formato non supportato**: Converti in CSV o Excel (.xlsx)
            """)
            
            return None
    
    def load_file(self, uploaded_file, file_type: str) -> pd.DataFrame:
        """Load CSV or XLSX file with automatic chunking for large files"""
        return self.load_file_chunked(uploaded_file)
    
    def validate_columns(self, df: pd.DataFrame, file_type: str, selected_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate required columns exist"""
        required = self.required_columns[file_type]
        all_required = required + selected_columns
        missing = [col for col in all_required if col not in df.columns]
        
        if missing:
            return False, missing
        return True, []
    
    def clean_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate column names and handle conflicts"""
        if len(df.columns) != len(set(df.columns)):
            st.warning("⚠️ Colonne duplicate rilevate, pulizia in corso...")
            
            # Create a mapping of old to new column names
            new_columns = []
            seen_columns = {}
            
            for col in df.columns:
                if col in seen_columns:
                    seen_columns[col] += 1
                    new_col = f"{col}_duplicate_{seen_columns[col]}"
                else:
                    seen_columns[col] = 0
                    new_col = col
                new_columns.append(new_col)
            
            # Assign new column names
            df.columns = new_columns
            
            # Remove obviously duplicate Address columns (keep only the main one)
            duplicate_address_cols = [col for col in df.columns if col.startswith('Address_duplicate_')]
            if duplicate_address_cols:
                df.drop(columns=duplicate_address_cols, inplace=True, errors='ignore')
                st.info(f"🗑️ Rimosse colonne duplicate: {duplicate_address_cols}")
            
            st.info(f"✅ Colonne pulite: {len(df.columns)} colonne rimanenti")
        
        return df

    def ensure_unique_columns(self, df: pd.DataFrame, operation_name: str = "") -> pd.DataFrame:
        """Ensure all columns have unique names before operations"""
        if len(df.columns) != len(set(df.columns)):
            st.warning(f"⚠️ Colonne duplicate trovate prima di {operation_name}. Correzione in corso...")
            
            # Make columns unique by adding suffix
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                mask = cols == dup
                cols[mask] = [f"{dup}_v{i}" if i != 0 else dup for i in range(mask.sum())]
            
            df.columns = cols
            
            st.info(f"✅ Colonne rese uniche per {operation_name}")
        
        return df
    
    def preprocess_data_batch(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                             matching_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess the data with memory optimization"""
        st.write("🔄 Preprocessing dei dati...")
        
        df_live = df_live.copy()
        df_staging = df_staging.copy()
        
        # Clean duplicate columns FIRST - this is critical
        df_live = self.ensure_unique_columns(df_live, "preprocessing df_live")
        df_staging = self.ensure_unique_columns(df_staging, "preprocessing df_staging")
        
        # Optimize data types to save memory
        for col in df_live.columns:
            if df_live[col].dtype == 'object':
                df_live[col] = df_live[col].astype('string')
        
        for col in df_staging.columns:
            if df_staging[col].dtype == 'object':
                df_staging[col] = df_staging[col].astype('string')
        
        # Ensure required columns exist
        for col in matching_columns:
            if col not in df_live.columns:
                df_live[col] = ""
            if col not in df_staging.columns:
                df_staging[col] = ""
        
        # Verify Address column exists and is unique
        if 'Address' not in df_live.columns:
            st.error("❌ Colonna 'Address' mancante nel file PRE")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        if 'Address' not in df_staging.columns:
            st.error("❌ Colonna 'Address' mancante nel file POST")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Drop duplicates with progress tracking
        initial_live_count = len(df_live)
        initial_staging_count = len(df_staging)
        
        df_live.drop_duplicates(subset="Address", inplace=True)
        df_staging.drop_duplicates(subset="Address", inplace=True)
        
        st.write(f"📊 Duplicati rimossi: Live {initial_live_count - len(df_live)}, Staging {initial_staging_count - len(df_staging)}")
        
        # Extract non-redirectable URLs (3xx & 5xx)
        df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)].copy()
        df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)].copy()
        df_non_redirectable = pd.concat([df_3xx, df_5xx]) if not df_3xx.empty or not df_5xx.empty else pd.DataFrame()
        
        # Keep 2xx and 4xx status codes for redirecting
        df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)].copy()
        df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)].copy()
        df_live_clean = pd.concat([df_live_200, df_live_400]) if not df_live_200.empty or not df_live_400.empty else pd.DataFrame()
        
        # Clean up intermediate dataframes
        del df_3xx, df_5xx, df_live_200, df_live_400
        gc.collect()
        
        # Handle NaN values - populate with URL for 404s
        for col in matching_columns:
            if col in df_live_clean.columns:
                df_live_clean[col] = df_live_clean[col].fillna(df_live_clean['Address'])
            if col in df_staging.columns:
                df_staging[col] = df_staging[col].fillna(df_staging['Address'])
        
        # Final check for unique columns
        df_live_clean = self.ensure_unique_columns(df_live_clean, "final df_live_clean")
        df_staging = self.ensure_unique_columns(df_staging, "final df_staging")
        
        st.write(f"✅ Preprocessing completato: {len(df_live_clean):,} URL processabili, {len(df_non_redirectable):,} non-redirectable")
        
        return df_live_clean, df_staging, df_non_redirectable
    
    def perform_polyfuzz_matching_batch(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                                      matching_columns: List[str]) -> Dict[str, pd.DataFrame]:
        """Perform PolyFuzz matching in batches for large datasets"""
        matches = {}
        
        total_rows = max(len(df_live), len(df_staging))
        batch_size = self.calculate_optimal_batch_size(total_rows)
        
        st.write(f"🔍 Avvio matching con batch size: {batch_size:,}")
        st.write(f"💾 Memoria disponibile: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        main_progress = st.progress(0)
        status_text = st.empty()
        
        total_operations = len(matching_columns)
        
        for i, col in enumerate(matching_columns):
            status_text.text(f"🔍 Matching colonna: {col} ({i+1}/{total_operations})")
            
            if len(df_live) > batch_size or len(df_staging) > batch_size:
                all_matches = []
                
                live_batches = [df_live[j:j+batch_size] for j in range(0, len(df_live), batch_size)]
                staging_data = list(df_staging[col].astype(str))
                
                batch_progress = st.progress(0)
                
                for batch_idx, live_batch in enumerate(live_batches):
                    live_data = list(live_batch[col].astype(str))
                    
                    model = PolyFuzz("TF-IDF").match(live_data, staging_data)
                    batch_matches = model.get_matches()
                    
                    all_matches.append(batch_matches)
                    
                    batch_progress.progress((batch_idx + 1) / len(live_batches))
                    
                    del model, live_data
                    gc.collect()
                    
                    if self.get_memory_usage() > self.max_memory_usage:
                        st.warning("⚠️ Memoria quasi esaurita, consolidando batch...")
                        break
                
                df_match = pd.concat(all_matches, ignore_index=True)
                del all_matches
                
            else:
                live_data = list(df_live[col].astype(str))
                staging_data = list(df_staging[col].astype(str))
                
                model = PolyFuzz("TF-IDF").match(live_data, staging_data)
                df_match = model.get_matches()
                
                del model, live_data, staging_data
            
            similarity_col = f"{col} Similarity"
            from_col = f"From ({col})"
            to_col = f"To {col}"
            
            df_match.rename(columns={
                "Similarity": similarity_col,
                "From": from_col,
                "To": to_col
            }, inplace=True)
            
            matches[col] = df_match
            
            main_progress.progress((i + 1) / total_operations)
            
            gc.collect()
            
            memory_usage = self.get_memory_usage()
            if memory_usage > 80:
                st.warning(f"⚠️ Utilizzo memoria: {memory_usage:.1f}%")
        
        status_text.text("✅ Matching completato!")
        return matches
    
    def ai_enhance_matching(self, source_url: str, candidate_urls: List[str], 
                           source_content: Dict, candidate_contents: List[Dict]) -> Optional[str]:
        """Use OpenAI to enhance matching when similarity is low"""
        if not self.openai_client:
            return None
        
        try:
            prompt = f"""
            Sei un esperto SEO che deve identificare il miglior match per una URL di migrazione.
            
            URL di origine: {source_url}
            Contenuto origine: {source_content}
            
            Candidate URLs e contenuti:
            """
            
            for i, (url, content) in enumerate(zip(candidate_urls, candidate_contents)):
                prompt += f"\n{i+1}. URL: {url}\n   Contenuto: {content}\n"
            
            prompt += """
            
            Considera questi fattori:
            1. Similarità semantica del contenuto
            2. Struttura dell'URL
            3. Gerarchia del sito
            4. Intento della pagina
            
            Rispondi SOLO con il numero (1, 2, 3, etc.) della migliore opzione, o 0 se nessuna è appropriata.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            match = re.search(r'\d+', result)
            if match:
                choice = int(match.group())
                if 1 <= choice <= len(candidate_urls):
                    return candidate_urls[choice - 1]
            
            return None
            
        except Exception as e:
            st.warning(f"Errore AI enhancement: {str(e)}")
            return None
    
    def merge_matches_batch(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                           matches: Dict[str, pd.DataFrame], matching_columns: List[str]) -> pd.DataFrame:
        """Merge all matches into final dataframe with memory optimization"""
        
        # Ensure unique columns before starting
        df_live = self.ensure_unique_columns(df_live, "merge_matches_batch - df_live")
        df_staging = self.ensure_unique_columns(df_staging, "merge_matches_batch - df_staging")
        
        # Start with a base dataframe - only select columns that exist
        base_columns = ['Address', 'Status Code']
        available_columns = [col for col in base_columns if col in df_live.columns]
        available_matching_columns = [col for col in matching_columns if col in df_live.columns]
        
        df_final = df_live[available_columns + available_matching_columns].copy()
        
        # Ensure df_final has unique columns
        df_final = self.ensure_unique_columns(df_final, "df_final initial")
        
        # Add Address matching if available
        if 'Address' in matches and 'Address' in df_final.columns:
            try:
                # Double check that Address column is unique in df_final
                if df_final.columns.duplicated().any():
                    df_final = self.ensure_unique_columns(df_final, "before Address merge")
                
                # Perform the merge with explicit error handling
                df_final = pd.merge(df_final, matches['Address'], 
                                  left_on="Address", right_on="From (Address)", 
                                  how="left", suffixes=('', '_url_match'))
                
                # Rename the URL column
                if "To Address" in df_final.columns:
                    df_final.rename(columns={"To Address": "URL - URL Match"}, inplace=True)
                
            except ValueError as e:
                st.error(f"Errore nel merge Address: {str(e)}")
                st.write("Colonne df_final:", list(df_final.columns))
                st.write("Colonne matches['Address']:", list(matches['Address'].columns))
                # Skip Address matching if there's an error
        
        # Clean any duplicate columns that might have been created
        df_final = self.ensure_unique_columns(df_final, "after Address merge")
        
        # Merge other columns in batches to manage memory
        for col in matching_columns:
            if col == 'Address':
                continue
                
            if col in matches and col in df_staging.columns:
                try:
                    # Create mapping dataframe
                    df_col_map = df_staging[[col, 'Address']].drop_duplicates(col)
                    
                    # Rename Address column to avoid conflicts
                    df_col_map.rename(columns={'Address': f'TargetURL_{col}'}, inplace=True)
                    
                    # Merge with matches
                    df_match_with_url = pd.merge(
                        matches[col], df_col_map, 
                        left_on=f"To {col}", right_on=col, 
                        how="left"
                    )
                    
                    # Clean duplicates
                    df_match_with_url = df_match_with_url.drop_duplicates(f"From ({col})")
                    
                    # Ensure unique columns before merge
                    df_final = self.ensure_unique_columns(df_final, f"before {col} merge")
                    df_match_with_url = self.ensure_unique_columns(df_match_with_url, f"{col} match data")
                    
                    # Merge with final dataframe
                    df_final = df_final.merge(
                        df_match_with_url, 
                        how='left', left_on=col, right_on=f"From ({col})",
                        suffixes=('', f'_{col.lower().replace(" ", "_").replace("-", "_")}')
                    )
                    
                    # Rename URL column correctly
                    if f'TargetURL_{col}' in df_final.columns:
                        df_final.rename(columns={f'TargetURL_{col}': f"URL - {col} Match"}, inplace=True)
                    
                    # Clean duplicate columns after each merge
                    df_final = self.ensure_unique_columns(df_final, f"after {col} merge")
                    
                    # Memory cleanup
                    del df_col_map, df_match_with_url
                    gc.collect()
                    
                except Exception as e:
                    st.warning(f"⚠️ Errore nel merge della colonna {col}: {str(e)}. Saltata.")
                    continue
        
        return df_final
    
    def calculate_best_matches_batch(self, df_final: pd.DataFrame, matching_columns: List[str]) -> pd.DataFrame:
        """Calculate best, second best, and lowest matches with memory optimization"""
        
        similarity_cols = [f"{col} Similarity" for col in matching_columns if f"{col} Similarity" in df_final.columns]
        
        if not similarity_cols:
            return df_final
        
        chunk_size = 10000
        if len(df_final) > chunk_size:
            st.write(f"📊 Calcolo match in chunks di {chunk_size:,} righe...")
            
            chunks_processed = []
            total_chunks = math.ceil(len(df_final) / chunk_size)
            
            for i in range(0, len(df_final), chunk_size):
                chunk = df_final.iloc[i:i+chunk_size].copy()
                chunk = self._calculate_matches_for_chunk(chunk, similarity_cols, matching_columns)
                chunks_processed.append(chunk)
                
                if (i // chunk_size + 1) % 5 == 0:
                    st.write(f"   Processati {i // chunk_size + 1}/{total_chunks} chunks...")
            
            df_final = pd.concat(chunks_processed, ignore_index=True)
            del chunks_processed
            gc.collect()
        else:
            df_final = self._calculate_matches_for_chunk(df_final, similarity_cols, matching_columns)
        
        return df_final
    
    def _calculate_matches_for_chunk(self, chunk: pd.DataFrame, similarity_cols: List[str], 
                                   matching_columns: List[str]) -> pd.DataFrame:
        """Calculate matches for a single chunk"""
        
        chunk['Best Match On'] = chunk[similarity_cols].idxmax(axis=1)
        chunk['Highest Match Similarity'] = chunk[similarity_cols].max(axis=1)
        
        chunk['Lowest Match On'] = chunk[similarity_cols].idxmin(axis=1)
        chunk['Lowest Match Similarity'] = chunk[similarity_cols].min(axis=1)
        
        def get_second_best(row):
            values = [(col, row[col]) for col in similarity_cols if pd.notna(row[col])]
            if len(values) < 2:
                return None, None
            values.sort(key=lambda x: x[1], reverse=True)
            return values[1] if len(values) > 1 else (None, None)
        
        second_matches = chunk.apply(get_second_best, axis=1, result_type='expand')
        chunk['Second Match On'] = second_matches[0]
        chunk['Second Highest Match Similarity'] = second_matches[1]
        
        for col in matching_columns:
            similarity_col = f"{col} Similarity"
            url_col = f"URL - {col} Match"
            from_col = f"From ({col})"
            to_col = f"To {col}"
            
            if similarity_col in chunk.columns:
                mask = chunk['Best Match On'] == similarity_col
                if url_col in chunk.columns:
                    chunk.loc[mask, 'Best Matching URL'] = chunk.loc[mask, url_col]
                if from_col in chunk.columns:
                    chunk.loc[mask, 'Highest Match Source Text'] = chunk.loc[mask, from_col]
                if to_col in chunk.columns:
                    chunk.loc[mask, 'Highest Match Destination Text'] = chunk.loc[mask, to_col]
                
                mask = chunk['Second Match On'] == similarity_col
                if url_col in chunk.columns:
                    chunk.loc[mask, 'Second Highest Match'] = chunk.loc[mask, url_col]
                if from_col in chunk.columns:
                    chunk.loc[mask, 'Second Match Source Text'] = chunk.loc[mask, from_col]
                if to_col in chunk.columns:
                    chunk.loc[mask, 'Second Match Destination Text'] = chunk.loc[mask, to_col]
        
        if 'Best Matching URL' in chunk.columns and 'Second Highest Match' in chunk.columns:
            chunk["Double Matched?"] = (
                chunk['Best Matching URL'].str.lower() == 
                chunk['Second Highest Match'].str.lower()
            )
        
        chunk['Best Match On'] = chunk['Best Match On'].str.replace(' Similarity', '')
        if 'Second Match On' in chunk.columns:
            chunk['Second Match On'] = chunk['Second Match On'].str.replace(' Similarity', '')
        chunk['Lowest Match On'] = chunk['Lowest Match On'].str.replace(' Similarity', '')
        
        return chunk
    
    def apply_ai_enhancement_batch(self, df_final: pd.DataFrame, max_calls: int) -> pd.DataFrame:
        """Apply AI enhancement in controlled batches"""
        
        low_similarity_mask = df_final['Highest Match Similarity'] < 0.7
        low_similarity_indices = df_final[low_similarity_mask].index[:max_calls]
        
        progress_bar = st.progress(0)
        enhanced_count = 0
        
        batch_size = 5
        for i in range(0, len(low_similarity_indices), batch_size):
            batch_indices = low_similarity_indices[i:i+batch_size]
            
            for idx in batch_indices:
                try:
                    source_url = df_final.loc[idx, 'Address']
                    
                    candidates = []
                    url_cols = [col for col in df_final.columns if col.startswith('URL -') and 'Match' in col]
                    
                    for url_col in url_cols:
                        if pd.notna(df_final.loc[idx, url_col]):
                            candidates.append(df_final.loc[idx, url_col])
                    
                    if candidates:
                        ai_match = self.ai_enhance_matching(
                            source_url, candidates[:3], {}, [{}]*len(candidates[:3])
                        )
                        
                        if ai_match:
                            df_final.loc[idx, 'Best Matching URL'] = ai_match
                            df_final.loc[idx, 'Best Match On'] = 'AI Enhanced'
                            df_final.loc[idx, 'Highest Match Similarity'] = 0.95
                            enhanced_count += 1
                
                except Exception as e:
                    continue
            
            progress_bar.progress(min((i + batch_size) / len(low_similarity_indices), 1.0))
            time.sleep(0.1)
        
        st.write(f"🤖 AI enhancement applicato a {enhanced_count} URL")
        return df_final
    
    def finalize_results_batch(self, df_final: pd.DataFrame) -> pd.DataFrame:
        """Finalize results with memory optimization"""
        
        initial_count = len(df_final)
        df_final.drop_duplicates(subset="Address", inplace=True)
        duplicates_removed = initial_count - len(df_final)
        
        if duplicates_removed > 0:
            st.write(f"🗑️ Rimossi {duplicates_removed} duplicati finali")
        
        base_columns = [
            "Address", "Status Code", "Best Matching URL", "Best Match On", 
            "Highest Match Similarity"
        ]
        
        optional_columns = [
            "Highest Match Source Text", "Highest Match Destination Text", 
            "Second Highest Match", "Second Match On", "Second Highest Match Similarity", 
            "Second Match Source Text", "Second Match Destination Text", "Double Matched?"
        ]
        
        final_columns = [col for col in base_columns if col in df_final.columns]
        final_columns.extend([col for col in optional_columns if col in df_final.columns])
        
        df_final = df_final[final_columns]
        
        df_final.rename(columns={"Address": "URL - Source"}, inplace=True)
        
        if 'Highest Match Similarity' in df_final.columns:
            df_final = df_final.sort_values(["Highest Match Similarity"], ascending=[False], na_position='last')
        
        return df_final
    
    def process_migration_batch(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                              matching_columns: List[str], use_ai: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main processing function with batch optimization"""
        
        start_time = time.time()
        
        initial_memory = self.get_memory_usage()
        st.write(f"💾 Memoria iniziale: {initial_memory:.1f}%")
        
        total_rows = len(df_live) + len(df_staging)
        st.write(f"📊 Righe totali da processare: {total_rows:,}")
        
        if total_rows > 50000:
            st.info("🔄 Dataset grande rilevato. Utilizzo elaborazione ottimizzata a batch...")
            
        df_live_clean, df_staging_clean, df_non_redirectable = self.preprocess_data_batch(
            df_live, df_staging, matching_columns
        )
        
        matches = self.perform_polyfuzz_matching_batch(df_live_clean, df_staging_clean, matching_columns)
        
        st.write("🔗 Merge dei risultati...")
        df_final = self.merge_matches_batch(df_live_clean, df_staging_clean, matches, matching_columns)
        
        st.write("🎯 Calcolo dei migliori match...")
        df_final = self.calculate_best_matches_batch(df_final, matching_columns)
        
        if use_ai and self.openai_client and 'Highest Match Similarity' in df_final.columns:
            total_low_similarity = (df_final['Highest Match Similarity'] < 0.7).sum()
            
            if total_low_similarity > 0:
                max_ai_calls = min(100, total_low_similarity) if total_rows > 10000 else total_low_similarity
                
                st.write(f"🤖 Applicazione AI a {max_ai_calls} match con bassa similarità (di {total_low_similarity})...")
                
                df_final = self.apply_ai_enhancement_batch(df_final, max_ai_calls)
        
        df_final = self.finalize_results_batch(df_final)
        
        del df_live_clean, df_staging_clean, matches
        gc.collect()
        
        processing_time = time.time() - start_time
        final_memory = self.get_memory_usage()
        
        st.success(f"✅ Processamento completato in {processing_time:.2f} secondi!")
        st.write(f"💾 Memoria finale: {final_memory:.1f}% (Δ: {final_memory - initial_memory:+.1f}%)")
        
        return df_final, df_non_redirectable
    
    def process_migration(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                         matching_columns: List[str], use_ai: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main processing function - delegates to batch version for large datasets"""
        
        return self.process_migration_batch(df_live, df_staging, matching_columns, use_ai)

# Streamlit App
def main():
    st.title("🔗 URL Migration Mapping Tool")
    st.markdown("**Strumento avanzato per il mapping automatico di URL durante le migrazioni di siti web**")
    st.markdown("*Ottimizzato per file di grandi dimensioni (fino a 1GB)*")
    
    mapper = URLMigrationMapper()
    
    st.sidebar.header("⚙️ Configurazione")
    
    st.sidebar.subheader("💾 Info Sistema")
    available_memory = psutil.virtual_memory().available / (1024**3)
    total_memory = psutil.virtual_memory().total / (1024**3)
    st.sidebar.write(f"Memoria disponibile: {available_memory:.1f} GB / {total_memory:.1f} GB")
    st.sidebar.write(f"Utilizzo corrente: {psutil.virtual_memory().percent:.1f}%")
    
    st.sidebar.subheader("🚀 Impostazioni Performance")
    
    auto_batch = st.sidebar.checkbox("Auto-batch size", value=True, 
                                    help="Calcola automaticamente la dimensione ottimale del batch")
    
    if not auto_batch:
        custom_batch_size = st.sidebar.slider("Batch size personalizzato", 
                                             min_value=1000, max_value=20000, 
                                             value=5000, step=1000,
                                             help="Dimensione del batch per l'elaborazione")
        mapper.batch_size = custom_batch_size
    
    memory_limit = st.sidebar.slider("Limite memoria (%)", 
                                    min_value=70, max_value=95, 
                                    value=85, step=5,
                                    help="Limite massimo di utilizzo memoria prima dell'ottimizzazione")
    mapper.max_memory_usage = memory_limit
    
    st.sidebar.subheader("🤖 Configurazione OpenAI (Opzionale)")
    use_ai = st.sidebar.checkbox("Abilita miglioramento AI", 
                                help="Migliora i match con bassa similarità usando OpenAI GPT")
    
    api_key = None
    if use_ai:
        api_key = st.sidebar.text_input("API Key OpenAI", type="password", 
                                       help="Inserisci la tua API key OpenAI per abilitare il miglioramento AI")
        
        ai_limit = st.sidebar.number_input("Limite chiamate AI", 
                                          min_value=10, max_value=1000, 
                                          value=100, step=10,
                                          help="Numero massimo di chiamate AI per dataset grandi")
        
        if api_key:
            if mapper.setup_openai(api_key):
                st.sidebar.success("✅ OpenAI configurato correttamente")
            else:
                use_ai = False
    
    st.header("📁 Caricamento File")
    st.markdown("*Supporta file fino a 1GB (CSV, XLSX, XLS)*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File PRE-migrazione (Live)")
        pre_file = st.file_uploader(
            "Carica file pre-migrazione", 
            type=['csv', 'xlsx', 'xls'],
            key="pre_file",
            help="File contenente le URL del sito live prima della migrazione (max 1GB)"
        )
    
    with col2:
        st.subheader("File POST-migrazione (Staging)")
        post_file = st.file_uploader(
            "Carica file post-migrazione", 
            type=['csv', 'xlsx', 'xls'],
            key="post_file",
            help="File contenente le URL del sito staging dopo la migrazione (max 1GB)"
        )
    
    if pre_file and post_file:
        pre_size_mb = pre_file.size / (1024 * 1024)
        post_size_mb = post_file.size / (1024 * 1024)
        
        col1, col2 = st.columns(2)
        with col1:
            if pre_size_mb > 100:
                st.warning(f"⚠️ File grande: {pre_size_mb:.1f} MB")
            else:
                st.info(f"📊 Dimensione: {pre_size_mb:.1f} MB")
        
        with col2:
            if post_size_mb > 100:
                st.warning(f"⚠️ File grande: {post_size_mb:.1f} MB")
            else:
                st.info(f"📊 Dimensione: {post_size_mb:.1f} MB")
        
        with st.spinner("Caricamento file..."):
            df_pre = mapper.load_file(pre_file, 'pre')
            df_post = mapper.load_file(post_file, 'post')
        
        if df_pre is not None and df_post is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File PRE", f"{len(df_pre):,} righe", f"{len(df_pre.columns)} colonne")
            with col2:
                st.metric("File POST", f"{len(df_post):,} righe", f"{len(df_post.columns)} colonne")
            with col3:
                total_combinations = len(df_pre) * len(df_post)
                if total_combinations > 1e9:
                    st.metric("Complessità", "Molto Alta", "🔴")
                elif total_combinations > 1e7:
                    st.metric("Complessità", "Alta", "🟡")
                else:
                    st.metric("Complessità", "Media", "🟢")
            
            estimated_memory = (len(df_pre) + len(df_post)) * len(df_pre.columns) * 0.001
            if estimated_memory > available_memory * 1024 * 0.8:
                st.error("⚠️ Il dataset potrebbe superare la memoria disponibile. Considera di ridurre le dimensioni del file.")
            elif estimated_memory > available_memory * 1024 * 0.5:
                st.warning("⚠️ Dataset grande rilevato. L'elaborazione utilizzerà la modalità batch ottimizzata.")
            
            st.header("🎯 Selezione Colonne per Matching")
            
            common_columns = list(set(df_pre.columns) & set(df_post.columns))
            
            required_pre = ['Address', 'Status Code']
            required_post = ['Address', 'Status Code', 'Indexability']
            
            missing_pre = [col for col in required_pre if col not in df_pre.columns]
            missing_post = [col for col in required_post if col not in df_post.columns]
            
            if missing_pre or missing_post:
                st.error("❌ Colonne obbligatorie mancanti:")
                if missing_pre:
                    st.error(f"File PRE: {missing_pre}")
                if missing_post:
                    st.error(f"File POST: {missing_post}")
                return
            
            matching_options = [col for col in common_columns if col not in ['Status Code', 'Indexability']]
            
            default_matching = ['Address']
            if 'Title 1' in matching_options:
                default_matching.append('Title 1')
            if 'H1-1' in matching_options:
                default_matching.append('H1-1')
            
            total_rows = len(df_pre) + len(df_post)
            if total_rows > 100000:
                st.info("💡 **Consiglio per dataset grandi:** Limita il numero di colonne di matching (max 3-4) per ottimizzare le performance.")
            
            matching_columns = st.multiselect(
                "Seleziona colonne per il matching:",
                options=matching_options,
                default=default_matching,
                help="Seleziona le colonne su cui basare il matching delle URL. Più colonne = matching più accurato ma elaborazione più lenta."
            )
            
            if not matching_columns:
                st.warning("⚠️ Seleziona almeno una colonna per il matching")
                return
            
            if len(matching_columns) > 5 and total_rows > 50000:
                st.warning("⚠️ Molte colonne selezionate con dataset grande. L'elaborazione potrebbe richiedere molto tempo.")
            
            if st.checkbox("👀 Anteprima dati"):
                preview_rows = min(5, len(df_pre), len(df_post))
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Anteprima PRE (prime {preview_rows} righe):**")
                    display_cols = matching_columns + ['Status Code']
                    st.dataframe(df_pre[display_cols].head(preview_rows))
                
                with col2:
                    st.write(f"**Anteprima POST (prime {preview_rows} righe):**")
                    display_cols = matching_columns + ['Status Code', 'Indexability']
                    display_cols = [col for col in display_cols if col in df_post.columns]
                    st.dataframe(df_post[display_cols].head(preview_rows))
            
            st.header("🚀 Elaborazione")
            
            if total_rows > 10000:
                estimated_time = (total_rows / 10000) * len(matching_columns) * 30
                st.info(f"⏱️ Tempo stimato: {estimated_time/60:.1f} minuti (per {total_rows:,} righe totali)")
            
            col1, col2 = st.columns(2)
            with col1:
                process_button = st.button("▶️ Avvia Mapping", type="primary", use_container_width=True)
            with col2:
                if st.button("🗑️ Libera Memoria", help="Forza garbage collection per liberare memoria"):
                    gc.collect()
                    st.success("Memoria liberata!")
                    st.rerun()
            
            if process_button:
                try:
                    current_memory = psutil.virtual_memory().percent
                    if current_memory > 85:
                        st.warning(f"⚠️ Memoria attuale alta ({current_memory:.1f}%). Considerare di liberare memoria o ridurre il dataset.")
                    
                    with st.spinner("Elaborazione in corso..."):
                        start_time = time.time()
                        df_result, df_non_redirectable = mapper.process_migration(
                            df_pre, df_post, matching_columns, use_ai and api_key is not None
                        )
                        end_time = time.time()
                    
                    st.header("📊 Risultati")
                    
                    processing_time = end_time - start_time
                    rows_per_second = len(df_result) / processing_time
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("URL Processate", f"{len(df_result):,}")
                    with col2:
                        st.metric("Non-redirectable", f"{len(df_non_redirectable):,}")
                    with col3:
                        st.metric("Tempo Elaborazione", f"{processing_time:.1f}s")
                    with col4:
                        st.metric("Velocità", f"{rows_per_second:.0f} righe/s")
                    with col5:
                        if 'Highest Match Similarity' in df_result.columns:
                            avg_similarity = df_result['Highest Match Similarity'].mean()
                            st.metric("Similarità Media", f"{avg_similarity:.3f}")
                    
                    if 'Highest Match Similarity' in df_result.columns:
                        similarity_data = df_result['Highest Match Similarity'].dropna()
                        high_quality = (similarity_data >= 0.8).sum()
                        medium_quality = ((similarity_data >= 0.5) & (similarity_data < 0.8)).sum()
                        low_quality = (similarity_data < 0.5).sum()
                        
                        st.subheader("📈 Analisi Qualità")
                        qual_col1, qual_col2, qual_col3 = st.columns(3)
                        with qual_col1:
                            st.metric("Alta Qualità", f"{high_quality:,}", f"{high_quality/len(similarity_data)*100:.1f}%")
                        with qual_col2:
                            st.metric("Media Qualità", f"{medium_quality:,}", f"{medium_quality/len(similarity_data)*100:.1f}%")
                        with qual_col3:
                            st.metric("Bassa Qualità", f"{low_quality:,}", f"{low_quality/len(similarity_data)*100:.1f}%")
                    
                    st.subheader("🎯 Risultati Mapping")
                    
                    if len(df_result) > 1000:
                        st.info(f"Mostra prime 1000 righe di {len(df_result):,} risultati totali")
                        st.dataframe(df_result.head(1000), use_container_width=True)
                    else:
                        st.dataframe(df_result, use_container_width=True)
                    
                    st.header("💾 Download Risultati")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        estimated_size_mb = len(df_result) * len(df_result.columns) * 50 / (1024 * 1024)
                        
                        csv_result = df_result.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Scarica Mapping Completo (CSV) ~{estimated_size_mb:.1f}MB",
                            data=csv_result,
                            file_name=f"auto-migration-mapped-{len(df_result)}-urls-{int(time.time())}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        if len(df_non_redirectable) > 0:
                            csv_non_redirect = df_non_redirectable.to_csv(index=False)
                            st.download_button(
                                label=f"📥 URL Non-redirectable (CSV) - {len(df_non_redirectable)} righe",
                                data=csv_non_redirect,
                                file_name=f"auto-migration-non-redirectable-{int(time.time())}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("✅ Nessuna URL non-redirectable trovata")
                    
                    del df_result, df_non_redirectable
                    gc.collect()
                    
                    st.success("🎉 Elaborazione completata! Memoria liberata automaticamente.")
                
                except MemoryError:
                    st.error("❌ Memoria insufficiente. Prova a:")
                    st.write("- Ridurre le dimensioni del file")
                    st.write("- Diminuire il numero di colonne di matching") 
                    st.write("- Aumentare la memoria disponibile")
                except Exception as e:
                    st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
                    st.exception(e)
                    
                    gc.collect()
    
    with st.expander("ℹ️ Informazioni e Ottimizzazioni"):
        st.markdown("""
        ### 🚀 Ottimizzazioni per File Grandi:
        
        **Nuove Funzionalità v2.0:**
        - ✅ **Elaborazione Batch**: Processamento automatico in chunks per file fino a 1GB
        - ✅ **Monitoraggio Memoria**: Controllo real-time dell'utilizzo memoria con auto-ottimizzazione
        - ✅ **Batch Size Dinamico**: Calcolo automatico della dimensione ottimale del batch
        - ✅ **AI Limitato**: Per dataset grandi, limita le chiamate AI per controllare i costi
        - ✅ **Progress Tracking**: Barre di progresso dettagliate per ogni fase
        - ✅ **Garbage Collection**: Pulizia automatica della memoria durante l'elaborazione
        
        ### 📊 Raccomandazioni per Performance:
        
        **File < 10K righe:** Elaborazione standard, tutte le funzionalità disponibili  
        **File 10K-100K righe:** Modalità batch automatica, AI limitato  
        **File > 100K righe:** Batch ottimizzato, AI solo per casi critici  
        **File > 500K righe:** Limita colonne matching (max 3-4), disabilita anteprima  
        
        ### 💾 Gestione Memoria:
        
        - **Auto-batch**: Il sistema calcola automaticamente la dimensione ottimale
        - **Limite memoria**: Impostabile da 70% a 95% (default 85%)
        - **Cleanup automatico**: Garbage collection tra le fasi di elaborazione
        - **Monitoraggio real-time**: Visualizzazione utilizzo memoria durante l'elaborazione
        
        ### ⚡ Tips per Ottimizzare:
        
        1. **Usa meno colonne** per dataset molto grandi (Address + 1-2 colonne aggiuntive)
        2. **Disabilita l'AI** per file > 200K righe per velocizzare l'elaborazione
        3. **Aumenta il limite memoria** se hai RAM sufficiente
        4. **Processa in più sessioni** dividendo file enormi
        5. **Usa file CSV** invece di Excel quando possibile (più veloce)
        
        ### 🔧 Colonne richieste:
        
        **File PRE:** Address, Status Code  
        **File POST:** Address, Status Code, Indexability
        
        ### 🎯 Algoritmo Ottimizzato:
        
        1. **Caricamento Chunk-based** per file grandi
        2. **Preprocessing con ottimizzazione tipi dati**
        3. **Matching batch-wise** con PolyFuzz
        4. **Merge incrementale** per gestire memoria
        5. **AI enhancement controllato** per qualità
        6. **Export ottimizzato** con compressione
        """)

if __name__ == "__main__":
    main()
