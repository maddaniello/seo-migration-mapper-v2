import streamlit as st
import pandas as pd
import numpy as np
from polyfuzz import PolyFuzz
import time
import io
from typing import List, Dict, Tuple, Optional, Union
import openai
import json
import re
from urllib.parse import urlparse
import openpyxl

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
        
    def setup_openai(self, api_key: str) -> bool:
        """Setup OpenAI client"""
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            # Test the connection
            self.openai_client.models.list()
            return True
        except Exception as e:
            st.error(f"Errore nella configurazione OpenAI: {str(e)}")
            return False
    
    def load_file(self, uploaded_file, file_type: str) -> pd.DataFrame:
        """Load CSV or XLSX file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Formato file non supportato")
            
            return df
        except Exception as e:
            st.error(f"Errore nel caricamento del file {uploaded_file.name}: {str(e)}")
            return None
    
    def validate_columns(self, df: pd.DataFrame, file_type: str, selected_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate required columns exist"""
        required = self.required_columns[file_type]
        all_required = required + selected_columns
        missing = [col for col in all_required if col not in df.columns]
        
        if missing:
            return False, missing
        return True, []
    
    def preprocess_data(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                       matching_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess the data"""
        # Create copies
        df_live = df_live.copy()
        df_staging = df_staging.copy()
        
        # Ensure required columns exist
        for col in matching_columns:
            if col not in df_live.columns:
                df_live[col] = ""
            if col not in df_staging.columns:
                df_staging[col] = ""
        
        # Drop duplicates
        df_live.drop_duplicates(subset="Address", inplace=True)
        df_staging.drop_duplicates(subset="Address", inplace=True)
        
        # Extract non-redirectable URLs (3xx & 5xx)
        df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)]
        df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)]
        df_non_redirectable = pd.concat([df_3xx, df_5xx])
        
        # Keep 2xx and 4xx status codes for redirecting
        df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)]
        df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)]
        df_live_clean = pd.concat([df_live_200, df_live_400])
        
        # Handle NaN values - populate with URL for 404s
        for col in matching_columns:
            df_live_clean[col] = df_live_clean[col].fillna(df_live_clean['Address'])
            df_staging[col] = df_staging[col].fillna(df_staging['Address'])
        
        return df_live_clean, df_staging, df_non_redirectable
    
    def perform_polyfuzz_matching(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                                matching_columns: List[str]) -> Dict[str, pd.DataFrame]:
        """Perform PolyFuzz matching for each column"""
        matches = {}
        
        progress_bar = st.progress(0)
        total_columns = len(matching_columns)
        
        for i, col in enumerate(matching_columns):
            st.write(f"🔍 Matching su colonna: {col}")
            
            # Perform matching
            model = PolyFuzz("TF-IDF").match(
                list(df_live[col].astype(str)), 
                list(df_staging[col].astype(str))
            )
            
            df_match = model.get_matches()
            
            # Rename columns
            similarity_col = f"{col} Similarity"
            from_col = f"From ({col})"
            to_col = f"To {col}"
            
            df_match.rename(columns={
                "Similarity": similarity_col,
                "From": from_col,
                "To": to_col
            }, inplace=True)
            
            matches[col] = df_match
            progress_bar.progress((i + 1) / total_columns)
        
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
            
            # Extract number from response
            match = re.search(r'\d+', result)
            if match:
                choice = int(match.group())
                if 1 <= choice <= len(candidate_urls):
                    return candidate_urls[choice - 1]
            
            return None
            
        except Exception as e:
            st.warning(f"Errore AI enhancement: {str(e)}")
            return None
    
    def merge_matches(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                     matches: Dict[str, pd.DataFrame], matching_columns: List[str]) -> pd.DataFrame:
        """Merge all matches into final dataframe"""
        
        # Start with Address matching (always present)
        if 'Address' in matches:
            df_final = pd.merge(df_live, matches['Address'], 
                              left_on="Address", right_on="From (Address)", how="inner")
            df_final.rename(columns={"To Address": "URL - URL Match"}, inplace=True)
        else:
            df_final = df_live.copy()
            df_final['Address Similarity'] = 0
            df_final['URL - URL Match'] = ""
        
        # Merge other columns
        for col in matching_columns:
            if col == 'Address':
                continue
                
            if col in matches:
                # Create mapping dataframe
                df_col_map = df_staging[[col, 'Address']].copy()
                df_match_with_url = pd.merge(
                    matches[col], df_col_map, 
                    left_on=f"To {col}", right_on=col, how="inner"
                )
                
                # Merge with final dataframe
                df_final = df_final.merge(
                    df_match_with_url.drop_duplicates(col), 
                    how='left', left_on=col, right_on=f"From ({col})"
                )
                
                df_final.rename(columns={"Address": f"URL - {col} Match"}, inplace=True)
        
        return df_final
    
    def calculate_best_matches(self, df_final: pd.DataFrame, matching_columns: List[str]) -> pd.DataFrame:
        """Calculate best, second best, and lowest matches"""
        
        similarity_cols = [f"{col} Similarity" for col in matching_columns if f"{col} Similarity" in df_final.columns]
        
        if not similarity_cols:
            return df_final
        
        # Get best match
        df_final['Best Match On'] = df_final[similarity_cols].idxmax(axis=1)
        df_final['Highest Match Similarity'] = df_final[similarity_cols].max(axis=1)
        
        # Get worst match
        df_final['Lowest Match On'] = df_final[similarity_cols].idxmin(axis=1)
        df_final['Lowest Match Similarity'] = df_final[similarity_cols].min(axis=1)
        
        # Calculate middle match (second best)
        def get_middle_similarity(row):
            values = [(col, row[col]) for col in similarity_cols if pd.notna(row[col])]
            if len(values) < 2:
                return None, None
            values.sort(key=lambda x: x[1], reverse=True)
            return values[1] if len(values) > 1 else (None, None)
        
        middle_matches = df_final.apply(get_middle_similarity, axis=1, result_type='expand')
        df_final['Second Match On'] = middle_matches[0]
        df_final['Second Highest Match Similarity'] = middle_matches[1]
        
        # Set best matching URLs and text
        for col in matching_columns:
            similarity_col = f"{col} Similarity"
            url_col = f"URL - {col} Match"
            from_col = f"From ({col})"
            to_col = f"To {col}"
            
            if similarity_col in df_final.columns:
                # Best match
                mask = df_final['Best Match On'] == similarity_col
                if url_col in df_final.columns:
                    df_final.loc[mask, 'Best Matching URL'] = df_final.loc[mask, url_col]
                if from_col in df_final.columns:
                    df_final.loc[mask, 'Highest Match Source Text'] = df_final.loc[mask, from_col]
                if to_col in df_final.columns:
                    df_final.loc[mask, 'Highest Match Destination Text'] = df_final.loc[mask, to_col]
                
                # Second match
                mask = df_final['Second Match On'] == similarity_col
                if url_col in df_final.columns:
                    df_final.loc[mask, 'Second Highest Match'] = df_final.loc[mask, url_col]
                if from_col in df_final.columns:
                    df_final.loc[mask, 'Second Match Source Text'] = df_final.loc[mask, from_col]
                if to_col in df_final.columns:
                    df_final.loc[mask, 'Second Match Destination Text'] = df_final.loc[mask, to_col]
        
        # Check for double matches
        if 'Best Matching URL' in df_final.columns and 'Second Highest Match' in df_final.columns:
            df_final["Double Matched?"] = (
                df_final['Best Matching URL'].str.lower() == 
                df_final['Second Highest Match'].str.lower()
            )
        
        # Clean up match labels
        df_final['Best Match On'] = df_final['Best Match On'].str.replace(' Similarity', '')
        df_final['Second Match On'] = df_final['Second Match On'].str.replace(' Similarity', '')
        df_final['Lowest Match On'] = df_final['Lowest Match On'].str.replace(' Similarity', '')
        
        return df_final
    
    def process_migration(self, df_live: pd.DataFrame, df_staging: pd.DataFrame, 
                         matching_columns: List[str], use_ai: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main processing function"""
        
        start_time = time.time()
        
        st.write("📊 Preprocessing dei dati...")
        df_live_clean, df_staging_clean, df_non_redirectable = self.preprocess_data(
            df_live, df_staging, matching_columns
        )
        
        st.write("🔍 Esecuzione matching con PolyFuzz...")
        matches = self.perform_polyfuzz_matching(df_live_clean, df_staging_clean, matching_columns)
        
        st.write("🔗 Merge dei risultati...")
        df_final = self.merge_matches(df_live_clean, df_staging_clean, matches, matching_columns)
        
        st.write("🎯 Calcolo dei migliori match...")
        df_final = self.calculate_best_matches(df_final, matching_columns)
        
        # AI Enhancement for low similarity matches
        if use_ai and self.openai_client and 'Highest Match Similarity' in df_final.columns:
            st.write("🤖 Miglioramento AI per match con bassa similarità...")
            low_similarity_mask = df_final['Highest Match Similarity'] < 0.7
            
            if low_similarity_mask.sum() > 0:
                st.write(f"Trovati {low_similarity_mask.sum()} match con bassa similarità, applicando AI...")
                
                # Process in batches to avoid API limits
                batch_size = 10
                for i in range(0, low_similarity_mask.sum(), batch_size):
                    batch_indices = df_final[low_similarity_mask].index[i:i+batch_size]
                    
                    for idx in batch_indices:
                        try:
                            source_url = df_final.loc[idx, 'Address']
                            source_content = {col: df_final.loc[idx, col] for col in matching_columns if col in df_final.columns}
                            
                            # Get top 3 candidates
                            candidates = []
                            for col in matching_columns:
                                url_col = f"URL - {col} Match"
                                if url_col in df_final.columns and pd.notna(df_final.loc[idx, url_col]):
                                    candidates.append(df_final.loc[idx, url_col])
                            
                            if candidates:
                                ai_match = self.ai_enhance_matching(
                                    source_url, candidates[:3], source_content, [{}]*len(candidates[:3])
                                )
                                
                                if ai_match:
                                    df_final.loc[idx, 'Best Matching URL'] = ai_match
                                    df_final.loc[idx, 'Best Match On'] = 'AI Enhanced'
                                    df_final.loc[idx, 'Highest Match Similarity'] = 0.95  # AI confidence score
                        
                        except Exception as e:
                            continue
        
        # Final cleanup and sorting
        df_final.drop_duplicates(subset="Address", inplace=True)
        
        # Select final columns
        final_columns = [
            "Address", "Status Code", "Best Matching URL", "Best Match On", 
            "Highest Match Similarity", "Highest Match Source Text", 
            "Highest Match Destination Text", "Second Highest Match", 
            "Second Match On", "Second Highest Match Similarity", 
            "Second Match Source Text", "Second Match Destination Text"
        ]
        
        # Add Double Matched column if it exists
        if "Double Matched?" in df_final.columns:
            final_columns.append("Double Matched?")
        
        # Only include columns that exist
        final_columns = [col for col in final_columns if col in df_final.columns]
        df_final = df_final[final_columns]
        
        # Rename Address column
        df_final.rename(columns={"Address": "URL - Source"}, inplace=True)
        
        # Sort by similarity
        if 'Highest Match Similarity' in df_final.columns:
            df_final.sort_values(["Highest Match Similarity"], ascending=[False], inplace=True)
        
        processing_time = time.time() - start_time
        st.success(f"✅ Processamento completato in {processing_time:.2f} secondi!")
        
        return df_final, df_non_redirectable

# Streamlit App
def main():
    st.title("🔗 URL Migration Mapping Tool")
    st.markdown("**Strumento avanzato per il mapping automatico di URL durante le migrazioni di siti web**")
    
    # Initialize the mapper
    mapper = URLMigrationMapper()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configurazione")
    
    # OpenAI Configuration
    st.sidebar.subheader("🤖 Configurazione OpenAI (Opzionale)")
    use_ai = st.sidebar.checkbox("Abilita miglioramento AI", help="Migliora i match con bassa similarità usando OpenAI GPT")
    
    api_key = None
    if use_ai:
        api_key = st.sidebar.text_input("API Key OpenAI", type="password", 
                                       help="Inserisci la tua API key OpenAI per abilitare il miglioramento AI")
        if api_key:
            if mapper.setup_openai(api_key):
                st.sidebar.success("✅ OpenAI configurato correttamente")
            else:
                use_ai = False
    
    # File upload section
    st.header("📁 Caricamento File")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File PRE-migrazione (Live)")
        pre_file = st.file_uploader(
            "Carica file pre-migrazione", 
            type=['csv', 'xlsx', 'xls'],
            key="pre_file",
            help="File contenente le URL del sito live prima della migrazione"
        )
    
    with col2:
        st.subheader("File POST-migrazione (Staging)")
        post_file = st.file_uploader(
            "Carica file post-migrazione", 
            type=['csv', 'xlsx', 'xls'],
            key="post_file",
            help="File contenente le URL del sito staging dopo la migrazione"
        )
    
    if pre_file and post_file:
        # Load files
        with st.spinner("Caricamento file..."):
            df_pre = mapper.load_file(pre_file, 'pre')
            df_post = mapper.load_file(post_file, 'post')
        
        if df_pre is not None and df_post is not None:
            st.success("✅ File caricati correttamente!")
            
            # Show file info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File PRE:** {pre_file.name}")
                st.write(f"- Righe: {len(df_pre):,}")
                st.write(f"- Colonne: {len(df_pre.columns)}")
            
            with col2:
                st.write(f"**File POST:** {post_file.name}")
                st.write(f"- Righe: {len(df_post):,}")
                st.write(f"- Colonne: {len(df_post.columns)}")
            
            # Column selection
            st.header("🎯 Selezione Colonne per Matching")
            
            # Get common columns
            common_columns = list(set(df_pre.columns) & set(df_post.columns))
            
            # Required columns check
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
            
            # Optional columns for matching
            matching_options = [col for col in common_columns if col not in ['Status Code', 'Indexability']]
            
            default_matching = ['Address']
            if 'Title 1' in matching_options:
                default_matching.append('Title 1')
            if 'H1-1' in matching_options:
                default_matching.append('H1-1')
            
            matching_columns = st.multiselect(
                "Seleziona colonne per il matching:",
                options=matching_options,
                default=default_matching,
                help="Seleziona le colonne su cui basare il matching delle URL. Più colonne = matching più accurato."
            )
            
            if not matching_columns:
                st.warning("⚠️ Seleziona almeno una colonna per il matching")
                return
            
            # Show preview
            if st.checkbox("👀 Anteprima dati"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Anteprima PRE (prime 5 righe):**")
                    st.dataframe(df_pre[matching_columns + ['Status Code']].head())
                
                with col2:
                    st.write("**Anteprima POST (prime 5 righe):**")
                    display_cols = matching_columns + ['Status Code', 'Indexability']
                    display_cols = [col for col in display_cols if col in df_post.columns]
                    st.dataframe(df_post[display_cols].head())
            
            # Processing section
            st.header("🚀 Elaborazione")
            
            if st.button("▶️ Avvia Mapping", type="primary"):
                try:
                    with st.spinner("Elaborazione in corso..."):
                        df_result, df_non_redirectable = mapper.process_migration(
                            df_pre, df_post, matching_columns, use_ai and api_key is not None
                        )
                    
                    # Show results
                    st.header("📊 Risultati")
                    
                    # Stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("URL Processate", len(df_result))
                    with col2:
                        st.metric("URL Non-redirectable", len(df_non_redirectable))
                    with col3:
                        if 'Highest Match Similarity' in df_result.columns:
                            avg_similarity = df_result['Highest Match Similarity'].mean()
                            st.metric("Similarità Media", f"{avg_similarity:.3f}")
                    with col4:
                        if 'Double Matched?' in df_result.columns:
                            double_matches = df_result['Double Matched?'].sum()
                            st.metric("Double Match", double_matches)
                    
                    # Show results table
                    st.subheader("🎯 Risultati Mapping")
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Download section
                    st.header("💾 Download Risultati")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Main results
                        csv_result = df_result.to_csv(index=False)
                        st.download_button(
                            label="📥 Scarica Mapping Completo (CSV)",
                            data=csv_result,
                            file_name="auto-migration-mapped-all-output.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Non-redirectable URLs
                        if len(df_non_redirectable) > 0:
                            csv_non_redirect = df_non_redirectable.to_csv(index=False)
                            st.download_button(
                                label="📥 Scarica URL Non-redirectable (CSV)",
                                data=csv_non_redirect,
                                file_name="auto-migration-non-redirectable-urls.csv",
                                mime="text/csv"
                            )
                    
                    # Quality analysis
                    if 'Highest Match Similarity' in df_result.columns:
                        st.header("📈 Analisi Qualità")
                        
                        # Similarity distribution
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        similarity_data = df_result['Highest Match Similarity'].dropna()
                        ax.hist(similarity_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Similarità')
                        ax.set_ylabel('Frequenza')
                        ax.set_title('Distribuzione Punteggi di Similarità')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Quality insights
                        high_quality = (similarity_data >= 0.8).sum()
                        medium_quality = ((similarity_data >= 0.5) & (similarity_data < 0.8)).sum()
                        low_quality = (similarity_data < 0.5).sum()
                        
                        st.write("**Analisi Qualità Match:**")
                        st.write(f"- 🟢 Alta qualità (≥0.8): {high_quality} ({high_quality/len(similarity_data)*100:.1f}%)")
                        st.write(f"- 🟡 Media qualità (0.5-0.8): {medium_quality} ({medium_quality/len(similarity_data)*100:.1f}%)")
                        st.write(f"- 🔴 Bassa qualità (<0.5): {low_quality} ({low_quality/len(similarity_data)*100:.1f}%)")
                        
                        if low_quality > 0:
                            st.warning(f"⚠️ {low_quality} URL hanno una bassa similarità. Considera di rivedere manualmente questi match.")
                
                except Exception as e:
                    st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
                    st.exception(e)
    
    # Information section
    with st.expander("ℹ️ Informazioni sullo strumento"):
        st.markdown("""
        ### Come funziona:
        
        1. **Carica i file:** Supporta formati CSV e Excel
        2. **Seleziona colonne:** Scegli su quali campi basare il matching
        3. **Elaborazione:** Utilizza PolyFuzz con TF-IDF per trovare le corrispondenze
        4. **AI Enhancement:** (Opzionale) Migliora i match con bassa similarità usando OpenAI
        5. **Risultati:** Ottieni mapping completo con punteggi di similarità
        
        ### Caratteristiche:
        
        - ✅ Supporto CSV e Excel
        - ✅ Matching multi-colonna personalizzabile
        - ✅ Miglioramento AI opzionale
        - ✅ Analisi qualità automatica
        - ✅ Export risultati
        - ✅ Gestione URL non-redirectable
        
        ### Colonne richieste:
        
        **File PRE:** Address, Status Code  
        **File POST:** Address, Status Code, Indexability
        
        ### Tips:
        
        - Più colonne di matching = risultati più accurati
        - Usa l'AI per migliorare match con bassa similarità
        - Rivedi manualmente i match con similarità < 0.5
        """)

if __name__ == "__main__":
    main()