# --- 1. SYSTEM-FIX FÖR CHROMADB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import google.generativeai as genai
from tavily import TavilyClient
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import pandas as pd
from PIL import Image

# --- 2. CONFIG & UI ---
st.set_page_config(
    page_title="Pots-EDS-Experten",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .source-box { 
        background-color: #e9ecef; 
        padding: 10px; 
        border-radius: 5px; 
        border-left: 5px solid #343a40;
        margin-bottom: 10px;
    }
    .zebra-text { color: #000000; font-weight: bold; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Central logotyp och titel
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Kontrollera båda filändelserna för säkerhets skull
    logo_path = "zebra_logo.png" if os.path.exists("zebra_logo.png") else "zebra_logo.PNG"
    
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        # Ändrat till en fast bredd (200 pixlar) för att den inte ska vara för stor
        st.image(logo, width=200) 
    st.title("Pots-EDS-Experten")
    st.markdown("---")
    
# --- 3. KUNSKAPSBAS (RAG) LOGIK ---
DB_PATH = "chroma_db"
KB_FOLDER = "knowledge_base" 

def initialize_rag():
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(name="eds_pots_docs", embedding_function=emb_fn)
    
    if collection.count() == 0 and os.path.exists(KB_FOLDER):
        with st.spinner("Indexerar forskningsdatabasen..."):
            for filename in os.listdir(KB_FOLDER):
                if filename.endswith(".pdf"):
                    path = os.path.join(KB_FOLDER, filename)
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    chunks = [text[i:i+2000] for i in range(0, len(text), 1500)]
                    ids = [f"{filename}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": filename} for _ in range(len(chunks))]
                    
                    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            st.success("✅ Kunskapsbas redo!")
    return collection

# --- 4. AI & SÖK LOGIK ---
def get_latest_updates(collection):
    query = "senaste viktiga forskningsrön och uppdateringar om EDS och POTS"
    
    web_context = ""
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        # Hämtar nyheter med avancerad sökning
        web_search = tavily.search(query=query, search_depth="advanced", max_results=5, topic="news")
        
        # Inkluderar publiceringsdatum i kontexten om det finns tillgängligt
        web_context = "\n".join([
            f"Källa: {r['url']}\nDatum: {r.get('published_date', 'Okänt datum')}\nInnehåll: {r['content']}" 
            for r in web_search['results']
        ])
    except Exception as e:
        st.warning(f"Kunde inte hämta senaste nyheter: {e}")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        prompt = f"""
        Du är Pots-EDS-Experten. Sammanfatta de viktigaste senaste nyheterna inom EDS och POTS baserat på informationen nedan.
        
        INFORMATION:
        {web_context}
        
        INSTRUKTIONER:
        1. Ge en kortfattad sammanfattning av rönen.
        2. För varje punkt ska du UTTRYCKLIGEN inkludera publiceringsdatumet från källan så att användaren vet hur gammalt rönat är.
        3. Ange tydliga källhänvisningar (URL).
        4. Svara på svenska.
        5. Avsluta med medicinsk ansvarsfriskrivning.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ett fel uppstod vid kommunikation med Gemini: {e}"

def perform_ai_analysis(query, collection):
    results = collection.query(query_texts=[query], n_results=3)
    local_context = "\n".join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))
    
    web_context = ""
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        search_query = f"medical research {query} EDS POTS MCAS"
        web_search = tavily.search(query=search_query, max_results=5)
        web_context = "\n".join([
            f"Datum: {r.get('published_date', 'Okänt')}\nInnehåll: {r['content']}" 
            for r in web_search['results']
        ])
    except Exception as e:
        st.warning(f"Kunde inte utföra webbsökning: {e}")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        prompt = f"""
        Du är Pots-EDS-Experten, en medicinsk forskningsassistent för EDS och POTS.
        
        ANVÄNDARFRÅGA: {query}
        LOKAL KONTEXT: {local_context}
        WEBBINFORMATION: {web_context}
        
        INSTRUKTIONER:
        1. Svara på svenska, professionellt och pedagogiskt.
        2. Vid referens till webbinformation, försök inkludera datum om det finns i kontexten.
        3. Skilj på lokal data och nya rön från webben.
        4. Avsluta alltid med ansvarsfriskrivning: Du är AI, inte läkare.
        5. Ange källor.
        """
        
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as e:
        return f"Fel vid AI-analys: {e}", []

# --- 5. MAIN APP ---
def main():
    collection = initialize_rag()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Hämta senaste uppdateringarna inom EDS/POTS"):
            with st.spinner("Söker efter senaste nytt..."):
                latest_info = get_latest_updates(collection)
                st.markdown("### 📢 Senaste nytt & rön (med datum)")
                st.info(latest_info)
                st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_main_1, col_main_2, col_main_3 = st.columns([1, 4, 1])
    with col_main_2:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Vad vill du veta om EDS/POTS idag?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with col_main_2:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyserar medicinsk data..."):
                    full_response, sources = perform_ai_analysis(prompt, collection)
                    st.markdown(full_response)
                    
                    if sources:
                        with st.expander("Använda lokala källor"):
                            for s in sources:
                                st.write(f"📄 {s}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()