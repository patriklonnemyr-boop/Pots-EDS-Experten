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
from PIL import Image
import base64

# --- 2. CONFIG & UI ---
st.set_page_config(
    page_title="Pots-EDS-Experten",
    layout="wide"
)

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# CSS för att dölja menyer, footer och göra layouten responsiv
st.markdown("""
<style>
    /* Dölj Streamlits översta rad (Share, inställningar) och footer (Manage app) */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Förhindra sidledsskroll */
    .main .block-container {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
        overflow-x: hidden;
    }

    /* Header: Logga och Rubrik bredvid varandra */
    .custom-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: -50px; /* Justera uppåt eftersom headern är dold */
        margin-bottom: 20px;
    }
    
    .logo-img { width: 50px; height: auto; }
    
    .header-title {
        font-size: 1.4rem !important;
        margin: 0;
        font-weight: 800;
        color: #1E1E1E;
    }

    /* Nyhetsboxen */
    .news-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #000000;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.95rem;
        line-height: 1.6;
        word-wrap: break-word;
    }

    @media (max-width: 480px) {
        .header-title { font-size: 1.1rem !important; }
        .logo-img { width: 40px; }
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
logo_path = "zebra_logo.png" if os.path.exists("zebra_logo.png") else "zebra_logo.PNG"
img_b64 = get_image_base64(logo_path)

if img_b64:
    st.markdown(f'<div class="custom-header"><img src="data:image/png;base64,{img_b64}" class="logo-img"><h1 class="header-title">Pots-EDS-Experten</h1></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="custom-header"><h1 class="header-title">Pots-EDS-Experten</h1></div>', unsafe_allow_html=True)

st.markdown("---")

# --- 3. OPTIMERAD KUNSKAPSBAS (RAG) ---
@st.cache_resource
def initialize_rag():
    DB_PATH = "chroma_db"
    KB_FOLDER = "knowledge_base"
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(name="eds_pots_docs", embedding_function=emb_fn)
    
    if collection.count() == 0 and os.path.exists(KB_FOLDER):
        pdf_files = [f for f in os.listdir(KB_FOLDER) if f.endswith(".pdf")]
        if pdf_files:
            with st.spinner("Optimerar forskningsdatabasen..."):
                for filename in pdf_files:
                    try:
                        path = os.path.join(KB_FOLDER, filename)
                        reader = PdfReader(path)
                        text = "".join([p.extract_text() + "\n" for p in reader.pages])
                        chunks = [text[i:i+2000] for i in range(0, len(text), 1500)]
                        ids = [f"{filename}_{i}" for i in range(len(chunks))]
                        metadatas = [{"source": filename} for _ in range(len(chunks))]
                        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                    except Exception: continue
    return collection

# --- 4. AI & SÖK LOGIK ---
def get_latest_updates():
    query = "latest clinical research findings EDS Ehlers-Danlos POTS Syndrome 2025 2026"
    web_context = ""
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        web_search = tavily.search(query=query, search_depth="advanced", max_results=5, topic="news")
        
        web_context = "\n".join([
            f"KÄLLA: {r['url']}\nDATUM: {r.get('published_date', 'Datum saknas')}\nINNEHÅLL: {r['content']}" 
            for r in web_search['results']
        ])
    except:
        return "Kunde inte hämta nyheter just nu."

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    prompt = f"""
    Du är en medicinsk expert. Sammanfatta nyheterna inom EDS och POTS från 2025-2026.
    
    KÄLLOR:
    Avsluta varje punkt med en klickbar länk: "Källa: [Namn](URL)".
    
    DATUM:
    Ange publiceringsdatumet från webben för varje nyhet.
    
    INFORMATION:
    {web_context}
    
    Svara på svenska i punktform. Inga ikoner. Avsluta med ansvarsfriskrivning.
    """
    response = model.generate_content(prompt)
    return response.text

def perform_ai_analysis(query, collection):
    results = collection.query(query_texts=[query], n_results=3)
    local_context = "\n".join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))
    
    web_context = ""
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        web_search = tavily.search(query=f"clinical study 2025 {query} POTS EDS", max_results=3)
        web_context = "\n".join([f"Källa: {r['url']} (Datum: {r.get('published_date', 'N/A')}) - {r['content']}" for r in web_search['results']])
    except: pass

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    prompt = f"""
    Svara på svenska på: {query}
    LOKAL DATA: {local_context}
    WEBB-DATA: {web_context}
    
    Ange alltid källans URL och publiceringsdatum för webbinformation.
    Avsluta med ansvarsfriskrivning.
    """
    response = model.generate_content(prompt)
    return response.text, sources

# --- 5. MAIN APP ---
def main():
    collection = initialize_rag()
    
    if st.button("✨ Hämta senaste nytt"):
        with st.spinner("Hämtar rön..."):
            latest_info = get_latest_updates()
            st.markdown("### Senaste nytt")
            st.markdown(f'<div class="news-box">{latest_info}</div>', unsafe_allow_html=True)
            st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Skriv din fråga här..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyserar..."):
                res, src = perform_ai_analysis(prompt, collection)
                st.markdown(res)
                if src:
                    with st.expander("Lokala PDF-källor"):
                        for s in src: st.write(f"📄 {s}")
        st.session_state.messages.append({"role": "assistant", "content": res})

if __name__ == "__main__":
    main()