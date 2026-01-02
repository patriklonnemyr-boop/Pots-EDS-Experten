# --- 1. SYSTEM-FIX FÖR CHROMADB (Om det behövs på Streamlit Cloud) ---
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
    /* Centrera innehåll om man vill ha en renare look utan sidebar */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Central logotyp och titel (istället för sidebar)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists("zebra_logo.PNG"):
        logo = Image.open("zebra_logo.PNG")
        st.image(logo, use_container_width=True)
    st.title("Pots-EDS-Experten")
    st.markdown("---")
    
# --- 3. KUNSKAPSBAS (RAG) LOGIK ---
DB_PATH = "chroma_db"
KB_FOLDER = "../knowledge_base" # Pejar mot mappen på skrivbordet

def initialize_rag():
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(name="eds_pots_docs", embedding_function=emb_fn)
    
    # Om kollektionen är tom, indexera dokumenten
    if collection.count() == 0 and os.path.exists(KB_FOLDER):
        with st.spinner("Indexerar forskningsdatabasen..."):
            for filename in os.listdir(KB_FOLDER):
                if filename.endswith(".pdf"):
                    path = os.path.join(KB_FOLDER, filename)
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Dela upp i mindre segment (chunks)
                    chunks = [text[i:i+2000] for i in range(0, len(text), 1500)]
                    ids = [f"{filename}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": filename} for _ in range(len(chunks))]
                    
                    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            st.success("✅ Kunskapsbas redo!")
    return collection

# --- 4. AI & SÖK LOGIK ---
def get_latest_updates(collection):
    query = "senaste viktiga forskningsrön och uppdateringar om EDS och POTS"
    
    # 1. Webbsökning fokuserad på senaste nyheter
    web_context = ""
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        # Sök efter senaste nyheter (news)
        web_search = tavily.search(query=query, search_depth="advanced", max_results=5, topic="news")
        web_context = "\n".join([f"Källa: {r['url']}\nInnehåll: {r['content']}" for r in web_search['results']])
    except Exception as e:
        st.warning(f"Kunde inte hämta senaste nyheter: {e}")

    # 2. Generera sammanfattning med Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Du är Pots-EDS-Experten. Sammanfatta de viktigaste senaste nyheterna inom EDS (Ehlers-Danlos syndrom) och POTS baserat på följande information.
    
    INFORMATION:
    {web_context}
    
    INSTRUKTIONER:
    1. Ge en kortfattad sammanfattning av de absolut viktigaste rönen.
    2. För varje nyhet/punkt, ange publiceringsdatum (om tillgängligt) och en tydlig källhänvisning (URL).
    3. Var extremt källkritisk. Prioritera information från medicinska institut och betrodda vetenskapliga källor.
    4. Om ingen ny relevant information hittas, säg det.
    5. Svara på svenska.
    6. Avsluta med ansvarsfriskrivning.
    """
    
    response = model.generate_content(prompt)
    return response.text

def perform_ai_analysis(query, collection):
    # 1. Hämta kontext från RAG
    results = collection.query(query_texts=[query], n_results=3)
    local_context = "\n".join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))
    
    # 2. Utför webbsökning via Tavily (om API-nyckel finns)
    web_context = ""
    try:
        tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        search_query = f"medical research {query} EDS POTS MCAS"
        web_search = tavily.search(query=search_query, max_results=5)
        web_context = "\n".join([r['content'] for r in web_search['results']])
    except Exception as e:
        st.warning(f"Kunde inte utföra webbsökning: {e}")

    # 3. Generera svar med Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash') # Gemini 2.5 Flash i prompten men ofta gemini-1.5-flash i API:et just nu
    
    prompt = f"""
    Du är Pots-EDS-Experten, en medicinsk forskningsassistent specialiserad på Ehlers-Danlos syndrom och POTS.
    
    ANVÄNDARFRÅGA: {query}
    
    LOKAL KONTEXT FRÅN FORSKNINGSRAPPORTER:
    {local_context}
    
    REALTIDSINFORMATION FRÅN WEBBEN:
    {web_context}
    
    INSTRUKTIONER:
    1. Svara på svenska på ett professionellt och pedagogiskt sätt.
    2. Prioritera information från välrenommerade källor som PubMed, The Lancet, Mayo Clinic.
    3. Skilj tydligt på lokal forskningsdata och nya rön från webben.
    4. Om informationen är motstridig, nämna detta.
    5. Avsluta alltid med en ansvarsfriskrivning att du är en AI och inte en läkare.
    6. Ange källor där det är möjligt.
    """
    
    response = model.generate_content(prompt)
    return response.text, sources

# --- 5. MAIN APP ---
def main():
    # Central logotyp och titel hanteras nu i CONFIG & UI sektionen men vi kan säkerställa ordningen här
    collection = initialize_rag()
    
    # Knapp för senaste uppdateringar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Hämta senaste uppdateringarna inom EDS/POTS"):
            with st.spinner("Söker efter senaste nytt..."):
                latest_info = get_latest_updates(collection)
                st.markdown("### 📢 Senaste nytt & rön")
                st.info(latest_info)
                st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Behållare för chatten för att centrera den
    col_main_1, col_main_2, col_main_3 = st.columns([1, 4, 1])
    with col_main_2:
        # Visa chatthistorik
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat-input (st.chat_input är alltid längst ner)
    if prompt := st.chat_input("Vad vill du veta om EDS/POTS idag?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Rendera chatten igen i den centrerade kolumnen
        # (Streamlit kör om hela main, så vi behöver bara hantera visningen ovan)
        st.rerun()

    # Logik för att hantera sista meddelandet om det är nytt
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with col_main_2:
            with st.chat_message("assistant"):
                with st.spinner("Analyserar data och söker i medicinska databaser..."):
                    query = st.session_state.messages[-1]["content"]
                    full_response, sources = perform_ai_analysis(query, collection)
                    st.markdown(full_response)
                    
                    if sources:
                        with st.expander("Använda lokala källor"):
                            for s in sources:
                                st.write(f"📄 {s}")
                    
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

if __name__ == "__main__":
    main()
