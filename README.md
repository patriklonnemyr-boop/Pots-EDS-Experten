# Pots-EDS-Experten

En intelligent forskningsassistent för Ehlers-Danlos syndrom (EDS) och Posturalt ortostatiskt takykardisyndrom (POTS).

## Installation

1. Installera beroenden:
   ```bash
   pip install -r requirements.txt
   ```

2. Konfigurera API-nycklar i `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "din_gemini_nyckel"
   TAVILY_API_KEY = "din_tavily_nyckel"
   ```

3. Lägg till PDF-filer i mappen `knowledge_base`.

4. Kör applikationen:
   ```bash
   streamlit run app.py
   ```
