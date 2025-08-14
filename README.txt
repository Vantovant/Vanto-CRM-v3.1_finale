Vanto CRM v3.1 (Contacts) — Localhost Package

What’s inside
- app.py — UI with Contacts (search, add form, inline edit, segments), all pages restored
- db.py — SQLite schema (20 columns + Tags), auto-migration, global search
- requirements.txt — dependencies
- .streamlit/config.toml — theme & server
- v3_import_template.csv — 20 columns + Tags
- crm.sqlite3 — sample data preloaded so you can test immediately

Run locally
1) In Command Prompt, cd into this folder.
2) Install deps:
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
3) Start the app:
   python -m streamlit run app.py
4) Open http://localhost:8501

Notes
- Replace crm.sqlite3 with your own to keep existing data (the app will auto-add any missing columns).
