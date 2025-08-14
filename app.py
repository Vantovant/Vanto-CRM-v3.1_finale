# app.py  — Vanto CRM v3.1 (Contacts) — Professional Upgrade
# - City is free text (no dropdown)
# - Enforced picklists with import normalization + non-blocking warnings
# - Global search, Add New Contact, filters, inline edit save

import io
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ----------------------------
# Paths / DB helpers
# ----------------------------
DB_PATH = Path(__file__).with_name("crm.sqlite3")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Core columns you already use + new picklist fields (all TEXT; no schema change type)
COLUMNS = [
    # your existing common fields
    "name", "phone", "email", "source", "interest", "status", "tags",
    "assigned", "notes", "action_needed", "action_taken",
    "username", "password",
    # geo/date (kept as TEXT)
    "date", "country", "province", "city",
    # NEW distinct status/picklist fields
    "registration_status",   # Activated / Registered / Not Registered / To Be Registered
    "lead_type",             # Customer / Distributor
    "associate_status",      # Promoter / Associate / Builder / Mentor / VIP / Diamond
]

def ensure_schema():
    conn = get_conn()
    cur = conn.cursor()
    # Minimal table (all TEXT; we won't change existing types)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, phone TEXT, email TEXT, source TEXT, interest TEXT, status TEXT, tags TEXT,
            assigned TEXT, notes TEXT, action_needed TEXT, action_taken TEXT,
            username TEXT, password TEXT,
            date TEXT, country TEXT, province TEXT, city TEXT,
            registration_status TEXT, lead_type TEXT, associate_status TEXT,
            created_at TEXT DEFAULT (datetime('now')), updated_at TEXT
        );
    """)
    # Add any missing columns safely
    cur.execute("PRAGMA table_info(contacts);")
    existing = {row["name"] for row in cur.fetchall()}
    for col in COLUMNS:
        if col not in existing:
            cur.execute(f"ALTER TABLE contacts ADD COLUMN {col} TEXT;")
    conn.commit()
    conn.close()

def fetch_contacts():
    ensure_schema()
    conn = get_conn()
    rows = conn.execute("SELECT * FROM contacts ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_one(row: dict):
    ensure_schema()
    payload = {c: row.get(c, "") for c in COLUMNS}
    conn = get_conn()
    cols = ",".join(COLUMNS)
    ph = ",".join(["?"] * len(COLUMNS))
    conn.execute(f"INSERT INTO contacts ({cols}) VALUES ({ph})", [payload[c] for c in COLUMNS])
    conn.commit()
    conn.close()

def bulk_insert(rows):
    if not rows: return 0
    ensure_schema()
    conn = get_conn()
    cols = ",".join(COLUMNS)
    ph = ",".join(["?"] * len(COLUMNS))
    data = [[r.get(c, "") for c in COLUMNS] for r in rows]
    conn.executemany(f"INSERT INTO contacts ({cols}) VALUES ({ph})", data)
    conn.commit()
    n = conn.total_changes
    conn.close()
    return n

def bulk_update_from_editor(df: pd.DataFrame):
    """Save inline table edits (id must be present)."""
    if "id" not in df.columns or df.empty:
        return 0
    ensure_schema()
    conn = get_conn()
    cur = conn.cursor()
    saved = 0
    for _, r in df.iterrows():
        if pd.isna(r["id"]): continue
        set_cols = []
        vals = []
        for c in COLUMNS:
            if c in df.columns:
                vals.append("" if pd.isna(r[c]) else str(r[c]))
                set_cols.append(f"{c} = ?")
        vals.append(r["id"])
        cur.execute(f"UPDATE contacts SET {', '.join(set_cols)}, updated_at = datetime('now') WHERE id = ?", vals)
        saved += 1
    conn.commit()
    conn.close()
    return saved

# ----------------------------
# Picklists & normalization
# ----------------------------
REG_STATUS_ALLOWED = ["Activated", "Registered", "Not Registered", "To Be Registered"]
LEAD_TYPE_ALLOWED = ["Customer", "Distributor"]
ASSOC_STATUS_ALLOWED = ["Promoter", "Associate", "Builder", "Mentor", "VIP", "Diamond"]

def _clean(s):
    if s is None: return ""
    return str(s).strip()

def _norm_key(s):
    # normalize text for mapping (remove punctuation, collapse spaces, lowercase)
    s = _clean(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_registration_status(val):
    key = _norm_key(val)
    lookup = {
        "activated": "Activated",
        "registerd": "Registered", "registered": "Registered",
        "not registered": "Not Registered", "notregistered": "Not Registered",
        "to be registered": "To Be Registered", "tbr": "To Be Registered",
        "to_be_registered": "To Be Registered", "to be  registered": "To Be Registered",
    }
    return lookup.get(key, "")

def normalize_lead_type(val):
    key = _norm_key(val)
    if key in ("customer",): return "Customer"
    if key in ("distributor",): return "Distributor"
    return ""

def normalize_associate_status(val):
    key = _norm_key(val)
    # handle VIP variations like "V.I.P"
    if key in ("vip", "v i p", "v i.p", "v i p.", "v i p ."): return "VIP"
    mapping = {
        "promoter": "Promoter",
        "associate": "Associate",
        "builder": "Builder",
        "mentor": "Mentor",
        "diamond": "Diamond",
    }
    return mapping.get(key, "")

def normalize_row_for_picklists(rec, warn_counts):
    """Normalize the three enforced picklists; unknowns -> blank + count warning."""
    # Registration Status
    raw = rec.get("registration_status", "")
    norm = normalize_registration_status(raw)
    if raw and not norm:
        warn_counts["registration_status"] = warn_counts.get("registration_status", 0) + 1
    rec["registration_status"] = norm

    # Lead Type
    raw = rec.get("lead_type", "")
    norm = normalize_lead_type(raw)
    if raw and not norm:
        warn_counts["lead_type"] = warn_counts.get("lead_type", 0) + 1
    rec["lead_type"] = norm

    # Associate Status
    raw = rec.get("associate_status", "")
    norm = normalize_associate_status(raw)
    if raw and not norm:
        warn_counts["associate_status"] = warn_counts.get("associate_status", 0) + 1
    rec["associate_status"] = norm

    # City: accept any text (do not coerce)
    rec["city"] = _clean(rec.get("city", ""))

    return rec, warn_counts

# ----------------------------
# UI Helpers
# ----------------------------
st.set_page_config(page_title="Vanto CRM v3.1 — Contacts", layout="wide")

def page_header():
    st.sidebar.title("Vanto CRM v3.1")
    return st.sidebar.radio("Navigate", [
        "Dashboard", "Contacts", "Orders", "Campaigns",
        "WhatsApp Tools", "Import / Export", "Settings", "Help"
    ])

def global_search_filter(df: pd.DataFrame, query: str):
    if not query.strip():
        return df
    tokens = [t.strip() for t in query.split() if t.strip()]
    if not tokens:
        return df
    # search across key cols
    hay_cols = ["name", "phone", "email", "username", "assigned", "interest",
                "city", "province", "country", "tags",
                "registration_status", "lead_type", "associate_status"]
    hay_cols = [c for c in hay_cols if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for t in tokens:
        tmask_any = pd.Series(False, index=df.index)
        for c in hay_cols:
            tmask_any = tmask_any | df[c].astype(str).str.contains(re.escape(t), case=False, na=False)
        mask = mask & tmask_any
    return df[mask]

# ----------------------------
# PAGES
# ----------------------------
page = page_header()
ensure_schema()

# === Dashboard ===
if page == "Dashboard":
    st.title("Dashboard")
    rows = fetch_contacts()
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["lead_type","registration_status","associate_status","city"])
    # Simple KPIs
    total = len(df)
    reg_counts = df["registration_status"].value_counts(dropna=True) if "registration_status" in df else pd.Series()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("Activated", int(reg_counts.get("Activated", 0)))
    col3.metric("Registered", int(reg_counts.get("Registered", 0)))
    col4.metric("Not Registered", int(reg_counts.get("Not Registered", 0)))
    st.caption("KPI cards use the **Registration Status** field.")

# === Contacts ===
elif page == "Contacts":
    st.title("Contacts")

    # Tabs: Add / View
    tab_add, tab_list = st.tabs(["➕ Add New Contact", "📋 View & Edit"])

    with tab_add:
        with st.form("add_contact"):
            c1, c2, c3 = st.columns(3)
            name  = c1.text_input("Full Name")
            phone = c2.text_input("Phone")
            email = c3.text_input("Email")

            c4, c5, c6, c7 = st.columns(4)
            date = c4.text_input("Date (YYYY-MM-DD)", value="")
            country = c5.text_input("Country", value="")
            province = c6.text_input("Province", value="")
            city = c7.text_input("City (free text)", value="")  # free text per spec

            c8, c9, c10 = st.columns(3)
            registration_status = c8.selectbox("Registration Status (picklist)", [""] + REG_STATUS_ALLOWED, index=0)
            lead_type = c9.selectbox("Lead Type (picklist)", [""] + LEAD_TYPE_ALLOWED, index=0)
            associate_status = c10.selectbox("Associate Status (picklist)", [""] + ASSOC_STATUS_ALLOWED, index=0)

            c11, c12, c13 = st.columns(3)
            source = c11.text_input("Source", value="")
            interest = c12.text_input("Interest", value="")
            assigned = c13.text_input("Assigned To", value="")

            notes = st.text_area("Notes", value="")
            tags = st.text_input("Tags (comma separated)", value="")

            c14, c15, c16, c17 = st.columns(4)
            action_needed = c14.text_input("Next Action (ActionNeeded)", value="")
            action_taken  = c15.text_input("Action Taken", value="")
            username      = c16.text_input("Username (APL Go ID)", value="")
            password      = c17.text_input("Account Password", value="")

            submitted = st.form_submit_button("Save Contact", type="primary")
            if submitted:
                rec = {
                    "name": name, "phone": phone, "email": email,
                    "source": source, "interest": interest, "assigned": assigned,
                    "notes": notes, "tags": tags, "action_needed": action_needed, "action_taken": action_taken,
                    "username": username, "password": password,
                    "date": date, "country": country, "province": province, "city": city,
                    "registration_status": registration_status,
                    "lead_type": lead_type,
                    "associate_status": associate_status,
                }
                # normalize picklists (City remains free)
                warn_counts = {}
                rec, _ = normalize_row_for_picklists(rec, warn_counts)
                insert_one(rec)
                st.success("Saved ✅  (City kept as free text; picklists enforced for the three fields)")

    with tab_list:
        rows = fetch_contacts()
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["id"] + COLUMNS)
        for c in ["id"] + COLUMNS:
            if c not in df.columns: df[c] = ""

        # Search + Filters
        with st.expander("Search & Filters", expanded=True):
            q = st.text_input("Global search (name, phone, email, APL Go ID, location, tags, statuses...)", "")
            colA, colB, colC, colD = st.columns(4)
            f_reg = colA.multiselect("Registration Status", REG_STATUS_ALLOWED)
            f_lead = colB.multiselect("Lead Type", LEAD_TYPE_ALLOWED)
            f_assoc = colC.multiselect("Associate Status", ASSOC_STATUS_ALLOWED)
            f_city_contains = colD.text_input("City contains (free text filter)", "")

        view = df.copy()
        view = global_search_filter(view, q)
        if f_reg:   view = view[view["registration_status"].isin(f_reg)]
        if f_lead:  view = view[view["lead_type"].isin(f_lead)]
        if f_assoc: view = view[view["associate_status"].isin(f_assoc)]
        if f_city_contains.strip():
            view = view[view["city"].astype(str).str.contains(re.escape(f_city_contains.strip()), case=False, na=False)]

        # Show editable table (id fixed as read-only)
        show_cols = ["id"] + [
            "name","phone","email","source","interest","status","tags","assigned","notes",
            "action_needed","action_taken","username","password",
            "date","country","province","city",
            "registration_status","lead_type","associate_status",
        ]
        for c in show_cols:
            if c not in view.columns: view[c] = ""
        st.caption("Tip: edit cells, then click Save below. City stays free text.")
        edited = st.data_editor(view[show_cols], use_container_width=True, num_rows="dynamic", disabled=["id"])
        if st.button("💾 Save table changes"):
            saved = bulk_update_from_editor(edited)
            st.success(f"Saved {saved} rows.")

# === Import / Export ===
elif page == "Import / Export":
    st.title("Import / Export")

    st.subheader("📥 Import Contacts (CSV/XLSX)")
    file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
    if file is not None:
        # Load dataframe
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(file, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

        st.write("1) Map your file columns to CRM fields")
        # Build mapping UI (left = your file headers, right = CRM field)
        human_labels = [
            # identity / common
            ("Name", "name"), ("Phone", "phone"), ("Email", "email"),
            ("Source", "source"), ("Interest", "interest"), ("Status", "status"),
            ("Tags", "tags"), ("Assigned To", "assigned"), ("Notes", "notes"),
            ("ActionNeeded", "action_needed"), ("ActionTaken", "action_taken"),
            ("Username (APL Go ID)", "username"), ("Password", "password"),
            # geo/date
            ("Date", "date"), ("Country", "country"), ("Province", "province"), ("City", "city"),
            # enforced picklists
            ("Registration Status", "registration_status"),
            ("Lead Type", "lead_type"),
            ("Associate Status", "associate_status"),
        ]

        selections = {}
        cols = [""] + list(df.columns)
        grid = st.columns(3)
        for i, (label, key) in enumerate(human_labels):
            # auto‑guess header by case‑insensitive match
            guess = next((c for c in df.columns if c.lower() == label.lower()), "")
            with grid[i % 3]:
                selections[key] = st.selectbox(f"{label}", cols, index=(cols.index(guess) if guess in cols else 0))

        st.write("2) Preview (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Import Now", type="primary"):
            # Build rows according to mapping
            warn_counts = {}
            rows = []
            for _, r in df.iterrows():
                rec = {c: "" for c in COLUMNS}
                for key, src in selections.items():
                    if src and src in df.columns:
                        val = r.get(src, "")
                        rec[key] = "" if pd.isna(val) else str(val).strip()

                # Normalize picklists; City stays free text (no coercion)
                rec, warn_counts = normalize_row_for_picklists(rec, warn_counts)

                rows.append(rec)

            n = bulk_insert(rows)
            st.success(f"Imported {n} rows ✅")

            # Non-blocking warning summary
            if warn_counts:
                msgs = []
                if warn_counts.get("registration_status"): msgs.append(f"{warn_counts['registration_status']} rows had unknown Registration Status → left blank")
                if warn_counts.get("lead_type"): msgs.append(f"{warn_counts['lead_type']} rows had unknown Lead Type → left blank")
                if warn_counts.get("associate_status"): msgs.append(f"{warn_counts['associate_status']} rows had unknown Associate Status → left blank")
                st.warning("Import notes:\n\n- " + "\n- ".join(msgs))

    st.divider()
    st.subheader("📤 Export All Contacts (CSV)")
    if st.button("Download CSV"):
        rows = fetch_contacts()
        exp = pd.DataFrame(rows)[["id"] + COLUMNS] if rows else pd.DataFrame(columns=["id"] + COLUMNS)
        csv_bytes = exp.to_csv(index=False).encode("utf-8")
        st.download_button("Save contacts.csv", data=csv_bytes, file_name=f"contacts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

# === Placeholder pages so nothing is blank ===
elif page == "Orders":
    st.title("Orders")
    st.info("Orders page placeholder (no schema change).")

elif page == "Campaigns":
    st.title("Campaigns")
    st.info("Campaigns page placeholder.")

elif page == "WhatsApp Tools":
    st.title("WhatsApp Tools")
    st.info("Compose & send using your templates. (This page is a placeholder in this file.)")

elif page == "Settings":
    st.title("Settings")
    st.info("Permissions/Roles can be added here later (view vs edit).")

else:  # Help
    st.title("Help")
    st.markdown("""
**What changed (Professional Upgrade)**  
- **City is free text** in forms, inline edits, and import.  
- **Registration Status** picklist: Activated / Registered / Not Registered / **To Be Registered**  
- **Lead Type** picklist: Customer / Distributor  
- **Associate Status** picklist: Promoter / Associate / Builder / Mentor / VIP / Diamond  
- Import leaves unknown picklist values **blank** and shows a **summary warning**.  
- Global search (tokens) across name/phone/email/APL Go ID/location/tags/statuses.  
""")
