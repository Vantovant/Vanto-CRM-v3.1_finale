# app.py
import io
from datetime import date
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

from db import (
    ensure_schema, list_contacts, upsert_many, insert_one, export_all, unique_values,
    update_rows, update_tags, COLUMNS, load_segments, save_segment, delete_segment
)

st.set_page_config(page_title="Vanto CRM v3.1", layout="wide")
ensure_schema()

KEYS = [k for k,_ in COLUMNS]
LABELS = {k: lbl for k,lbl in COLUMNS}

STATUS_OPTIONS = {
    "lead_temperature": ["Hot","Warm","Cold"],
    "communication_status": ["New","In Progress","Pending","Completed"],
    "registration_status": ["Activated","Registered","Not Registered","To Be Registered"],
}

# -------- Sidebar Nav --------
st.sidebar.title("Vanto CRM v3.1")
page = st.sidebar.radio("Navigate", [
    "Dashboard","Contacts","Orders","Campaigns","WhatsApp Tools","Import / Export","Settings","Help"
])

# -------- Helpers --------
def df_from_rows(rows):
    if not rows:
        return pd.DataFrame([{k:"" for k in ["id"]+KEYS}])
    df = pd.DataFrame(rows)
    for k in ["id"]+KEYS:
        if k not in df.columns:
            df[k] = ""
    return df[["id"]+KEYS]

def to_human(df):
    return df.rename(columns=LABELS)

def parse_date(val):
    if not val or str(val).strip()=="" or pd.isna(val):
        return ""
    try:
        return dtparser.parse(str(val)).date().isoformat()
    except Exception:
        return str(val).strip()

def kpi_link(label, value, set_filters):
    c = st.container()
    if c.button(f"{label}\n**{value}**"):
        st.session_state["contacts_filters"] = set_filters
        st.session_state["page"] = "Contacts"
        st.experimental_rerun()

# -------- Dashboard --------
if page == "Dashboard":
    st.title("Dashboard")
    data = df_from_rows(list_contacts())

    total = len(data)
    lt_counts = data["lead_temperature"].value_counts(dropna=False).to_dict()
    cs_counts = data["communication_status"].value_counts(dropna=False).to_dict()
    rs_counts = data["registration_status"].value_counts(dropna=False).to_dict()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Contacts", total)
    c2.metric("Hot", lt_counts.get("Hot",0))
    c3.metric("Warm", lt_counts.get("Warm",0))
    c4.metric("Cold", lt_counts.get("Cold",0))

    r1,r2,r3 = st.columns(3)
    r1.metric("Activated", rs_counts.get("Activated",0))
    r2.metric("Registered", rs_counts.get("Registered",0))
    r3.metric("Not Registered", rs_counts.get("Not Registered",0))

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("New", cs_counts.get("New",0))
    s2.metric("In Progress", cs_counts.get("In Progress",0))
    s3.metric("Pending", cs_counts.get("Pending",0))
    s4.metric("Completed", cs_counts.get("Completed",0))

    st.markdown("#### Recent Contacts")
    st.dataframe(to_human(data[["full_name","phone_number","date_captured","lead_temperature","registration_status","communication_status"]].head(25)), use_container_width=True)

# -------- Contacts --------
elif page == "Contacts":
    st.title("Contacts")

    # Quick Add
    with st.expander("➕ Add New Contact", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        full_name = c1.text_input("Full Name *")
        phone = c2.text_input("Phone Number *")
        email = c3.text_input("Email Address")
        date_captured = c4.text_input("Date Captured (YYYY-MM-DD)", value=date.today().isoformat())

        c5,c6,c7,c8 = st.columns(4)
        country = c5.text_input("Country", value="South Africa")
        province = c6.text_input("Province")
        city = c7.text_input("City")
        state = c8.text_input("State")

        c9,c10,c11 = st.columns(3)
        lt = c9.selectbox("Lead Temperature", STATUS_OPTIONS["lead_temperature"], index=1)
        cs = c10.selectbox("Communication Status", STATUS_OPTIONS["communication_status"], index=0)
        rs = c11.selectbox("Registration Status", STATUS_OPTIONS["registration_status"], index=2)

        c12,c13,c14 = st.columns(3)
        source = c12.text_input("Source")
        interest = c13.text_input("Interest Level")
        assigned = c14.text_input("Assigned To")

        tags = st.text_input("Tags (comma separated)")
        c15,c16,c17,c18 = st.columns(4)
        next_action = c15.text_input("Next Action (YYYY-MM-DD)")
        action_taken = c16.text_input("Action Taken")
        username = c17.text_input("APL Go ID")
        password = c18.text_input("Account Password")
        sponsor = st.text_input("Sponsor Name")
        lead_type = st.selectbox("Lead Type", ["","Customer","Distributor"])
        associate_status = st.selectbox("Associate Status", ["","Promoter","Associate","Builder","Mentor","VIP","Diamond"])

        if st.button("Save Contact", type="primary"):
            if not full_name.strip() or not phone.strip():
                st.error("Full Name and Phone Number are required.")
            else:
                rec = {
                    "full_name": full_name.strip(),
                    "phone_number": phone.strip(),
                    "email_address": email.strip(),
                    "date_captured": parse_date(date_captured),
                    "country": country.strip(),
                    "province": province.strip(),
                    "city": city.strip(),
                    "state": state.strip(),
                    "lead_temperature": lt,
                    "communication_status": cs,
                    "registration_status": rs,
                    "source": source.strip(),
                    "interest_level": interest.strip(),
                    "assigned_to": assigned.strip(),
                    "tags": tags.strip(),
                    "next_action": parse_date(next_action),
                    "action_taken": action_taken.strip(),
                    "username": username.strip(),
                    "account_password": password.strip(),
                    "sponsor_name": sponsor.strip(),
                    "lead_type": lead_type.strip(),
                    "associate_status": associate_status.strip(),
                    "apl_go_id": username.strip() if username else "",
                }
                insert_one(rec)
                st.success("Contact added ✅")
                st.experimental_rerun()

    # Search + Filters
    search = st.text_input("Global search (name / phone / email / sponsor / APL Go ID / city / province / country / interest / tags)", value="")

    # Load saved segments
    with st.expander("Saved segments"):
        segs = load_segments()
        col_s1, col_s2 = st.columns([2,1])
        with col_s1:
            if segs:
                chosen = st.selectbox("Apply segment", [""] + list(segs.keys()))
                if chosen:
                    st.session_state["contacts_filters"] = segs[chosen]
                    st.success(f"Segment '{chosen}' applied. Scroll down to see results.")
            else:
                st.caption("No segments saved yet.")
        with col_s2:
            seg_name = st.text_input("New segment name")
            if st.button("Save current filters as segment", disabled=not seg_name.strip()):
                # we'll store current filters after they are built below
                st.session_state["save_segment_name"] = seg_name.strip()

    # Build filters UI
    rows_all = list_contacts()
    df_all = df_from_rows(rows_all)
    c1,c2,c3,c4 = st.columns(4)
    f_lt = c1.multiselect("Lead Temperature", STATUS_OPTIONS["lead_temperature"])
    f_cs = c2.multiselect("Communication Status", STATUS_OPTIONS["communication_status"])
    f_rs = c3.multiselect("Registration Status", STATUS_OPTIONS["registration_status"])
    f_country = c4.multiselect("Country", sorted(df_all["country"].dropna().unique().tolist()))

    c5,c6,c7,c8 = st.columns(4)
    f_prov = c5.multiselect("Province", sorted(df_all["province"].dropna().unique().tolist()))
    f_city = c6.multiselect("City", sorted(df_all["city"].dropna().unique().tolist()))
    f_assigned = c7.multiselect("Assigned To", sorted(df_all["assigned_to"].dropna().unique().tolist()))
    f_sponsor = c8.multiselect("Sponsor Name", sorted(df_all["sponsor_name"].dropna().unique().tolist()))

    # Use session_state preset filters (from KPIs or segments)
    preset = st.session_state.get("contacts_filters", {})
    filters = {}
    if f_lt: filters["lead_temperature"] = f_lt
    if f_cs: filters["communication_status"] = f_cs
    if f_rs: filters["registration_status"] = f_rs
    if f_country: filters["country"] = f_country
    if f_prov: filters["province"] = f_prov
    if f_city: filters["city"] = f_city
    if f_assigned: filters["assigned_to"] = f_assigned
    if f_sponsor: filters["sponsor_name"] = f_sponsor
    if preset:
        # merge preset (do not overwrite explicit UI choices)
        for k,v in preset.items():
            filters.setdefault(k, v)

    # Save segment now if requested
    if "save_segment_name" in st.session_state:
        save_segment(st.session_state.pop("save_segment_name"), filters)
        st.success("Segment saved.")

    rows = list_contacts(filters=filters, search_query=search)
    df = df_from_rows(rows)

    # Inline editor
    show_cols = ["full_name","phone_number","email_address","lead_temperature","communication_status","registration_status",
                 "assigned_to","sponsor_name","lead_type","associate_status","date_captured","country","province","city","state",
                 "tags","next_action","action_taken","apl_go_id","account_password","source","interest_level"]
    for c in show_cols:
        if c not in df.columns: df[c] = ""
    st.caption("Tip: Edit cells below, then click **Save table changes**.")
    edited = st.data_editor(df[["id"]+show_cols], use_container_width=True, num_rows="fixed", disabled={"id": True}, key="contacts_editor")

    if st.button("Save table changes", type="primary"):
        n = update_rows(edited.to_dict("records"))
        st.success(f"Saved {n} updates.")
        st.experimental_rerun()

    # Bulk tags
    with st.expander("Bulk: Tags"):
        id_str = st.text_input("IDs to update (comma-separated)", "")
        ids = []
        if id_str.strip():
            ids = [int(x.strip()) for x in id_str.split(",") if x.strip().isdigit()]
        add_t = st.text_input("Add tags (comma-separated)")
        rem_t = st.text_input("Remove tags (comma-separated)")
        if st.button("Apply tag changes", disabled=not ids):
            add = [t.strip() for t in add_t.split(",")] if add_t.strip() else None
            rem = [t.strip() for t in rem_t.split(",")] if rem_t.strip() else None
            update_tags(ids, add_tags=add, remove_tags=rem)
            st.success("Tags updated.")

# -------- Orders --------
elif page == "Orders":
    st.title("Orders")
    st.info("Orders module placeholder.")

# -------- Campaigns --------
elif page == "Campaigns":
    st.title("Campaigns")
    st.info("Campaigns module placeholder.")

# -------- WhatsApp Tools --------
elif page == "WhatsApp Tools":
    st.title("WhatsApp Tools")
    st.info("Select a contact in Contacts and use your template in a future version.")

# -------- Import / Export --------
elif page == "Import / Export":
    st.title("Import / Export")
    st.subheader("Import Contacts")
    up = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_in = pd.read_csv(up, dtype=str, keep_default_na=False)
        else:
            df_in = pd.read_excel(up, dtype=str)
        df_in.columns = [str(c).strip() for c in df_in.columns]

        # mapping UI (try best guess by label)
        st.caption("Step 1: Map your headers to CRM fields")
        label_by_key = {k: lbl for k,lbl in COLUMNS}
        mapping = {}
        for k,lbl in COLUMNS:
            options = [""] + list(df_in.columns)
            guess = ""
            if lbl in df_in.columns:
                guess = lbl
            else:
                for c in df_in.columns:
                    if c.lower().replace(" ","_") == k:
                        guess = c
                        break
            mapping[lbl] = st.selectbox(lbl, options, index=(options.index(guess) if guess in options else 0))

        st.caption("Step 2: Preview first 10 rows")
        st.dataframe(df_in.head(10), use_container_width=True)

        if st.button("Import Now", type="primary"):
            inv = {v:k for k,v in mapping.items() if v}
            rows = []
            warn_counts = {}
            for _, r in df_in.iterrows():
                rec = {k:"" for k,_ in COLUMNS}
                for lbl, src in mapping.items():
                    if not src: continue
                    key = [kk for kk,ll in COLUMNS if ll==lbl][0]
                    val = r.get(src, "")
                    if "Date" in lbl:
                        try:
                            val = dtparser.parse(str(val)).date().isoformat()
                        except Exception:
                            val = str(val)
                    rec[key] = "" if pd.isna(val) else str(val).strip()
                rec, warn_counts = normalize_picklists(rec, warn_counts)
                rows.append(rec)
            n = upsert_many(rows)
            # surface non-blocking warnings
            msg = []
            if warn_counts.get('registration_status'): msg.append(f"{warn_counts['registration_status']} rows had unknown Registration Status → left blank")
            if warn_counts.get('lead_type'): msg.append(f"{warn_counts['lead_type']} rows had unknown Lead Type → left blank")
            if warn_counts.get('associate_status'): msg.append(f"{warn_counts['associate_status']} rows had unknown Associate Status → left blank")
            if msg:
                st.warning("; ".join(msg))
            st.success(f"Imported {n} contacts.")

    st.divider()
    st.subheader("Export All Contacts (CSV)")
    if st.button("Download CSV"):
        rows = export_all()
        df_out = df_from_rows(rows).drop(columns=["id"], errors="ignore")
        df_out = to_human(df_out)
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Save v3_export.csv", data=csv, file_name="v3_export.csv", mime="text/csv")

# -------- Settings --------
elif page == "Settings":
    st.title("Settings")
    st.info("Viewer/Editor roles and auth can be added in v3.2.")

# -------- Help --------
else:
    st.title("Help")
    st.markdown("""
**What’s included**
- 20 columns + Tags (Date, Country, Province, City, Lead Temperature, Communication Status, Registration Status, etc.)
- Global search (token-based) across name, phone, email, sponsor, APL Go ID, location, interest, tags
- Add New Contact form
- Inline editing & bulk tags
- Import / Export with mapping UI
- Saved segments (create and apply in Contacts)

**Next ideas**
- Saved KPI links to auto-apply filters
- Dedupe on import by APL Go ID
- Multi-user auth (Supabase) and cloud sync
""")
