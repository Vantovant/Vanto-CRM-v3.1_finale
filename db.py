# db.py
import sqlite3, json
from pathlib import Path

DB_PATH = Path(__file__).with_name("crm.sqlite3")
SEGMENTS_PATH = Path(__file__).with_name("segments.json")

# 20 agreed columns (snake keys ↔ labels) + optional tags
COLUMNS = [
    ("date_captured", "Date Captured"),
    ("state", "State"),
    ("country", "Country"),
    ("province", "Province"),
    ("city", "City"),
    ("full_name", "Full Name"),
    ("phone_number", "Phone Number"),
    ("interest_level", "Interest Level"),
    ("assigned_to", "Assigned To"),
    ("action_taken", "Action Taken"),
    ("next_action", "Next Action"),
    ("lead_temperature", "Lead Temperature"),
    ("communication_status", "Communication Status"),
    ("sponsor_name", "Sponsor Name"),
    ("lead_type", "Lead Type"),
    ("associate_status", "Associate Status"),
    ("registration_status", "Registration Status"),
    ("apl_go_id", "APL Go ID"),
    ("account_password", "Account Password"),
    ("email_address", "Email Address"),
    ("tags", "Tags"),
]

TABLE = "contacts"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema():
    conn = get_conn()
    cur = conn.cursor()
    cols_sql = ", ".join([f"{k} TEXT" for k, _ in COLUMNS])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          {cols_sql},
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT
        );
    """)
    cur.execute(f"PRAGMA table_info({TABLE})")
    existing = {row["name"] for row in cur.fetchall()}
    for k, _ in COLUMNS:
        if k not in existing:
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {k} TEXT;")
    conn.commit()
    conn.close()
    # ensure segments file
    if not SEGMENTS_PATH.exists():
        SEGMENTS_PATH.write_text("{}", encoding="utf-8")

def list_contacts(filters=None, search_query=""):
    ensure_schema()
    conn = get_conn()
    conn.create_function("like_nocase", 2, lambda a,b: (a or "").lower().find((b or "").lower()) != -1)
    cur = conn.cursor()
    where = []
    params = []
    if filters:
        for k,v in filters.items():
            if not v: 
                continue
            if isinstance(v, (list,tuple)) and v:
                place = ",".join(["?"]*len(v))
                where.append(f"{k} IN ({place})")
                params.extend(v)
            else:
                where.append(f"{k} = ?")
                params.append(v)
    if search_query:
        tokens = [t for t in search_query.strip().split() if t]
        for t in tokens:
            where.append("""(
                like_nocase(full_name, ?) OR like_nocase(phone_number, ?) OR like_nocase(email_address, ?) OR
                like_nocase(sponsor_name, ?) OR like_nocase(apl_go_id, ?) OR
                like_nocase(city, ?) OR like_nocase(province, ?) OR like_nocase(country, ?) OR
                like_nocase(interest_level, ?) OR like_nocase(tags, ?)
            )""")
            params.extend([t]*10)
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    cur.execute(f"SELECT * FROM {TABLE}{where_sql} ORDER BY id DESC", params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def unique_values(column):
    ensure_schema()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {column} AS v FROM {TABLE} WHERE {column} IS NOT NULL AND {column} <> '' ORDER BY 1 ASC")
    vals = [r["v"] for r in cur.fetchall()]
    conn.close()
    return vals

def upsert_many(rows):
    ensure_schema()
    if not rows:
        return 0
    conn = get_conn()
    cur = conn.cursor()
    keys = [k for k,_ in COLUMNS]
    q = ",".join(["?"]*len(keys))
    sql = f"INSERT INTO {TABLE} ({','.join(keys)}) VALUES ({q})"
    data = []
    for r in rows:
        data.append([r.get(k,"") for k in keys])
    cur.executemany(sql, data)
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n

def insert_one(row):
    return upsert_many([row])

def update_rows(rows):
    ensure_schema()
    if not rows: return 0
    conn = get_conn()
    cur = conn.cursor()
    keys = [k for k,_ in COLUMNS]
    count = 0
    for r in rows:
        rid = r.get("id")
        if not rid: 
            continue
        fields = [k for k in keys if k in r]
        if not fields:
            continue
        set_clause = ", ".join([f"{k}=?" for k in fields])
        params = [r.get(k,"") for k in fields] + [rid]
        cur.execute(f"UPDATE {TABLE} SET {set_clause}, updated_at=datetime('now') WHERE id=?", params)
        count += cur.rowcount
    conn.commit()
    conn.close()
    return count

def update_tags(ids, add_tags=None, remove_tags=None):
    ensure_schema()
    conn = get_conn()
    cur = conn.cursor()
    for rid in ids:
        row = cur.execute(f"SELECT tags FROM {TABLE} WHERE id=?", (rid,)).fetchone()
        current = set()
        if row and row["tags"]:
            current = {t.strip() for t in str(row["tags"]).split(",") if t.strip()}
        if add_tags:
            for t in add_tags:
                if t.strip(): current.add(t.strip())
        if remove_tags:
            for t in remove_tags:
                current.discard(t.strip())
        newv = ", ".join(sorted(current))
        cur.execute(f"UPDATE {TABLE} SET tags=?, updated_at=datetime('now') WHERE id=?", (newv, rid))
    conn.commit()
    conn.close()

def export_all():
    ensure_schema()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {TABLE} ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# Saved segments (simple JSON file)
def load_segments():
    if not SEGMENTS_PATH.exists(): return {}
    try:
        return json.loads(SEGMENTS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}

def save_segment(name, filters):
    segs = load_segments()
    segs[name] = filters
    SEGMENTS_PATH.write_text(json.dumps(segs, indent=2), encoding='utf-8')

def delete_segment(name):
    segs = load_segments()
    if name in segs:
        del segs[name]
        SEGMENTS_PATH.write_text(json.dumps(segs, indent=2), encoding='utf-8')
