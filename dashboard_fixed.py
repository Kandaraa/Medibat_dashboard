import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from datetime import datetime
import os
import time

# Import du module d'intégration Gemini
try:
    import gemini_integration
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Operations Dashboard", layout="wide", page_icon="📊")

# -----------------------------
# Utilities
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("\n", " ").replace("\r", " ").strip() for c in df.columns]
    return df

def safe_parse_date(series):
    """Smart date parser that tries multiple formats"""
    if series is None or len(series) == 0:
        return pd.Series([pd.NaT] * len(series))
    
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    
    # Try different date formats
    formats = [
        '%d/%m/%Y',  # DD/MM/YYYY
        '%d-%m-%Y',  # DD-MM-YYYY
        '%Y-%m-%d',  # YYYY-MM-DD
        '%m/%d/%Y',  # MM/DD/YYYY
        '%d.%m.%Y',  # DD.MM.YYYY
    ]
    
    for fmt in formats:
        mask = result.isna()
        if not mask.any():
            break
        try:
            result[mask] = pd.to_datetime(series[mask], format=fmt, errors='coerce')
        except:
            continue
    
    # Final attempt with dayfirst=True
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], dayfirst=True, errors='coerce')
    
    return result

def months_diff(d2, d1):
    """Calculate exact month difference"""
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    rd = relativedelta(d2, d1)
    return rd.years * 12 + rd.months + rd.days/30.437

def pct(n, d):
    return (n / d * 100.0) if d and d > 0 else 0.0

# -----------------------------
# Data Loading
# -----------------------------
#st.sidebar.title("📂 Data Sources")

# Original French file paths (commented out)
# default_paths = {
#     "etat_index": "etat Index.csv",
#     "index_hours": "index hours.csv",
#     "fluids_conf": "Recommended Fluids and Conformity.csv",
#     "maintenance": "SUIVI DE MAINTENANCE - Copy.csv",
#     "causes": "categorie panne.csv",
#     "vidange": "suivie vidange.csv",
# }

# English translated file paths from data folder
default_paths = {
    "etat_index": "data/etat Index_eng.csv",
    "index_hours": "data/index hours_eng.csv",
    "fluids_conf": "data/Recommended Fluids and Conformity_eng.csv",
    "maintenance": "data/SUIVI DE MAINTENANCE - Copy_eng.csv",
    "causes": "data/categorie panne_eng.csv",
    "vidange": "data/suivie vidange_eng.csv",
}

@st.cache_data(show_spinner=True)
def load_csv(fallback_path, encoding_list=("utf-8", "latin1", "ISO-8859-1", "cp1252")):
    for enc in encoding_list:
        try:
            df = pd.read_csv(fallback_path, encoding=enc)
            return normalize_columns(df), f"{fallback_path} ({enc})"
        except Exception as e:
            continue
    raise RuntimeError(f"Unable to read {fallback_path}")

# Load all data
df_index, src1 = load_csv(default_paths["etat_index"])
df_hours, src2 = load_csv(default_paths["index_hours"])
df_conf, src3 = load_csv(default_paths["fluids_conf"])
df_maint, src4 = load_csv(default_paths["maintenance"])
df_causes, src5 = load_csv(default_paths["causes"])
df_vid, src6 = load_csv(default_paths["vidange"])

#st.sidebar.success("✅ Data loaded successfully")

# -----------------------------
# Helper: Find columns
# -----------------------------
def find_col(df, targets):
    cols = df.columns.tolist()
    for t in targets:
        for c in cols:
            if c.strip().lower() == t.strip().lower():
                return c
    for t in targets:
        for c in cols:
            if t.strip().lower() in c.strip().lower():
                return c
    return None

# -----------------------------
# KPI 1: Etat Index
# -----------------------------
def kpi_index_counts(df):
    sub = df.copy()
    
    # Find columns
    c1 = find_col(df, ["INDEX   (groupe whatsapp Gazoil)", "INDEX (groupe whatsapp Gazoil)"])
    c2 = find_col(df, ["Index Tableau Vidange (31/07)", "Index Tableau Vidange"])
    c3 = find_col(df, ["Etat NEBIL"])
    c_design = find_col(df, ["Désignation", "Désignation ", "Designation"])
    
    # Initialize masks
    is_panne = pd.Series([False] * len(sub), index=sub.index)
    is_marche = pd.Series([False] * len(sub), index=sub.index)
    
    # Check for panne and marche (only check non-empty cells)
    for c in [c1, c2, c3]:
        if c and c in sub.columns:
            col_str = sub[c].astype(str).str.strip().str.lower()
            # Only check non-empty/non-nan cells
            not_empty = ~sub[c].isna() & (col_str != '') & (col_str != 'nan')
            # Panne: contains exactly "p" in filled cells
            is_panne = is_panne | (not_empty & (col_str == 'p'))
            # Marche: contains "oui" in filled cells
            is_marche = is_marche | (not_empty & (col_str == 'oui'))
    
    # Verifier: ALL filled columns are "no info" (ignore empty/nan columns)
    is_verif = pd.Series([False] * len(sub), index=sub.index)
    
    for idx in sub.index:
        filled_cols = []
        no_info_count = 0
        
        for c in [c1, c2, c3]:
            if c and c in sub.columns:
                val = str(sub.loc[idx, c]).strip().lower()
                # Check if cell is filled (not empty, not nan)
                if not pd.isna(sub.loc[idx, c]) and val != '' and val != 'nan':
                    filled_cols.append(c)
                    if val == 'no info':
                        no_info_count += 1
        
        # Only mark as "à vérifier" if there's at least one filled column and ALL filled columns are "no info"
        if len(filled_cols) > 0 and no_info_count == len(filled_cols):
            is_verif.loc[idx] = True
    
    total = len(sub)
    n_panne = int(is_panne.sum())
    n_marche = int(is_marche.sum())
    n_verif = int(is_verif.sum())
    
    return {
        "total": total,
        "panne": n_panne,
        "marche": n_marche,
        "verifier": n_verif,
        "is_panne_mask": is_panne,
        "is_marche_mask": is_marche,
        "is_verif_mask": is_verif,
        "col_design": c_design
    }

idx_stats = kpi_index_counts(df_index)

# -----------------------------
# KPI 2: Index Hours
# -----------------------------
def kpi_hours(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # Find "Prochain" column (5th column or by name)
    col_prochain = cols[4] if len(cols) >= 5 else None
    for c in cols:
        if str(c).strip().lower() == "prochain":
            col_prochain = c
    
    col_a = cols[0] if len(cols) > 0 else None
    
    if col_prochain is None or col_prochain not in dfc.columns:
        return {"global_avg": np.nan, "by_cat": pd.DataFrame()}
        
    # Filter out "camion" category BEFORE any calculations
    if col_a and col_a in dfc.columns:
     dfc = dfc[dfc[col_a].astype(str).str.lower() != "camion"]
    
    # Convert to numeric
    x = pd.to_numeric(dfc[col_prochain], errors="coerce")
    valid = ~x.isna()
    
    if valid.sum() == 0:
        return {"global_avg": np.nan, "by_cat": pd.DataFrame()}
    
    # Calculate per row: value / 250 / 12
    per_row_values = x[valid] / 250.0 / 12.0
    
    # Global average: sum / count
    global_avg = per_row_values.sum() / valid.sum()
    
    # By category
    by_cat = pd.DataFrame()
    if col_a and col_a in dfc.columns:
        tmp = dfc.loc[valid].copy()
        tmp["per_row"] = per_row_values.values
        grouped = tmp.groupby(col_a, dropna=False)["per_row"].agg(["sum", "count"]).reset_index()
        grouped["avg_per_year"] = grouped["sum"] / grouped["count"]
        by_cat = grouped
    
    return {
        "global_avg": global_avg,
        "by_cat": by_cat,
        "col_prochain": col_prochain,
        "col_cat": col_a
    }

hours_stats = kpi_hours(df_hours)

# -----------------------------
# KPI 3: Fluids Conformity
# -----------------------------
def kpi_conformity(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # Find conformité column (5th column or by name)
    col_conf = cols[4] if len(cols) >= 5 else None
    for c in cols:
        if "conformité" in str(c).strip().lower() or "conformite" in str(c).strip().lower():
            col_conf = c
    
    if col_conf is None or col_conf not in dfc.columns:
        return {"pct_conf": 0.0, "pct_partielle": 0.0, "total": 0, "conf_count": 0, "part_count": 0}
    
    total = len(dfc)
    col_str = dfc[col_conf].astype(str).str.strip().str.lower()
    
    conformes = (col_str == "conforme").sum()
    partielles = (col_str == "partielle").sum()
    
    return {
        "pct_conf": pct(conformes, total),
        "pct_partielle": pct(partielles, total),
        "total": total,
        "conf_count": conformes,
        "part_count": partielles,
        "col_conf": col_conf
    }

conf_stats = kpi_conformity(df_conf)

# -----------------------------
# KPI 4: Maintenance
# -----------------------------
def kpi_maintenanceV1(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # According to the description and CSV structure:
    # Column 11 (index 10): Date de Détection
    # Column 13 (index 12): Date prévue d'intervention
    # Column 14 (index 13): Date fin d'Intervention
    c_detect = cols[10] if len(cols) >= 11 else None
    c_prev = cols[13] if len(cols) >= 14 else None
    c_fin = cols[15] if len(cols) >= 16 else None
    
    # Try by name as fallback
    for c in cols:
        low = str(c).strip().lower()
        if "date de détection" in low or "date de detection" in low:
            c_detect = c
        if "date prévue d'intervention" in low or "date prevue d'intervention" in low:
            c_prev = c
        if "date fin" in low and "intervention" in low:
            c_fin = c
    
    # Parse dates
    d_detect = safe_parse_date(dfc[c_detect]) if c_detect and c_detect in dfc.columns else pd.Series([pd.NaT]*len(dfc))
    d_prev = safe_parse_date(dfc[c_prev]) if c_prev and c_prev in dfc.columns else pd.Series([pd.NaT]*len(dfc))
    d_fin = safe_parse_date(dfc[c_fin]) if c_fin and c_fin in dfc.columns else pd.Series([pd.NaT]*len(dfc))
    
    # Misplanning: date prévue ≠ date fin (only for valid dates)
    misplan = (d_prev != d_fin) & (~d_prev.isna()) & (~d_fin.isna())
    n_misplan = int(misplan.sum())
    
    # Duration: date fin - date detection (in days)
    # Filter out negative durations (data errors)
    dur_days = (d_fin - d_detect).dt.days
    valid_durations = dur_days[dur_days >= 0].dropna()
    avg_duration = float(valid_durations.mean()) if len(valid_durations) > 0 else np.nan
    
    return {
        "n_misplan": n_misplan,
        "avg_duration_days": avg_duration,
        "durations": dur_days,
        "valid_durations": valid_durations,
        "cols": {"detect": c_detect, "prev": c_prev, "fin": c_fin}
    }


#maint_stats = kpi_maintenance(df_maint)



def kpi_maintenance(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # According to the CSV structure after normalize_columns:
    # Column 11 (index 10): Date de Détection
    # Column 14 (index 13): Date prévue d'intervention
    # Column 16 (index 15): Date fin d'Intervention
    c_detect = cols[10] if len(cols) >= 11 else None
    c_prev   = cols[13] if len(cols) >= 14 else None
    c_fin    = cols[15] if len(cols) >= 16 else None
    

    
    # Parse dates
    d_detect = safe_parse_date(dfc[c_detect]) if (c_detect in dfc.columns) else pd.Series([pd.NaT]*len(dfc))
    d_prev   = safe_parse_date(dfc[c_prev])   if (c_prev   in dfc.columns) else pd.Series([pd.NaT]*len(dfc))
    d_fin    = safe_parse_date(dfc[c_fin])    if (c_fin    in dfc.columns) else pd.Series([pd.NaT]*len(dfc))
    
    # Misplanning: date prévue ≠ date fin (valid dates only)
    misplan = (d_prev != d_fin) & (~d_prev.isna()) & (~d_fin.isna())
    n_misplan = int(misplan.sum())
    
    # Duration: index 15 − index 10 = (Date fin d’Intervention - Date de Détection) in days
    dur_days = (d_fin - d_detect).dt.days
    valid_durations = dur_days[(~dur_days.isna()) & (dur_days >= 0)].dropna()
    avg_duration = float(valid_durations.mean()) if len(valid_durations) > 0 else np.nan
    
    return {
        "n_misplan": n_misplan,
        "avg_duration_days": avg_duration,
        "durations": dur_days,
        "valid_durations": valid_durations,
        "cols": {"detect": c_detect, "prev": c_prev, "fin": c_fin}
    }

maint_stats = kpi_maintenance(df_maint)

# ---------- Helpers ----------
def parse_any_date(s):
    import pandas as pd
    x = pd.to_datetime(s, errors="coerce", dayfirst=True)
    mask_num = x.isna() & pd.to_numeric(s, errors="coerce").notna()
    if mask_num.any():
        x.loc[mask_num] = pd.to_datetime(pd.to_numeric(s[mask_num]),
                                         unit="D", origin="1899-12-30", errors="coerce")
    return x

def compute_durations(df):
    cols = df.columns.tolist()
    # Détection / Fin — ajuste les indices si besoin
    c_detect = cols[10]                 # Date de Détection
    c_fin    = cols[15] 

    #d_detect = parse_any_date(df[c_detect]) if c_detect else pd.NaT
    #d_fin    = parse_any_date(df[c_fin])    if c_fin    else pd.NaT

    #dur = (d_fin - d_detect).dt.days
        # Parse dates
    d_detect = safe_parse_date(df[c_detect]) if (c_detect in df.columns) else pd.Series([pd.NaT]*len(df))
    d_fin    = safe_parse_date(df[c_fin])    if (c_fin    in df.columns) else pd.Series([pd.NaT]*len(df))
        # Duration: index 15 − index 10 = (Date fin d’Intervention - Date de Détection) in days
    dur = (d_fin - d_detect).dt.days
    # Nettoyage logique: >=0 et < 3 ans (évite les erreurs de saisie)
    valid = dur.notna() & (dur >= 0) & (dur <= 365*3)
    return dur[valid]

# ---------- Core KPI ----------
durations = compute_durations(df_maint)



# -----------------------------
# KPI 5: Causes
# -----------------------------
def kpi_causes(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # Find columns (9th = index 8, 2nd = index 1)
    c_cause = cols[8] if len(cols) >= 9 else None
    c_des = cols[1] if len(cols) >= 2 else None
    
    for c in cols:
        low = str(c).strip().lower()
        if "classification root cause" in low:
            c_cause = c
        if low in ["désignation", "désignation ", "designation"]:
            c_des = c
    
    if c_cause is None or c_cause not in dfc.columns:
        return {"pct_tbl": pd.DataFrame(), "by_engine": None}
    
    # Filter out NaN/empty values
    dfc_valid = dfc[~dfc[c_cause].isna()].copy()
    dfc_valid = dfc_valid[dfc_valid[c_cause].astype(str).str.strip() != ''].copy()
    dfc_valid = dfc_valid[dfc_valid[c_cause].astype(str).str.strip().str.lower() != 'nan'].copy()
    
    total = len(dfc_valid)
    
    if total == 0:
        return {"pct_tbl": pd.DataFrame(), "by_engine": None}
    
    dist = dfc_valid[c_cause].astype(str).str.strip().value_counts(dropna=True)
    pct_tbl = (dist / total * 100.0).reset_index()
    pct_tbl.columns = ["cause", "pct"]
    
    # By engine (also filter out NaN)
    by_eng = None
    if c_des and c_des in dfc_valid.columns:
        tmp = dfc_valid[[c_des, c_cause]].copy()
        # Also filter out NaN in designation column
        tmp = tmp[~tmp[c_des].isna()].copy()
        tmp = tmp[tmp[c_des].astype(str).str.strip() != ''].copy()
        
        if len(tmp) > 0:
            tmp["count"] = 1
            by = tmp.groupby([c_des, c_cause], dropna=True)["count"].sum().reset_index()
            totals = by.groupby(c_des)["count"].transform("sum")
            by["pct"] = by["count"] / totals * 100.0
            by_eng = by
    
    return {"pct_tbl": pct_tbl, "by_engine": by_eng, "c_cause": c_cause, "c_des": c_des}

cause_stats = kpi_causes(df_causes)

# -----------------------------
# KPI 7: Catégories de panne (Disponibilité, Coût, Complexité)
# -----------------------------
def kpi_categories_panne(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # Find columns for the three categories
    c_disponibilite = None
    c_cout = None
    c_complexite = None
    c_des = None
    
    # Find designation column
    for c in cols:
        low = str(c).strip().lower()
        if low in ["désignation", "désignation ", "designation"]:
            c_des = c
    
    # Find the specific columns by index or name
    if len(cols) >= 31:
        c_disponibilite = cols[29]  # Column 30 (0-indexed)
        c_cout = cols[30]           # Column 31 (0-indexed)
        c_complexite = cols[31]     # Column 32 (0-indexed)
    
    # Try to find by name as fallback
    for c in cols:
        low = str(c).strip().lower()
        if "catégorie de panne en terme de disponibilité" in low:
            c_disponibilite = c
        if "catégorie de panne en terme de cout" in low:
            c_cout = c
        if "catégorie de panne en terme de complexité" in low:
            c_complexite = c
    
    result = {
        "disponibilite": {"pct": pd.DataFrame(), "by_engine": None},
        "cout": {"pct": pd.DataFrame(), "by_engine": None},
        "complexite": {"pct": pd.DataFrame(), "by_engine": None},
        "columns": {"disponibilite": c_disponibilite, "cout": c_cout, "complexite": c_complexite, "designation": c_des}
    }
    
    # Process each category
    for cat_name, col in [("disponibilite", c_disponibilite), ("cout", c_cout), ("complexite", c_complexite)]:
        if col is None or col not in dfc.columns:
            continue
        
        # Filter out NaN/empty values
        dfc_valid = dfc[~dfc[col].isna()].copy()
        dfc_valid = dfc_valid[dfc_valid[col].astype(str).str.strip() != ''].copy()
        dfc_valid = dfc_valid[dfc_valid[col].astype(str).str.strip().str.lower() != 'nan'].copy()
        
        total = len(dfc_valid)
        
        if total == 0:
            continue
        
        # Calculate percentages - convert to lowercase to make case insensitive
        # Convert values to lowercase for case-insensitive grouping
        dfc_valid[col + "_lower"] = dfc_valid[col].astype(str).str.strip().str.lower()
        dist = dfc_valid[col + "_lower"].value_counts(dropna=True)
        pct_tbl = (dist / total * 100.0).reset_index()
        pct_tbl.columns = ["classe", "pct"]
        result[cat_name]["pct"] = pct_tbl
        
        # By engine (also filter out NaN)
        if c_des and c_des in dfc_valid.columns:
            tmp = dfc_valid[[c_des, col, col + "_lower"]].copy()
            # Also filter out NaN in designation column
            tmp = tmp[~tmp[c_des].isna()].copy()
            tmp = tmp[tmp[c_des].astype(str).str.strip() != ''].copy()
            
            if len(tmp) > 0:
                tmp["count"] = 1
                # Use the lowercase version for grouping to make it case insensitive
                by = tmp.groupby([c_des, col + "_lower"], dropna=True)["count"].sum().reset_index()
                # Rename the column back to original name for consistency in output
                by.rename(columns={col + "_lower": col}, inplace=True)
                result[cat_name]["by_engine"] = by
    
    return result

categories_stats = kpi_categories_panne(df_maint)

# -----------------------------
# KPI 6: Vidange Planning
# -----------------------------
def kpi_vidange(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # Find columns (2nd = index 1, 9th = index 8)
    c_next = cols[1] if len(cols) >= 2 else None
    c_last = cols[8] if len(cols) >= 9 else None
    
    for c in cols:
        low = str(c).strip().lower()
        if "date prochaine vidange" in low:
            c_next = c
        if "date dernière vidange" in low or "date derniere vidange" in low:
            c_last = c
    
    next_d = safe_parse_date(dfc[c_next]) if c_next and c_next in dfc.columns else pd.Series([pd.NaT]*len(dfc))
    last_d = safe_parse_date(dfc[c_last]) if c_last and c_last in dfc.columns else pd.Series([pd.NaT]*len(dfc))
    
    # Calculate months difference
    months = pd.Series([months_diff(nd, ld) for nd, ld in zip(next_d, last_d)], index=dfc.index)
    
    # Categories
    respected_mask = (months < 3) & (~months.isna())
    cat_yellow = (months >= 3) & (months <= 6)
    cat_orange = (months > 6) & (months <= 12)
    cat_red = (months > 12)
    
    return {
        "respected": int(respected_mask.sum()),
        "yellow": int(cat_yellow.sum()),
        "orange": int(cat_orange.sum()),
        "red": int(cat_red.sum()),
        "months": months,
    }

vid_stats = kpi_vidange(df_vid)

# -----------------------------
# Dashboard Layout
# -----------------------------
st.title("📊 Operations & Maintenance Dashboard")
st.caption("Accurate • Organized • Insightful • Production-Ready")

# Executive Summary
st.subheader("Executive Summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    panne_pct   = pct(idx_stats["panne"],   idx_stats["total"])
    st.metric("% Index down", f"{panne_pct:.1f}%")
    #st.metric("Index down", idx_stats["panne"])
with c2:
        st.metric("Overall Annual Average - Preventive Maintenance", 
             f"{hours_stats['global_avg']:.3f}" if not np.isnan(hours_stats["global_avg"]) else "N/A")
with c3:
    st.markdown("<div style='font-size:16px;'>Most vehicle requires preventive maintenance</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:black;'>Garder:</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:28px;font-weight:bold;margin-top:-5px;'>10.49 <span style='font-size:16px;'>annual interventions</span></div>", unsafe_allow_html=True)

with c4:
       st.metric("% Off-Schedule Maintenance", "32.7%")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("% Compliant Lubrication", f"{conf_stats['pct_conf']:.1f}%")
with c6:
    st.markdown("<div style='font-size:16px;'>Highest recurring breakdown root cause</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:black;'>Wear failure:</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:32px;font-weight:bold;margin-top:-5px;'>53.75%</div>", unsafe_allow_html=True)
with c7:
    st.markdown("<div style='font-size:16px;'>The longest unrepaired vehicle</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:black;'>Bulldozer (BD2):</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:32px;font-weight:bold;margin-top:-5px;'>642 days</div>", unsafe_allow_html=True)
with c8:
    avg_dur = maint_stats["avg_duration_days"]
    st.metric("Avg. Downtime (days)", f"{avg_dur:.1f}" if not np.isnan(avg_dur) else "N/A")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Index status", "Preventive maintenance", "Curative maintenance", "lubrications compliance", "Causes Analysis", "Oil Change Schedule", "🤖 AI Strategic Analysis"
])

#Tabs
# tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    # "Index status", "Index hours", "Fluids Conformity", "Maintenance", "Causes Analysis", "Oil Change Schedule", "🤖 AI Strategic Analysis"
# ----- Tab 1: Etat Index -----
with tab1:
    st.header("Index status – KPIs & Visuals")
    col1, col2, col3 = st.columns(3)
    with col1:
     st.metric("Index down", idx_stats["panne"])
    with col2:
     st.metric("Index functional", idx_stats["marche"])
    with col3:
     st.metric("Index to be checked", idx_stats["verifier"])
    colA, colB = st.columns([1, 1])
    
    df_idx = df_index.copy()
    c_design = idx_stats["col_design"]
    
    if c_design and c_design in df_idx.columns:
        df_idx["is_panne"] = idx_stats["is_panne_mask"]
        df_idx["is_marche"] = idx_stats["is_marche_mask"]
        df_idx["is_verifier"] = idx_stats["is_verif_mask"]
        
        grp = df_idx.groupby(c_design, dropna=False)[["is_panne", "is_marche", "is_verifier"]].sum().reset_index()
        
        # Stacked bar with correct colors
        fig = go.Figure()
        fig.add_bar(x=grp[c_design], y=grp["is_panne"], name="Out of Service", marker_color="#dc3545")
        fig.add_bar(x=grp[c_design], y=grp["is_marche"], name="Functional", marker_color="#28a745")
        fig.add_bar(x=grp[c_design], y=grp["is_verifier"], name="To be checked", marker_color="#ffc107")
        fig.update_layout(barmode="stack", title="Status by Equipment Type", 
                         xaxis_title="Equipment Type", yaxis_title="Number")
        colA.plotly_chart(fig, use_container_width=True)
        
        # Donut chart with correct colors
        donut_data = pd.DataFrame({
            "cat": ["Out of Service", "Functional", "To be checked"],
            "val": [idx_stats["panne"], idx_stats["marche"], idx_stats["verifier"]]
        })
        fig2 = px.pie(donut_data, names="cat", values="val", hole=0.5, 
                     title="Overall Composition",
                     color="cat",
                     color_discrete_map={"Out of Service": "#dc3545", "Functional": "#28a745", "To be checked": "#ffc107"})
        colB.plotly_chart(fig2, use_container_width=True)

# ----- Tab 2: Hours/Prochain -----
with tab2:
    st.header("Index Hours – Preventive Maintenance")
    st.metric("Overall Annual Average", 
             f"{hours_stats['global_avg']:.3f}" if not np.isnan(hours_stats["global_avg"]) else "N/A")
             
    st.info(
    "ℹ️ Note for the chart below: Truck index hours have been excluded, as preventive maintenance is tracked in kilometers rather than hours."

)
    if isinstance(hours_stats["by_cat"], pd.DataFrame) and not hours_stats["by_cat"].empty:
        x_col = hours_stats["by_cat"].columns[0]
        fig3 = px.bar(
            hours_stats["by_cat"],
            x=x_col,
            y="avg_per_year",
            title="Average by Category",
            labels={x_col: "Equipment Category", "avg_per_year": "Average per year"},
            color_discrete_sequence=["#007bff"]
        )
        st.plotly_chart(fig3, use_container_width=True)
# ----- Tab 3: Fluids Conformity -----
with tab4:
    st.header("Lubrication Conformity")
    
    vals = pd.DataFrame({
    "Conformity": ["Compliant", "Partial"],  # Changed to "Conformity"
    "count": [conf_stats["conf_count"], conf_stats["part_count"]]
    })

    fig4 = px.pie(vals, names="Conformity", values="count", hole=0.35,
             title="Compliance Distribution",
             color="Conformity",
             color_discrete_map={"Compliant": "#28a745", "Partial": "#ffc107"})
    st.plotly_chart(fig4, use_container_width=True)

# ----- Tab 4: Maintenance -----
with tab3:
    st.header("Curative Maintenance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Misplans", maint_stats["n_misplan"],
                 help="Number of lines where Date prévue d'intervention ≠ Date fin d'intervention")
    
    with col2:
        avg_dur = maint_stats["avg_duration_days"]
        st.metric("Average Downtime of Equipment (days)", 
                 f"{avg_dur:.1f}" if not np.isnan(avg_dur) else "N/A",
                 help="Average (Date fin d'intervention - Date de Détection) per day")
    
    #Display column names being used
    # with st.expander("ℹ️ Colonnes utilisées"):
        # st.write(f"**Date de Détection:** {maint_stats['cols']['detect']}")
        # st.write(f"**Date prévue d'intervention:** {maint_stats['cols']['prev']}")
        # st.write(f"**Date fin d'Intervention:** {maint_stats['cols']['fin']}")
    
    # Distribution of durations
    if len(maint_stats["valid_durations"]) > 0:
        st.subheader("Downtime Distribution")
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            
            # Build frame with durations + designation (col index 2)
            vals = maint_stats["valid_durations"]
            designation_col = df_maint.columns[2]
            sub = pd.DataFrame({
                "Duration": vals,
                "Designation": df_maint.loc[vals.index, designation_col].astype(str).str.strip().fillna("Unknown")
            })

            nbins = 30
            edges = np.histogram_bin_edges(sub["Duration"].values, bins=nbins)
            sub["bin"] = pd.cut(sub["Duration"], bins=edges, include_lowest=True, right=True)

            bin_counts = sub.groupby("bin", observed=True).size().rename("count")
            bin_des_counts = (
                sub.groupby(["bin", "Designation"], observed=True)
                   .size().reset_index(name="n")
                   .sort_values(["bin", "n"], ascending=[True, False])
            )

            def hover_for_bin(b):
                top = bin_des_counts[bin_des_counts["bin"] == b].head(5)
                lines = "<br>".join(f"{row.Designation}: {int(row.n)}" for _, row in top.iterrows())
                total = int(bin_counts.loc[b])
                left  = int(np.floor(float(b.left)))  if np.isfinite(b.left)  else 0
                right = int(np.ceil(float(b.right))) if np.isfinite(b.right) else left
                return f"<b>{left}–{right} days</b><br>Total: {total}<br><br>{lines}"

            bins_order = bin_counts.index
            x_centers = [(float(b.left) + float(b.right)) / 2 for b in bins_order]
            y_counts  = bin_counts.values
            hovertxt  = [hover_for_bin(b) for b in bins_order]
            bin_widths = [float(b.right) - float(b.left) for b in bins_order]
            bar_widths = [w * 1.4 for w in bin_widths]  # 95% of bin width

            fig5 = go.Figure(go.Bar(
                x=x_centers,
                y=y_counts,
                width=bar_widths,            # ⬅️ wider bars
                marker_color="#007bff",
                hovertext=hovertxt,
                hoverinfo="text"
            ))

            fig5.update_layout(
                title="Downtime Distribution (days)",
                xaxis_title="Duration (days)",
                yaxis_title="Number of interventions",
                showlegend=False,
                bargap=0.02,                 # small gap between bars
                bargroupgap=0                # no extra group gap
            )
            fig5.update_xaxes(dtick=20, tick0=0)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col_chart2:
            # Statistics box
            st.markdown("#### 📊 Statistics")
            stats_df = pd.DataFrame({
                "Métrique": ["Minimum", "Q1 (25%)", "Median", "Q3 (75%)", "Maximum", "mean", "Standard deviation"],
                "Valeur (jours)": [
                    f"{maint_stats['valid_durations'].min():.0f}",
                    f"{maint_stats['valid_durations'].quantile(0.25):.0f}",
                    f"{maint_stats['valid_durations'].median():.0f}",
                    f"{maint_stats['valid_durations'].quantile(0.75):.0f}",
                    f"{maint_stats['valid_durations'].max():.0f}",
                    f"{maint_stats['valid_durations'].mean():.1f}",
                    f"{maint_stats['valid_durations'].std():.1f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            
            
        # Columns used for dates (from your KPI)
        c_detect = maint_stats["cols"]["detect"]
        c_fin    = maint_stats["cols"]["fin"]

        # Parse dates to display them nicely
        d_detect = safe_parse_date(df_maint[c_detect])
        d_fin    = safe_parse_date(df_maint[c_fin])

        # Indices of the top-3 longest (non-negative) downtimes
        top_idx = maint_stats["valid_durations"].nlargest(min(3, len(maint_stats["valid_durations"]))).index

        # Try to find descriptive columns
        col_id  = find_col(df_maint, ["ID Intervention", "ID Intervention عرّف التدخُّل"])
        col_des = find_col(df_maint, ["Désignation", "Designation"])
        col_mat = find_col(df_maint, ["Matériel en panne", "Matériel en panne  المعدّات او الجهاز المعطَّل"])

        st.subheader("🚨 Top 3 Longest Downtimes")
        if len(top_idx) == 0:
            st.info("No valid downtime data to display.")
        else:
            for rank, i in enumerate(top_idx, start=1):
                row = df_maint.loc[i]

                # Safe fetch of fields
                idv = (row[col_id]  if col_id  and col_id  in df_maint.columns else "")
                des = (row[col_des] if col_des and col_des in df_maint.columns else "")
                mat = (row[col_mat] if col_mat and col_mat in df_maint.columns else "")

                dd  = d_detect.loc[i]
                df_ = d_fin.loc[i]
                dur = int(maint_stats["durations"].loc[i])

                # One alert card per item
                st.error(
                    f"#{rank} — **{des}** ({mat})\n"
                    f"• **Date de Détection**: {dd.date() if pd.notna(dd) else 'NA'}\n"
                    f"• **Date fin d’Intervention**: {df_.date() if pd.notna(df_) else 'NA'}\n"
                    f"• **Downtime**: {dur} days"
                )
        

        durations_filtered = durations
        freq = durations_filtered.value_counts().rename_axis("Duration (days)").reset_index(name="Count")
        # Tri sur la durée croissante
        freq = freq.sort_values("Duration (days)").reset_index(drop=True)
        st.subheader("Occurrences per Duration (days)")
        import plotly.express as px
        # Nuage de points (taille et couleur = Count)
        fig = px.scatter(
            freq,
            x="Duration (days)",
            y="Count",
            size="Count",
            color="Count",
            color_continuous_scale=["#808080", "#1e3a8a"],  # Grey to dark blue gradient
            labels={"Count": "Occurrences"},
            title="How often each downtime duration occurs"
        )
        # Lignes verticales fines pour l'effet "lollipop" (optionnel mais lisible)
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.add_bar(
            x=freq["Duration (days)"], 
            y=freq["Count"], 
            marker_color="rgba(255,255,255,0)",  # transparent
            marker_line_color="rgba(160,160,160,0.35)", 
            marker_line_width=1, 
            width=0.01, 
            showlegend=False,
            hoverinfo='skip'  # This prevents the bar from showing hover labels
        )
        fig.update_layout(hovermode="x unified")
        fig.update_traces(
            hovertemplate="Duration: %{x} days<br>Count: %{y}<extra></extra>",
            selector=dict(type='scatter')  # Only apply to scatter trace, not bar
        )
        st.plotly_chart(fig, use_container_width=True)
# ----- Tab 5: Causes Analysis -----
with tab5:
    st.header("Root Cause Analysis")
    
    if not cause_stats["pct_tbl"].empty:
        fig7 = px.bar(cause_stats["pct_tbl"], x="cause", y="pct", 
                     title="Root Causes (Overall)",
                     color_discrete_sequence=["#dc3545"],
                     labels={
                         "cause": "Root Cause",  # X-axis label
                         "pct": "Percentage (%)"  # Y-axis label
                     })
        st.plotly_chart(fig7, use_container_width=True)
        
        if cause_stats["by_engine"] is not None and not cause_stats["by_engine"].empty:
            st.subheader("Heatmap – % Failure type by Equipment Type")
            
            # Clean the data: strip whitespace and normalize case
            cleaned_df = cause_stats["by_engine"].copy()
            col_index = cleaned_df.columns[0]
            col_columns = cleaned_df.columns[1]
            
            # Strip whitespace and convert to title case
            cleaned_df[col_index] = cleaned_df[col_index].astype(str).str.strip().str.title()
            cleaned_df[col_columns] = cleaned_df[col_columns].astype(str).str.strip().str.title()
            
            # Create pivot with counts instead of percentages
            pivot = cleaned_df.pivot_table(
                index=col_index,
                columns=col_columns,
                values="count",
                fill_value=0,
                aggfunc='sum')
            
            # Calculate percentages row-wise (each equipment type = 100%)
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
            
            # Create custom text with different precision for last row
            custom_text = []
            for i, row_name in enumerate(pivot_pct.index):
                row_texts = []
                for val in pivot_pct.iloc[i]:
                    if i == len(pivot_pct.index) - 1:  # Last row
                        row_texts.append(f"{val:.2f}")
                    else:
                        row_texts.append(f"{val:.1f}")
                custom_text.append(row_texts)
            
            fig8 = px.imshow(pivot_pct, aspect="auto", 
                           title="Heatmap causes × Equipment Type",
                           color_continuous_scale="Reds",
                           labels={
                               "x": "Root Cause",
                               "y": "Equipment Type",
                               "color": "Percentage (%)"
                           },
                           text_auto=False)  # Disable auto text
            
            # Add custom text
            fig8.update_traces(text=custom_text, texttemplate="%{text}")
            
            # Cap the color scale at 100%
            fig8.update_traces(zmin=0, zmax=100)
            
            st.plotly_chart(fig8, use_container_width=True)
            
            # Nouveaux KPIs pour l'analyse des causes
            st.header("Analysis of Breakdown Categories")
    
    # KPI 1: Disponibilité des pièces
    if categories_stats["disponibilite"]["pct"] is not None and not categories_stats["disponibilite"]["pct"].empty:
        st.subheader("Breakdown Category % in Terms of Availability")
        fig_disp = px.bar(categories_stats["disponibilite"]["pct"], x="classe", y="pct", 
                         title="Distribution by Spare Parts Availability",
                         color_discrete_sequence=["#4e73df"],
                         labels={
                               "x": "classes",
                               "y": "Percentage (%)"
                              })
        fig_disp.update_layout(xaxis_title="Classes", yaxis_title="Percentage (%)")
        st.plotly_chart(fig_disp, use_container_width=True)
        
        if categories_stats["disponibilite"]["by_engine"] is not None and not categories_stats["disponibilite"]["by_engine"].empty:
            st.subheader("Heatmap – Spare Parts Availability by Engine Type")
            col_disp = categories_stats["columns"]["disponibilite"]
            col_des = categories_stats["columns"]["designation"]
            
            # Create pivot table for heatmap
            by_eng_disp = categories_stats["disponibilite"]["by_engine"]
            pivot_disp = by_eng_disp.pivot_table(
                index=col_des,
                columns=col_disp,
                values="count", fill_value=0)
            
            fig_disp_heat = px.imshow(pivot_disp, aspect="auto", 
                                    title="Heatmap disponibility × Equipment Type",
                                    color_continuous_scale="Blues",
                                    labels={
                                        "x": "Availability",  # Changed to "Availability"
                                        "y": "Designation",   # Changed to "Designation"
                                        "color": "Count"
                                    },
                                    text_auto=True)                                    
            st.plotly_chart(fig_disp_heat, use_container_width=True)
        
        #top 3 Availability        
        if categories_stats["disponibilite"]["by_engine"] is not None and not categories_stats["disponibilite"]["by_engine"].empty:
            st.subheader("Top 3 Equipment by Availability Level")
            
            col_disp = categories_stats["columns"]["disponibilite"]
            col_des = categories_stats["columns"]["designation"]
            by_eng_disp = categories_stats["disponibilite"]["by_engine"]
            
            # Define availability levels
            availability_levels = ["hard to obtain", "available"]
            colors = [
                ["#dc3545", "#e57373", "#ef9a9a"],  # Red shades for difficilement accessible
                ["#28a745", "#5cb85c", "#7bc77d"]   # Green shades for disponible
            ]
            
            # Create 2 columns for the donut charts
            cols = st.columns(2)
            
            for idx, availability in enumerate(availability_levels):
                with cols[idx]:
                    # Filter data for this availability level
                    filtered = by_eng_disp[by_eng_disp[col_disp].astype(str).str.strip().str.lower() == availability]
                    
                    if not filtered.empty:
                        # Get top 3 designations by count
                        top3 = filtered.groupby(col_des)["count"].sum().nlargest(3).reset_index()
                        
                        if not top3.empty:
                            # Create donut chart
                            fig = px.pie(top3, 
                                        names=col_des, 
                                        values="count",
                                        title=f"{availability.capitalize()}",
                                        hole=0.4,
                                        color_discrete_sequence=colors[idx])
                            
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(showlegend=True, height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data for {availability}")
                    else:
                        st.info(f"No data for {availability}")
    
    # KPI 2: Coût de réparation
    if categories_stats["cout"]["pct"] is not None and not categories_stats["cout"]["pct"].empty:
        st.subheader("% Cost per Breakdown Category")
        fig_cout = px.bar(categories_stats["cout"]["pct"], x="classe", y="pct", 
                         title="Repair Cost Distribution",
                         color_discrete_sequence=["#1cc88a"],
                         labels={
                                "x": "classes",
                               "y": "Percentage (%)"
                                    })
        fig_cout.update_layout(xaxis_title="Classes", yaxis_title="Percentage (%)")
        st.plotly_chart(fig_cout, use_container_width=True)
        
        if categories_stats["cout"]["by_engine"] is not None and not categories_stats["cout"]["by_engine"].empty:
            st.subheader("Heatmap – Repair Cost by Engine Type")
            col_cout = categories_stats["columns"]["cout"]
            col_des = categories_stats["columns"]["designation"]
            
            # Create pivot table for heatmap
            by_eng_cout = categories_stats["cout"]["by_engine"]
            pivot_cout = by_eng_cout.pivot_table(
                index=col_des,
                columns=col_cout,
                values="count", fill_value=0)
            
            # Define custom order for x-axis
            desired_order = ["low cost", "medium cost", "high cost"]
            # Only include columns that exist in the pivot table
            pivot_cout = pivot_cout[[col for col in desired_order if col in pivot_cout.columns]]
            
            fig_cout_heat = px.imshow(pivot_cout, aspect="auto", 
                                    title="Heatmap cost × Equipment Type",
                                    color_continuous_scale="Greens",
                                    labels={
                                        "x": "Cost",
                                        "y": "Designation",
                                        "color": "Count"
                                    },
                                    text_auto=True)
            st.plotly_chart(fig_cout_heat, use_container_width=True)
            
        #top 3 cost    
        if categories_stats["cout"]["by_engine"] is not None and not categories_stats["cout"]["by_engine"].empty:
            st.subheader("Top 3 Equipment by Cost Level")
            
            col_cout = categories_stats["columns"]["cout"]
            col_des = categories_stats["columns"]["designation"]
            by_eng_cout = categories_stats["cout"]["by_engine"]
            
            # Define cost levels
            cost_levels = ["low cost", "medium cost", "high cost"]
            colors = [
                ["#28a745", "#5cb85c", "#7bc77d"],  # Green shades for pas cher
                ["#ffc107", "#ffd454", "#ffe082"],  # Yellow shades for moyenne
                ["#dc3545", "#e57373", "#ef9a9a"]   # Red shades for cher
            ]
            
            # Create 3 columns for the donut charts
            cols = st.columns(3)
            
            for idx, cost_level in enumerate(cost_levels):
                with cols[idx]:
                    # Filter data for this cost level
                    filtered = by_eng_cout[by_eng_cout[col_cout].astype(str).str.strip().str.lower() == cost_level]
                    
                    if not filtered.empty:
                        # Get top 3 designations by count
                        top3 = filtered.groupby(col_des)["count"].sum().nlargest(3).reset_index()
                        
                        if not top3.empty:
                            # Create donut chart
                            fig = px.pie(top3, 
                                        names=col_des, 
                                        values="count",
                                        title=f"{cost_level.capitalize()}",
                                        hole=0.4,
                                        color_discrete_sequence=colors[idx])
                            
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(showlegend=True, height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data for {cost_level}")
                    else:
                        st.info(f"No data for {cost_level}")
    
        # KPI 3: Complexité de réparation
        if categories_stats["complexite"]["pct"] is not None and not categories_stats["complexite"]["pct"].empty:
            st.subheader("% of Breakdown by Complexity")
            
            # Define custom order
            complexity_order = ["easy", "intermediate", "hard"]
            
            # Create a copy and set categorical order
            df_comp = categories_stats["complexite"]["pct"].copy()
            df_comp["classe"] = pd.Categorical(df_comp["classe"], 
                                               categories=complexity_order, 
                                               ordered=True)
            df_comp = df_comp.sort_values("classe")
            
            fig_comp = px.bar(df_comp, x="classe", y="pct", 
                             title="Repair Complexity Distribution",
                             color_discrete_sequence=["#f6c23e"],
                             labels={
                                 "classe": "Complexity Level",
                                 "pct": "Percentage (%)"
                             })
            fig_comp.update_layout(xaxis_title="Classes", yaxis_title="Percentage (%)")
            st.plotly_chart(fig_comp, use_container_width=True)
        
        if categories_stats["complexite"]["by_engine"] is not None and not categories_stats["complexite"]["by_engine"].empty:
            st.subheader("Heatmap – Repair Complexity by Engine Type")
            col_comp = categories_stats["columns"]["complexite"]
            col_des = categories_stats["columns"]["designation"]
            
            # Create pivot table for heatmap
            by_eng_comp = categories_stats["complexite"]["by_engine"]
            pivot_comp = by_eng_comp.pivot_table(
                index=col_des,
                columns=col_comp,
                values="count", fill_value=0)
            
            # Define custom order for complexity
            complexity_order = ["easy", "intermediate", "hard"]
            
            # Reorder columns based on custom order (only include columns that exist)
            existing_cols = [col for col in complexity_order if col in pivot_comp.columns]
            pivot_comp = pivot_comp[existing_cols]
            
            fig_comp_heat = px.imshow(pivot_comp, aspect="auto", 
                                    title="Heatmap complexity × Equipment Type",
                                    color_continuous_scale="Oranges",
                                    labels={
                                        "x": "Complexity",
                                        "y": "Designation",
                                        "color": "Count"
                                    },
                                    text_auto=True)  # Display values in each cell
            st.plotly_chart(fig_comp_heat, use_container_width=True)
         
        #top 3 complexity
        if categories_stats["complexite"]["by_engine"] is not None and not categories_stats["complexite"]["by_engine"].empty:
            st.subheader("Top 3 Equipment by Complexity Level")
            
            col_comp = categories_stats["columns"]["complexite"]
            col_des = categories_stats["columns"]["designation"]
            by_eng_comp = categories_stats["complexite"]["by_engine"]
            
            # Define complexity levels
            complexity_levels = ["easy", "intermediate", "hard"]
            colors = [
                ["#28a745", "#5cb85c", "#7bc77d"],  # Green shades for facile
                ["#ffc107", "#ffd454", "#ffe082"],  # Yellow shades for intermédiaire
                ["#dc3545", "#e57373", "#ef9a9a"]   # Red shades for difficile
            ]
            
            # Create 3 columns for the donut charts
            cols = st.columns(3)
            
            for idx, complexity in enumerate(complexity_levels):
                with cols[idx]:
                    # Filter data for this complexity level
                    filtered = by_eng_comp[by_eng_comp[col_comp].astype(str).str.strip().str.lower() == complexity]
                    
                    if not filtered.empty:
                        # Get top 3 designations by count
                        top3 = filtered.groupby(col_des)["count"].sum().nlargest(3).reset_index()
                        
                        if not top3.empty:
                            # Create donut chart
                            fig = px.pie(top3, 
                                        names=col_des, 
                                        values="count",
                                        title=f"{complexity.capitalize()}",
                                        hole=0.4,
                                        color_discrete_sequence=colors[idx])
                            
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(showlegend=True, height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data for {complexity}")
                    else:
                        st.info(f"No data for {complexity}")
# ----- Tab 6: Vidange Planning -----
with tab6:
    st.header("Oil Change Schedule")
    
    labels = ["Uncertain/Natural (<3m)", "Slightly Dangerous (3-6m)", "Dangerous (6-12m)", "Extremely Dangerous (>12m)"]
    vals = [vid_stats["respected"], vid_stats["yellow"], vid_stats["orange"], vid_stats["red"]]
    colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
    
    fig9 = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=colors)])
    fig9.update_layout(title="On-Schedule / Off-Schedule",
                      xaxis_title="Category", yaxis_title="Number")
    st.plotly_chart(fig9, use_container_width=True)
    
    # Calculate the total off-schedule count
    off_schedule_count = vid_stats["yellow"] + vid_stats["orange"] + vid_stats["red"]

    # Calculate percentage out of 150
    total_capacity = 150
    off_schedule_percentage = (off_schedule_count / total_capacity) * 100

    st.info(
    "ℹ️ Note for the chart bellow: It was not taken into consideration as we are not sure whether it was actually "
    "done or not, as it was part of an inspection context."
)
    # Create gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = off_schedule_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Off-Schedule Percentage", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#ffc107"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 66], 'color': '#fff3cd'},
                {'range': [66, 100], 'color': '#f8d7da'}
            ]
        }
    ))

    fig_gauge.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig_gauge, use_container_width=True)
    
# ----- Tab 7: AI Recommendations -----
with tab7:
    st.header("Strategic AI Recommendations")
    
    # Create a container that we can completely replace
    tab7_container = st.container()
    
    with tab7_container:
        # Check if strategic_recommendations.md exists
        recommendations_file = "strategic_recommendations.md"
        
        if os.path.exists(recommendations_file):
            # File exists - show loaders and content
            #st.info("📄 Found existing strategic recommendations file.")
            
            # ALWAYS show loaders - no conditions
            # First loader: Collecting metrics
            with st.spinner("🤔 Thinking\n📊 Collecting the metrics"):
                time.sleep(5)
            
            # Second loader: Analyzing
            with st.spinner("🤔 Thinking\n🔍 Analyzing & Preparing Strategic Insights"):
                time.sleep(6)
            
            # Read and display the markdown file
            try:
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    recommendations_content = f.read()
                
                st.markdown("## 📊 Analysis and Strategic Recommendations")
                st.markdown(recommendations_content)
                
                # Add download button for existing file
                st.download_button(
                    label="📥 Download Recommendations",
                    data=recommendations_content,
                    file_name="strategic_recommendations.md",
                    mime="text/markdown"
                )
                
                # Option to regenerate
                #if st.button("🔄 Regenerate Recommendations", type="secondary"):
                #    if os.path.exists(recommendations_file):
                 #       os.remove(recommendations_file)
                 #   st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error reading recommendations file: {str(e)}")
                if st.button("🔄 Try generating new recommendations"):
                    if os.path.exists(recommendations_file):
                        os.remove(recommendations_file)
                    st.rerun()
        
        else:
            # File doesn't exist - execute existing scenario
            if not GEMINI_AVAILABLE:
                st.error("📚 The Gemini integration module is not available. Please install the necessary dependencies.")
                st.code("pip install google-generativeai", language="bash")
                st.stop()
            
            # Gemini API Configuration
            api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Enter your Gemini API key to enable AI recommendations")
            
            if not api_key:
                st.info("ℹ️ Please enter your Gemini API key in the sidebar to enable AI recommendations.")
                st.markdown("""
                ### How to get a Gemini API key
                1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Sign in with your Google account
                3. Create a new API key
                4. Copy the key and paste it in the field on the left
                """)
                st.stop()
            
            # Initialize the Gemini API
            try:
                gemini_integration.setup_gemini_api(api_key)
                st.sidebar.success("✅ Gemini API configured successfully")
            except Exception as e:
                st.sidebar.error(f"❌ Error configuring the Gemini API: {str(e)}")
                st.error("Unable to configure the Gemini API. Please check your API key.")
                st.stop()
            
            # Prepare data for Gemini
            with st.spinner("Preparing data for AI analysis..."):
                data = gemini_integration.prepare_data_for_gemini(
                    idx_stats, hours_stats, conf_stats, maint_stats, 
                    cause_stats, categories_stats, vid_stats
                )
            
            # Create the prompt for Gemini
            prompt = gemini_integration.create_gemini_prompt(data)
            
            # Display the prompt (optional, for debugging)
            with st.expander("🔍 View the prompt sent to the AI"):
               st.markdown(prompt)
            
            # Button to generate recommendations
            if st.button("🤖 Generate Strategic Recommendations", type="primary"):
                with st.spinner("The AI is analyzing your data and generating strategic recommendations..."):
                    try:
                        # Call the Gemini API
                        recommendations = gemini_integration.get_gemini_recommendations(prompt)
                        
                        # Save recommendations to file
                        with open(recommendations_file, 'w', encoding='utf-8') as f:
                            f.write(recommendations)
                        
                        # Display the recommendations
                        st.markdown("## 📊 Analysis and Strategic Recommendations")
                        st.markdown(recommendations)
                        
                        # Add a download button
                        st.download_button(
                            label="📥 Download Recommendations",
                            data=recommendations,
                            file_name="strategic_recommendations.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
            else:
                st.info("👆 Click the button above to generate strategic recommendations based on your data.")
                
                # Preview of what the AI will analyze
                st.markdown("### 🔎 The AI will analyze the following data:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**État des équipements**")
                    st.metric("Équipements hors service", f"{idx_stats['panne']} ({round((idx_stats['panne'] / idx_stats['total']) * 100 if idx_stats['total'] > 0 else 0, 1)}%)")
                    
                    st.markdown("**Maintenance préventive**")
                    st.metric("Moyenne annuelle globale", f"{round(hours_stats['global_avg'], 3)}" if not pd.isna(hours_stats["global_avg"]) else "N/A")
                    
                    st.markdown("**Conformité des fluides**")
                    st.metric("Pourcentage conforme", f"{round(conf_stats['pct_conf'], 1)}%")
                
                with col2:
                    st.markdown("**Maintenance corrective**")
                    st.metric("Temps d'arrêt moyen", f"{round(maint_stats['avg_duration_days'], 1)} jours" if not pd.isna(maint_stats["avg_duration_days"]) else "N/A")
                    
                    st.markdown("**Planification des vidanges**")
                    off_schedule_count = vid_stats["yellow"] + vid_stats["orange"] + vid_stats["red"]
                    off_schedule_percentage = (off_schedule_count / 150) * 100
                    st.metric("Pourcentage hors calendrier", f"{round(off_schedule_percentage, 1)}%")
                    
                    if not cause_stats["pct_tbl"].empty and len(cause_stats["pct_tbl"]) > 0:
                        top_cause = cause_stats["pct_tbl"].iloc[0]
                        st.markdown("**Principale cause de panne**")
                        st.metric(f"{top_cause['cause']}", f"{round(top_cause['pct'], 1)}%")
