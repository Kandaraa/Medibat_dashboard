import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from datetime import datetime

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
st.sidebar.title("📂 Data Sources")

default_paths = {
    "etat_index": "etat Index.csv",
    "index_hours": "index hours.csv",
    "fluids_conf": "Recommended Fluids and Conformity.csv",
    "maintenance": "SUIVI DE MAINTENANCE.csv",
    "causes": "categorie panne.csv",
    "vidange": "suivie vidange.csv",
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

st.sidebar.success("✅ Data loaded successfully")

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
def kpi_maintenance(df):
    dfc = df.copy()
    cols = dfc.columns.tolist()
    
    # According to the description and CSV structure:
    # Column 11 (index 10): Date de Détection
    # Column 14 (index 13): Date prévue d'intervention
    # Column 16 (index 15): Date fin d'Intervention
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

maint_stats = kpi_maintenance(df_maint)

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
        if "classification des causes racine" in low:
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
    st.metric("Index down", idx_stats["panne"])
with c2:
    st.metric("Index functional", idx_stats["marche"])
with c3:
    st.metric("Index to be checked", idx_stats["verifier"])
with c4:
    panne_pct = pct(idx_stats["panne"], idx_stats["total"])
    st.metric("% Breakdown vs Total", f"{panne_pct:.1f}%")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("% Compliant oils", f"{conf_stats['pct_conf']:.1f}%")
with c6:
    st.metric("% Partial oils", f"{conf_stats['pct_partielle']:.1f}%")
with c7:
    st.metric("Miscalculations in Planning", maint_stats["n_misplan"])
with c8:
    avg_dur = maint_stats["avg_duration_days"]
    st.metric("Avg. Downtime (days)", f"{avg_dur:.1f}" if not np.isnan(avg_dur) else "N/A")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Index status", "Index hours", "Fluids Conformity", "Maintenance", "Causes Analysis", "Oil Change Schedule"
])

# ----- Tab 1: Etat Index -----
with tab1:
    st.header("Index status – KPIs & Visuals")
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
                     title="Global Composition",
                     color="cat",
                     color_discrete_map={"Out of Service": "#dc3545", "Functional": "#28a745", "To be checked": "#ffc107"})
        colB.plotly_chart(fig2, use_container_width=True)

# ----- Tab 2: Hours/Prochain -----
with tab2:
    st.header("Index Hours – Average Oil Changes per Year")
    st.metric("Global Annual Average", 
             f"{hours_stats['global_avg']:.3f}" if not np.isnan(hours_stats["global_avg"]) else "N/A")
    
    if isinstance(hours_stats["by_cat"], pd.DataFrame) and not hours_stats["by_cat"].empty:
        fig3 = px.bar(hours_stats["by_cat"], 
                     x=hours_stats["by_cat"].columns[0], 
                     y="avg_per_year", 
                     title="Average by Category",
                     color_discrete_sequence=["#007bff"])
        st.plotly_chart(fig3, use_container_width=True)

# ----- Tab 3: Fluids Conformity -----
with tab3:
    st.header("Oil Conformity")
    
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
with tab4:
    st.header("Maintenance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Misplans", maint_stats["n_misplan"],
                 help="Number of lines where Date prévue d'intervention ≠ Date fin d'intervention")
    
    with col2:
        avg_dur = maint_stats["avg_duration_days"]
        st.metric("Average Downtime of Equipment (days)", 
                 f"{avg_dur:.1f}" if not np.isnan(avg_dur) else "N/A",
                 help="Average (Date fin d'intervention - Date de Détection) per day")
    
    # Display column names being used
    with st.expander("ℹ️ Colonnes utilisées"):
        st.write(f"**Date de Détection:** {maint_stats['cols']['detect']}")
        st.write(f"**Date prévue d'intervention:** {maint_stats['cols']['prev']}")
        st.write(f"**Date fin d'Intervention:** {maint_stats['cols']['fin']}")
    
    # Distribution of durations
    if len(maint_stats["valid_durations"]) > 0:
        st.subheader("Downtime Distribution")
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            # Histogram
            fig5 = px.histogram(maint_stats["valid_durations"], 
                               nbins=30, 
                               title="Downtime Distribution (days)",
                               labels={"value": "Duration (days)", "count": "Frequency"},
                               color_discrete_sequence=["#007bff"])
            fig5.update_layout(showlegend=False, xaxis_title="Duration (days)", yaxis_title="interventions number")
            st.plotly_chart(fig5, use_container_width=True)
        
        with col_chart2:
            # Statistics box
            st.markdown("#### 📊 Statistics")
            stats_df = pd.DataFrame({
                "Métrique": ["Minimum", "Q1 (25%)", "Median", "Q3 (75%)", "Maximum", "mean", "Écart-type"],
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
        
        # Box plot for outliers
        st.subheader("Outliers Analysis")
        fig6 = px.box(maint_stats["valid_durations"], 
                      title="Box Plot - Identification of Abnormal Durations",
                      labels={"value": "Duration (days)"},
                      color_discrete_sequence=["#28a745"])
        fig6.update_layout(showlegend=False, yaxis_title="Duration (days)")
        st.plotly_chart(fig6, use_container_width=True)
        
        # Timeline scatter
        st.subheader("Durations in Chronological Order")
        timeline_df = pd.DataFrame({
            "Index": range(len(maint_stats["valid_durations"])),
            "Durée": maint_stats["valid_durations"].values
        })
        fig7 = px.scatter(timeline_df, x="Index", y="Duration",
                         title="Evolution of Downtime",
                         labels={"Index": "intervention number", "Duration": "Duration (days)"},
                         color="Durée",
                         color_continuous_scale="RdYlGn_r")
        fig7.update_traces(marker=dict(size=8))
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("⚠️ No valid duration data available.")

# ----- Tab 5: Causes Analysis -----
with tab5:
    st.header("Root Cause Analysis")
    
    if not cause_stats["pct_tbl"].empty:
        fig7 = px.bar(cause_stats["pct_tbl"], x="cause", y="pct", 
                     title="Root Causes (global)",
                     color_discrete_sequence=["#dc3545"])
        st.plotly_chart(fig7, use_container_width=True)
        
        if cause_stats["by_engine"] is not None and not cause_stats["by_engine"].empty:
            st.subheader("Heatmap – % by Equipment Type")
            pivot = cause_stats["by_engine"].pivot_table(
                index=cause_stats["by_engine"].columns[0],
                columns=cause_stats["by_engine"].columns[1],
                values="pct", fill_value=0.0)
            fig8 = px.imshow(pivot, aspect="auto", 
                           title="Heatmap causes × Equipment Type",
                           color_continuous_scale="Reds")
            st.plotly_chart(fig8, use_container_width=True)
    
    # Nouveaux KPIs pour l'analyse des causes
    st.header("Analysis of Breakdown Categories")
    
    # KPI 1: Disponibilité des pièces
    if categories_stats["disponibilite"]["pct"] is not None and not categories_stats["disponibilite"]["pct"].empty:
        st.subheader("Breakdown Category % in Terms of Availability")
        fig_disp = px.bar(categories_stats["disponibilite"]["pct"], x="classe", y="pct", 
                         title="Distribution by Spare Parts Availability",
                         color_discrete_sequence=["#4e73df"])
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
                                    title="Heatmap disponibilité × Equipment Type",
                                    color_continuous_scale="Blues")
            st.plotly_chart(fig_disp_heat, use_container_width=True)
    
    # KPI 2: Coût de réparation
    if categories_stats["cout"]["pct"] is not None and not categories_stats["cout"]["pct"].empty:
        st.subheader("% Cost per Breakdown Category")
        fig_cout = px.bar(categories_stats["cout"]["pct"], x="classe", y="pct", 
                         title="Repair Cost Distribution",
                         color_discrete_sequence=["#1cc88a"])
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
            
            fig_cout_heat = px.imshow(pivot_cout, aspect="auto", 
                                    title="Heatmap cost × Equipment Type",
                                    color_continuous_scale="Greens")
            st.plotly_chart(fig_cout_heat, use_container_width=True)
    
    # KPI 3: Complexité de réparation
    if categories_stats["complexite"]["pct"] is not None and not categories_stats["complexite"]["pct"].empty:
        st.subheader("% of Breakdown by Complexity")
        fig_comp = px.bar(categories_stats["complexite"]["pct"], x="classe", y="pct", 
                         title="Repair Complexity Distribution",
                         color_discrete_sequence=["#f6c23e"])
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
            
            fig_comp_heat = px.imshow(pivot_comp, aspect="auto", 
                                    title="Heatmap complexity × Equipment Type",
                                    color_continuous_scale="Oranges")
            st.plotly_chart(fig_comp_heat, use_container_width=True)

# ----- Tab 6: Vidange Planning -----
with tab6:
    st.header("Planification de vidange")
    
    labels = ["Respected (<3m)", "Not respected 3-6m", "Not respected 6-12m", "Not respected >12m"]
    vals = [vid_stats["respected"], vid_stats["yellow"], vid_stats["orange"], vid_stats["red"]]
    colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
    
    fig9 = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=colors)])
    fig9.update_layout(title="On-Schedule / Off-Schedule",
                      xaxis_title="Catégorie", yaxis_title="Nombre")
    st.plotly_chart(fig9, use_container_width=True)
    
    # Gauge
    total_v = sum(vals)
    share_respected = pct(vid_stats["respected"], total_v) if total_v > 0 else 0
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=share_respected,
        number={'suffix': "%"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#28a745"},
               'steps': [
                   {'range': [0, 50], 'color': "#f8d7da"},
                   {'range': [50, 80], 'color': "#fff3cd"},
                   {'range': [80, 100], 'color': "#d1e7dd"},
               ]},
        title={'text': "Taux de respect du planning"}
    ))
    st.plotly_chart(gauge, use_container_width=True)


