import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_openai import ChatOpenAI
import hashlib

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR

st.set_page_config(page_title="Survey Explorer", layout="wide")

# -----------------------------
# Survey definition parsing
# - Used for determining chart types to show on dashboard
# - For the Q&A/LLM context the definitions were added to column headers as part of pre-processing to simplify structure
# -----------------------------
@dataclass
class SurveyQuestion:
    name: str
    qtype: str
    title: str
    rows: List[str]
    columns: List[str]
    choices: List[str]

def load_survey_definition_json(path: str) -> Dict[str, Any]:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]", " ", txt) # gets rid of spacing introduced by the rtf format
    return json.loads(txt)

def _choice_values(raw: Any) -> List[str]:
    out: List[str] = []
    for c in (raw or []):
        if isinstance(c, dict):
            v = c.get("value", "")
            if v is None or v == "":
                v = c.get("text", "")
            out.append(str(v))
        else:
            out.append(str(c))
    return [x.strip() for x in out if str(x).strip()]

def build_questions(defn: Dict[str, Any]) -> Dict[str, SurveyQuestion]:
    qs: Dict[str, SurveyQuestion] = {}
    for p in defn.get("pages", []):
        for el in p.get("elements", []):
            name = el.get("name")
            if not name:
                continue
            qtype = el.get("type", "")
            title = el.get("title", name)
            rows = [str(r) for r in el.get("rows", [])] if qtype == "matrix" else []
            cols = [str(c) for c in el.get("columns", [])] if qtype == "matrix" else []
            choices = _choice_values(el.get("choices", [])) if qtype == "checkbox" else []
            qs[name] = SurveyQuestion(name=name, qtype=qtype, title=title, rows=rows, columns=cols, choices=choices)
    return qs

# -----------------------------
# DuckDB + schema helpers (scalable for large datasets)
# -----------------------------
EXCLUDE_HEADERS = {
    " Respondent Id",
    "SURVEY_OUTCOME",
    "D_Postcode",
    "D_State",
    "SurveyDateTime",
    "S3_Consent",
}

def to_sql_ident(col: str) -> str:
    """Make safe column names for DuckDB by lowercasing and stripping non-alphanumeric characters"""
    s = str(col).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "col"

@st.cache_resource(show_spinner=False)
def build_duckdb_from_parquet(parquet_path: str) -> Tuple[duckdb.DuckDBPyConnection, Dict[str, str]]:
    """Build DuckDB table `responses` from parquet without loading full dataset into pandas."""
    con = duckdb.connect(database=":memory:")
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchall()
    headers = [r[0] for r in desc]

    sql_to_header: Dict[str, str] = {}
    used = set()
    select_parts = []
    for h in headers:
        base = to_sql_ident(h)
        name = base
        i = 1
        while name in used: # suffix incremental numbers in case of duplicated column names intoduced by processing
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        sql_to_header[name] = h # map back to original header
        qh = '"' + str(h).replace('"', '""') + '"'
        select_parts.append(f"{qh} AS {name}")

    select_list = ", ".join(select_parts)
    con.execute(f"CREATE VIEW responses AS SELECT {select_list} FROM read_parquet('{parquet_path}')")
    return con, sql_to_header

def get_user_sql_cols(sql_to_header: Dict[str, str]) -> List[str]:
    """Remove columns that start with the excluded headers or are comment fields"""
    return [
        c for c, h in sql_to_header.items()
        if "-Comment" not in h and not any(h.startswith(ex) for ex in EXCLUDE_HEADERS)
    ]

def fetch_distinct_values(con, col: str, limit: int = 30) -> List[Any]:
    """Fetch example values for schema context"""
    q = f"SELECT DISTINCT {col} AS v FROM responses WHERE {col} IS NOT NULL LIMIT {limit}"
    return [r[0] for r in con.execute(q).fetchall()]

def build_schema_context(con, sql_cols: List[str], sql_to_header: Dict[str, str], for_llm: bool) -> str:
    lines = []
    code_to_sql: Dict[str, List[str]] = {}
    for c in sql_cols:
        header = sql_to_header[c]
        code = header.split(" - ", 1)[0].split("_", 1)[0].strip()
        code_to_sql.setdefault(code, []).append(c)

        vals = []
        try:
            vals = fetch_distinct_values(con, c, 12)
        except Exception:
            pass
        vals_str = ", ".join([str(v)[:60] for v in vals])
        if for_llm:
            lines.append(f"- responses.{c} | {header} | example values: {vals_str}")
        else:
            lines.append(f"- {header} | example values: {vals_str}")

    if not for_llm:
        return "\n".join(lines)

    # Create mapping of question codes to possible SQL columns to pass to LLM for SQL generation
    map_lines = ["", "QUESTION CODE -> SQL COLUMNS (use these SQL column names exactly):"]
    for code in sorted(code_to_sql.keys()):
        cols = sorted(code_to_sql[code])
        if len(cols) <= 8:
            rhs = ", ".join([f"responses.{x}" for x in cols])
        else:
            rhs = ", ".join([f"responses.{x}" for x in cols[:8]]) + f", ... (+{len(cols)-8} more)"
        map_lines.append(f"- {code}: {rhs}")

    return "\n".join(lines + map_lines)

# -----------------------------
# LLM helpers
# -----------------------------
SYSTEM_SQL = """You are a data analyst. You will be given a DuckDB table schema for a table named responses.

Return ONLY a single SQL query (no markdown, no explanation).

Critical rules:
- Use ONLY the SQL column names shown in the schema (the tokens after 'responses.').
- Do NOT invent or abbreviate column names.
- Always reference the table as responses.
- Prefer aggregations for summaries.
- Limit output rows to at most 200.
"""

def extract_sql(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.I)
    t = re.sub(r"```\s*$", "", t, flags=re.I).strip()
    m = re.search(r"(?is)\b(select|with)\b", t)
    if m:
        t = t[m.start():].strip()
    t = re.split(r"```", t)[0].strip()
    return t

def validate_sql(sql: str) -> str:
    s = extract_sql(sql).strip().strip(";")
    if not re.match(r"(?is)^(select|with)\b", s):
        raise ValueError("Only SELECT statements are allowed.")
    banned = ["insert", "update", "delete", "drop", "alter", "create", "attach", "copy", "pragma"]
    if any(re.search(rf"(?is)\b{b}\b", s) for b in banned):
        raise ValueError("Disallowed statement detected.")
    if "responses" not in s.lower():
        raise ValueError("Query must reference responses.")
    return s + ";"

def _sql_literal(val: Any) -> str:
    """Escape a Python value for safe use as a SQL string literal in f-strings."""
    if val is None:
        return "NULL"
    s = str(val)
    # escape single quotes by doubling them for SQL string literal safety
    s = s.replace("'", "''")
    return f"'{s}'"

def _auto_chart_key(df: pd.DataFrame) -> str:
    """Stable short key for a DataFrame based on columns + shape."""
    s = ",".join(df.columns.astype(str)) + f":{df.shape}"
    return "auto_chart_" + hashlib.md5(s.encode("utf-8")).hexdigest()

def auto_chart(df: pd.DataFrame) -> None:
    """
    Heuristic chart chooser based on column datatypes (not specific column names).

    Rules:
      - If single column (categorical/non-numeric): show pie for few categories else bar counts.
      - If two columns:
          * (cat, numeric): bar x=cat, y=numeric
          * (numeric, cat): bar x=cat, y=numeric (swap)
          * (cat, cat): stacked bar using counts (x=first cat, color=second cat)
          * (numeric, numeric): scatter
      - If >=3 columns:
          * If there is >=1 numeric and >=1 categorical: pick one numeric as y and up to two categoricals as x/color, aggregate as needed and show stacked/grouped bar
          * Otherwise fallback to showing head of dataframe
    """
    if df is None or df.empty:
        return

    cols = list(df.columns)
    key = _auto_chart_key(df)

    # classify columns by dtype
    is_num = lambda c: pd.api.types.is_numeric_dtype(df[c])
    num_cols = [c for c in cols if is_num(c)]
    cat_cols = [c for c in cols if not is_num(c)]

    try:
        # 1-column: categorical counts
        if len(cols) == 1:
            c = cols[0]
            vc = df[c].value_counts(dropna=False).reset_index()
            vc.columns = [c, "count"]
            if vc.shape[0] <= 8:
                fig = px.pie(vc, names=c, values="count")
            else:
                fig = px.bar(vc, x=c, y="count")
            st.plotly_chart(fig, use_container_width=True, key=key)
            return

        # 2-columns: choose chart by datatypes
        if len(cols) == 2:
            a, b = cols[0], cols[1]
            a_num, b_num = is_num(a), is_num(b)

            if a_num and b_num:
                fig = px.scatter(df, x=a, y=b)
                st.plotly_chart(fig, use_container_width=True, key=key)
                return

            if a_num and not b_num:
                # prefer categorical on x axis for readability
                fig = px.bar(df, x=b, y=a)
                st.plotly_chart(fig, use_container_width=True, key=key)
                return

            if not a_num and b_num:
                fig = px.bar(df, x=a, y=b)
                st.plotly_chart(fig, use_container_width=True, key=key)
                return

            # both categorical -> stacked bar of counts (x=a, color=b)
            # compute counts
            agg = df.groupby([a, b]).size().reset_index(name="count")
            pivot = agg.pivot(index=a, columns=b, values="count").fillna(0)
            fig = px.bar(pivot.reset_index(), x=a, y=pivot.columns.tolist(), barmode="stack")
            st.plotly_chart(fig, use_container_width=True, key=key)
            return

        # 3+ columns: try to pick one numeric and up to two categoricals
        if len(cols) >= 3:
            # prefer a numeric column to use as y
            if num_cols and cat_cols:
                y = num_cols[0]
                x = cat_cols[0]
                color = cat_cols[1] if len(cat_cols) > 1 else None
                # aggregate if necessary (if values are already aggregated it's okay)
                if color:
                    agg = df.groupby([x, color])[y].sum().reset_index()
                    fig = px.bar(agg, x=x, y=y, color=color, barmode="stack")
                else:
                    agg = df.groupby(x)[y].sum().reset_index()
                    fig = px.bar(agg, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True, key=key)
                return

            # if no numeric but >=2 categoricals, show stacked counts for first two
            if not num_cols and len(cat_cols) >= 2:
                a, b = cat_cols[0], cat_cols[1]
                agg = df.groupby([a, b]).size().reset_index(name="count")
                pivot = agg.pivot(index=a, columns=b, values="count").fillna(0)
                fig = px.bar(pivot.reset_index(), x=a, y=pivot.columns.tolist(), barmode="stack")
                st.plotly_chart(fig, use_container_width=True, key=key)
                return

        # fallback: try a sensible 2-column plot if possible
        if len(cols) >= 2 and num_cols:
            # pick first categorical (or cast first non-numeric) and first numeric
            x = cat_cols[0] if cat_cols else cols[0]
            y = num_cols[0]
            agg = df.groupby(x)[y].sum().reset_index()
            fig = px.bar(agg, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True, key=key)
            return

    except Exception:
        # final fallback: show a small table
        st.dataframe(df.head(200), use_container_width=True)
        return

    # if nothing matched, show table
    st.dataframe(df.head(200), use_container_width=True)

# -----------------------------
# Aggregation queries for dashboard
# -----------------------------
def _where_from_filters(filters: List[Tuple[str, List[str]]]) -> Tuple[str, List[Any]]:
    clauses = []
    params: List[Any] = []
    for col, vals in filters:
        if not vals:
            continue
        placeholders = ", ".join(["?"] * len(vals))
        clauses.append(f"{col} IN ({placeholders})")
        params.extend(vals)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params

def agg_counts(con, col: str, filters: List[Tuple[str, List[str]]]) -> pd.DataFrame:
    where, params = _where_from_filters(filters)
    q = f"SELECT {col} AS option, COUNT(*) AS count FROM responses {where} GROUP BY 1 ORDER BY count DESC"
    return con.execute(q, params).fetchdf()

def agg_crosstab(con, xcol: str, ycol: str, filters: List[Tuple[str, List[str]]]) -> pd.DataFrame:
    where, params = _where_from_filters(filters)
    q = f"SELECT {xcol} AS x, {ycol} AS y, COUNT(*) AS count FROM responses {where} GROUP BY 1,2"
    return con.execute(q, params).fetchdf()

def find_question_sql_cols(sql_to_header: Dict[str, str], q: SurveyQuestion) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Takes the survey question and finds matching SQL columns in the DuckDB table.
    In particular for matrix and checkbox questions, returns the multiple relevant columns.
    """
    base_label = f"{q.name} - {q.title}"
    if q.qtype in ("matrix", "checkbox"):
        pairs: List[Tuple[str, str]] = []
        for sqlc, h in sql_to_header.items():
            if h.startswith(base_label + "_"):
                suffix = h.split(base_label + "_", 1)[1]
                pairs.append((suffix, sqlc))
        pairs.sort(key=lambda t: t[0])
        return None, pairs
    else:
        for sqlc, h in sql_to_header.items():
            if h == base_label:
                return sqlc, []
        return None, []

def matrix_chart_data(con, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    parts = []
    for row_label, col in pairs:
        parts.append(
            f"SELECT {_sql_literal(row_label)} AS row, {col} AS response, COUNT(*) AS count "
            f"FROM responses WHERE {col} IS NOT NULL GROUP BY 2"
        )
    if not parts:
        return pd.DataFrame(columns=["row", "response", "count"])
    q = " UNION ALL ".join(parts)
    return con.execute(q).fetchdf()

def checkbox_chart_data(con, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    parts = []
    for choice_label, col in pairs:
        parts.append(
            f"SELECT {_sql_literal(choice_label)} AS choice, "
            f"SUM(CASE WHEN {col} THEN 1 ELSE 0 END) AS count, "
            f"COUNT(*) AS total "
            f"FROM responses"
        )
    if not parts:
        return pd.DataFrame(columns=["choice", "count", "total"])
    q = " UNION ALL ".join(parts)
    df = con.execute(q).fetchdf()
    df["proportion"] = df["count"] / df["total"].replace({0: 1})
    return df

@st.cache_data(show_spinner=False)
def cached_schema_context(parquet_path: str, cap: int) -> tuple[str, str]:
    con, sql_to_header = build_duckdb_from_parquet(parquet_path)  # cached_resource
    user_cols = get_user_sql_cols(sql_to_header)[:cap]
    schema_for_llm = build_schema_context(con, user_cols, sql_to_header, for_llm=True)
    schema_preview = build_schema_context(con, user_cols, sql_to_header, for_llm=False)
    return schema_for_llm, schema_preview

# -----------------------------
# UI
# -----------------------------
#st.write(st.session_state)
st.sidebar.header("Function selection")
page = st.sidebar.radio("Toggle between dashboard and Q&A pages", ["Dashboard", "Q&A"], index=0)

#st.sidebar.header("Data and definition paths")
#responses_path = st.sidebar.text_input("Survey response path (parquet)", value=str(REPO_ROOT / "prepared_data" / "responses.parquet"))
#surveydef_path = st.sidebar.text_input("Survey definition JSON (optional, for question-type charts)", value=str(REPO_ROOT / "PS_SurveyDefinition.json"))
responses_path = str(REPO_ROOT / "prepared_data" / "responses.parquet")
surveydef_path = str(REPO_ROOT / "PS_SurveyDefinition.json")

con, sql_to_header = build_duckdb_from_parquet(responses_path)
user_sql_cols = get_user_sql_cols(sql_to_header)
schema_for_llm, schema_preview = cached_schema_context(responses_path, cap=1000)

# initialize Q&A session_state keys once so their values persist across page switches
st.session_state.setdefault("qa_question", "")
st.session_state.setdefault("qa_sql_raw", "")
st.session_state.setdefault("qa_sql_valid", "")
st.session_state.setdefault("qa_res", None)
st.session_state.setdefault("qa_answer", "")
st.session_state.setdefault("qa_confidence", None)
st.session_state.setdefault("qa_model_choice", "Deeper reasoning (GPT-4.1)")

questions: Dict[str, SurveyQuestion] = {}
if surveydef_path and Path(surveydef_path).exists():
    try:
        defn = load_survey_definition_json(surveydef_path)
        questions = build_questions(defn)
    except Exception as e:
        st.sidebar.warning(f"Could not parse survey definition JSON: {e}")

if page == "Dashboard":
    st.title("Survey Dashboard")
    st.caption("This dashboard area allows you to see the results of each question, and use the 'combine and filter' section below to view results for subsets of the data and to see relationships between responses.")
    st.caption("Switch to the Q&A tab to ask deeper specific questions about the data.")

    st.subheader("Select a question to see results")
    if not questions:
        st.info("Provide a valid survey definition JSON path in the sidebar to enable question-type charts.")
    else:
        available = []
        for qq in questions.values():
            single, pairs = find_question_sql_cols(sql_to_header, qq)
            if single or pairs:
                available.append(qq)
        available.sort(key=lambda x: x.name)

        # Use stable string options and session_state so selection persists across page switches
        option_labels = [f"{qq.name} - {qq.title}" for qq in available]
        label_to_q = {lbl: qq for lbl, qq in zip(option_labels, available)}

        # Normalise any prior session_state value to a valid label
        DEFAULT_VARIABLE = "CC1 - Over the next 12 months, do you think Australia's economy, as a whole, will…"
        prev = st.session_state.get("q_sel_name")
        if not isinstance(prev, str) or prev not in option_labels:
            # set default
            st.session_state["q_sel_name"] = DEFAULT_VARIABLE #option_labels[0] if option_labels else None

        sel_label = st.selectbox("Question", options=option_labels, key="q_sel_name")
        q_sel = label_to_q.get(sel_label)

        chart_type_wording = "Chart type (note: chart options depend on question type)"
        if q_sel:
            if q_sel.qtype == "matrix":
                _, pairs = find_question_sql_cols(sql_to_header, q_sel)
                dfm = matrix_chart_data(con, pairs)
                if dfm.empty:
                    st.warning("No data found for this matrix question.")
                else:
                    mode = st.radio(chart_type_wording, ["Count", "100% Stacked"], horizontal=True, key="mx_mode")
                    if mode == "100% Stacked":
                        dfm["pct"] = dfm.groupby("row")["count"].transform(lambda s: s / max(s.sum(), 1) * 100.0)
                        y = "pct"
                    else:
                        y = "count"
                    fig = px.bar(dfm, x="row", y=y, color="response", barmode="stack")
                    st.plotly_chart(fig, use_container_width=True, key=f"q_chart_matrix_{q_sel.name}_{mode.replace(' ', '_')}")

            elif q_sel.qtype == "checkbox":
                _, pairs = find_question_sql_cols(sql_to_header, q_sel)
                dfc = checkbox_chart_data(con, pairs)
                if dfc.empty:
                    st.warning("No data found for this checkbox question.")
                else:
                    mode = st.radio(chart_type_wording, ["Count", "Proportion"], horizontal=True, key="cb_mode")
                    y = "count" if mode == "Count" else "proportion"
                    fig = px.bar(dfc.sort_values(y, ascending=False), x="choice", y=y)
                    st.plotly_chart(fig, use_container_width=True, key=f"q_chart_checkbox_{q_sel.name}_{mode.replace(' ', '_')}")

            else:
                single, _ = find_question_sql_cols(sql_to_header, q_sel)
                if not single:
                    st.warning("No matching column found for this question in the dataset.")
                else:
                    dfc = agg_counts(con, single, filters=[])
                    if dfc.empty:
                        st.warning("No data found for this question.")
                    else:
                        mode = st.radio(chart_type_wording, ["Histogram", "Pie"], horizontal=True, key="sg_mode")
                        if mode == "Pie":
                            fig = px.pie(dfc, names="option", values="count")
                        else:
                            # histogram-like categorical distribution (counts)
                            fig = px.bar(dfc, x="option", y="count")
                        st.plotly_chart(fig, use_container_width=True, key=f"q_chart_single_{q_sel.name}_{mode.replace(' ', '_')}")

    st.divider()

    st.subheader("Combine and filter questions")
    st.caption("Note that in this section each multi select question type has a selection for each possible choice/row in the survey. This allows you to view, for example, if people who selected that they were part of a political party (D18A) are more or less likely to think favourably of Ursula Von Der Leyen (selectable from PI5B).")

    headers = [sql_to_header[c] for c in user_sql_cols]
    header_to_sql = {sql_to_header[c]: c for c in user_sql_cols}

    # Hardcoded defaults (change here if you want different initial behaviour)
    DEFAULT_PRIMARY = "CC1 - Over the next 12 months, do you think Australia's economy, as a whole, will…" #headers[0] if headers else None
    DEFAULT_SECONDARY = "PD5_18A - What is your view of each of the following?_Prime Minister – Anthony Albanese"
    DEFAULT_CHART_TYPE = "Bar"         # used when no secondary selected
    DEFAULT_STACKED_MODE = "100% Stacked"   # used when secondary is selected

    # Initialize session_state with hardcoded defaults (preserve across reruns/page switches)
    if "primary_h" not in st.session_state or st.session_state.get("primary_h") not in headers:
        st.session_state["primary_h"] = DEFAULT_PRIMARY
    if "secondary_h" not in st.session_state or st.session_state.get("secondary_h") not in ["(none)"] + headers:
        st.session_state["secondary_h"] = DEFAULT_SECONDARY
    if "chart_type" not in st.session_state:
        st.session_state["chart_type"] = DEFAULT_CHART_TYPE
    if "stacked_mode" not in st.session_state:
        st.session_state["stacked_mode"] = DEFAULT_STACKED_MODE

    # Primary / Secondary selectors: rely on session_state defaults, do not pass index/default to avoid conflict
    secondary_options = ["(none)"] + headers
    primary_h = st.selectbox("Primary column", options=headers, key="primary_h")
    secondary_h = st.selectbox("Secondary column (optional)", options=secondary_options, key="secondary_h")

    # below: filters (left) and chart (right)
    left, right = st.columns([1, 2], gap="large")

    with left:
        # control row: show chart-type when no secondary selected, else show stacked-mode
        if secondary_h == "(none)":
            chart_type = st.radio("Chart type", ["Bar", "Pie"], horizontal=True, key="chart_type")
            stacked_mode = None
        else:
            stacked_mode = st.radio("Stack mode", ["Stacked", "100% Stacked"], horizontal=True, key="stacked_mode")
            chart_type = None

        st.subheader("Filters")
        filters: List[Tuple[str, List[str]]] = []
        for i in range(3):
            f_h = st.selectbox(f"Filter {i+1} column", options=["(none)"] + headers, index=0, key=f"fcol_{i}")
            if f_h != "(none)":
                f_sql = header_to_sql[f_h]
                vals = [str(v) for v in fetch_distinct_values(con, f_sql, limit=200)]
                sel = st.multiselect(f"Filter {i+1} values", options=vals, default=vals, key=f"fval_{i}")
                filters.append((f_sql, sel))

    with right:
        primary = header_to_sql[primary_h]
        if secondary_h == "(none)":
            dfc = agg_counts(con, primary, filters)
            if dfc.empty:
                st.warning("No results for current filters.")
            else:
                if chart_type == "Pie":
                    fig = px.pie(dfc, names="option", values="count")
                else:
                    fig = px.bar(dfc, x="option", y="count")
                st.plotly_chart(fig, use_container_width=True, key=f"combine_chart_{id(fig)}")
        else:
            secondary = header_to_sql[secondary_h]
            dfxy = agg_crosstab(con, primary, secondary, filters)
            if dfxy.empty:
                st.warning("No results for current filters.")
            else:
                if stacked_mode == "100% Stacked":
                    dfxy["pct"] = dfxy.groupby("x")["count"].transform(lambda s: s / max(s.sum(), 1) * 100.0)
                    y = "pct"
                else:
                    y = "count"
                fig = px.bar(dfxy, x="x", y=y, color="y", barmode="stack")
                st.plotly_chart(fig, use_container_width=True, key=f"combine_chart_{id(fig)}")

else:
    st.title("Survey Q&A")

    #st.caption("Schema context is capped to keep prompts stable. Increase if needed.")
    cap = 1000 #st.slider("Max columns in schema context", 30, min(600, len(user_sql_cols)), min(220, len(user_sql_cols)), 10)
    schema_cols = user_sql_cols[:cap]
    #schema_for_llm = build_schema_context(con, schema_cols, sql_to_header, for_llm=True)
    #schema_preview = build_schema_context(con, schema_cols, sql_to_header, for_llm=False)

    with st.expander("Schema context (expand to see survey questions to help decide what to ask)"):
        st.text(schema_preview)

    # Model selector: "fast" -> gpt-4o-mini, "deeper reasoning" -> gpt-4.1 (default)
    model_map = {"Fast (GPT-4o-mini)": "gpt-4o-mini", "Deeper reasoning (GPT-4.1)": "gpt-4.1"}
    model_label = st.radio("Model", ["Fast (GPT-4o-mini)", "Deeper reasoning (GPT-4.1)"], index=1, key="qa_model_choice")
    selected_model = model_map.get(model_label, "gpt-4.1")

    # Restore and show previous run (if any) so returning users see last results immediately
    # (previous run results are persisted in session_state; the question widget below will be pre-filled)

    # put the question + run button in a form so Ctrl+Enter submits as well as clicking Run
    with st.form("qa_form", clear_on_submit=False):
        # use a key so the text is persisted in session_state
        question = st.text_area("Question", height=120, key="qa_question")
        submit = st.form_submit_button("Run")

    sql_raw = None
    if submit and question.strip():
        llm = ChatOpenAI(model=selected_model, temperature=0)
        prompt = f"{SYSTEM_SQL}\n\nSCHEMA:\n{schema_for_llm}\n\nQUESTION:\n{question}\n"
        sql_raw = llm.invoke(prompt).content

    if sql_raw:
         try:
            sql_valid = validate_sql(sql_raw)
            res = con.execute(sql_valid).fetchdf()
         except Exception as e:
             st.error(f"Failed to execute SQL: {e}")
             with st.expander("Show model output"):
                 st.code(sql_raw, language="text")
             with st.expander("Show SQL we tried"):
                 st.code(extract_sql(sql_raw), language="sql")
             st.stop()
 
         # Ask the LLM for an answer
         st.subheader("Answer")
         preview_csv = res.head(50).to_csv(index=False)
         answer_prompt = (
             "You are a survey data analyst. Answer the user's question using ONLY the query result table provided. "
             "Be concise and specific. If the result table is aggregated, interpret it. "
             "If you cannot answer from the table, say what is missing.\n\n"
             f"QUESTION:\n{question}\n\nRESULT_TABLE_CSV (first 50 rows):\n{preview_csv}"
         )
         answer = llm.invoke(answer_prompt).content
         st.write(answer)

         # persist the Q&A run results so switching pages preserves them
         st.session_state["qa_sql_raw"] = sql_raw
         st.session_state["qa_sql_valid"] = sql_valid
         st.session_state["qa_res"] = res
         st.session_state["qa_answer"] = answer
         # leave confidence/structured parsing unchanged here (None)

         st.subheader("Result table")
         st.dataframe(res, use_container_width=True)
         with st.expander("Chart (only shown if applicable)"):
             auto_chart(res)
 
         with st.expander("Show SQL"):
             st.code(sql_valid, language="sql")
    else:
        # No new run this render — if there is a previous run, display it inline in the same widgets area
        if st.session_state.get("qa_res") is not None:
            st.subheader("Answer")
            if st.session_state.get("qa_answer"):
                st.write(st.session_state.get("qa_answer"))
            conf = st.session_state.get("qa_confidence")
            if conf is not None:
                st.caption(f"Confidence: {conf}%")
            st.subheader("Result table")
            st.dataframe(st.session_state["qa_res"], use_container_width=True)
            with st.expander("Chart (only shown if applicable)"):
                auto_chart(st.session_state["qa_res"])
            with st.expander("Show SQL"):
                st.code(st.session_state.get("qa_sql_valid", st.session_state.get("qa_sql_raw", "")), language="sql")
