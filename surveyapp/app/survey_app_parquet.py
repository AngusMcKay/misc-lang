import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px

# Optional: OpenAI via LangChain (uses OPENAI_API_KEY / Streamlit secrets)
from langchain_openai import ChatOpenAI

# -----------------------------
# Utilities: Survey definition
# -----------------------------

def _normalize_json_whitespace(txt: str) -> str:
    """
    Python's built-in json parser only treats ASCII whitespace as whitespace.
    Some editors insert Unicode space separators (e.g., U+2006, U+2007) which break json.loads.
    This normalizes common Unicode whitespace to regular spaces.
    """
    # Replace any Unicode 'space separator' characters with ASCII space
    txt = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]", " ", txt)
    # Also normalize odd thin spaces that sometimes appear
    txt = txt.replace("\uFEFF", "")  # BOM
    return txt


def _rtf_hex_unescape(s: str) -> str:
    """Convert RTF hex escapes like \\'96 into the corresponding cp1252 character."""
    def repl(m: re.Match) -> str:
        b = bytes([int(m.group(1), 16)])
        return b.decode("cp1252", errors="ignore")
    return re.sub(r"\\'([0-9a-fA-F]{2})", repl, s)

def _extract_first_json_object(s: str) -> str:
    """Extract the first complete JSON object from a string by brace matching."""
    in_str = False
    escape = False
    depth = 0
    start = None

    for i, ch in enumerate(s):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    raise ValueError("Could not extract JSON object from the survey definition file.")

def load_survey_definition_rtf(rtf_path: str) -> Dict[str, Any]:
    raw = open(rtf_path, "rb").read().decode("utf-8", errors="ignore")
    # Find JSON start near a known key.
    idx = raw.find('"pages"')
    if idx == -1:
        idx = raw.find('"completedHtmlOnCondition"')
    if idx == -1:
        raise ValueError("Could not find JSON keys in the RTF file.")

    start = raw.rfind("{", 0, idx)
    snippet = raw[start:]
    snippet = snippet.replace("\\\n", "\n").replace("\\{", "{").replace("\\}", "}")
    snippet = _rtf_hex_unescape(snippet)
    json_text = _extract_first_json_object(snippet)
    return json.loads(json_text)

def canon_key(s: str) -> str:
    """Canonical key: alnum lower only (for robust matching)."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

@dataclass
class QuestionMeta:
    name: str
    title: str
    qtype: str
    choices: Optional[List[str]] = None

def build_question_meta(survey_def: Dict[str, Any]) -> Dict[str, QuestionMeta]:
    """
    Build a mapping from canonicalized column keys -> QuestionMeta.
    Handles matrix by expanding rows into separate canonical keys.
    """
    out: Dict[str, QuestionMeta] = {}

    pages = survey_def.get("pages", [])
    for p in pages:
        for el in p.get("elements", []):
            qtype = el.get("type", "")
            name = el.get("name", "")
            title = el.get("title", name)
            choices = None

            if qtype in ("radiogroup", "dropdown"):
                raw_choices = el.get("choices", [])
                # choices may include dicts with value/text
                choices = []
                for c in raw_choices:
                    if isinstance(c, dict):
                        choices.append(str(c.get("text", c.get("value", ""))))
                    else:
                        choices.append(str(c))
            elif qtype == "checkbox":
                raw_choices = el.get("choices", [])
                choices = []
                for c in raw_choices:
                    if isinstance(c, dict):
                        choices.append(str(c.get("text", c.get("value", ""))))
                    else:
                        choices.append(str(c))
            elif qtype == "matrix":
                # Export often yields separate columns per row
                cols = el.get("columns", [])
                rows = el.get("rows", [])
                # Store base matrix meta as well
                out[canon_key(name)] = QuestionMeta(name=name, title=title, qtype=qtype, choices=[str(c) for c in cols])
                for r in rows:
                    combined = f"{name}_{r}"
                    out[canon_key(combined)] = QuestionMeta(
                        name=combined,
                        title=f"{title} — {r}",
                        qtype=qtype,
                        choices=[str(c) for c in cols],
                    )
                continue

            out[canon_key(name)] = QuestionMeta(name=name, title=title, qtype=qtype, choices=choices)

    return out



def load_survey_definition_json(json_path: str) -> Dict[str, Any]:
    """
    Robust loader for survey definition.
    Accepts:
      - strict JSON
      - JSON embedded in other text (extracts first {...})
      - python-dict-like text (single quotes) via ast.literal_eval as a fallback
    """
    txt = open(json_path, "rb").read().decode("utf-8", errors="ignore")
    txt = _normalize_json_whitespace(txt).strip()

    # If there's leading junk, extract first JSON object.
    try:
        candidate = _extract_first_json_object(txt)
    except Exception:
        candidate = txt

    # First attempt: strict JSON
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Fallback: python literal (handles single quotes). Safer than eval.
    try:
        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Survey definition is not a JSON object (expected a dict at top level).")
    except Exception as e:
        # Provide a compact, useful error for UI
        snippet = candidate[:300].replace("\n", "\\n")
        raise ValueError(f"Could not parse survey definition as JSON. Snippet: {snippet}. Error: {e}")



# -----------------------------
# Survey model for display labels
# -----------------------------
@dataclass
class SurveyQuestion:
    name: str
    title: str
    qtype: str  # radiogroup, checkbox, matrix, text, dropdown, etc.

@dataclass
class MatrixQuestion(SurveyQuestion):
    rows: List[str]
    columns: List[str]

@dataclass
class CheckboxQuestion(SurveyQuestion):
    choices: List[str]  # values (preferred)

def _choice_values(raw_choices: Any) -> List[str]:
    vals: List[str] = []
    if not raw_choices:
        return vals
    for c in raw_choices:
        if isinstance(c, dict):
            v = c.get("value", "")
            if v is None or v == "":
                v = c.get("text", "")
            vals.append(str(v))
        else:
            vals.append(str(c))
    return [v.strip() for v in vals if str(v).strip()]

def sanitize_option_for_col(opt: str) -> str:
    s = str(opt).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "option"

def prettify_choice_from_suffix(suffix: str) -> str:
    s = str(suffix).replace("_", " ").strip()
    if s.islower():
        s = s.title()
    return s

def build_survey_model(survey_def: Dict[str, Any]) -> Tuple[Dict[str, SurveyQuestion], Dict[str, MatrixQuestion], Dict[str, CheckboxQuestion]]:
    questions: Dict[str, SurveyQuestion] = {}
    matrix: Dict[str, MatrixQuestion] = {}
    checkbox: Dict[str, CheckboxQuestion] = {}

    for p in survey_def.get("pages", []):
        for el in p.get("elements", []):
            qtype = el.get("type", "")
            name = el.get("name", "")
            if not name:
                continue
            title = el.get("title", name)

            if qtype == "matrix":
                rows = [str(r) for r in el.get("rows", [])]
                cols = [str(c) for c in el.get("columns", [])]
                mq = MatrixQuestion(name=name, title=title, qtype=qtype, rows=rows, columns=cols)
                questions[name] = mq
                matrix[name] = mq
            elif qtype == "checkbox":
                choices = _choice_values(el.get("choices", []))
                cq = CheckboxQuestion(name=name, title=title, qtype=qtype, choices=choices)
                questions[name] = cq
                checkbox[name] = cq
            else:
                questions[name] = SurveyQuestion(name=name, title=title, qtype=qtype)

    return questions, matrix, checkbox


# -----------------------------
# Utilities: Data + DuckDB
# -----------------------------
def normalize_colname(col: str) -> str:
    """
    Normalize messy export column names into SQL-safe identifiers while preserving meaning.
    Example: ' C C1' -> 'cc1', ' D18 A_ A Trade Union' -> 'd18a_a_trade_union'
    """
    s = str(col).strip()

    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s)

    # remove spaces between single letters (common in export: 'C C1' -> 'CC1')
    s = re.sub(r"\b([A-Za-z])\s+([A-Za-z0-9])\b", r"\1\2", s)
    s = s.replace(" _", "_").replace("_ ", "_")

    # replace remaining non-word with underscore
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()

@st.cache_data(show_spinner=False)
def load_excel_to_df(excel_path: str) -> pd.DataFrame:
    return pd.read_excel(excel_path)

@st.cache_resource(show_spinner=False)
def build_duckdb(df: pd.DataFrame) -> Tuple[duckdb.DuckDBPyConnection, Dict[str, str], Dict[str, str]]:
    """
    Return:
      - connection (in-memory)
      - sql_name -> original column
      - sql_name -> canonical key

    IMPORTANT:
    Do not use a dict comprehension for normalized names, as it can silently drop
    columns when multiple original columns normalize to the same SQL-safe key.
    This implementation preserves all columns and disambiguates collisions.
    """
    used: Dict[str, int] = {}
    sql_cols: List[Tuple[str, str]] = []

    for orig in df.columns.astype(str).tolist():
        base_name = normalize_colname(orig)
        sql_name = base_name
        if sql_name in used:
            used[base_name] += 1
            sql_name = f"{base_name}_{used[base_name]}"
        else:
            used[base_name] = 0
        sql_cols.append((sql_name, orig))

    # Rename dataframe for duckdb
    df2 = df.copy()
    df2.columns = [sql for sql, _ in sql_cols]

    con = duckdb.connect(database=":memory:")
    con.register("responses_df", df2)
    con.execute("CREATE TABLE responses AS SELECT * FROM responses_df")

    sql_to_orig = {sql: orig for sql, orig in sql_cols}
    sql_to_canon = {sql: canon_key(orig) for sql, orig in sql_cols}
    return con, sql_to_orig, sql_to_canon

def get_categorical_columns(con: duckdb.DuckDBPyConnection, sql_to_orig: Dict[str, str]) -> List[str]:
    """
    Heuristic: treat all non-numeric columns as categorical; if numeric has small cardinality, allow too.
    """
    cols = list(sql_to_orig.keys())
    cat = []
    for c in cols:
        # type detection
        dtype = con.execute(f"SELECT typeof({c}) FROM responses LIMIT 1").fetchone()[0]
        if dtype in ("VARCHAR", "BOOLEAN"):
            cat.append(c)
        elif dtype in ("DOUBLE", "INTEGER", "BIGINT"):
            # allow numeric columns if low distinct count
            distinct = con.execute(f"SELECT COUNT(DISTINCT {c}) FROM responses").fetchone()[0]
            if distinct <= 30:
                cat.append(c)
    return sorted(cat)

def safe_where_clause(filters: List[Tuple[str, List[str]]]) -> Tuple[str, List[Any]]:
    """
    Build parameterized WHERE clause from filters.
    filters: list of (sql_col, selected_values)
    """
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
    return "WHERE " + " AND ".join(clauses), params

def fetch_distinct_values(con: duckdb.DuckDBPyConnection, col: str, limit: int = 200) -> List[str]:
    q = f"SELECT DISTINCT {col} AS v FROM responses WHERE {col} IS NOT NULL ORDER BY v LIMIT {limit}"
    vals = [r[0] for r in con.execute(q).fetchall()]
    return [str(v) for v in vals]


# -----------------------------
# Charts
# -----------------------------
def chart_single(con, col: str, where_sql: str, params: List[Any], chart_type: str):
    q = f"""
    SELECT {col} AS category, COUNT(*) AS n
    FROM responses
    {where_sql}
    GROUP BY 1
    ORDER BY n DESC
    """
    df = con.execute(q, params).df()
    df["category"] = df["category"].astype(str)
    if chart_type == "bar":
        fig = px.bar(df, x="category", y="n", title=f"Distribution of {col}")
    else:
        fig = px.pie(df, names="category", values="n", title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

def chart_stacked(con, col_x: str, col_color: str, where_sql: str, params: List[Any], normalize: bool):
    q = f"""
    SELECT {col_x} AS x, {col_color} AS c, COUNT(*) AS n
    FROM responses
    {where_sql}
    GROUP BY 1,2
    """
    df = con.execute(q, params).df()
    df["x"] = df["x"].astype(str)
    df["c"] = df["c"].astype(str)

    if normalize:
        # convert to % within each x
        totals = df.groupby("x")["n"].transform("sum")
        df["pct"] = (df["n"] / totals) * 100.0
        fig = px.bar(df, x="x", y="pct", color="c", barmode="stack",
                     title=f"100% Stacked distribution: {col_x} by {col_color}")
        fig.update_yaxes(title_text="Percent")
    else:
        fig = px.bar(df, x="x", y="n", color="c", barmode="stack",
                     title=f"Stacked distribution: {col_x} by {col_color}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.sort_values(["x", "c"]), use_container_width=True)


# -----------------------------
# LLM-to-SQL helpers
# -----------------------------
DISALLOWED_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|EXPORT|IMPORT|PRAGMA)\b", re.I)

def validate_sql(sql: str, allowed_cols: List[str]) -> str:
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    if DISALLOWED_SQL.search(s):
        raise ValueError("Disallowed SQL operation detected.")
    # Basic column allowlist: require any identifiers to be subset of allowed cols (heuristic)
    # We don't attempt full SQL parsing; instead we check that any `responses.<col>` references are allowed.
    for m in re.finditer(r"responses\.([a-zA-Z0-9_]+)", s):
        col = m.group(1).lower()
        if col not in allowed_cols:
            raise ValueError(f"Unknown or disallowed column referenced: {col}")
    return s

def build_schema_context(con, sql_cols: List[str], sql_to_orig: Dict[str, str], qmeta: Dict[str, QuestionMeta], sql_to_canon: Dict[str, str]) -> str:
    lines = []
    for c in sql_cols:
        orig = sql_to_orig.get(c, c)
        label = display_label_for_original(orig)
        try:
            vals = fetch_distinct_values(con, c, limit=12)
        except Exception:
            vals = []
        vals_str = ", ".join([str(v)[:50] for v in vals[:12]])
        lines.append(f"- responses.{c}: {label}. Example values: {vals_str}")
    return "\n".join(lines)

def llm_plan_query(llm: ChatOpenAI, question: str, schema_context: str) -> Dict[str, Any]:
    """
    Ask LLM for a JSON plan:
      - sql: SELECT query referencing table 'responses'
      - chart: optional {type, x, y, color, normalize, title}
    """
    system = (
        "You are a data analyst. Produce a JSON object with keys: "
        "`sql` (a single read-only SELECT query against table `responses`), "
        "`chart` (optional; choose from bar,pie,line,stacked_bar,histogram), "
        "`notes` (short). "
        "Use only columns listed in the schema. Prefer aggregated queries that return <= 500 rows. "
        "If the user asks for a chart, choose an appropriate chart and set x/y/color. "
        "If the question is ambiguous, make a reasonable assumption and mention it in notes."
    )
    user = f"SCHEMA:\n{schema_context}\n\nQUESTION:\n{question}\n\nReturn only valid JSON."

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content
    # Robustly extract first JSON object
    txt = resp.strip()
    json_obj = _extract_first_json_object(txt)
    return json.loads(json_obj)

def llm_explain(llm: ChatOpenAI, question: str, df: pd.DataFrame) -> str:
    preview = df.head(50).to_markdown(index=False)
    system = (
        "You are a careful analyst. Answer the user's question using ONLY the provided query result table. "
        "If the table is empty or insufficient, say so and suggest what to ask next. "
        "Keep the answer concise and numerical where possible."
    )
    user = f"QUESTION:\n{question}\n\nQUERY RESULT (first rows):\n{preview}"
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content

def render_llm_chart(df: pd.DataFrame, chart: Dict[str, Any]):
    ctype = chart.get("type")
    title = chart.get("title") or "Chart"
    x = chart.get("x")
    y = chart.get("y")
    color = chart.get("color")
    normalize = bool(chart.get("normalize", False))

    if df is None or df.empty:
        st.info("No rows returned for charting.")
        return

    # Basic charting over result set
    if ctype == "bar":
        fig = px.bar(df, x=x, y=y, color=color, title=title)
    elif ctype == "pie":
        fig = px.pie(df, names=x, values=y, title=title)
    elif ctype == "line":
        fig = px.line(df, x=x, y=y, color=color, title=title)
    elif ctype == "histogram":
        fig = px.histogram(df, x=x, title=title)
    elif ctype == "stacked_bar":
        if normalize and y:
            tmp = df.copy()
            # normalize within x
            if color:
                tmp["_tot"] = tmp.groupby(x)[y].transform("sum")
                tmp["_pct"] = (tmp[y] / tmp["_tot"]) * 100.0
                fig = px.bar(tmp, x=x, y="_pct", color=color, barmode="stack", title=title)
                fig.update_yaxes(title_text="Percent")
            else:
                fig = px.bar(df, x=x, y=y, barmode="stack", title=title)
        else:
            fig = px.bar(df, x=x, y=y, color=color, barmode="stack", title=title)
    else:
        st.info("No chart spec produced (or unsupported chart type).")
        return

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Survey Data Explorer + Q&A", layout="wide")

st.title("Survey Data Explorer + Q&A")

with st.sidebar:
    st.header("Data files")
    use_prepared = st.toggle("Use prepared Parquet (recommended)", value=False)
    if use_prepared:
        responses_parquet = st.text_input("Prepared responses.parquet path", value="prepared/responses.parquet")
        selections_parquet = st.text_input("Prepared selections.parquet path", value="prepared/selections.parquet")
        meta_json_path = st.text_input("Prepared meta.json path", value="prepared/meta.json")
    else:
        excel_path = st.text_input("Excel dataset path", value="PS Export_60.xlsx")
    surveydef_path = st.text_input("Survey definition (JSON) path", value="PS_SurveyDefinition.json")

    st.caption("Tip: Put data files in the same folder as this app, or update paths.")

    st.divider()
    st.header("LLM settings")
    use_llm = st.toggle("Enable LLM Q&A", value=True)
    model_name = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption("LLM requires OPENAI_API_KEY via environment variable or .streamlit/secrets.toml")

# Load files
try:
    if use_prepared:
        df_raw = pd.read_parquet(responses_parquet)
    else:
        df_raw = load_excel_to_df(excel_path)
except Exception as e:
    src = responses_parquet if use_prepared else excel_path
    st.error(f"Failed to load dataset from {src}: {e}")
    st.stop()

selections_df = None
if use_prepared:
    try:
        selections_df = pd.read_parquet(selections_parquet)
    except Exception:
        selections_df = None

meta_info = None
if use_prepared:
    try:
        with open(meta_json_path, 'r', encoding='utf-8') as f:
            meta_info = json.load(f)
    except Exception:
        meta_info = None

try:
    survey_def = load_survey_definition_json(surveydef_path)
    qmeta = build_question_meta(survey_def)
    questions_by_name, matrix_by_name, checkbox_by_name = build_survey_model(survey_def)
except Exception as e:
    qmeta = {}
    questions_by_name, matrix_by_name, checkbox_by_name = {}, {}, {}
    st.warning(
        f"Failed to parse survey definition JSON. Charts will still work, but labels may be less friendly. Details: {e}"
    )

# Fields to exclude from user-facing UI (dashboard + schema)
EXCLUDE_COL_PATTERNS = [
    re.compile(r"^respondent_id$", re.I),
    re.compile(r"^respondent id$", re.I),
    re.compile(r"^survey_outcome$", re.I),
    re.compile(r"^survey outcome$", re.I),
    re.compile(r"^surveyoutcome$", re.I),
    re.compile(r"^s3_?consent$", re.I),
    re.compile(r"consent", re.I),
    re.compile(r"comment", re.I),
    # Only exclude the survey timestamp field (avoid hiding legitimate time-based questions)
    re.compile(r"^survey[ _-]?date[ _-]?time$", re.I),
]

def is_excluded_user_field(original_name: str) -> bool:
    s = str(original_name).strip()
    return any(p.search(s) for p in EXCLUDE_COL_PATTERNS)

con, sql_to_orig, sql_to_canon = build_duckdb(df_raw)
all_sql_cols = list(sql_to_orig.keys())
allowed_cols_for_llm = [c for c in all_sql_cols if not is_excluded_user_field(sql_to_orig.get(c,''))]
cat_cols = [c for c in get_categorical_columns(con, sql_to_orig) if not is_excluded_user_field(sql_to_orig.get(c,''))]

def display_label_for_original(original_name: str) -> str:
    """
    Required formats:
      - Radiogroup/Dropdown/Text:  NAME - TITLE
      - Matrix:                   NAME - TITLE - ROW
      - Checkbox (one-hot):       NAME - TITLE - CHOICE
    """
    orig = str(original_name).strip()

    # Checkbox one-hot or Matrix: "<QNAME> <suffix>"
    m = re.match(r"^([A-Za-z0-9]+)\s+(.+)$", orig)
    if m:
        base, suffix = m.group(1), m.group(2)
        if base in checkbox_by_name:
            q = checkbox_by_name[base]
            lookup = {sanitize_option_for_col(c): c for c in q.choices}
            # Fallback to meta.json choices if available (prepared pipeline)
            if meta_info and isinstance(meta_info, dict):
                try:
                    meta_cb = meta_info.get("checkbox_questions", {}).get(base, {}).get("choices", [])
                    for c in meta_cb:
                        lookup.setdefault(sanitize_option_for_col(c), c)
                except Exception:
                    pass
            choice = lookup.get(sanitize_option_for_col(suffix))
            if choice is None:
                choice = prettify_choice_from_suffix(suffix)
            return f"{base} - {q.title} - {choice}"

        if base in matrix_by_name:
            q = matrix_by_name[base]
            return f"{base} - {q.title} - {suffix}"

    # Base question exact match
    if orig in questions_by_name:
        q = questions_by_name[orig]
        return f"{q.name} - {q.title}"

    return orig

def friendly_label(sql_col: str) -> str:
    orig = sql_to_orig.get(sql_col, sql_col)
    return display_label_for_original(orig)

tab1, tab2 = st.tabs(["Dashboard", "Q&A"])

# -----------------------------
# Dashboard Page
# -----------------------------
with tab1:
    st.subheader("Dashboard")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Chart controls")

        col1 = st.selectbox(
            "Primary question/column (categorical)",
            options=cat_cols,
            format_func=friendly_label,
        )

        chart_type = st.radio("Chart type", options=["bar", "pie"], horizontal=True, index=0)

        use_second = st.toggle("Add a second column (stacked bar)", value=False)
        col2 = None
        if use_second:
            col2 = st.selectbox(
                "Second question/column (for breakdown)",
                options=[c for c in cat_cols if c != col1],
                format_func=friendly_label,
            )
            normalize = st.toggle("100% stacked", value=False)
        else:
            normalize = False

        st.divider()
        st.markdown("### Filters (optional)")
        st.caption("Add one or more filters. This is applied to all charts on this page.")

        filters: List[Tuple[str, List[str]]] = []
        num_filters = st.number_input("Number of filters", min_value=0, max_value=8, value=0, step=1)

        for i in range(int(num_filters)):
            fcol = st.selectbox(
                f"Filter {i+1} column",
                options=cat_cols,
                format_func=friendly_label,
                key=f"filter_col_{i}",
            )
            vals = fetch_distinct_values(con, fcol, limit=200)
            fvals = st.multiselect(
                f"Filter {i+1} values",
                options=vals,
                key=f"filter_vals_{i}",
            )
            filters.append((fcol, fvals))

        where_sql, params = safe_where_clause(filters)

    with right:
        st.markdown("### Results")

        if not col2:
            chart_single(con, col1, where_sql, params, chart_type=chart_type)
        else:
            chart_stacked(con, col1, col2, where_sql, params, normalize=normalize)

        with st.expander("Show active filters"):
            if not filters or all(len(v)==0 for _, v in filters):
                st.write("No filters applied.")
            else:
                for fcol, fvals in filters:
                    if fvals:
                        st.write(f"- **{friendly_label(fcol)}** in {fvals}")

# -----------------------------
# Q&A Page
# -----------------------------
with tab2:
    st.subheader("Ask questions about the survey data")

    if not use_llm:
        st.info("LLM Q&A is disabled in the sidebar.")
        st.stop()

    # Fail-fast key check (more user-friendly than the SDK exception)
    if not os.getenv("OPENAI_API_KEY") and not st.secrets.get("OPENAI_API_KEY", None):
        st.error(
            "OPENAI_API_KEY is not available to this Streamlit process. "
            "Set it as an environment variable or add it to .streamlit/secrets.toml."
        )
        st.stop()

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Build schema context for prompting.
    # Previously this was capped at 80 columns; you can increase if needed.
    st.caption("Schema summary is capped by default to avoid overly long prompts. Increase the cap if needed.")
    schema_cap = st.slider("Max columns to include in schema context", min_value=20, max_value=min(400, len(cat_cols)), value=min(160, len(cat_cols)), step=10)

    schema_cols = cat_cols[:schema_cap]
    schema_context = build_schema_context(con, schema_cols, sql_to_orig, qmeta, sql_to_canon)

    with st.expander("Schema summary (used for Q&A)", expanded=False):
        st.code(schema_context)

    question = st.text_area("Question", placeholder="e.g., What share of respondents approve of the Federal government overall?", height=110)

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        show_sql = st.toggle("Show SQL plan", value=True)
    with colB:
        show_table = st.toggle("Show result table", value=True)
    with colC:
        show_chart = st.toggle("Render chart (if suggested)", value=True)

    if st.button("Run", type="primary", disabled=not bool(question.strip())):
        with st.spinner("Planning query..."):
            plan = llm_plan_query(llm, question.strip(), schema_context)

        sql = plan.get("sql", "")
        chart = plan.get("chart", None)
        notes = plan.get("notes", "")

        try:
            validated = validate_sql(sql, allowed_cols=allowed_cols_for_llm)
        except Exception as e:
            st.error(f"SQL validation failed: {e}")
            st.code(sql or "(no sql produced)")
            st.stop()

        if show_sql:
            st.markdown("### Query plan")
            st.code(validated, language="sql")
            if notes:
                st.caption(f"Planner notes: {notes}")

        with st.spinner("Executing query..."):
            try:
                result_df = con.execute(validated).df()
            except Exception as e:
                st.error(f"Failed to execute SQL: {e}")
                st.stop()

        if show_table:
            st.markdown("### Query result")
            st.dataframe(result_df, use_container_width=True)

        if show_chart and isinstance(chart, dict):
            st.markdown("### Chart")
            try:
                render_llm_chart(result_df, chart)
            except Exception as e:
                st.warning(f"Could not render chart from spec: {e}")
                st.json(chart)

        with st.spinner("Generating explanation..."):
            answer = llm_explain(llm, question.strip(), result_df)

        st.markdown("### Answer")
        st.write(answer)
