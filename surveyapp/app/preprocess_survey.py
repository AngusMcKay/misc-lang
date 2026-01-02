#!/usr/bin/env python3
"""
Preprocess survey export into analysis-friendly Parquet files:

Outputs (in out_dir/):
  - responses.parquet   : one row per respondent; includes one-hot columns for checkbox questions
  - selections.parquet  : long-form table of checkbox selections (respondent_id, question, option)
  - meta.json           : helpful metadata (checkbox questions, one-hot columns, etc.)

Key behaviors:
  - Uses existing 'Respondent ID' as respondent_id (stringified for safety).
  - Standardizes column headings:
      * Removes spaces between single-letter prefixes: "P I6_0" -> "PI6_0", "C C1" -> "CC1"
      * Replaces "_ " with " " (removes underscore but keeps space) to tidy matrix-style headings.
  - For checkbox questions:
      * Detects columns like PI6_0, PI6_1, ... (after standardization)
      * Builds selections.parquet (long)
      * Adds one-hot columns PI6__<option> to responses.parquet
      * Drops the original PI6_0..PI6_n columns from responses

Requirements:
  - pandas
  - openpyxl
  - pyarrow (recommended) or fastparquet for Parquet writing
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


RESPONDENT_ID_COL = "Respondent Id"
ALSO_SAVE_AS_CSV = True


def normalize_unicode_whitespace(txt: str) -> str:
    """Normalize common Unicode whitespace to ASCII spaces."""
    txt = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]", " ", txt)
    txt = txt.replace("\ufeff", "")  # BOM
    return txt


def load_survey_definition_json(path: str) -> Dict[str, Any]:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = normalize_unicode_whitespace(txt).strip()
    return json.loads(txt)


def extract_checkbox_questions(survey_def: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping:
      question_name -> {"title": str, "choices": List[str]}

    For choice objects, prefer the 'value' field.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for page in survey_def.get("pages", []):
        for el in page.get("elements", []):
            if el.get("type") == "checkbox":
                name = el.get("name")
                title = el.get("title", name)
                raw_choices = el.get("choices", [])
                choices: List[str] = []
                for c in raw_choices:
                    if isinstance(c, dict):
                        choices.append(str(c.get("value", "")))
                    else:
                        choices.append(str(c))
                choices = [c for c in (x.strip() for x in choices) if c]
                out[name] = {"title": title, "choices": choices}
    return out


def standardize_export_column(col: str) -> str:
    """
    Standardize column headings to align survey definition names with export columns.

    Fixes:
      - Two-letter spacing artefact:
          'P I6_0' -> 'PI6_0', 'C C1' -> 'CC1'
      - Matrix-style base name spacing:
          'D18 A A sports club' -> 'D18A A Sports Club'
        (base code 'D18A' + row label 'A Sports Club')
      - Remove underscore before a space (keep the space):
          'PI7_ A ...' -> 'PI7 A ...'
      - Preserves spaces in row labels (avoids 'A sports' -> 'Asports')
    """
    s = str(col)
    s = normalize_unicode_whitespace(s).strip()
    s = re.sub(r"\s+", " ", s)

    # 1) Remove spaces between a single-letter prefix and a token that CONTAINS a digit (e.g., 'P I6' -> 'PI6').
    s = re.sub(r"\b([A-Za-z])\s+([A-Za-z]*\d[A-Za-z0-9_]*)", r"\1\2", s)

    # 2) Remove spaces between an alphanumeric code ending in digits and a single letter (e.g., 'D18 A' -> 'D18A').
    s = re.sub(r"\b([A-Za-z]*\d+)\s+([A-Za-z])\b", r"\1\2", s)

    # 3) Remove underscore when it is immediately followed by a space (keep the space).
    s = s.replace("_ ", " ")

    # 4) Cosmetic: Title-case row label when it's fully lowercase.
    m = re.match(r"^([A-Za-z0-9]+)\s+(.+)$", s)
    if m:
        code_part, label_part = m.group(1), m.group(2)
        if any(ch.isalpha() for ch in label_part) and label_part == label_part.lower():
            label_part = label_part.title()
        s = f"{code_part} {label_part}"

    return s


def rename_columns_safely(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standardization to all columns, disambiguating collisions."""
    new_cols = []
    seen: Dict[str, int] = {}
    for c in df.columns:
        base = standardize_export_column(c)
        name = base
        if name in seen:
            seen[name] += 1
            name = f"{base}__{seen[base]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df2 = df.copy()
    df2.columns = new_cols
    return df2


def find_checkbox_export_columns(columns: List[str], qname: str) -> List[str]:
    pat = re.compile(rf"^{re.escape(qname)}_\d+$")
    cols = [c for c in columns if pat.match(c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def sanitize_option_for_col(opt: str) -> str:
    s = str(opt).strip()#.lower()
    #s = re.sub(r"\s+", "_", s)
    #s = re.sub(r"[^a-z0-9_]+", "", s)
    #s = re.sub(r"_+", "_", s).strip("_")
    return s or "option"


def preprocess(
    excel_path: str,
    survey_def_json_path: str,
    out_dir: str,
    respondent_id_col: str = RESPONDENT_ID_COL,
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_excel(excel_path)
    df = rename_columns_safely(df_raw)

    if respondent_id_col not in df.columns:
        raise ValueError(
            f"Expected respondent id column '{respondent_id_col}' not found after standardization."
        )

    # Stringify for safety (leading zeros, etc.)
    df["respondent_id"] = df[respondent_id_col].astype(str)

    survey_def = load_survey_definition_json(survey_def_json_path)
    checkbox_meta = extract_checkbox_questions(survey_def)

    selections_parts: List[pd.DataFrame] = []
    onehot_cols_added: List[str] = []
    checkbox_export_cols_used: Dict[str, List[str]] = {}

    for qname, meta in checkbox_meta.items():
        export_cols = find_checkbox_export_columns(df.columns.tolist(), qname)
        if not export_cols:
            continue

        checkbox_export_cols_used[qname] = export_cols

        melted = df[["respondent_id"] + export_cols].melt(
            id_vars=["respondent_id"], value_vars=export_cols, var_name="export_col", value_name="option"
        )
        melted = melted.dropna(subset=["option"])
        melted["option"] = melted["option"].astype(str).str.strip()
        melted = melted[melted["option"].str.len() > 0]
        melted = melted[melted["option"].str.lower() != "nan"]

        if melted.empty:
            df = df.drop(columns=export_cols)
            continue

        melted["question"] = qname
        selections_parts.append(melted[["respondent_id", "question", "option"]])

        defined = [c for c in (x.strip() for x in meta.get("choices", [])) if c]
        observed = melted["option"].drop_duplicates().tolist()

        all_options: List[str] = []
        seen = set()
        for v in defined + observed:
            vv = v.strip()
            if vv and vv not in seen:
                seen.add(vv)
                all_options.append(vv)

        ct = pd.crosstab(melted["respondent_id"], melted["option"])
        for opt in all_options:
            if opt not in ct.columns:
                ct[opt] = 0
        ct = ct[all_options]
        ct = ct.gt(0)

        renamed = {opt: f"{qname} {sanitize_option_for_col(opt)}" for opt in ct.columns}
        ct = ct.rename(columns=renamed)
        onehot_cols_added.extend(list(ct.columns))

        df = df.merge(ct, how="left", left_on="respondent_id", right_index=True)
        df[list(ct.columns)] = df[list(ct.columns)].fillna(False).astype(bool)

        # Drop original checkbox export columns
        df = df.drop(columns=export_cols)

    selections = (
        pd.concat(selections_parts, ignore_index=True)
        if selections_parts
        else pd.DataFrame(columns=["respondent_id", "question", "option"])
    )

    responses_path = outp / "responses.parquet"
    selections_path = outp / "selections.parquet"

    df.to_parquet(responses_path, index=False)
    selections.to_parquet(selections_path, index=False)

    if ALSO_SAVE_AS_CSV:
        responses_csv_path = outp / "responses.csv"
        selections_csv_path = outp / "selections.csv"
        df.to_csv(responses_csv_path, index=False)
        selections.to_csv(selections_csv_path, index=False)

    meta_out = {
        "respondent_id_col": respondent_id_col,
        "column_name_map": dict(zip(df_raw.columns.astype(str).tolist(), df.columns.astype(str).tolist())),
        "checkbox_questions": checkbox_meta,
        "checkbox_export_cols_used": checkbox_export_cols_used,
        "onehot_columns_added": sorted(set(onehot_cols_added)),
        "row_count": int(len(df)),
        "selections_row_count": int(len(selections)),
        "notes": {
            "standardization": [
                "collapsed whitespace",
                "removed spaces between single-letter prefixes (e.g., 'P I6' -> 'PI6')",
                "replaced '_ ' with ' ' for tidier matrix-style headings",
                "dropped original checkbox columns qname_0..qname_n after creating one-hot columns",
            ]
        },
    }
    (outp / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"Wrote: {responses_path}")
    print(f"Wrote: {selections_path}")
    print(f"Wrote: {outp / 'meta.json'}")


if __name__ == "__main__":
    preprocess(
        excel_path="PS Export_60.xlsx",
        survey_def_json_path="PS_SurveyDefinition.json",
        out_dir="prepared",
        respondent_id_col=RESPONDENT_ID_COL,
    )
