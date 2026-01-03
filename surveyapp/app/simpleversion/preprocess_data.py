#!/usr/bin/env python3
"""
Inputs:
  - Excel export ("PS Export_60.xlsx")
  - Survey definition JSON ("PS_SurveyDefinition.json")

Outputs:
  - prepared_simple/responses.parquet (and .csv)
  - prepared_simple/selections.parquet (and .csv)
  - prepared_simple/meta.json

Cleaning rules:
  1) Column heading normalization:
     - Remove the blank space before each capital letter (e.g., " P I6_0" -> "PI6_0", " Survey Date Time" -> "SurveyDateTime").
  2) Add question titles into column names:
     - For non-checkbox/matrix questions: rename exact-match column NAME -> "NAME - TITLE"
     - For checkbox & matrix: split on the LAST "_" to get base and suffix. If base matches NAME exactly,
       rename to "NAME - TITLE_<suffix>", where <suffix> is the checkbox choice or matrix row.
  3) Checkbox handling:
     - Identify numbered columns NAME_0, NAME_1, ... (after normalization).
     - Build one-hot boolean columns named "NAME - TITLE_<choice>" where <choice> is the observed value (plus definition choices).
     - Drop original numbered columns.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def normalize_unicode_whitespace(txt: str) -> str:
    txt = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]", " ", txt)
    txt = txt.replace("\ufeff", "")
    return txt


def load_survey_definition_json(path: str) -> Dict[str, Any]:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = normalize_unicode_whitespace(txt).strip()
    return json.loads(txt)


def remove_spaces_before_capitals(s: str) -> str:
    #s = normalize_unicode_whitespace(str(s))
    #s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r" (?=[A-Z])", "", s)


def choice_values(raw_choices: Any) -> List[str]:
    vals: List[str] = []
    for c in (raw_choices or []):
        if isinstance(c, dict):
            v = c.get("value", "")
            if v is None or v == "":
                v = c.get("text", "")
            vals.append(str(v))
        else:
            vals.append(str(c))
    return [v.strip() for v in vals if str(v).strip()]


def extract_questions(survey_def: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for page in survey_def.get("pages", []):
        for el in page.get("elements", []):
            qtype = el.get("type", "")
            name = el.get("name", "")
            if not name:
                continue
            title = el.get("title", name)
            out[name] = {
                "type": qtype,
                "title": title,
                "choices": choice_values(el.get("choices", [])) if qtype == "checkbox" else [],
            }
    return out


def split_last_underscore(col: str) -> Tuple[str, str]:
    if "_" not in col:
        return col, ""
    return col.rsplit("_", 1)


def sanitize_choice(choice: str) -> str:
    c = normalize_unicode_whitespace(str(choice)).strip()
    c = re.sub(r"\s+", " ", c)
    return c


def preprocess(excel_path: str, survey_def_json_path: str, out_dir: str, respondent_id_col: str = " Respondent Id") -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_excel(excel_path)
    
    original_cols = df_raw.columns.astype(str).tolist()
    norm_cols = [remove_spaces_before_capitals(c) if c!=respondent_id_col else c for c in original_cols]

    # Disambiguate collisions
    seen: Dict[str, int] = {}
    final_cols: List[str] = []
    for c in norm_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)

    df = df_raw.copy()
    df.columns = final_cols

    if respondent_id_col not in df.columns:
        raise ValueError(f"Expected respondent id column '{respondent_id_col}' not found after normalization.")

    df["respondent_id"] = df[respondent_id_col].astype(str)

    survey_def = load_survey_definition_json(survey_def_json_path)
    q = extract_questions(survey_def)

    # Rename map
    rename_map: Dict[str, str] = {}

    # First: rename exact matches for non checkbox/matrix
    for name, meta in q.items():
        if meta["type"] not in ("checkbox", "matrix"):
            if name in df.columns:
                rename_map[name] = f"{name} - {meta['title']}"

    # Then: rename matrix-style columns by last-underscore split
    for col in df.columns:
        base, suf = split_last_underscore(col)
        if not suf:
            continue
        meta = q.get(base)
        if meta and meta["type"] in ("checkbox", "matrix"):
            rename_map[col] = f"{base} - {meta['title']}_{suf}"

    df = df.rename(columns=rename_map)

    # Checkbox one-hot
    selections_parts: List[pd.DataFrame] = []
    onehot_cols_added: List[str] = []
    dropped_numbered_cols: List[str] = []

    # Detect numbered columns for each checkbox question.
    for name, meta in q.items():
        if meta["type"] != "checkbox":
            continue
        title = meta["title"]
        base_label = f"{name} - {title}"

        # columns may be either name_0 or base_label_0 depending on rename above; we standardize on base_label.
        numbered = []
        for c in df.columns:
            b, suf = split_last_underscore(str(c))
            if suf.isdigit() and b == base_label:
                numbered.append(str(c))
        numbered.sort(key=lambda x: int(split_last_underscore(x)[1]))

        if not numbered:
            continue

        melted = df[["respondent_id"] + numbered].melt(
            id_vars=["respondent_id"], value_vars=numbered, var_name="export_col", value_name="choice"
        )
        melted = melted.dropna(subset=["choice"])
        melted["choice"] = melted["choice"].astype(str).map(sanitize_choice)
        melted = melted[melted["choice"].str.len() > 0]
        melted = melted[melted["choice"].str.lower() != "nan"]

        if not melted.empty:
            melted["question"] = name
            selections_parts.append(melted[["respondent_id", "question", "choice"]])

        defined = [sanitize_choice(c) for c in meta.get("choices", []) if str(c).strip()]
        observed = melted["choice"].drop_duplicates().tolist() if not melted.empty else []
        all_choices: List[str] = []
        seen_c = set()
        for c in defined + observed:
            if c and c not in seen_c:
                seen_c.add(c)
                all_choices.append(c)

        if all_choices and not melted.empty:
            ct = pd.crosstab(melted["respondent_id"], melted["choice"])
            for ch in all_choices:
                if ch not in ct.columns:
                    ct[ch] = 0
            ct = ct[all_choices].gt(0)

            ct = ct.rename(columns={ch: f"{base_label}_{ch}" for ch in ct.columns})
            onehot_cols_added.extend(list(ct.columns))

            df = df.merge(ct, how="left", left_on="respondent_id", right_index=True)
            df[list(ct.columns)] = df[list(ct.columns)].fillna(False).astype(bool)

        df = df.drop(columns=numbered)
        dropped_numbered_cols.extend(numbered)

    selections = pd.concat(selections_parts, ignore_index=True) if selections_parts else pd.DataFrame(
        columns=["respondent_id", "question", "choice"]
    )

    responses_path = outp / "responses.parquet"
    selections_path = outp / "selections.parquet"

    df.to_parquet(responses_path, index=False)
    selections.to_parquet(selections_path, index=False)

    # Also store as csv for inspection
    df.to_csv(outp / "responses.csv", index=False)
    selections.to_csv(outp / "selections.csv", index=False)

    meta_out = {
        "respondent_id_col": respondent_id_col,
        "row_count": int(len(df)),
        "responses_col_count": int(df.shape[1]),
        "selections_row_count": int(len(selections)),
        "onehot_cols_added": onehot_cols_added,
        "dropped_numbered_cols": dropped_numbered_cols,
    }
    (outp / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"Wrote: {responses_path}")
    print(f"Wrote: {selections_path}")
    print(f"Wrote: {outp / 'meta.json'}")


if __name__ == "__main__":
    preprocess(
        excel_path="PS Export_60.xlsx",
        survey_def_json_path="PS_SurveyDefinition.json",
        out_dir="prepared_data",
        respondent_id_col=" Respondent Id",
    )
