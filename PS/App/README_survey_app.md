# Survey Data Explorer + Q&A (Streamlit)

This project provides:
- **Dashboard** (histogram/bar or pie; optional stacked bar with 100% toggle; optional multi-filtering)
- **LLM Q&A** over the survey data (LLM-to-SQL via DuckDB; optional chart output)

## Files expected
Place these files in the same directory as `app.py` (or update the paths in the sidebar):
- `PS Export_60.xlsx` (survey responses)
- `PS_SurveyDefinition.rtf` (RTF container holding JSON survey definition)

## Setup
Create and activate virtual environment (recommended)
```bash
python -m venv surveyapp
source surveyapp/bin/activate
```

then:
```bash
pip install -r requirements.txt
```

### Set your OpenAI API key
Either set an environment variable in the same terminal where you start Streamlit:

```bash
export OPENAI_API_KEY="sk-..."
```

Or create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

## Run
```bash
streamlit run app.py
```

## Notes on scalability
- The app uses **DuckDB** for query execution and aggregations.
- Charts are built from **aggregated results**, not by loading all rows into pandas for each interaction.
- For 1M+ rows, consider converting the Excel export to Parquet and pointing the app at a Parquet-backed DuckDB table.
