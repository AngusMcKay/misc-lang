# Survey Data Explorer + Q&A

This project provides an app containing two functions:
- Dashboard: containing options to select survey questions and view various charts to explore the data
- Q&A: where an LLM can be asked deeper question about the survey data and will crate basic charts where appropriate

## Files contained
The project contains the following files:
- `PS Export_60.xlsx` (survey responses raw data)
- `PS_SurveyDefinition.json` (json with survey definition, converted from provided rtf file)
- `requirements.txt` (python libraries required to run the app)
- `preprocess_data.py` (script to clean and preprocess data)
- `survey_app.py` (the actual app)

## App infrastructure
- Data preprocessing has two main outcomes:
	- Uses the survey definitions to inject meaningful context directly into data headers for LLM to use
	- Also reformats 'checkbox' style questions into on-hot-encoded columns for each possible answer
	- Saves output as .parquet for faster operation with DuckDB
- Survey app:
	- Streamlit app with a DuckDB backend for querying the data
	- DuckDB will easily work with datasets up to 1m rows for preprocessing data
	- Aggrgated data is then handled within the app by pandas and plotly for visualisations
	- OpenAI APIs are used for converting user queries to SQL queries in order to extract relevant survey data for the questions asked

## Setup
NOTE: tested with Python 3.11.14
- Create and activate virtual environment in the directory where the files are contained
- Install requirements
- Set OpenAI API key as environment variable (alternatively set up using streamlit secrets)

```bash
python -m venv surveyapp
source surveyapp/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Run preprocessing
This will create a subdirectory with the pre-processed data in parquet form which is used by DuckDB (also saves as csv for easier manual inspection)
```bash
python preprocess_data.py
```

## Run app
```bash
streamlit run app.py
```
