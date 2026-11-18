# Who says yes to Boston College?

A Streamlit dashboard for our capstone project. It looks at **yield**, the share of admitted
students who enroll, at Boston College and other selective private colleges, and shows which
factors move it.

**The story (7 chapters, one scrolling page)**
1. BC's yield over time vs. the 14 original peers and any school you pick
2. Where BC ranks in a given year
3. Within-school model: what moves a school's yield from one year to the next
4. Across-school model: what kind of school has high yield
5. Prediction for BC's next fall from the train/test-split model, with sliders
6. How accurate that model is on colleges it never saw
7. Class planner: offers needed and admit rate for a target class size and expected yield

**Models**
- Chapters 3-4: `Capstone_Yield_Model.ipynb` (school + year fixed effects, and across-school
  with traits). Files: `data/yield_model_data.csv`, `data/yield_within_effects.csv`,
  `data/yield_across_effects.csv`.
- Chapters 5-7: `Yield_TTS.ipynb`, in `tts_model.py`. Schools are split 80/20 (BC always held
  out, seed 42). Ridge + gradient boosting, averaged, on last year's values of 8 variables plus
  year. File: `data/yield_data.csv` (only the columns the model needs). To update, replace it
  with a new `yield_data.csv` from the notebook; the full file works too.

## Data files

The dashboard reads three files in `data/`, the same three the Colab notebook saves to Drive:

- `yield_model_data.csv`
- `yield_within_effects.csv`
- `yield_across_effects.csv`

They hold the full **93-school** results from Colab. To update, replace these three files
(and `data/raw/admissions_yield.csv`) with new ones from Colab. The text, charts and sliders
update from the files.

Forecast inputs (next fall) come from `data/raw/scorecard_93_schools.csv` and any
`data/raw/admissions*.csv` file, via `features.py`.

`scripts/build_yield_data.py` rebuilds the three files without Colab from
`data/raw/scorecard_93_schools.csv` plus `data/raw/admissions_yield.csv`. It reproduces the Colab results exactly.

## Run it on your own computer

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
