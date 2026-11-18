# Who says yes to Boston College?

A Streamlit dashboard for our capstone project. It looks at **yield**, the share of admitted
students who enroll, at Boston College and other selective private colleges, and shows which
factors move it.

**The story (7 chapters, one scrolling page)**
1. BC's yield over time vs. the 14 original peers and any school you pick
2. Where BC ranks in a given year
3. Within-school model: what moves a school's yield from one year to the next
4. Across-school model: what kind of school has high yield (with a pick-a-factor scatter)
5. Forecast next fall's yield for BC or any school, with sliders that start at the real values,
   plus how the model did when it forecast past years a year ahead
6. Build a school: set traits and numbers, get a predicted yield and where it would rank
7. Class planner: offers needed and admit rate for a target class size and expected yield

**Model:** same as `Capstone_Yield_Model.ipynb`. Yield = enrolled / admitted (IPEDS).
Predictors are last year's College Scorecard values (admit rate left out on purpose).
Within model: school + year fixed effects, variables picked by forward selection with an
expanding-window test. Across model: year fixed effects plus region and school traits.
Standard errors clustered by school.

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
