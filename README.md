# Who says yes to Boston College?

A Streamlit dashboard for our capstone project. It looks at **yield**, the share of admitted
students who enroll, at Boston College and other selective private colleges, and shows which
factors move it.

**The story (6 chapters, one scrolling page)**
1. BC's yield over time vs. the 14 original peers and any school you pick
2. Where BC ranks in a given year
3. Within-school model: what moves a school's yield from one year to the next
4. Across-school model: what kind of school has high yield (with a pick-a-factor scatter)
5. BC's actual yield vs. what the model expected
6. What if: move last year's factors and see the estimated change in BC's yield

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

Right now they hold the **15-school** version (BC + 14 peers). To switch to all 93 schools,
replace these three files with the ones from Colab. Nothing else needs to change: the text,
charts and sliders update from the files.

`scripts/build_yield_data.py` rebuilds the three files without Colab from
`data/raw/scorecard_93_schools.csv` plus an admissions file (default:
`data/raw/admissions_15_schools.csv`).

## Run it on your own computer

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
