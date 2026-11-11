# What drives applications to Boston College?

A Streamlit dashboard for our capstone project. It forecasts next year's first-year
applications to Boston College and shows which factors (price after aid, sticker price,
endowment, SAT, one-time events) the model found matter most, compared with 14 peer schools.

**The story (8 chapters, one scrolling page)**
1. A bumpy ride: BC vs. any peers you pick, raw or indexed, with BC policy changes marked
2. Everyone moves together: yearly growth for all 15 schools with a year spotlight slider
3. BC's own choices: BC's growth minus the peer average
4. The test-optional rush: BC vs. the six peers that went test-optional the same cycle
5. Does price matter?: fixed-effects price effects and a binned scatter for each price measure
6. What predicts growth: 11-factor fixed-effects regression with 95% CIs (plus best-subset and SHAP views)
7. Quality signals: retention vs. graduation, across schools and within schools
8. Looking ahead: next-year forecast with what-if sliders

**Data:** College Scorecard (prices, aid, SAT, endowment) merged with IPEDS admissions
(application counts, fall 2005-2023) on UNITID + year. File: `data/bc_peers.csv`.

**Model:** `model.py`. Ridge regression (scikit-learn) on all 15 schools, predicting each
year's growth in applications. Tested with an expanding-window backtest (each year predicted
using only earlier years).

## Run it on your own computer

1. Install Python 3.10 or newer from python.org.
2. Open Terminal in this folder and run:

   ```
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

3. Your browser opens the dashboard at http://localhost:8501.
