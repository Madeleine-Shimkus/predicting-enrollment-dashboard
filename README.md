# What drives applications to Boston College?

A Streamlit dashboard for our capstone project. It forecasts next year's first-year
applications to Boston College and shows which factors (price after aid, sticker price,
endowment, SAT, one-time events) the model found matter most, compared with 14 peer schools.

**Tabs**
- **Forecast:** next-year prediction, plus actual vs. predicted applications for past years
- **What matters most:** ranked factor importance (mean |SHAP| from a Ridge regression)
- **What if?:** sliders for net price, aid for low-income families, sticker price, SAT and endowment
- **BC vs. peers:** animated map of applications by school, 2005-2023

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
