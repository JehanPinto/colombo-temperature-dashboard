# Colombo Next-Day Temperature: Ensemble Learning Dashboard

Streamlit dashboard comparing bagging, boosting, stacking and time-series models
(ARIMA, SARIMAX and a hybrid ARIMA + Random Forest) for forecasting tomorrow's
mean temperature in Colombo, Sri Lanka.

This repository holds only what the dashboard needs: `app.py`, the weather data
and the result tables produced by the notebook. The full project (Jupyter
notebook, saved models, slides and video) was submitted separately.

Run locally:

```
pip install -r requirements.txt
streamlit run app.py
```

Data: [Open-Meteo Historical Weather API](https://open-meteo.com), licensed CC BY 4.0.
