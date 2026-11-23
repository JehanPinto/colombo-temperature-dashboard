import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Colombo Temperature Ensembles", layout="wide")

@st.cache_data
def load_data():
    weather = pd.read_csv("data/colombo_weather.csv", index_col="date", parse_dates=True)
    results = pd.read_csv("models/results.csv", index_col=0)
    results.index.name = "Model"
    preds = pd.read_csv("models/test_predictions.csv", index_col="date", parse_dates=True)
    importance = pd.read_csv("models/feature_importance.csv", index_col=0)
    forecast = pd.read_csv("models/next_day_forecast.csv", index_col=0)["forecast"]
    return weather, results, preds, importance, forecast

weather, results, preds, importance, forecast = load_data()

st.title("Ensemble Learning Dashboard: Colombo Next-Day Temperature")
st.caption("Data: Open-Meteo Historical Weather API (CC BY 4.0) · Colombo · 2015-01-01 to 2026-11-15")

# Sidebar: model selector (list is sorted best → worst MAE)
model = st.sidebar.selectbox("Choose a model", results.index)
st.sidebar.markdown(
    "**Model families**\n"
    "- Bagging: Bagging, Random Forest\n"
    "- Boosting: XGBoost, LightGBM, CatBoost\n"
    "- Stacking: Stacking (Ridge meta-model)\n"
    "- Time series: ARIMA, SARIMAX, Hybrid ARIMA + RF\n"
    "- Baselines: Naive, Linear Regression, Decision Tree"
)

# 1) Data overview
st.header("1. Data overview")
c1, c2, c3 = st.columns(3)
c1.metric("Days of data", f"{len(weather):,}")
c2.metric("Average mean temperature", f"{weather['temperature_2m_mean'].mean():.1f} °C")
c3.metric("Test period", f"{len(preds)} days")
c3.caption(f"{preds.index.min():%d %b %Y} → {preds.index.max():%d %b %Y} (most recent year, never seen in training)")
st.plotly_chart(px.line(weather, y="temperature_2m_mean", title="Daily mean temperature",
                        labels={"temperature_2m_mean": "°C", "date": ""}))

# 2) Model comparison
st.header("2. Model comparison (test year)")
best = "background-color: #2e7d32; color: white; font-weight: bold"
left, right = st.columns([1, 1.3])
with left:
    st.dataframe(results.style
                 .highlight_min(subset=["MAE", "RMSE", "MAPE_%"], props=best)
                 .highlight_max(subset=["R2"], props=best)
                 .format("{:.3f}"))
    st.caption("Green = best value in each column. Lower is better except R².")

bar = results.reset_index()
bar["group"] = bar["Model"].eq(model).map({True: "Selected", False: "Other"})
fig = px.bar(bar, x="MAE", y="Model", orientation="h", color="group", text_auto=".3f",
             color_discrete_map={"Selected": "crimson", "Other": "steelblue"},
             title="Test MAE (°C), lower is better")
fig.add_vline(x=results.loc["Naive", "MAE"], line_dash="dash", line_color="darkorange", line_width=2,
              annotation_text="Naive baseline", annotation_font_color="darkorange")
fig.update_layout(yaxis={"categoryorder": "total descending"}, showlegend=False)
with right:
    st.plotly_chart(fig)

# 3) Actual vs predicted for the selected model
st.header(f"3. Actual vs predicted: {model}")
m = results.loc[model]
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{m['MAE']:.3f} °C")
c2.metric("RMSE", f"{m['RMSE']:.3f} °C")
c3.metric("R²", f"{m['R2']:.3f}")
c4.metric("MAPE", f"{m['MAPE_%']:.2f} %")

days = st.slider("Days to show (most recent)", 30, 360, 90, step=30)
recent = preds[["Actual", model]].tail(days)
st.plotly_chart(px.line(recent, labels={"value": "°C", "date": "", "variable": ""},
                        color_discrete_map={"Actual": "gray", model: "crimson"}))

# 4) Feature importance for the selected model
st.header("4. Feature importance")
if model in importance.columns:
    imp = importance[model].sort_values()
    st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h",
                           labels={"x": "Share of total importance", "y": ""},
                           title=f"What {model} relies on"))
else:
    st.info(f"Feature importance is only available for the tree-based models. "
            f"{model} is not tree-based, so choose e.g. Random Forest or LightGBM.")

# 5) Next-day forecast
st.header("5. Next-day forecast")
next_day = (weather.index[-1] + pd.Timedelta(days=1)).strftime("%d %b %Y")
today = weather["temperature_2m_mean"].iloc[-1]
st.metric(f"{model}: forecast mean temperature for {next_day}", f"{forecast[model]:.1f} °C",
          delta=f"{forecast[model] - today:+.1f} °C vs last observed day ({today:.1f} °C)",
          delta_color="off")
st.dataframe(forecast.sort_values().round(2).rename("Forecast (°C)"))