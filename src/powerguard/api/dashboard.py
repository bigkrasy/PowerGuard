"""
Streamlit Dashboard за PowerGuard anomaly detection.

Real-time мониторинг UI, който вика REST API-то.

Стартиране (локално):
    streamlit run src/powerguard/api/dashboard.py

Стартиране (Docker):
    Автоматично през docker-compose.yml

Environment:
    API_URL — URL на REST API-то (default: http://localhost:8000)
              В Docker: http://api:8000
"""
import os

# Environment настройки (преди import streamlit)
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

# API URL — чете от environment (Docker: http://api:8000)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PowerGuard Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "PowerGuard — AI Engineer Portfolio",
    },
)


# ═══════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════

def api_health() -> dict | None:
    """Проверява дали API-то работи."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_predict(reading: dict) -> dict | None:
    """Прави prediction за един sample."""
    try:
        r = requests.post(f"{API_URL}/predict", json=reading, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Грешка при API call: {e}")
        return None


def api_predict_batch(readings: list[dict]) -> dict | None:
    """Прави prediction за batch."""
    try:
        r = requests.post(
            f"{API_URL}/predict_batch",
            json={"readings": readings},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Грешка при batch API call: {e}")
        return None


# ═══════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════

def render_header():
    """Заглавна секция."""
    st.title("⚡ PowerGuard Dashboard")
    st.markdown(
        "Real-time anomaly detection за електрическа подстанция. "
        "Използва XGBoost с физично-обосновани features."
    )


def render_sidebar():
    """Sidebar с API status и navigation."""
    with st.sidebar:
        st.header("🔌 API Status")

        health = api_health()
        if health is None:
            st.error("❌ API-то не отговаря")
            st.caption(f"Провери дали работи на {API_URL}")
            st.code(
                "python -m uvicorn powerguard.api.rest_api:app --port 8000",
                language="powershell",
            )
            return False

        if health["model_loaded"]:
            st.success("✅ Модел зареден")
        else:
            st.warning("⚠️ Degraded режим (моделът не е зареден)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Версия", health["version"])
        with col2:
            st.metric("Uptime", f"{health['uptime_seconds']:.0f}s")

        st.divider()

        st.header("⚙️ Настройки")
        confidence_threshold = st.slider(
            "Anomaly threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Вероятност над която се счита за аномалия",
        )

        st.divider()

        st.caption("PowerGuard v1.0.0")
        st.caption("© 2024 — AI Engineer Portfolio")

        return True


def render_sensor_inputs():
    """Слайдери за входни стойности."""
    st.subheader("📡 Сензорни показания")

    col1, col2, col3 = st.columns(3)

    with col1:
        U = st.slider(
            "Напрежение U (V)", 0.0, 20000.0, 11000.0, step=100.0,
            help="Номинално: 11 000 V"
        )
        I = st.slider(
            "Ток I (A)", 0.0, 1000.0, 100.0, step=1.0,
            help="Номинално: 100 A"
        )

    with col2:
        R = st.slider(
            "Съпротивление R (Ω)", 1.0, 500.0, 110.0, step=1.0,
            help="Номинално: 110 Ω"
        )
        T = st.slider(
            "Температура T (°C)", -20.0, 150.0, 25.0, step=1.0,
            help="Номинално: 25 °C"
        )

    with col3:
        f = st.slider(
            "Честота f (Hz)", 45.0, 55.0, 50.0, step=0.1,
            help="Номинално: 50 Hz"
        )
        cos_phi = st.slider(
            "cos φ", 0.0, 1.0, 0.92, step=0.01,
            help="Номинално: 0.92"
        )

    # Автоматично изчисли P и Q от физиката
    # Но позволи на потребителя да ги "счупи" (за sensor_fault)
    st.markdown("**Мощности** (може да се различават от физиката — за sensor_fault)")

    col1, col2 = st.columns(2)
    with col1:
        # Изчисли "правилните" P и Q
        P_physical = U * I * cos_phi
        P = st.number_input(
            "Активна мощност P (W)",
            value=float(P_physical),
            step=1000.0,
            help=f"Физически правилно: {P_physical:,.0f} W",
        )
    with col2:
        Q_physical = U * I * np.sqrt(max(0, 1 - cos_phi**2))
        Q = st.number_input(
            "Реактивна мощност Q (VAR)",
            value=float(Q_physical),
            step=1000.0,
            help=f"Физически правилно: {Q_physical:,.0f} VAR",
        )

    return {
        "U": U, "I": I, "R": R,
        "P": P, "Q": Q, "T": T,
        "f": f, "cos_phi": cos_phi,
    }


def render_prediction(prediction: dict):
    """Показва prediction резултата."""
    st.subheader("🎯 Prediction")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if prediction["is_anomaly"]:
            st.error("⚠️ АНОМАЛИЯ")
        else:
            st.success("✅ Нормално")

    with col2:
        prob = prediction["anomaly_probability"]
        st.metric("Вероятност", f"{prob:.1%}")

    with col3:
        st.metric("Тип", prediction["anomaly_type"])

    with col4:
        st.metric("Confidence", prediction["confidence"].upper())

    # Gauge chart за вероятността
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction["anomaly_probability"] * 100,
        title={"text": "Anomaly Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred" if prediction["is_anomaly"] else "darkgreen"},
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_batch_tab():
    """Tab за batch prediction."""
    st.subheader("📂 Batch Prediction")
    st.markdown(
        "Качи CSV файл със сензорни показания. "
        "Колоните трябва да са: `U, I, R, P, Q, T, f, cos_phi`"
    )

    uploaded = st.file_uploader("Избери CSV файл", type=["csv"])

    if uploaded is None:
        # Примерен CSV за сваляне
        st.markdown("**Примерен файл:**")
        example_df = pd.DataFrame({
            "U": [11000.0, 11000.0, 1500.0],
            "I": [100.0, 160.0, 500.0],
            "R": [110.0, 110.0, 110.0],
            "P": [1012000.0, 1600000.0, 500000.0],
            "Q": [430000.0, 700000.0, 200000.0],
            "T": [25.0, 90.0, 70.0],
            "f": [50.0, 50.0, 49.8],
            "cos_phi": [0.92, 0.92, 0.85],
        })
        st.dataframe(example_df)
        st.download_button(
            "⬇️ Свали примерен CSV",
            example_df.to_csv(index=False),
            "example_readings.csv",
            "text/csv",
        )
        return

    # Прочети CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Грешка при четене на CSV: {e}")
        return

    st.markdown(f"**Качени {len(df)} реда:**")
    st.dataframe(df.head(20))

    if st.button("🚀 Пусни predictions", type="primary"):
        with st.spinner("Изчисляване..."):
            readings = df.to_dict(orient="records")
            result = api_predict_batch(readings)

        if result is None:
            return

        predictions_df = pd.DataFrame(result["predictions"])
        combined = pd.concat([df, predictions_df], axis=1)

        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Общо samples", len(combined))
        with col2:
            st.metric("Аномалии", result["n_anomalies"])
        with col3:
            st.metric("Anomaly ratio", f"{result['anomaly_ratio']:.1%}")

        # Резултати
        st.markdown("### Резултати")
        st.dataframe(
            combined.style.apply(
                lambda row: [
                    "background-color: #f8d7da" if row["is_anomaly"] else ""
                    for _ in row
                ],
                axis=1,
            )
        )

        # Download
        st.download_button(
            "⬇️ Свали резултатите",
            combined.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )


def render_history_tab():
    """Tab за история на predictions (session state)."""
    st.subheader("📜 История на predictions")

    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.info("Все още няма predictions. Пусни един в 'Live' tab.")
        return

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    # Timeline
    if len(history_df) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df["anomaly_probability"],
            mode="lines+markers",
            name="Anomaly prob",
            line=dict(color="steelblue"),
            marker=dict(
                size=10,
                color=[
                    "red" if is_anom else "green"
                    for is_anom in history_df["is_anomaly"]
                ],
            ),
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Anomaly Probability Timeline",
            xaxis_title="Prediction #",
            yaxis_title="Probability",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🗑️ Изчисти историята"):
        st.session_state.history = []
        st.rerun()


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    render_header()

    if not render_sidebar():
        st.stop()

    # Tabs
    tab_live, tab_batch, tab_history = st.tabs([
        "🎯 Live Prediction",
        "📂 Batch Prediction",
        "📜 История",
    ])

    with tab_live:
        reading = render_sensor_inputs()

        if st.button("🚀 Пусни prediction", type="primary"):
            with st.spinner("Изчисляване..."):
                prediction = api_predict(reading)

            if prediction is not None:
                render_prediction(prediction)

                # Запази в история
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    **reading,
                    **prediction,
                })

    with tab_batch:
        render_batch_tab()

    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()