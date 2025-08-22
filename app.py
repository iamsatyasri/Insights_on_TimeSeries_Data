import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import google.generativeai as genai
import plotly.graph_objects as go
import json
import re
import os

# ---------------------------
# Configure Gemini API
# ---------------------------
genai.configure(api_key="AIzaSyC-kGmLlBifFL0XI0CoHmGcjIC5WQqZWdE")

# ---------------------------
# File for storing user data
# ---------------------------
USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# ---------------------------
# Session state defaults
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "prophet_df" not in st.session_state:
    st.session_state.prophet_df = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "date_col" not in st.session_state:
    st.session_state.date_col = None
if "forecast_months" not in st.session_state:
    st.session_state.forecast_months = 1

# ---------------------------
# Authentication Pages
# ---------------------------
def login_page():
    st.title("🔐 Login to Forecast Dashboard")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    if st.button("Login"):
        users = load_users()
        if username in users and users[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.rerun()  # Updated from experimental_rerun
        else:
            st.error("❌ Invalid username or password")
    if st.button("Go to Sign Up"):
        st.session_state["signup"] = True
        st.rerun()  # Updated from experimental_rerun

def signup_page():
    st.title("📝 Sign Up for Forecast Dashboard")
    username = st.text_input("New Username", placeholder="Choose a username")
    password = st.text_input("New Password", type="password", placeholder="Choose a password")
    if st.button("Sign Up"):
        users = load_users()
        if username in users:
            st.error("⚠️ Username already exists. Try another.")
        else:
            users[username] = password
            save_users(users)
            st.success("✅ Account created! You can now login.")
            st.session_state["signup"] = False
            st.rerun()  # Updated from experimental_rerun
    if st.button("Go to Login"):
        st.session_state["signup"] = False
        st.rerun()  # Updated from experimental_rerun

# ---------------------------
# Logout button
# ---------------------------
def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state["signup"] = False
        st.rerun()  # Updated from experimental_rerun

# ---------------------------
# Main App (only if logged in)
# ---------------------------
def main_app():
    st.set_page_config(page_title="📊Forecast Dashboard", layout="wide")
    st.title(f"📊Forecast Dashboard - Welcome {st.session_state.username}!")
    st.sidebar.title("📂 Navigation")
    page = st.sidebar.radio("Go to", ["📤 Upload Dataset", "📊 Generate Forecast", "🤖 AI Insights"])
    logout_button()

    # ---------------------------
    # Page 1: Upload Dataset
    # ---------------------------
    if page == "📤 Upload Dataset":
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.write("👀 Dataset Preview:")
            st.dataframe(df.head(5))

            # Detect date column
            date_candidates = []
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.datetime64):
                    date_candidates.append(col)
                else:
                    try:
                        parsed = pd.to_datetime(df[col], errors='coerce')
                        if parsed.notna().sum() > len(df) * 0.5:
                            date_candidates.append(col)
                    except:
                        continue

            if len(date_candidates) == 0:
                st.error("❌ No date column detected.")
            elif len(date_candidates) == 1:
                st.session_state.date_col = date_candidates[0]
            else:
                st.session_state.date_col = st.selectbox("🗓 Select Date Column", date_candidates)

            # Detect numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                default_col = df[numeric_cols].var().idxmax()
                st.session_state.target_col = st.selectbox(
                    "📊 Select Column to Forecast", numeric_cols, index=numeric_cols.index(default_col)
                )

            st.success("✅ Data uploaded and columns selected! Move to Generate Forecast")

    # ---------------------------
    # Page 2: Generate Forecast
    # ---------------------------
    elif page == "📊 Generate Forecast":
        st.subheader("📊 Generate Forecast")
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first in '📤 Upload Dataset'.")
        else:
            df = st.session_state.df
            target_col = st.session_state.target_col
            date_col = st.session_state.date_col

            if date_col is None or target_col is None:
                st.error("❌ Date or target column not selected.")
                st.stop()

            # Prepare DataFrame for Prophet
            prophet_df = df[[target_col, date_col]].copy()
            prophet_df = prophet_df.rename(columns={date_col: 'ds', target_col: 'y'})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
            prophet_df = prophet_df.dropna(subset=['ds','y'])

            if prophet_df.empty:
                st.error("❌ No valid data available after cleaning.")
                st.stop()

            # Optional log-transform
            if (prophet_df['y'] <= 0).any():
                st.warning("⚠️ Some values <= 0, skipping log-transform.")
            else:
                prophet_df['y'] = np.log(prophet_df['y'])

            st.session_state.prophet_df = prophet_df

            # Forecast horizon in months
            forecast_months = st.number_input("⏳ Forecast Horizon (months)", 1, 12, 1)
            st.session_state.forecast_months = forecast_months
            forecast_days = forecast_months * 30

            # Fit Prophet
            @st.cache_data
            def fit_prophet(df):
                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                model.fit(df)
                return model

            model = fit_prophet(prophet_df)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            forecast['yhat_exp'] = np.exp(forecast['yhat'])
            st.session_state.forecast = forecast

            # Plot Forecast
            st.markdown("### 📈 Forecast vs Actual")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_exp'], mode='lines', name='Forecast', line=dict(color='#FF7F50', width=3)))
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=np.exp(prophet_df['y']), mode='lines+markers', name='Actual', line=dict(color='#1E90FF', width=3)))
            fig.update_layout(title=f"Forecast for `{target_col}`", xaxis_title="Date", yaxis_title=target_col,
                              plot_bgcolor='#F8F8FF', paper_bgcolor='#F8F8FF')
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Page 3: AI Insights
    # ---------------------------
    elif page == "🤖 AI Insights":
        st.subheader("🤖 AI-Generated Insights")
        if st.session_state.forecast is None:
            st.warning("⚠️ Please generate forecast first in '📊 Generate Forecast'.")
        else:
            prophet_df = st.session_state.prophet_df
            forecast = st.session_state.forecast
            target_col = st.session_state.target_col
            forecast_months = st.session_state.forecast_months

            last_actual = np.exp(prophet_df['y'].iloc[-1])
            last_forecast = forecast['yhat_exp'].iloc[forecast_months*30-1]
            growth = (last_forecast - last_actual)/last_actual*100

            # AI Insights prompt
            prompt = f"""
            You are a senior data analyst. Latest actual value is {last_actual:.2f}, forecast for next {forecast_months} months is {last_forecast:.2f} ({growth:.2f}% change).

            Format exactly as JSON:
            {{
                "Opportunity": "Describe main growth opportunity" 2 lines,
                "Risk": "Describe main risk" 2 lines,
                "Recommendation": "Provide actionable recommendation" 2 lines,
                "Summary": "Provide a 3-line business summary"
            }}
            """
            model_ai = genai.GenerativeModel("gemini-1.5-flash")
            response = model_ai.generate_content(prompt)

            if response and response.text:
                text = response.text.strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    try:
                        insight_json = json.loads(match.group())
                        st.markdown("### 📊 Key Insights")
                        st.info(f"📈 **Opportunity:** {insight_json.get('Opportunity','N/A')}")
                        st.warning(f"⚠️ **Risk:** {insight_json.get('Risk','N/A')}")
                        st.success(f"💡 **Recommendation:** {insight_json.get('Recommendation','N/A')}")
                        st.markdown("---")
                        st.markdown("### 📑 Business Summary")
                        st.text(insight_json.get("Summary","N/A"))
                    except:
                        st.error("⚠️ AI returned JSON but could not parse it.")
                else:
                    st.warning("⚠️ AI did not return valid JSON. Showing raw text:")
                    st.text(text)
            else:
                st.warning("⚠️ AI did not return a response.")

# ---------------------------
# Router: Login / Signup / App
# ---------------------------
if not st.session_state.authenticated:
    if "signup" not in st.session_state:
        st.session_state["signup"] = False

    if st.session_state["signup"]:
        signup_page()
    else:
        login_page()
else:
    main_app()