import streamlit as st
import pandas as pd
import time
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="Dynamic ML Prediction Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CSS
# =========================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117 !important;
}
.main {
    background-color: #0E1117 !important;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #31333F;
    min-width: 260px !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
    background-color: #262730 !important;
    color: white !important;
    border: 1px solid #4B4B4B !important;
}
.stButton > button:hover {
    border: 1px solid #00FFAA !important;
    color: #00FFAA !important;
}
div[data-testid="metric-container"] {
    background-color: #161B22 !important;
    border: 1px solid #31333F;
    padding: 18px;
    border-radius: 14px;
}
#MainMenu, footer, header {
    visibility: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HERO
# =========================================
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f4037, #99f2c8);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
">
<h1 style="color: black; font-size: 42px;">
Dynamic Machine Learning Prediction Platform
</h1>
<p style="color: black; font-size: 20px; font-weight: 600;">
Upload datasets, train models, compare algorithms, and generate predictions.
</p>
</div>
""", unsafe_allow_html=True)

# =========================================
# METRICS
# =========================================
col_a, col_b, col_c = st.columns(3)

col_a.metric("Supported Algorithms", "4")
col_b.metric("ML Type", "Classification")
col_c.metric("Framework", "Streamlit")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

sample_choice = st.sidebar.selectbox(
    "📊 Or Use Sample Dataset",
    ["None", "Visa Approval", "Telecom Churn"]
)

# =========================================
# LOAD DATA
# =========================================
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

elif sample_choice == "Visa Approval":
    df = pd.read_csv("sample_data/Visadataset.csv")

elif sample_choice == "Telecom Churn":
    df = pd.read_csv("sample_data/telecom_churn_data.csv")

# =========================================
# MAIN APP
# =========================================
if df is not None:

    st.success("✅ Dataset Loaded Successfully")

    target_column = st.sidebar.selectbox("🎯 Target Column", df.columns)

    algorithm = st.sidebar.selectbox(
        "🧠 Algorithm",
        ["Decision Tree", "Random Forest", "Logistic Regression", "KNN"]
    )

    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🧠 Training", "🎯 Prediction"])

    # =========================================
    # DATA TAB
    # =========================================
    with tab1:
        st.dataframe(df.head())
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.bar_chart(df[target_column].value_counts())

    # =========================================
    # TRAINING TAB
    # =========================================
    with tab2:

        data = df.copy()

        # Handle missing values properly
        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = data[col].fillna("Missing")
            else:
                data[col] = data[col].fillna(data[col].mean())

        # Encoding
        encoders = {}
        for col in data.columns:
            if data[col].dtype == "object":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                encoders[col] = le

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 🔥 Scaling (important)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model selection
        if algorithm == "Decision Tree":
            model = DecisionTreeClassifier()

        elif algorithm == "Random Forest":
            model = RandomForestClassifier(n_estimators=50, random_state=42)

        elif algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        else:
            model = KNeighborsClassifier()

        try:
            with st.spinner("Training Model..."):
                model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)

            st.metric("Accuracy", f"{accuracy * 100:.2f}%")
            st.progress(float(accuracy))

            # Save
            st.session_state["model"] = model
            st.session_state["encoders"] = encoders
            st.session_state["scaler"] = scaler
            st.session_state["columns"] = X.columns

        except Exception as e:
            st.error(f"Training Failed: {e}")

    # =========================================
    # PREDICTION TAB
    # =========================================
    with tab3:

        st.subheader("Make Prediction")

        if "model" not in st.session_state:
            st.warning("⚠️ Train model first!")
            st.stop()

        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        scaler = st.session_state["scaler"]
        columns = st.session_state["columns"]

        input_data = {}

        for col in columns:
            if col in encoders:
                val = st.selectbox(col, encoders[col].classes_)
                input_data[col] = encoders[col].transform([val])[0]
            else:
                input_data[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):

            with st.spinner("🤖 Thinking..."):
                time.sleep(2)

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]

            if target_column in encoders:
                prediction = encoders[target_column].inverse_transform([prediction])[0]

            if str(prediction).lower() in ["denied", "reject", "no", "0"]:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#ff4d4d;
                        padding:15px;
                        border-radius:10px;
                        text-align:center;
                        font-size:20px;
                        font-weight:bold;
                        color:white;">
                        ❌ Prediction: {prediction}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#00cc66;
                        padding:15px;
                        border-radius:10px;
                        text-align:center;
                        font-size:20px;
                        font-weight:bold;
                        color:white;">
                        ✅ Prediction: {prediction}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================
# FOOTER
# =========================================
st.markdown("""
<div style="text-align:center; padding:20px; margin-top:60px; border-top:1px solid #31333F;">
<p>Developed by Harika Reddy 🚀</p>
<p style="color:gray;">Python • Streamlit • Scikit-learn • ML</p>
</div>
""", unsafe_allow_html=True)