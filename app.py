import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Dynamic ML Prediction Platform",
    layout="centered"
)

# =========================================
# HERO SECTION (RESPONSIVE)
# =========================================
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f4037, #99f2c8);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
">
<h1 style="color: black; font-size: clamp(24px, 5vw, 42px);">
Dynamic Machine Learning Prediction Platform
</h1>
<p style="color: black; font-size: clamp(14px, 3vw, 20px); font-weight: 600;">
Upload datasets, train models, compare algorithms, and generate predictions.
</p>
</div>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

sample_choice = st.sidebar.selectbox(
    "Or use sample dataset",
    ["None", "Visa Approval", "Telecom Churn"]
)

# =========================================
# LOAD DATA
# =========================================
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif sample_choice == "Visa Approval":
    df = pd.read_csv("sample_data/Visadataset.csv")

elif sample_choice == "Telecom Churn":
    df = pd.read_csv("sample_data/telecom_churn_data.csv")

# =========================================
# MAIN APP
# =========================================
if df is not None:

    st.success("✅ Dataset Loaded")

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    # ✅ Strict validation
    if df[target_column].nunique() > 0.5 * len(df):
        st.error("❌ Wrong target selected! Please choose a categorical column like Yes/No, Approved/Denied.")
        st.stop()

    algorithm = st.selectbox(
        "🧠 Select Algorithm",
        ["Decision Tree", "Random Forest", "Logistic Regression", "KNN"]
    )

    tab1, tab2, tab3 = st.tabs(["📊 Data", "🧠 Train", "🎯 Predict"])

    # =========================================
    # DATA TAB
    # =========================================
    with tab1:
        st.dataframe(df.head(), width="stretch")
        st.write("Shape:", df.shape)

    # =========================================
    # TRAINING TAB
    # =========================================
    with tab2:

        data = df.copy()

        # Remove missing target
        data = data.dropna(subset=[target_column])

        encoders = {}

        # Process features
        for col in data.columns:
            if col == target_column:
                continue

            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].astype(str).fillna("Missing")
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                encoders[col] = le

        # Process target
        data[target_column] = data[target_column].astype(str)
        le_target = LabelEncoder()
        data[target_column] = le_target.fit_transform(data[target_column])
        encoders[target_column] = le_target

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Clean X
        X = X.loc[:, X.nunique() > 1]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model
        if algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algorithm == "Random Forest":
            model = RandomForestClassifier(n_estimators=50)
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = KNeighborsClassifier()

        try:
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)

            acc = model.score(X_test, y_test)

            st.success(f"Accuracy: {acc*100:.2f}%")
            st.progress(float(acc))

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

        if "model" not in st.session_state:
            st.warning("⚠️ Train model first")
            st.stop()

        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        scaler = st.session_state["scaler"]
        columns = st.session_state["columns"]

        input_data = {}

        st.subheader("Enter Input Data")

        cols = st.columns(2)

        for i, col in enumerate(columns):
            with cols[i % 2]:
                if col in encoders:
                    val = st.selectbox(col, encoders[col].classes_)
                    input_data[col] = encoders[col].transform([val])[0]
                else:
                    input_data[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):

            with st.spinner("🤖 Thinking..."):
                time.sleep(1.5)

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=columns, fill_value=0)

            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]

            pred = encoders[target_column].inverse_transform([pred])[0]

            if str(pred).lower() in ["denied", "reject", "no", "0"]:
                st.error(f"❌ Prediction: {pred}")
            else:
                st.success(f"✅ Prediction: {pred}")

# =========================================
# FOOTER (ADDED BACK)
# =========================================
st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    margin-top:50px;
    border-top:1px solid #31333F;
">
<p style="margin:5px;">Developed by Harika Reddy 🚀</p>
<p style="color:gray; font-size:14px;">
Python • Streamlit • Scikit-learn • Machine Learning
</p>
</div>
""", unsafe_allow_html=True)