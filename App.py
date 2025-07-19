import streamlit as st
import pickle
import datetime
import pandas as pd
from PIL import Image
import time

# Set page config
# st.set_page_config(
#     page_title="🏠 House Price Estimator",
#     page_icon="🏡",
#     layout="centered",
#     initial_sidebar_state="auto"
# )

# Custom CSS with background color
st.markdown("""
    <style>
        body {
            background-color: #eaf4f4;
        }
        .main {
            background-color: #fefefe;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1d391kg, .css-18e3th9 {
            background-color: #f0fbfc;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and data
with open("house_price_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data.csv")
df.dropna(inplace=True)

# Title and banner
# st.image("https://img.freepik.com/free-vector/house-sale-concept-illustration_114360-7331.jpg", use_column_width=True)
st.title("🏠 BrickWorth")
st.subheader("House Price Prediction App")
st.subheader("Clear Smart Estimation for Smarter Investment Decisions")
st.markdown("""
This app uses a trained machine learning model to estimate the selling price of a house based on its features. 
Simply enter the details below and get an instant estimate.
""")

# Sidebar - Branding and Help
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=100)
st.sidebar.title("About the App")
st.sidebar.info("""
This app was built using:
- Python 🐍
- Streamlit ⚡
- Scikit-learn 📊

Developed by: Ebaadullah kosgi
""")

# User Inputs
st.header("📝 Enter House Details")
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 0, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", 0, 10000, 1500)
    sqft_lot = st.number_input("Lot Area (sqft)", 0, 100000, 5000)
    sqft_above = st.number_input("Sqft Above Ground", 0, 10000, 1200)

with col2:
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 300)
    waterfront = st.selectbox("Waterfront View", [0, 1], format_func=lambda x: "Yes" if x else "No")
    city = st.selectbox("Select City", sorted(df["city"].unique()))
    statezip = st.selectbox("Select State Zip Code", sorted(df["statezip"].unique()))
    yr_built = st.number_input("Year Built", 1900, datetime.datetime.now().year, 2000)

# Year fields
yr_renovated = st.number_input("Year Renovated (0 if never)", 0, datetime.datetime.now().year, 0)
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition (1–5)", 1, 5, 3)

# Derived fields
current_year = datetime.datetime.now().year
house_age = current_year - yr_built
has_been_renovated = 1 if yr_renovated > 0 else 0

# Prediction button
st.markdown("---")
st.header("🔮 Price Estimation")
input_df = pd.DataFrame([{
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'sqft_lot': sqft_lot,
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement,
    'house_age': house_age,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'city': city,
    'statezip': statezip,
    'has_been_renovated': has_been_renovated
}])

if st.button("📈 Predict House Price"):
    with st.spinner('Estimating price... please wait...'):
        time.sleep(2)
        prediction = model.predict(input_df)
        st.balloons()
        st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")
        st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=100)

# Footer
st.markdown("""
---
<div style='text-align: center;'>
    Built with ❤️ using Streamlit | 2025
</div>
""", unsafe_allow_html=True)

