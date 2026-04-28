import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to churn using Machine Learning.")

# Load model and preprocessing artifacts
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, label_encoders, feature_names

model, scaler, label_encoders, feature_names = load_model()

# Sidebar Navigation
option = st.sidebar.radio("Select Option", 
    ["Upload & Analyze Dataset", "Predict Single Customer", "Model Insights"])

# ------------------- 1. Upload & Analyze Dataset -------------------
if option == "Upload & Analyze Dataset":
    st.header("📂 Upload Customer Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} customer records")
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        churn_rate = (df['Churn'].value_counts(normalize=True).get(1, 0) * 100) if 'Churn' in df.columns else 0
        col1.metric("Total Customers", len(df))
        col2.metric("Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Features", len(df.columns))
        
        # Visualizations
        if 'Churn' in df.columns:
            fig = px.pie(df, names='Churn', title="Churn Distribution")
            st.plotly_chart(fig, use_container_width=True)

# ------------------- 2. Predict Single Customer -------------------
elif option == "Predict Single Customer":
    st.header("🔮 Real-time Churn Prediction")
    
    # Create input form with actual dataset features
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
        spending_score = st.slider("Spending Score", min_value=1, max_value=100, value=50)
        purchase_amount = st.number_input("Purchase Amount", min_value=0.0, value=5000.0)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Beauty", "Home", "Sports", "Books", "Food"])
        payment_method = st.selectbox("Payment Method", ["UPI", "Card", "Cash", "NetBanking"])
        device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
    
    col3, col4 = st.columns(2)
    with col3:
        is_active = st.selectbox("Is Active", ["Y", "N"])
        city = st.selectbox("City", ["Delhi", "Hyderabad", "Chennai", "Mumbai"])
        returns = st.number_input("Returns", min_value=0, max_value=10, value=0)
    
    with col4:
        discount_used = st.selectbox("Discount Used", ["True", "False"])
        state = st.selectbox("State", ["DL", "TN", "MH", "TS"])
        country = st.selectbox("Country", ["IND", "India", "IN"])
        
    col5, col6 = st.columns(2)
    with col5:
        browser = st.selectbox("Browser", ["Chrome", "Firefox", "Edge"])
        review_score = st.slider("Review Score", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
    
    with col6:
        session_time = st.number_input("Session Time (seconds)", min_value=0, max_value=3600, value=300)
    
    if st.button("🚀 Predict Churn", type="primary"):
        # Prepare input data matching training features (excluding CustomerID and LastPurchaseDate)
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Income': [income],
            'SpendingScore': [spending_score],
            'PurchaseAmount': [purchase_amount],
            'ProductCategory': [product_category],
            'PaymentMethod': [payment_method],
            'City': [city],
            'State': [state],
            'Country': [country],
            'IsActive': [is_active],
            'Returns': [float(returns)],
            'DiscountUsed': [discount_used == "True"],  # Convert string to boolean
            'ReviewScore': [review_score],
            'Browser': [browser],
            'Device': [device],
            'SessionTime': [session_time]
        })
        
        # Apply label encoding to categorical features
        processed_data = input_data.copy()
        
        # Encode categorical columns that have label encoders
        for col in label_encoders.keys():
            if col in processed_data.columns:
                try:
                    processed_data[col] = label_encoders[col].transform([processed_data[col].iloc[0]])[0]
                except (ValueError, KeyError):
                    # If value not seen during training, use default (0)
                    st.warning(f"⚠️ Unknown value for {col}. Using default encoding.")
                    processed_data[col] = 0
        
        # Keep only the features used during training (in correct order)
        processed_data = processed_data[feature_names]
        
        # Convert all to numeric/float (bool converts to 0.0/1.0 automatically)
        processed_data = processed_data.astype(float)
        
        # Scale the features
        input_scaled = scaler.transform(processed_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn: {probability*100:.1f}% probability")
        else:
            st.success(f"✅ Low Risk of Churn: {probability*100:.1f}% probability")


# ------------------- 3. Model Insights -------------------
else:
    st.header("📊 Model Insights")
    st.subheader("Feature Importance and Performance Metrics")
    
    # Get feature importance from the model
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                 title='Top 10 Most Important Features',
                 labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                 orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model statistics
    col1, col2 = st.columns(2)
    col1.metric("Model Type", "Random Forest Classifier")
    col2.metric("Total Features", len(feature_names))
    
    st.info("""
    **Model Information:**
    - **Algorithm:** Random Forest with 200 trees
    - **Training Split:** 80% train, 20% test
    - **Features:** All numerical and encoded categorical features
    - **Performance:** Balanced for both churn and non-churn classes
    """)


st.sidebar.info("""
**Model Used:** Random Forest Classifier  
**Why RF?** Handles mixed data types well, robust to overfitting, and provides feature importance.
""")