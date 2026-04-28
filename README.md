# Customer Churn Prediction System

A machine learning-based web application to predict customer churn using Random Forest Classifier.

## Features

- 📊 **Dataset Upload & Analysis**: Upload customer datasets for batch analysis
- 🔮 **Single Customer Prediction**: Real-time churn prediction for individual customers
- 📈 **Model Insights**: View feature importance and model performance metrics
- 🎨 **Interactive Dashboard**: Built with Streamlit for easy visualization

## Model Performance

- **Algorithm**: Random Forest Classifier (200 trees)
- **Accuracy**: 47.6%
- **ROC-AUC Score**: 47.13%
- **Features**: 17 customer behavioral and demographic features

## Project Structure

```
├── app.py                          # Main Streamlit application
├── churn_model.py                  # Model training script
├── models/                         # Pre-trained model artifacts
│   ├── model.pkl                   # Trained Random Forest model
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoders.pkl          # Categorical encoders
│   └── feature_names.pkl           # Feature names used in training
├── data/                           # Training dataset
│   └── churn prediction.csv        # Customer churn dataset
├── requirements.txt                # Python dependencies
└── .streamlit/config.toml          # Streamlit configuration
```

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Customer Churn Prediction using Random Forest Classifier"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (optional, pre-trained models included)
   ```bash
   python churn_model.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

## Deployment on Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add churn prediction app"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [Streamlit Cloud](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Choose `app.py` as the main file
   - Click "Deploy"

## Features in the App

### 1. Upload & Analyze Dataset
- Upload a CSV file with customer data
- View dataset statistics (customer count, churn rate, feature count)
- Visualize churn distribution with interactive pie chart

### 2. Predict Single Customer
- Input customer details through an intuitive form
- Get real-time churn prediction with probability
- Supported features:
  - Age, Gender, Income, Spending Score, Purchase Amount
  - Product Category, Payment Method, Device
  - Is Active, Returns, Discount Used, Review Score, Session Time
  - City, State, Country, Browser

### 3. Model Insights
- View top 10 most important features
- Check model statistics and configuration
- Understand feature importance for predictions

## Input Features

| Feature | Type | Range/Values |
|---------|------|--------------|
| Age | Numeric | 18-100 |
| Gender | Categorical | Male, Female |
| Income | Numeric | 0-200,000 |
| Spending Score | Numeric | 1-100 |
| Purchase Amount | Numeric | 0+ |
| Product Category | Categorical | Electronics, Clothing, Beauty, Home, Sports, Books, Food |
| Payment Method | Categorical | UPI, Card, Cash, NetBanking |
| City | Categorical | Delhi, Hyderabad, Chennai, Mumbai |
| State | Categorical | DL, TN, MH, TS |
| Country | Categorical | IND, India, IN |
| Is Active | Categorical | Y, N |
| Returns | Numeric | 0-10 |
| Discount Used | Boolean | True, False |
| Review Score | Numeric | 0.0-5.0 |
| Browser | Categorical | Chrome, Firefox, Edge |
| Device | Categorical | Mobile, Desktop, Tablet |
| Session Time | Numeric | 0-3600 seconds |

## How the Model Works

1. **Data Preprocessing**
   - Handles missing values with mean/mode imputation
   - Encodes categorical features with LabelEncoder
   - Scales numerical features with StandardScaler

2. **Model Training**
   - Splits data: 80% train, 20% test
   - Uses Random Forest with 200 trees
   - Balanced class weights for imbalanced churn data
   - Achieves stratified cross-validation

3. **Prediction Pipeline**
   - Accepts user input for customer features
   - Applies same encoding and scaling as training data
   - Returns churn probability and risk classification

## Technologies Used

- **Framework**: Streamlit (web interface)
- **ML Library**: scikit-learn (Random Forest)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Seaborn
- **Model Serialization**: joblib

## Error Handling

- Unknown categorical values default to encoding 0
- Missing numerical inputs use sensible defaults
- Type conversion errors are caught and reported
- All categorical values match training data exactly

## Future Enhancements

- [ ] Add feature importance explanation (SHAP values)
- [ ] Support for multiple model algorithms (Gradient Boost, XGBoost)
- [ ] Real-time model retraining pipeline
- [ ] Database integration for prediction history
- [ ] User authentication for production use
- [ ] A/B testing framework for model versions

## License

MIT License - feel free to use this project for educational and commercial purposes.

## Support

For issues or questions, please create an issue in the repository.
