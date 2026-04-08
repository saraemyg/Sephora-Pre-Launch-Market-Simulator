# Sephora Market Intelligence Dashboard

A machine learning-powered Streamlit application that predicts product success in the beauty market using historical Sephora data. This dashboard helps businesses simulate new product launches and make data-driven pricing and marketing decisions.

## Features

- **🔮 Pre-Launch Simulator**: Predict success probability for new product concepts based on brand, category, price, and marketing attributes
- **📈 Historical Market Analysis**: Competitive analysis comparing selected brand against market with price vs. quality positioning matrix
- **🧠 Model Explainability**: Deep dive into model performance, feature importance, and strategic correlations
- **💡 Price Sensitivity Analysis**: Visualize how pricing impacts success probability and identify optimal price points
- **📊 Marketing Strategy Impact**: Understand how "New Launch", "Exclusive", and "Limited Edition" tags affect predicted success

## Dataset

- **Source**: Sephora product database
- **Scope**: 5,000+ beauty products with attributes including:
  - Brand name and pricing
  - Product categories (3-tier hierarchy)
  - Customer ratings
  - Marketing flags (exclusive, limited edition, new launch status)

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the following files are in the project directory:
   - `rf_model.pkl` - Trained Random Forest model
   - `label_encoders.pkl` - Label encoders for categorical features
   - `cleaned_product_info.csv` - Historical product data

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Workflow

1. **Configure Product**: Use the sidebar to define your new product concept
   - Select brand strategy
   - Choose product category (primary, secondary, tertiary)
   - Set price point and marketing attributes

2. **Run Simulation**: Click "Execute Simulation" to get predictions
   - View predicted success tier (Top Tier / Average / Below Average)
   - Analyze marketing strategy impact
   - Review automated strategic analysis

3. **Explore Market**: Navigate to "Historical Market Data" tab
   - Compare your selected brand against competitors
   - View price vs. quality positioning
   - Analyze category strength and distribution

4. **Understand Model**: Check "Model Explainability" tab
   - Review model accuracy metrics
   - Explore feature importance
   - Analyze price-rating correlations

## Model Details

- **Algorithm**: Random Forest Classifier
- **Output Classes**: 3-tier rating classification
  - High (> 4.44 stars): Top 33% performers
  - Medium (4.11 - 4.44 stars): Competitive products
  - Low (< 4.11 stars): Below-average performers

- **Features Analyzed**:
  1. Brand name
  2. Price (USD)
  3. Product categories (3 levels)
  4. Online exclusivity
  5. Limited edition flag
  6. New launch flag
  7. Sephora exclusive flag

## Business Logic

The app applies intelligent pricing penalties:
- **Price too high** (>40% above category average): May downgrade predictions
- **Price too low** (<30% below category average): Risk of brand devaluation
- **Model confidence < 50%**: Marked as uncertain predictions

## Project Structure

```
├── app.py                      # Main Streamlit application
├── rf_model.pkl               # Pre-trained Random Forest model (not in git)
├── label_encoders.pkl         # Categorical encoders (not in git)
├── cleaned_product_info.csv   # Historical dataset (not in git)
├── Data Preproccessing.ipynb  # Data cleaning & preprocessing
├── model.ipynb                # Model training notebook
└── README.md                  # This file
```

## Files Not Included

Large files required for operation are not in the repository:
- Model file (~32MB)
- Dataset file (~1MB)
- Label encoders (~8KB)

These should be generated from the Jupyter notebooks or obtained separately.

## Author

Created as part of Data Mining Assignment 2 (CDS6314)

## License

Academic use only
