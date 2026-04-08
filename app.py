import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sephora Market Intelligence Dashboard", page_icon="💄", layout="wide")

# Force standard font (Inter/Sans-Serif) everywhere to prevent font glitches
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", sans-serif !important;
    }
    div[data-testid="stAlert"] * {
        font-family: "Inter", sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- LOAD RESOURCES ---
@st.cache_resource
def load_data_and_model():
    # Load Model & Encoders
    try:
        model = joblib.load('rf_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
    except:
        st.error("⚠️ Model files not found! Please ensure 'rf_model.pkl' and 'label_encoders.pkl' are in the folder.")
        st.stop()
    
    # Load the actual data for the "Overview" tab
    try:
        df = pd.read_csv("cleaned_product_info.csv")
    except:
        st.error("⚠️ Data file 'cleaned_product_info.csv' not found!")
        st.stop()
    
    # Create the 'rating_class' column for visualization if not saved
    if 'rating_class' not in df.columns:
        def categorize(r):
            # UPDATED: Matching the new Quantile thresholds (33% split)
            if r < 4.11: return 'Low'
            elif r < 4.44: return 'Medium'
            else: return 'High'
        df['rating_class'] = df['rating'].apply(categorize)
    
    return model, encoders, df

try:
    rf_model, label_encoders, df = load_data_and_model()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# =========================================================
# SIDEBAR: SCENARIO BUILDER
# =========================================================
st.sidebar.title("🛠️ Scenario Builder")
st.sidebar.markdown("Define your **New Product** attributes below:")

# 1. Brand Selection
st.sidebar.subheader("1. Brand Strategy")

all_brands = sorted(label_encoders['brand_name'].classes_)
search_brand = st.sidebar.text_input("Search Brand Portfolio", placeholder="E.g., Dior, Fenty...")

if search_brand:
    brand_list = [b for b in all_brands if search_brand.lower() in b.lower()]
else:
    brand_list = all_brands

if not brand_list:
    st.sidebar.warning("No matching brands found.")
    selected_brand = st.sidebar.selectbox("Select Brand", all_brands) 
else:
    selected_brand = st.sidebar.selectbox("Select Brand", brand_list)

# Filter dataframe by Brand FIRST
df_brand = df[df['brand_name'] == selected_brand]

# 2. Dynamic Category Selection
st.sidebar.subheader("2. Product Taxonomy")

# Step A: Primary
primary_cats = sorted(df_brand['primary_category'].unique())
# Fallback if brand has no data
if not primary_cats:
    primary_cats = sorted(label_encoders['primary_category'].classes_)

selected_primary = st.sidebar.selectbox("Primary Category", primary_cats)

# Step B: Secondary
df_primary = df_brand[df_brand['primary_category'] == selected_primary]
secondary_options = sorted(df_primary['secondary_category'].unique())
if not secondary_options:
    secondary_options = sorted(label_encoders['secondary_category'].classes_)

selected_secondary = st.sidebar.selectbox("Secondary Category", secondary_options)

# Step C: Tertiary
df_secondary = df_primary[df_primary['secondary_category'] == selected_secondary]
tertiary_options = sorted(df_secondary['tertiary_category'].unique())
if not tertiary_options:
    tertiary_options = sorted(label_encoders['tertiary_category'].classes_)

selected_tertiary = st.sidebar.selectbox("Tertiary Category", tertiary_options)

# 3. Price & Attributes
st.sidebar.subheader("3. Market Positioning")
price = st.sidebar.slider("Proposed Retail Price (USD)", 5.0, 300.0, 35.0)
is_new = st.sidebar.checkbox("Tag as 'New Launch'", True)
is_exclusive = st.sidebar.checkbox("Sephora Exclusive Strategy", False)
is_limited = st.sidebar.checkbox("Limited Edition Release", False)

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 Pre-Launch Simulator", "📈 Historical Market Data", "🧠 Model Explainability"])

# =========================================================
# TAB 1: MARKET SIMULATOR
# =========================================================
with tab1:
    st.header("🔮 Simulation: Predict Success for NEW Concepts")
    st.markdown("Use this tool to simulate how an **UNSEEN** product might perform based on learned market patterns.")

    if st.button("🚀 Execute Simulation", type="primary"):
        
        # --- A. PREDICTION ---
        input_data = pd.DataFrame({
            'brand_name': [selected_brand], 'price_usd': [price],
            'primary_category': [selected_primary], 'secondary_category': [selected_secondary],
            'tertiary_category': [selected_tertiary], 'online_only': [0],
            'limited_edition': [int(is_limited)], 'new': [int(is_new)], 'sephora_exclusive': [int(is_exclusive)]
        })

        # Encode
        for col in ['brand_name', 'primary_category', 'secondary_category', 'tertiary_category']:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # 1. Get Base Prediction
        raw_pred_class = rf_model.predict(input_data)[0]
        prob = rf_model.predict_proba(input_data).max()

        # 2. Contextual Validation (SMARTER AVERAGE)
        specific_data = df[(df['brand_name'] == selected_brand) & (df['primary_category'] == selected_primary)]
        
        if not specific_data.empty:
            avg_price = specific_data['price_usd'].mean()
            context_msg = f"average for {selected_brand} {selected_primary}"
            short_context = f"{selected_primary} Avg" 
        else:
            brand_data = df[df['brand_name'] == selected_brand]
            if not brand_data.empty:
                avg_price = brand_data['price_usd'].mean()
                context_msg = f"{selected_brand} brand average"
                short_context = "Brand Avg"
            else:
                avg_price = price
                context_msg = "market average"
                short_context = "Market Avg"

        # --- BUSINESS LOGIC ---
        TOO_HIGH = avg_price * 1.40 
        TOO_LOW = avg_price * 0.70  

        final_class = raw_pred_class
        penalty_msg = ""

        # --- 3. APPLY PENALTIES ---
        if prob < 0.50:
            final_class = 1
            penalty_msg = " (Uncertain: Mixed Reviews Expected)"

        if penalty_msg == "":
            if price > TOO_HIGH:
                if raw_pred_class == 2:
                    final_class = 1
                    prob = prob - 0.20
                    penalty_msg = " (Penalty: Price is unrealistically high)"
            elif price < TOO_LOW:
                if raw_pred_class == 2:
                    final_class = 1
                    prob = prob - 0.15
                    penalty_msg = " (Penalty: Price is suspiciously low)"

        # 4. Display Final Result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Price", f"${price}")
            st.metric(f"{short_context}", f"${avg_price:.2f}")
            
        with col2:
            st.markdown(f"**Brand Context**")
            st.markdown(f"#### {selected_brand}") 
            
        with col3:
            st.markdown("**Prediction Verdict**")
            
            # --- A. The Result (Clear Label) ---
            if final_class == 2:
                st.success(f"🌟 **Top Tier** (> 4.44 Stars)")
                desc = "Predicted Market Leader (Top 33%)"
            elif final_class == 1:
                st.warning(f"⚖️ **Average** (4.1 - 4.4 Stars)")
                desc = "Competitive but not a standout."
            else:
                st.error(f"📉 **Below Average** (< 4.11 Stars)")
                desc = "Likely to underperform competitors."
                
            # --- B. The Confidence Meter (Visual Bar) ---
            st.write("AI Confidence Score")
            st.progress(prob) 
            
            # --- C. The Explanation ---
            if penalty_msg:
                st.caption(f"**{prob:.1%} Probability** {penalty_msg}")
            else:
                st.caption(f"**{prob:.1%} Probability**")
            
            st.caption(f"*{desc}*")

        st.divider()

        # --- B. NEW: MARKETING IMPACT ANALYSIS (Connected to Final Score) ---
        st.subheader("Marketing Strategy Impact")
        st.markdown("How do your checkbox selections affect the probability?")
        
        # 1. Calculate Base Probability (Raw)
        base_data = input_data.copy()
        base_data['new'] = 0
        base_data['limited_edition'] = 0
        base_data['sephora_exclusive'] = 0
        base_prob = rf_model.predict_proba(base_data)[0][2]
        
        # 2. Calculate Current Probability (Raw)
        current_raw_prob = rf_model.predict_proba(input_data)[0][2]
        
        # 3. Difference (Marketing Lift)
        impact = current_raw_prob - base_prob
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if is_new: st.success("✅ 'New Launch' Active")
            else: st.caption("⬜ 'New Launch' Off")
        with c2:
            if is_exclusive: st.success("✅ 'Exclusive' Active")
            else: st.caption("⬜ 'Exclusive' Off")
        with c3:
            if is_limited: st.success("✅ 'Limited' Active")
            else: st.caption("⬜ 'Limited' Off")
            
        # 4. Generate Reasoning
        reasoning = []
        if price > (avg_price * 1.5) and is_limited:
            reasoning.append("The combination of **High Price** and **Limited Edition** implies a 'Niche' product, limiting mass appeal.")
        if is_exclusive and impact > 0:
            reasoning.append("Marking this as **Sephora Exclusive** is driving positive engagement due to perceived scarcity.")
        if is_new and impact > -0.01: 
            reasoning.append("The **New Launch** tag is providing a standard 'Honeymoon Phase' boost.")

        # 5. Display Impact (Using FINAL 'prob' to decide color)
        if impact > 0.01:
            # CASE A: Impact is positive AND Final Score is Good (Green)
            # We use 'prob' here because it includes the Price Penalty from Section A
            if prob >= 0.50:
                st.success(f"📈 **Strategy Boost:** Adding these tags increased success probability by **+{impact:.1%}**.")
                for r in reasoning: st.markdown(f"- *{r}*")
            
            # CASE B: Impact is positive BUT Final Score is Bad (Yellow)
            else:
                st.warning(f"**Marginal Help:** These tags improved the score by **+{impact:.1%}**, but the product is **still predicted to underperform**.")
                st.warning(f"**Reason:** The negative impact of the high Price (${price}) is overpowering your marketing strategy.")
                
        elif impact < -0.01:
            st.error(f"📉 **Negative Impact:** These tags actually lowered the success probability by **{impact:.1%}**.")
            if reasoning:
                for r in reasoning: st.markdown(f"- *{r}*")
            else:
                st.markdown("- *Historical data suggests these tags do not align with this specific Brand/Price strategy.*")
        else:
            st.info(f"⚖️ **Neutral Impact:** These marketing tags have a negligible effect on this specific Brand/Category combination.")

        # --- C. AUTOMATED STRATEGIC ANALYSIS ---
        st.subheader("Automated Strategic Analysis")
        st.markdown("**Executive Report**")
        
        insight_points = []
        
        if price > TOO_HIGH:
            insight_points.append(f"**Price Positioning Alert:** The proposed price (\${price}) is >40% higher than the **{context_msg}** (\${avg_price:.2f}). This exceeds the standard markup strategy.")
        elif price < TOO_LOW:
            insight_points.append(f"**Brand Risk:** The proposed price (\${price}) is significantly below the **{context_msg}** (\${avg_price:.2f}). This risks devaluing the brand image or incurring loss (rugi).")
        else:
            insight_points.append(f"**Market Alignment:** The proposed price is within the optimal range (±30-40%) of the **{context_msg}** (\${avg_price:.2f}).")

        if prob < 0.50:
            insight_points.append("**Low Confidence Warning:** The model is undecided (Probability < 50%). This suggests the product attributes do not strongly match any historical success pattern.")
        
        for point in insight_points:
            st.info(point, icon="📊")

        st.divider()
        
        # --- D. PRICE SENSITIVITY CURVE ---
        st.subheader("Price Sensitivity & Optimization")
        
        prices_to_test = np.linspace(5, 150, 50)
        probs = []
        
        b_code = label_encoders['brand_name'].transform([selected_brand])[0]
        p_code = label_encoders['primary_category'].transform([selected_primary])[0]
        s_code = label_encoders['secondary_category'].transform([selected_secondary])[0]
        t_code = label_encoders['tertiary_category'].transform([selected_tertiary])[0]

        for p in prices_to_test:
            row = [[b_code, p, p_code, s_code, t_code, 0, int(is_limited), int(is_new), int(is_exclusive)]]
            p_success = rf_model.predict_proba(row)[0][2]
            probs.append(p_success)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=prices_to_test, y=probs, color='#2E8B57', linewidth=2.5, ax=ax)
        ax.axvline(x=price, color='#B22222', linestyle='--', label=f"Current Selection (${price})")
        ax.set_ylabel("Probability of High Success")
        ax.set_xlabel("Price Point ($)")
        ax.set_title(f"Price Elasticity Analysis: {selected_brand}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        st.caption("Strategic Note: Identify the inflection point where probability declines to determine the maximum revenue potential before consumer satisfaction is impacted.")

# =========================================================
# TAB 2: HISTORICAL OVERVIEW
# =========================================================
with tab2:
    st.header(f"Competitive Analysis: {selected_brand}")
    st.markdown("Compare your selected brand against the entire Sephora market ecosystem.")

    # --- 1. COMPARATIVE KPIs ---
    # We calculate the "Market" stats (Everyone) vs "Brand" stats (Selected)
    
    # Market Stats
    market_avg_price = df['price_usd'].mean()
    market_avg_rating = df['rating'].mean()
    
    # Brand Stats
    brand_data = df[df['brand_name'] == selected_brand]
    if brand_data.empty:
        st.warning(f"No historical data found for {selected_brand}.")
        brand_avg_price = 0
        brand_avg_rating = 0
        brand_count = 0
    else:
        brand_avg_price = brand_data['price_usd'].mean()
        brand_avg_rating = brand_data['rating'].mean()
        brand_count = len(brand_data)

    # Display Metrics with Delta (Green/Red arrows)
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.metric(
            label="Brand Avg Price",
            value=f"${brand_avg_price:.2f}",
            delta=f"${brand_avg_price - market_avg_price:.2f} vs Market",
            delta_color="inverse" # Red if higher (expensive), Green if lower (cheaper)
        )
    
    with kpi2:
        st.metric(
            label="Brand Avg Rating",
            value=f"{brand_avg_rating:.2f} ⭐",
            delta=f"{brand_avg_rating - market_avg_rating:.2f} vs Market",
            delta_color="normal" # Green if higher (better), Red if lower (worse)
        )
        
    with kpi3:
        st.metric(
            label="Portfolio Size",
            value=f"{brand_count} Products",
            help="Number of unique products currently listed on Sephora."
        )

    st.divider()

    # --- 2. THE STRATEGY MATRIX (Scatter Plot) ---
    st.subheader("Market Positioning Matrix (Price vs. Quality)")
    st.markdown("Where does this brand sit compared to competitors? (Red = **{brand}**, Grey = **Competitors**)".format(brand=selected_brand))

    # Create a "Highlight" column for the chart
    plot_df = df.copy()
    plot_df['Type'] = np.where(plot_df['brand_name'] == selected_brand, 'Selected Brand', 'Competitor')
    
    # Define colors: Red for Brand, Light Grey for others
    palette = {'Selected Brand': '#FF4B4B', 'Competitor': '#4D4C4C'}
    sizes = {'Selected Brand': 60, 'Competitor': 15}
    alphas = {'Selected Brand': 1.0, 'Competitor': 0.3}

    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
    
    # Plot Competitors first (so they are in background)
    sns.scatterplot(
        data=plot_df[plot_df['Type'] == 'Competitor'], 
        x='price_usd', y='rating', 
        color="#4D4C4C", alpha=0.3, s=15, label='Competitors', ax=ax_scatter
    )
    
    # Plot Selected Brand on top
    sns.scatterplot(
        data=plot_df[plot_df['Type'] == 'Selected Brand'], 
        x='price_usd', y='rating', 
        color='#FF4B4B', alpha=1.0, s=80, edgecolor='black', label=selected_brand, ax=ax_scatter
    )

    ax_scatter.set_title(f"Price vs. Rating Landscape", fontsize=12)
    ax_scatter.set_xlabel("Price ($)")
    ax_scatter.set_ylabel("Star Rating")
    ax_scatter.axhline(y=market_avg_rating, color='blue', linestyle='--', alpha=0.5, label='Market Avg Rating')
    ax_scatter.legend()
    st.pyplot(fig_scatter)
    
    st.caption("Insight: If the red dots are above the blue line, the brand outperforms the market average.")

    st.divider()

    # --- 3. CATEGORY STRENGTH ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader(f"What does {selected_brand} sell?")
        # Bar Chart of Top Categories for THIS Brand
        if not brand_data.empty:
            cat_counts = brand_data['primary_category'].value_counts().head(5)
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            sns.barplot(x=cat_counts.values, y=cat_counts.index, palette='magma', ax=ax_bar)
            ax_bar.set_xlabel("Number of Products")
            st.pyplot(fig_bar)
        else:
            st.info("No data available for category analysis.")

    with c2:
        st.subheader("Price Distribution Strategy")
        # Histogram comparing Brand vs Market
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        
        # Market (Grey)
        sns.kdeplot(df['price_usd'], color='grey', fill=True, alpha=0.2, label='Total Market', ax=ax_hist)
        # Brand (Red)
        if not brand_data.empty:
            sns.kdeplot(brand_data['price_usd'], color='red', fill=True, alpha=0.5, label=selected_brand, ax=ax_hist)
        
        ax_hist.set_xlim(0, 300) # Limit to $300 to avoid outliers squishing the chart
        ax_hist.set_xlabel("Price ($)")
        ax_hist.legend()
        st.pyplot(fig_hist)
# =========================================================
# TAB 3: MODEL DNA (Advanced Insights)
# =========================================================

with tab3:
    st.header("Model DNA & Performance Audit")
    st.markdown("Deep dive into the algorithm's logic, accuracy, and bias.")

    # --- 1. MODEL HEALTH CHECK (Live Validation) ---
    st.subheader("1. Model Health Check")
    
    # We will run a quick prediction on the loaded data to check accuracy
    X_full = df.copy()
    
    try:
        # We just use the variables directly since we know they are loaded
        for col in ['brand_name', 'primary_category', 'secondary_category', 'tertiary_category']:
            # Transform data using the encoders loaded at the top
            X_full[col] = label_encoders[col].transform(X_full[col])
            
        X_test = X_full[['brand_name', 'price_usd', 'primary_category', 'secondary_category', 
                         'tertiary_category', 'online_only', 'limited_edition', 'new', 'sephora_exclusive']]
        
        # Get Predictions using the model loaded at the top
        y_pred = rf_model.predict(X_test)
        
        # Map existing rating_class to numbers (Low=0, Medium=1, High=2)
        y_true = df['rating_class'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        # Calculate Accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Model Accuracy", f"{accuracy:.1%}", "Based on Historical Data")
        m2.metric("Training Sample Size", f"{len(df)}", "Products Analyzed")
        m3.metric("Algorithm Type", "Random Forest", "Ensemble Method")
        
    except Exception as e:
        st.warning(f"Could not calculate live accuracy. Error details: {e}")

    st.divider()

    # --- 2. FEATURE IMPORTANCE (The "Why") ---
    st.subheader("2. Determinants of Success")
    st.markdown("Which attributes have the biggest impact on the **Beauty Insider Score**?")
    
    # Ensure we use the correct model variable
    # Use the model directly
    importances = rf_model.feature_importances_
    
    feature_names = ['Brand', 'Price', 'Category (L1)', 'Category (L2)', 'Category (L3)', 'Online Only', 'Limited Ed.', 'New Launch', 'Exclusive']
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    col_chart, col_text = st.columns([2, 1])
    
    with col_chart:
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        # Sephora Palette: Red for Top Feature
        sns.barplot(x='Importance', y='Feature', data=feat_df, palette=['#CE0F2D' if x == feat_df['Importance'].max() else '#333333' for x in feat_df['Importance']], ax=ax_imp)
        ax_imp.set_xlabel("Relative Importance Score")
        st.pyplot(fig_imp)
        
    with col_text:
        st.info("""
        **How to read this:**
        
        The **Brand Name** and **Price** are the strongest predictors. 
        
        This means the model looks at *Who made it* and *How much it costs* before it even looks at marketing tags like 'New' or 'Exclusive'.
        """)

    st.divider()

    # --- 3. CORRELATION & RELATIONSHIPS ---
    st.subheader("3. Strategic Correlations")
    st.markdown("Do higher prices actually lead to higher ratings?")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Price vs Rating Scatter
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        # Filter outliers for better view (< $200)
        viz_df = df[df['price_usd'] < 200]
        sns.regplot(data=viz_df, x='price_usd', y='rating', scatter_kws={'alpha':0.1, 'color': 'grey'}, line_kws={'color': '#CE0F2D'}, ax=ax_corr)
        ax_corr.set_title("Price vs. Rating Correlation")
        ax_corr.set_xlabel("Price ($)")
        ax_corr.set_ylabel("Rating (Stars)")
        st.pyplot(fig_corr)
        st.caption("The Red Line shows the trend. If it goes up, higher prices generally mean better ratings.")

    with c2:
        # Rating Class Distribution
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        class_counts = df['rating_class'].value_counts()
        # Custom colors: High=Black, Med=Grey, Low=Red
        sns.barplot(x=class_counts.index, y=class_counts.values, palette={'High': '#333333', 'Medium': 'grey', 'Low': '#CE0F2D'}, ax=ax_dist)
        ax_dist.set_title("Target Class Distribution (Balanced)")
        st.pyplot(fig_dist)
        st.caption("We have balanced the training classes (33% each) to prevent the model from bias. This ensures 'High' only represents the true top-tier.")
