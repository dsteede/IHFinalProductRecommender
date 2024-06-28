import streamlit as st
import pandas as pd
import joblib

# Function to load data
@st.cache_data
def load_data():
    return pd.read_csv('/Users/daniellesteede/IH_Labs/IHFinalProductRecommender/productsreviews_cleaned_filled.csv')

# Function to load model
@st.cache_resource
def load_model():
    return joblib.load('/Users/daniellesteede/IH_Labs/IHFinalProductRecommender/pasting_model.pkl')

# Load data and model
ProductsReviewsCC = load_data()
model = load_model()

# Clean data
ProductsReviewsCC = ProductsReviewsCC[pd.to_numeric(ProductsReviewsCC['tertiary_category'], errors='coerce').isna()]

# Define recommendation function
def recommend_products(category, num_recommendations):
    filtered = ProductsReviewsCC[ProductsReviewsCC['tertiary_category'].str.contains(category, case=False, na=False)]
    if filtered.empty:
        return "No products found."
    X_filtered = filtered[['price_usd', 'rating', 'reviews']]
    filtered['predicted_loves_count'] = model.predict(X_filtered)
    top_products = filtered.sort_values(by='predicted_loves_count', ascending=False).head(num_recommendations)
    top_products = top_products[['product_name', 'brand_name', 'price_usd']]  # Keep only relevant columns
    top_products.columns = ['Product', 'Brand', 'Price (USD)']  # Rename columns for display
    return top_products

# Streamlit interface
st.image('/Users/daniellesteede/IH_Labs/IHFinalProductRecommender/ProductRex.png', use_column_width=True)

categories = sorted(ProductsReviewsCC['tertiary_category'].dropna().unique())

selected_category = st.selectbox("Select Product Category:", categories)
num_recommendations = st.slider("Number of Recommendations:", 1, 50, 5)

if st.button('Show Recommendations'):
    recommendations = recommend_products(selected_category, num_recommendations)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        recommendations = recommendations.reset_index(drop=True)  # Reset index to show sequential product numbers
        recommendations.index += 1  # Start index from 1 instead of 0
        recommendations.index.name = 'No.'  # Name the index column to 'No.'
        st.write(recommendations)
