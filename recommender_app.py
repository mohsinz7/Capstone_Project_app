import streamlit as st
import pandas as pd
from recommendations import HybridRecommender

# ✅ Set Streamlit config first
st.set_page_config(page_title="Hybrid Recommender", layout="wide")

# ✅ Load data
user_product_time = pd.read_csv("users_products_time.csv")
products_df = pd.read_csv("products_df_recomm.csv")
users_df = pd.read_csv("users_df_recomm.csv")

valid_user_ids = users_df['user_id'].unique()

# ✅ Sidebar Filters
st.sidebar.title("🔧 Recommendation Settings")

input_user_id = st.sidebar.text_input("Enter User ID", value=str(valid_user_ids[0]))
cf_weight = st.sidebar.slider("Collaborative Filtering Weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
top_n = st.sidebar.slider("Top N Recommendations", min_value=5, max_value=20, value=10)
diversity_penalty = st.sidebar.slider("Diversity Penalty Strength", min_value=0.0, max_value=0.8, value=0.15, step=0.05)

# ✅ Validate user_id input
try:
    user_id = int(input_user_id)
except ValueError:
    st.error("Please enter a valid numeric user ID.")
    st.stop()

if user_id not in valid_user_ids:
    st.error("User ID not found in the dataset.")
    st.stop()

# ✅ Initialize recommender
recommender = HybridRecommender(
    user_product_time=user_product_time,
    products_df=products_df,
    users_df=users_df,
    cf_weight=cf_weight,
    diversity_penalty=diversity_penalty,
    top_n=top_n
)

# ✅ Show user interaction history
st.subheader("📜 User's Past Interactions")

user_history = user_product_time[user_product_time['user_id'] == user_id]
user_history = user_history.sort_values(by='time_spend_by_user_on_product_page', ascending=False)
user_history = user_history.merge(products_df, left_on='product_id_visited', right_on='id', how='left')

if user_history.empty:
    st.info("No interaction history found for this user.")
else:
    history_display = user_history[['id', 'name', 'brand', 'category', 'department', 'retail_price']]
    st.dataframe(history_display.rename(columns={'id': 'product_id'}).head(10), use_container_width=True)

# ✅ Get and show recommendations
st.subheader("🔮 Recommended Products")

top_products = recommender.recommend(user_id=user_id)
recommended_display = top_products[['id', 'name', 'brand', 'category', 'department', 'retail_price', 'score']]
st.dataframe(recommended_display.rename(columns={'id': 'product_id'}), use_container_width=True)