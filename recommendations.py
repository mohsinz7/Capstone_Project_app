import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import faiss

class HybridRecommender:
    def __init__(self, user_product_time, products_df, users_df, cf_weight=0.5, diversity_penalty=0.15, top_n=10):
        self.user_product_time = user_product_time
        self.products_df = products_df
        self.users_df = users_df
        self.cf_weight = cf_weight
        self.diversity_penalty = diversity_penalty
        self.top_n = top_n
        self._prepare()

    def _prepare(self):
        self._encode_users_products()
        self._build_cf_model()
        self._build_cb_model()

    def _encode_users_products(self):
        df = self.user_product_time.dropna(subset=['user_id', 'product_id_visited'])
        df['user_code'] = df['user_id'].astype("category").cat.codes
        df['product_code'] = df['product_id_visited'].astype("category").cat.codes
        self.user_id_map = dict(enumerate(df['user_id'].astype("category").cat.categories))
        self.product_id_map = dict(enumerate(df['product_id_visited'].astype("category").cat.categories))
        self.reverse_user_id_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_product_id_map = {v: k for k, v in self.product_id_map.items()}
        self.df_encoded = df

    def _build_cf_model(self):
        matrix = csr_matrix((
            self.df_encoded['time_spend_by_user_on_product_page'],
            (self.df_encoded['user_code'], self.df_encoded['product_code'])
        ))
        svd = TruncatedSVD(n_components=50, random_state=42)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T.astype('float32')
        self.faiss_index = faiss.IndexFlatIP(self.item_factors.shape[1])
        self.faiss_index.add(self.item_factors)

    def _build_cb_model(self):
        product_features = ['brand', 'category', 'department', 'retail_price']
        self.product_content_df = self.products_df[['id'] + product_features].dropna()
        self.cb_product_ids = set(self.product_content_df['id'])

        self.preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'category', 'department']),
            ('num', StandardScaler(), ['retail_price'])
        ])
        self.product_features_matrix = self.preprocessor.fit_transform(self.product_content_df)
        self.product_id_to_index = dict(zip(self.product_content_df['id'], range(len(self.product_content_df))))
        self.index_to_product_id = dict(zip(range(len(self.product_content_df)), self.product_content_df['id']))

    def _normalize_scores(self, scores):
        """Normalize an array of scores between 0 and 1"""
        if np.max(scores) == np.min(scores):  # Avoid division by zero
            return np.zeros_like(scores)
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    def recommend(self, user_id):
        is_cold_start = user_id not in self.users_df['user_id'].values

        if is_cold_start:
            cb_scores = self._get_cb_scores(user_id)
            scores = cb_scores
            product_ids = self.cb_product_ids
        else:
            cf_scores_full = self._get_cf_scores(user_id)
            cb_scores_full = self._get_cb_scores(user_id)

            # Align product IDs
            cf_product_ids = set(self.product_id_map.values())
            cb_product_ids = set(self.cb_product_ids)
            common_ids = list(cf_product_ids.intersection(cb_product_ids))

            # Reindex both CF and CB scores
            cf_indices = [self.reverse_product_id_map[pid] for pid in common_ids]
            cb_indices = [self.product_id_to_index[pid] for pid in common_ids]

            # Normalize both CF and CB scores before combining
            cf_scores = self._normalize_scores(cf_scores_full[cf_indices])
            cb_scores = self._normalize_scores(cb_scores_full[cb_indices])

            # Combine scores
            combined_scores = self.cf_weight * cf_scores + (1 - self.cf_weight) * cb_scores

            sorted_idx = np.argsort(combined_scores)[::-1][:self.top_n * 10]
            selected_ids = [common_ids[i] for i in sorted_idx]
            selected_scores = [combined_scores[i] for i in sorted_idx]  # ✅ Only those selected

# ✅ Apply diversity penalty
            selected_ids, scores = self._apply_diversity_penalty(selected_ids, user_id, selected_scores)

            # Apply diversity penalty
            #selected_ids, scores = self._apply_diversity_penalty(selected_ids, user_id, combined_scores)


            # Final top-N
            product_ids = selected_ids[:self.top_n]
            scores = combined_scores[sorted_idx[:self.top_n]]

        # Retrieve product info
        recommended_df = self.products_df[self.products_df['id'].isin(product_ids)].copy()
        recommended_df['score'] = scores
        return recommended_df.sort_values(by='score', ascending=False).reset_index(drop=True)

    def _apply_diversity_penalty(self, product_ids, user_id, candidate_scores):
        """
        Apply diversity penalty to candidate product scores based on categories already seen by the user.
        """
        # Step 1: Get user's interacted product IDs
        user_interacted_ids = set(
            self.user_product_time[self.user_product_time['user_id'] == user_id]['product_id_visited'].dropna()
        )

        # Step 2: Get user's historical categories and brands
        user_history = self.products_df[self.products_df['id'].isin(user_interacted_ids)]
        user_categories = set(user_history['category'].dropna())
        user_brands = set(user_history['brand'].dropna())

        # Step 3: For recommended products, get their brand and category
        product_info = self.products_df.set_index('id').loc[product_ids]
        product_categories = product_info['category']
        product_brands = product_info['brand']

        # Step 4: Penalize products from already seen brands or categories
        penalties = np.array([
            1.0 if (cat in user_categories or brand in user_brands) else 0.0
            for cat, brand in zip(product_categories, product_brands)
        ])


        # Step 3: Assign penalty = 1.0 if category was already seen, else 0.0
        #penalties = product_info['category'].apply(lambda x: 1.0 if x in user_categories else 0.0)

        # Step 4: Scale the penalty
        penalty_strength = self.diversity_penalty
        diversity_adjustment = 1.0 - penalty_strength * penalties
        print(diversity_adjustment)

        # Step 5: Adjust scores (must match shape)
        adjusted_scores = np.array(candidate_scores) * diversity_adjustment

        # Step 6: Reorder based on adjusted scores
        sorted_indices = np.argsort(adjusted_scores)[::-1]
        diversified_ids = [product_ids[i] for i in sorted_indices]
        diversified_scores = adjusted_scores[sorted_indices]

        return diversified_ids, diversified_scores

    def _get_cf_scores(self, user_id):
        user_index = self.reverse_user_id_map[user_id]
        user_vector = self.user_factors[user_index].astype('float32').reshape(1, -1)
        scores, indices = self.faiss_index.search(user_vector, self.item_factors.shape[0])
        score_vector = np.zeros(self.item_factors.shape[0])
        score_vector[indices[0]] = scores[0]
        return score_vector

    def _get_cb_scores(self, user_id):
        visited = self.user_product_time[self.user_product_time['user_id'] == user_id]['product_id_visited'].dropna().unique()
        visited_indices = [self.product_id_to_index[pid] for pid in visited if pid in self.cb_product_ids]
        if not visited_indices:
            return np.zeros(self.product_features_matrix.shape[0])

        user_profile = self.product_features_matrix[visited_indices].mean(axis=0)
        user_profile_dense = np.asarray(user_profile).reshape(1, -1)
        product_matrix_dense = self.product_features_matrix.toarray()
        similarities = cosine_similarity(user_profile_dense, product_matrix_dense).flatten()
        return similarities

recommender = HybridRecommender(pd.read_csv("users_products_time.csv"), pd.read_csv("products_df_recomm.csv"), pd.read_csv("users_df_recomm.csv"), cf_weight=0.9
,diversity_penalty=0.6, top_n=10)
top_products = recommender.recommend(user_id=100)
print(top_products)