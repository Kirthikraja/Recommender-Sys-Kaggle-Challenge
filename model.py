import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
import csv
from tqdm import tqdm
import re
import traceback
from collections import defaultdict, Counter
import pickle

# Configuration
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE_WITH_SEQUENCES = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
ITEM_META_FILE = os.path.join(DATA_DIR, 'clean_item_meta.csv')
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'submission.csv')

# saving the matrix and mappings for reusability
ITEMKNN_SIM_MATRIX_FILE = os.path.join(OUTPUT_DIR, 'itemknn_item_sim_matrix.npz')
ITEMKNN_USER_MAP_FILE = os.path.join(OUTPUT_DIR, 'itemknn_user_map.pkl')
ITEMKNN_ITEM_MAP_FILE = os.path.join(OUTPUT_DIR, 'itemknn_item_map.pkl')
ITEMKNN_IDX_TO_ITEM_MAP_FILE = os.path.join(OUTPUT_DIR, 'itemknn_idx_to_item_map.pkl')
SEQ_COOC_MAP_FILE = os.path.join(OUTPUT_DIR, 'seq_cooc_map.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
TOP_K = 10

# TF-IDF
MIN_TFIDF_SIMILARITY_THRESHOLD = 0.01
K_TFIDF_CANDIDATES = 50

# ItemKNN
TOP_N_SIMILAR_ITEMS_PER_HISTORY_ITEM_KNN = 30
K_ITEMKNN_CANDIDATES = 50

# Sequential Co-occurrence
NUM_RECENT_ITEMS_FOR_SEQ_CONTEXT = 1
MIN_COOC_FREQUENCY_SEQ = 1
K_SEQCOOC_CANDIDATES = 50

# Blended Score Weights
W_ITEMKNN = 0.6
W_SEQCOOC = 0.2
W_TFIDF = 0.1
W_POP = 0.1

USER_ID_COL = 'user_id'
ITEM_ID_COL = 'item_id'
TITLE_COL = 'title'
TIMESTAMP_COL = 'timestamp'

def standardize(df, column_name, df_name="DataFrame"):
    """ Drops rows where IDs cannot be converted or missing after conversion """
    if df is None or df.empty or column_name not in df.columns: return df
    df[column_name] = df[column_name].astype(str).str.strip().replace(
        {'': np.nan, 'nan': np.nan, 'None': np.nan, '<NA>': np.nan})
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df.dropna(subset=[column_name], inplace=True)
    if not df.empty:
        df[column_name] = df[column_name].astype('Int64').astype(str)
        df.drop(df[df[column_name].isin(['nan', '<NA>'])].index, inplace=True)
    return df


def text_format(text):
    """ Text processing for TF-IDF vectorization """
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def tf_idf_embeddings(item_meta_df_processed, item_id_col=ITEM_ID_COL, title_col=TITLE_COL):
    """ Generating embeddings for item titles """
    print("Generating TF-IDF embeddings...")
    tfidf_text_col = 'cleaned_title_for_tfidf_blend'
    item_meta_df_processed[tfidf_text_col] = item_meta_df_processed[title_col].apply(text_format)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(item_meta_df_processed[tfidf_text_col])
    tfidf_item_id_to_idx_map = {id_val: i for i, id_val in enumerate(item_meta_df_processed[item_id_col])}
    tfidf_idx_to_item_id_map = {i: id_val for i, id_val in enumerate(item_meta_df_processed[item_id_col])}
    return tfidf_matrix, tfidf_item_id_to_idx_map, tfidf_idx_to_item_id_map


def get_item_popularity(train_df_processed, item_id_col=ITEM_ID_COL):
    """ Item popularity calculation based on interaction counts in the training data """
    if train_df_processed.empty or item_id_col not in train_df_processed.columns: return pd.Series(dtype='int'), []
    popularity_series = train_df_processed[item_id_col].value_counts()
    popular_items_sorted_list = popularity_series.index.tolist()
    return popularity_series, popular_items_sorted_list


def itemknn_components(train_df, user_id_col, item_id_col):
    """ Builds user/item mappings and item-item similarity matrix """
    print("Building ItemKNN components...")
    unique_users = sorted(train_df[user_id_col].unique().tolist())
    user_to_internal_idx = {orig_id: i for i, orig_id in enumerate(unique_users)}
    unique_items = sorted(train_df[item_id_col].unique().tolist())
    item_to_internal_idx = {orig_id: i for i, orig_id in enumerate(unique_items)}
    internal_idx_to_item = {i: orig_id for orig_id, i in item_to_internal_idx.items()}
    user_indices = train_df[user_id_col].map(user_to_internal_idx)
    item_indices = train_df[item_id_col].map(item_to_internal_idx)
    valid_interactions = pd.DataFrame({'user_idx': user_indices, 'item_idx': item_indices}).dropna()
    user_item_matrix = csr_matrix(
        (np.ones(len(valid_interactions)), (valid_interactions['user_idx'], valid_interactions['item_idx'])),
        shape=(len(unique_users), len(unique_items)))
    item_user_matrix = user_item_matrix.T.tocsr()
    item_similarity_matrix = cosine_similarity(item_user_matrix, dense_output=False).tocsr()
    save_npz(ITEMKNN_SIM_MATRIX_FILE, item_similarity_matrix)
    with open(ITEMKNN_USER_MAP_FILE, 'wb') as f: pickle.dump(user_to_internal_idx, f)
    with open(ITEMKNN_ITEM_MAP_FILE, 'wb') as f: pickle.dump(item_to_internal_idx, f)
    with open(ITEMKNN_IDX_TO_ITEM_MAP_FILE, 'wb') as f: pickle.dump(internal_idx_to_item, f)
    print(
        f"  ItemKNN: Sim matrix {item_similarity_matrix.shape}, {len(user_to_internal_idx)} users, {len(item_to_internal_idx)} items.")
    return item_similarity_matrix, user_to_internal_idx, item_to_internal_idx, internal_idx_to_item


def sequential_cooccurrence_map(train_df, user_id_col, item_id_col, timestamp_col):
    """ Builds a map of item-to-item sequential co-occurances and stores
    How often one item follows another in user interaction sequences"""
    print("Building sequential co-occurrence map...")
    cooc_map = defaultdict(Counter)
    train_df_sorted = train_df.sort_values([user_id_col, timestamp_col])
    for _, group in tqdm(train_df_sorted.groupby(user_id_col), desc="Processing user sequences for co-occurrence"):
        item_sequence = group[item_id_col].tolist()
        for i in range(len(item_sequence) - 1):
            cooc_map[item_sequence[i]][item_sequence[i + 1]] += 1
    with open(SEQ_COOC_MAP_FILE, 'wb') as f:
        pickle.dump(dict(cooc_map), f)
    print(f"  Sequential co-occurrence map built. Keys: {len(cooc_map)}")
    return cooc_map


if __name__ == '__main__':
    rebuild_knn = not all(os.path.exists(f) for f in
                          [ITEMKNN_SIM_MATRIX_FILE, ITEMKNN_USER_MAP_FILE, ITEMKNN_ITEM_MAP_FILE,
                                     ITEMKNN_IDX_TO_ITEM_MAP_FILE])
    rebuild_cooc = not os.path.exists(SEQ_COOC_MAP_FILE)

    if rebuild_knn:
        print("ItemKNN files are missing. building ItemKNN again.")
        for f_path in [ITEMKNN_SIM_MATRIX_FILE, ITEMKNN_USER_MAP_FILE, ITEMKNN_ITEM_MAP_FILE,
                       ITEMKNN_IDX_TO_ITEM_MAP_FILE]:
            if os.path.exists(f_path): os.remove(f_path)
    if rebuild_cooc:
        print(f"Sequential Co-occurrence map {SEQ_COOC_MAP_FILE} is missing. building it again.")
        if os.path.exists(SEQ_COOC_MAP_FILE): os.remove(SEQ_COOC_MAP_FILE)

    print("Loading data...")
    try:
        item_meta_df_orig = pd.read_csv(ITEM_META_FILE, engine='python', on_bad_lines='warn',
                                        usecols=[ITEM_ID_COL, TITLE_COL])
        item_meta_df = standardize(item_meta_df_orig.copy(), ITEM_ID_COL, "item_meta_df")
        if TITLE_COL in item_meta_df.columns:
            item_meta_df[TITLE_COL] = item_meta_df[TITLE_COL].astype(str).fillna('')
        else:
            item_meta_df[TITLE_COL] = ""

        train_df_orig = pd.read_csv(TRAIN_FILE, usecols=[USER_ID_COL, ITEM_ID_COL, TIMESTAMP_COL])
        train_df = standardize(train_df_orig.copy(), USER_ID_COL, "train_df")
        train_df = standardize(train_df, ITEM_ID_COL, "train_df")

        if TIMESTAMP_COL not in train_df.columns:
            raise ValueError(f"Timestamp column missing in train_df.")
        train_df[TIMESTAMP_COL] = pd.to_numeric(train_df[TIMESTAMP_COL], errors='coerce')
        train_df.dropna(subset=[TIMESTAMP_COL], inplace=True)
        train_df[TIMESTAMP_COL] = train_df[TIMESTAMP_COL].astype(int)
        if train_df.empty:
            raise ValueError("Train_df is empty after standardization.")

        test_sequences_df_orig = pd.DataFrame(columns=[USER_ID_COL, ITEM_ID_COL, TIMESTAMP_COL])
        if os.path.exists(TEST_FILE_WITH_SEQUENCES):
            test_sequences_df_orig = pd.read_csv(TEST_FILE_WITH_SEQUENCES,
                                                 usecols=[USER_ID_COL, ITEM_ID_COL, TIMESTAMP_COL])
        test_sequences_df = standardize(test_sequences_df_orig.copy(), USER_ID_COL, "test_df")
        test_sequences_df = standardize(test_sequences_df, ITEM_ID_COL, "test_df")
        if TIMESTAMP_COL in test_sequences_df.columns and not test_sequences_df.empty:
            test_sequences_df[TIMESTAMP_COL] = pd.to_datetime(test_sequences_df[TIMESTAMP_COL], errors='coerce')
            test_sequences_df.dropna(subset=[TIMESTAMP_COL], inplace=True)

        sample_submission_df_orig = pd.read_csv(SAMPLE_SUBMISSION_FILE, usecols=[USER_ID_COL])
        sample_submission_df = standardize(sample_submission_df_orig.copy(), USER_ID_COL,
                                                        "sample_submission_df")
    except Exception as e:
        print(f"Error during data loading: {e}")
        traceback.print_exc()
        exit()

    # Build/Load Components
    if rebuild_knn:
        item_sim_matrix, _, item_to_internal_idx_knn, internal_idx_to_item_knn = itemknn_components(train_df,
                                                                                                    USER_ID_COL,
                                                                                                    ITEM_ID_COL)
    else:
        print("Loading ItemKNN...")
        item_sim_matrix = load_npz(ITEMKNN_SIM_MATRIX_FILE)
        with open(ITEMKNN_ITEM_MAP_FILE, 'rb') as f:
            item_to_internal_idx_knn = pickle.load(f)
        with open(ITEMKNN_IDX_TO_ITEM_MAP_FILE, 'rb') as f:
            internal_idx_to_item_knn = pickle.load(f)

    if rebuild_cooc:
        seq_cooc_map = sequential_cooccurrence_map(train_df, USER_ID_COL, ITEM_ID_COL, TIMESTAMP_COL)
    else:
        print("Loading SeqCooc map...")
        with open(SEQ_COOC_MAP_FILE, 'rb') as f:
            seq_cooc_map_loaded = pickle.load(f)
        seq_cooc_map = defaultdict(Counter)
        for k, v in seq_cooc_map_loaded.items():
            seq_cooc_map[k] = Counter(v)

    item_meta_for_tfidf = item_meta_df.drop_duplicates(subset=[ITEM_ID_COL]).reset_index(drop=True)
    if item_meta_for_tfidf.empty or TITLE_COL not in item_meta_for_tfidf.columns:
        tfidf_matrix, tfidf_item_id_to_idx, tfidf_idx_to_item_id = csr_matrix((0, 0)), {}, {}
    else:
        tfidf_matrix, tfidf_item_id_to_idx, tfidf_idx_to_item_id = tf_idf_embeddings(item_meta_for_tfidf)

    popularity_series, popular_items_sorted_list = get_item_popularity(train_df)

    # Generate Recommendations
    print("\nGenerating recommendations...")
    all_recommendations = []
    target_users_list = sample_submission_df[USER_ID_COL].unique().tolist()

    user_full_history_map = defaultdict(list)
    if not test_sequences_df.empty:  # Test history first
        test_sequences_df_sorted = test_sequences_df.sort_values([USER_ID_COL, TIMESTAMP_COL], ascending=[True, False])
        for user, group in test_sequences_df_sorted.groupby(USER_ID_COL): user_full_history_map[user].extend(
            group[ITEM_ID_COL].tolist())
    if not train_df.empty:  # Then train history
        train_df_for_history = train_df.sort_values([USER_ID_COL, TIMESTAMP_COL], ascending=[True, False])
        for user, group in train_df_for_history.groupby(USER_ID_COL):
            current_hist_set = set(user_full_history_map[user])
            for item_id_str in group[ITEM_ID_COL].tolist():
                if item_id_str not in current_hist_set: user_full_history_map[user].append(item_id_str)

    for target_user_id_str in tqdm(target_users_list, desc="Predicting for users"):
        user_recs_scores_agg = defaultdict(lambda: {'itemknn': 0.0, 'seq_cooc': 0.0, 'tfidf': 0.0})

        current_user_full_history_str = user_full_history_map.get(target_user_id_str, [])
        current_user_full_history_set_str = set(current_user_full_history_str)

        # ItemKNN Score
        history_item_indices_knn = [item_to_internal_idx_knn[item_id_str] for item_id_str in
                                    current_user_full_history_str if item_id_str in item_to_internal_idx_knn]
        if history_item_indices_knn:
            temp_itemknn_scores = defaultdict(float)
            for history_item_idx in history_item_indices_knn:
                sim_scores_row = item_sim_matrix[history_item_idx];
                sorted_sim_indices_for_row = np.argsort(-sim_scores_row.data)
                count_similar = 0
                for i in range(len(sim_scores_row.indices)):
                    if count_similar >= TOP_N_SIMILAR_ITEMS_PER_HISTORY_ITEM_KNN: break
                    similar_item_internal_idx = sim_scores_row.indices[sorted_sim_indices_for_row[i]]
                    if similar_item_internal_idx == history_item_idx: continue
                    score = sim_scores_row.data[sorted_sim_indices_for_row[i]];
                    if score <= 0: break
                    candidate_item_id_str = internal_idx_to_item_knn.get(similar_item_internal_idx)
                    if candidate_item_id_str and candidate_item_id_str not in current_user_full_history_set_str:
                        temp_itemknn_scores[candidate_item_id_str] += score;
                        count_similar += 1
            # Selects top K_ITEMKNN_CANDIDATES from ItemKNN source
            sorted_knn_candidates = sorted(temp_itemknn_scores.items(), key=lambda x: x[1], reverse=True)
            for item_id, score in sorted_knn_candidates[:K_ITEMKNN_CANDIDATES]: user_recs_scores_agg[item_id][
                'itemknn'] = score

        # Sequential Co-occurrence Score
        if current_user_full_history_str:
            context_items = current_user_full_history_str[:NUM_RECENT_ITEMS_FOR_SEQ_CONTEXT]
            temp_seq_cooc_scores = Counter()
            for context_item_str in context_items:
                if context_item_str in seq_cooc_map:
                    next_items_counts = seq_cooc_map[context_item_str]
                    for next_item_str, count in next_items_counts.items():
                        if count >= MIN_COOC_FREQUENCY_SEQ and next_item_str not in current_user_full_history_set_str:
                            temp_seq_cooc_scores[next_item_str] += count
            sorted_cooc_candidates = temp_seq_cooc_scores.most_common(K_SEQCOOC_CANDIDATES)
            for item_id, score in sorted_cooc_candidates: user_recs_scores_agg[item_id]['seq_cooc'] = float(score)

        # TF-IDF Score
        seed_item_id_for_tfidf = None
        if current_user_full_history_str: seed_item_id_for_tfidf = current_user_full_history_str[0]  # Most recent
        if seed_item_id_for_tfidf and seed_item_id_for_tfidf in tfidf_item_id_to_idx and tfidf_matrix.shape[0] > 0:
            tfidf_seed_idx = tfidf_item_id_to_idx[seed_item_id_for_tfidf]
            sim_scores_tfidf = cosine_similarity(tfidf_matrix[tfidf_seed_idx], tfidf_matrix).ravel()
            tfidf_candidates_with_scores = []
            for i_tfidf, score_tfidf in enumerate(sim_scores_tfidf):
                orig_item_id = tfidf_idx_to_item_id.get(i_tfidf)
                if score_tfidf >= MIN_TFIDF_SIMILARITY_THRESHOLD and orig_item_id and orig_item_id not in current_user_full_history_set_str:
                    tfidf_candidates_with_scores.append((orig_item_id, score_tfidf))
            sorted_tfidf_candidates = sorted(tfidf_candidates_with_scores, key=lambda x: x[1], reverse=True)
            for item_id, score in sorted_tfidf_candidates[:K_TFIDF_CANDIDATES]: user_recs_scores_agg[item_id][
                'tfidf'] = score

        # Combining scores and Rank
        final_candidates_df_data = []
        for item_id, scores_dict in user_recs_scores_agg.items():
            final_candidates_df_data.append({
                'item_id': item_id,
                'itemknn_score': scores_dict['itemknn'],
                'seq_cooc_score': scores_dict['seq_cooc'],
                'tfidf_score': scores_dict['tfidf'],
                'pop_score': np.log1p(popularity_series.get(item_id, 0))
            })

        user_recs = []
        if final_candidates_df_data:
            candidates_df = pd.DataFrame(final_candidates_df_data)
            # Normalization of scores (min-max scaling per user's candidate set)
            for col in ['itemknn_score', 'seq_cooc_score', 'tfidf_score', 'pop_score']:
                if candidates_df[col].nunique() > 1:
                    candidates_df[f'norm_{col}'] = minmax_scale(candidates_df[col])
                elif candidates_df[col].sum() > 0:
                    candidates_df[f'norm_{col}'] = 0.5
                else:
                    candidates_df[f'norm_{col}'] = 0.0

            candidates_df['blended_score'] = (W_ITEMKNN * candidates_df['norm_itemknn_score'] +
                                              W_SEQCOOC * candidates_df['norm_seq_cooc_score'] +
                                              W_TFIDF * candidates_df['norm_tfidf_score'] +
                                              W_POP * candidates_df['norm_pop_score'])

            candidates_df_sorted = candidates_df.sort_values(by='blended_score', ascending=False)
            user_recs = candidates_df_sorted['item_id'].tolist()[:TOP_K]

        # Fallback to global popularity
        if len(user_recs) < TOP_K:
            temp_existing_recs_set = current_user_full_history_set_str.copy()
            temp_existing_recs_set.update(user_recs)
            for pop_item_str in popular_items_sorted_list:
                if len(user_recs) >= TOP_K: break
                if pop_item_str not in temp_existing_recs_set: user_recs.append(pop_item_str)

        if len(user_recs) < TOP_K: user_recs.extend([f"placeholder_pop_{i}" for i in range(TOP_K - len(user_recs))])

        all_recommendations.append({'ID': target_user_id_str, 'user_id': target_user_id_str,
                                    'item_id': ",".join([str(item) for item in user_recs[:TOP_K]])})

    # Save Submission
    submission_df = pd.DataFrame(all_recommendations)
    target_users_df_final = sample_submission_df[[USER_ID_COL]].drop_duplicates();
    target_users_df_final[USER_ID_COL] = target_users_df_final[USER_ID_COL].astype(str)
    submission_df[USER_ID_COL] = submission_df[USER_ID_COL].astype(str)
    final_submission_df = pd.merge(target_users_df_final, submission_df, on=USER_ID_COL, how='left')
    placeholder_items_str_list = popular_items_sorted_list[:TOP_K] if popular_items_sorted_list else []
    if len(placeholder_items_str_list) < TOP_K: placeholder_items_str_list.extend(
        [f"placeholder_gen_{i}" for i in range(TOP_K - len(placeholder_items_str_list))])
    default_placeholder_items_str = ",".join([str(item) for item in placeholder_items_str_list])
    final_submission_df['item_id'] = final_submission_df['item_id'].fillna(default_placeholder_items_str)
    final_submission_df['ID'] = final_submission_df[USER_ID_COL];
    final_submission_df = final_submission_df[['ID', USER_ID_COL, 'item_id']]
    final_submission_df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Recommendations saved to {OUTPUT_FILE}. Users predicted for: {len(final_submission_df)}")