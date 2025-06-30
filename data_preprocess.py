import pandas as pd
import ast
import os

# Configuration
DATA_DIR = 'data'
ITEM_META_FILE = os.path.join(DATA_DIR, 'item_meta.csv')
CLEAN_ITEM_META_FILE = os.path.join(DATA_DIR, 'clean_item_meta.csv')

os.makedirs(DATA_DIR, exist_ok=True)


def is_item_discontinued(details_str):
    """
    Checks if the item is marked as discontinued in the 'details' string.
    """
    if pd.isna(details_str) or not isinstance(details_str, str):
        return False

    try:
        details_dict = ast.literal_eval(details_str)

        if isinstance(details_dict, dict):
            # Common variations of the key to check:
            keys_to_check = [
                'Is Discontinued By Manufacturer',
                'is_discontinued_by_manufacturer',
                'isDiscontinuedByManufacturer'
            ]
            for key in keys_to_check:
                discontinued_status = details_dict.get(key)
                if discontinued_status is not None:
                    if isinstance(discontinued_status, str):
                        return discontinued_status.lower() == 'yes'
                    elif isinstance(discontinued_status, bool):
                        return discontinued_status
                    return False

            return False

        return False

    except (ValueError, SyntaxError, TypeError) as e:
        return False


if __name__ == "__main__":
    print(f"Loading item metadata from: {ITEM_META_FILE}")
    try:
        item_meta_df = pd.read_csv(ITEM_META_FILE)
    except FileNotFoundError:
        print(f"ERROR: The file {ITEM_META_FILE} was not found. Please check the path.")
        exit()
    except Exception as e:
        print(f"ERROR: Could not read the file {ITEM_META_FILE}. Error: {e}")
        exit()

    original_columns = item_meta_df.columns.tolist()
    print(f"Successfully loaded. Initial item count: {len(item_meta_df)}")

    # Filtering Discontinued Items
    print("\nFiltering discontinued items...")
    if 'details' not in item_meta_df.columns:
        print("ERROR: 'details' column not found in the item metadata.")
    else:
        item_meta_df['is_discontinued_flag'] = item_meta_df['details'].apply(is_item_discontinued)
        discontinued_count = item_meta_df['is_discontinued_flag'].sum()
        print(f"Number of items identified as explicitly discontinued: {discontinued_count}")

        # Keeps only items that are NOT discontinued
        item_meta_df = item_meta_df[~item_meta_df['is_discontinued_flag']].copy()
        item_meta_df.drop(columns=['is_discontinued_flag'], inplace=True)
        print(f"Item count after removing discontinued items: {len(item_meta_df)}")

    # Filter by Average Rating
    print("\nFiltering items with average_rating < 4...")
    RATING_THRESHOLD = 4

    if 'average_rating' not in item_meta_df.columns:
        print("Warning: 'average_rating' column not found.")
    else:
        item_meta_df['average_rating'] = pd.to_numeric(item_meta_df['average_rating'], errors='coerce')

        items_before_rating_filter = len(item_meta_df)
        # Keeps items with rating >= RATING_THRESHOLD
        item_meta_df = item_meta_df[
            (item_meta_df['average_rating'] >= RATING_THRESHOLD) | (item_meta_df['average_rating'].isna())]
        items_removed_by_rating = items_before_rating_filter - len(item_meta_df)
        print(f"Items removed due to low average rating (< {RATING_THRESHOLD}): {items_removed_by_rating}")
        print(f"Item count after average rating filter: {len(item_meta_df)}")

    # Filter by Number of Ratings
    print("\nFiltering items with rating_number <= 10...")
    RATING_COUNT_THRESHOLD = 10

    if 'rating_number' not in item_meta_df.columns:
        print("Warning: 'rating_number' column not found.")
    else:
        item_meta_df['rating_number'] = pd.to_numeric(item_meta_df['rating_number'], errors='coerce')

        items_before_num_rating_filter = len(item_meta_df)
        # Keeps items with rating_number > RATING_COUNT_THRESHOLD
        item_meta_df = item_meta_df[
            (item_meta_df['rating_number'] > RATING_COUNT_THRESHOLD) | (item_meta_df['rating_number'].isna())]
        items_removed_by_num_rating = items_before_num_rating_filter - len(item_meta_df)
        print(
            f"Items removed due to low number of ratings (<= {RATING_COUNT_THRESHOLD}): {items_removed_by_num_rating}")
        print(f"Item count after number of ratings filter: {len(item_meta_df)}")
    final_df = item_meta_df.reindex(columns=original_columns)

    print(f"\nFinal item count after all filters: {len(final_df)}")

    try:
        final_df.to_csv(CLEAN_ITEM_META_FILE, index=False)
        print(f"Successfully saved cleaned item metadata to: {CLEAN_ITEM_META_FILE}")
    except Exception as e:
        print(f"ERROR: Could not save the cleaned file to {CLEAN_ITEM_META_FILE}. Error: {e}")
