# Hybrid Recommender System

This project implements a **hybrid recommender system** designed for a Kaggle competition. 
It blends multiple recommendation strategies—collaborative filtering (ItemKNN), sequential pattern mining, 
content-based filtering (TF-IDF), and popularity-based ranking—to predict the **top-10 items** a user is likely to interact with.

The system uses a combination of sparse matrix operations, TF-IDF vectorization, and co-occurrence statistics 
to make recommendations, and is optimized for sparse datasets where most users have only a few interactions.

---

## 📁 Project Structure

- `data_preprocess.py`: Cleans and filters the item metadata to remove discontinued or low-quality items.
- `model.py`: Main script to build the recommender system using:
  - Item-based collaborative filtering (ItemKNN)
  - Sequential co-occurrence analysis
  - Content-based TF-IDF matching
  - Popularity ranking fallback
- `data/`: Input CSV files (e.g., `train.csv`, `test.csv`, `item_meta.csv`, etc.)
- `output/`: Directory where recommendation results and intermediate data files are stored.

---

## ⚙️ Features

- **Data cleaning**:
  - Filters out discontinued products.
  - Removes items with low average ratings or few rating counts.
- **Hybrid model components**:
  - **ItemKNN** for collaborative filtering using cosine similarity.
  - **Sequential co-occurrence** maps based on recent user behavior.
  - **TF-IDF** for content-based filtering using item titles.
  - **Popularity model** as a fallback for cold-start scenarios.
- **Efficient inference** using sparse matrices and pickle caching.

---

## 🚀 How to Run

### ⚙️Requirements
Install dependencies via:
```bash
pip install -r requirements.txt
```

### 1. Prepare the Dataset

Place the following files in a folder named `data/`:
- `train.csv`: User-item interactions with timestamps.
- `test.csv`: Test interactions (for context).
- `item_meta.csv`: Metadata for items including title, details, ratings.
- `sample_submission.csv`: Target user IDs.

### 2. Run Data Preprocessing

```bash
python data_preprocess.py
```
This will:
- Remove discontinued items.
- Filter items with poor rating metrics.
- Save the cleaned item metadata to data/clean_item_meta.csv.

### 3. Run the Recommender System

```bash
python model.py
```
This will:

- Build/load necessary data structures (TF-IDF, KNN matrix, co-occurrence maps).
- Generate top-10 item recommendations per user.
- Save results to output/submission.csv.

## 📊 Output
The final submission file will be located at:
```bash
output/submission.csv
```
It contains:
- ID: Same as user ID.
- user_id: Target user.
- item_id: Comma-separated top-10 recommended item IDs