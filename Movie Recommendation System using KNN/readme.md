# 🎬 Movie Recommendation System using KNN

Welcome to the **Movie Recommendation System**!  
In today’s digital age, users are overwhelmed with vast content choices. This project helps deliver **personalized movie recommendations** using the **K-Nearest Neighbors (KNN)** algorithm and **collaborative filtering**, based on user preferences and historical ratings.

---

## 🧠 Problem Statement

With the exponential growth in digital content, users often face choice paralysis. The goal is to create a recommendation system that suggests movies tailored to a user’s taste by analyzing patterns in viewing history and preferences of similar users.

---

## 🚀 Concepts Used

- **K-Nearest Neighbors (KNN):**  
  A non-parametric machine learning algorithm that identifies the `k` closest users using **cosine similarity**.

- **Collaborative Filtering:**  
  Recommends movies based on preferences of users who exhibit similar behavior and taste.

---

## 🔍 Code Functionality

### 1. **Data Loading & Preprocessing**
- `movies.csv` – Movie metadata (Movie ID, Title, Genres)
- `ratings.csv` – User ratings (User ID, Movie ID, Rating)
- These datasets are merged to form a **user-item matrix**, where:
  - Rows = Users  
  - Columns = Movies  
  - Cells = Ratings (unrated movies are filled with 0)

### 2. **Sparse Matrix Conversion**
- The user-item matrix is converted into a **Compressed Sparse Row (CSR)** matrix using `scipy.sparse.csr_matrix` to optimize performance and memory.

### 3. **KNN Model Training**
- Using `sklearn.neighbors.NearestNeighbors` with:
  - `metric='cosine'`  
  - `algorithm='brute'`
- The model learns similarities between users based on their ratings.

### 4. **Recommendation Function**
- Given a user ID:
  - Locate the user’s position in the matrix  
  - Retrieve similar users using `kneighbors`  
  - Extract and rank movies highly rated by similar users  
  - Return a list of movie titles as recommendations

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python packages:

```bash
pip install numpy pandas scikit-learn scipy
```

### 📌 Steps to Run
1. Clone the repository
```bash
git clone https://github.com/ShreyansuPanda/Machine-Learning-Projects.git
```
2. Navigate to the project directory
```bash
cd "Machine-Learning-Projects/Movie Recommendation System using KNN"
```
3. Open CMD and run Python script
```bash
python movie_recommender.py
```
---

## 🗂️ File Structure
```sh
Movie Recommendation System using KNN/
│
├── movies.csv                 # Dataset with movie metadata
├── ratings.csv                # Dataset with user ratings
├── movie_recommender.py       # Main Python script (or Jupyter Notebook)
```
---

## 🌟 Show Your Support
If you like this project, consider giving it a ⭐ on GitHub. Contributions are also welcome!
