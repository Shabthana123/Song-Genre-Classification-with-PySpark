# 🎶 Song Genre Classification with PySpark

This project implements a **song genre classification** system using **PySpark MLlib**. The objective is to predict the genre of a song based on its lyrics. The pipeline includes data cleaning, preprocessing using **TF-IDF**, and model training with **Logistic Regression**. Trained models are saved for reuse, and a **Streamlit** web app is provided for user interaction. The app allows users to input lyrics, predict the corresponding genre, and visualize the genre compatibility across all 8 genres using a bar chart.

---

## 📁 Project Structure
```bash
├── clean_data.py                      # Cleans raw dataset into a uniform format
├── merge_dataset.py                   # Merges datasets into a single file (Merged_dataset.csv)
├── classify_logistic.py               # PySpark ML pipeline with Logistic Regression
├── app.py                             # # Streamlit app for genre prediction and visualization UI
├── run.bat                            # Batch file to run the full pipeline
├── genre_classifier_model_logistic/   # Saved PySpark model
├── label_indexer_model_logistic/      # Saved label indexer model
├── vectorizer_model_logistic/         # Saved vectorizer model
├── idf_model_logistic/                # Saved IDF model
├── Merged_dataset.csv                 # Final dataset used for training
├── student_dataset.csv                # Dataset of student-specific lyrics (pre-cleaning)
├── ska_dataset_raw.csv                # Original Ska dataset before cleaning
├── tcc_ceds_music.csv                 # Dataset with 7 genres before merging
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## ⚙️ Technologies Used

- Python 3.x
- PySpark (MLlib)
- NLTK for preprocessing
- Matplotlib for visualization
- Batch scripting (.bat)
- Python environment
- Streamlit

---

## ⚙️ How It Works

1. **Data Cleaning**: `clean_data.py` script standardizes raw datasets, handling missing values and unifying formats across different datasets.
2. **Dataset Merging**: The `merge_dataset.py` script merges multiple datasets into one unified dataset (`Merged_dataset.csv`) for model training.
3. **Model Training**: The `classify_logistic.py` script:
   - Preprocesses the lyrics (e.g., tokenization, stopword removal, lemmatization)
   - Converts lyrics into TF-IDF vectors
   - Trains a Logistic Regression model
   - Saves the trained models for future use (saved in respective directories)
4. **treamlit App for Genre Prediction**: The `app.py` script:
   - Uses Streamlit to create an interactive web app where users can input lyrics for song genre prediction.
   - The app loads the saved model (genre_classifier_model_logistic), applies the necessary transformations to the input lyrics (using the saved vectorizer and IDF model), and predicts the genre.
   - It displays the predicted genre along with a bar chart visualizing the model's compatibility score with all 8 genres, offering a clear and interactive user experience.

---

## 🚀 Getting Started

### 1. Clone the repository

Clone this project to your local machine using the following command:

```bash
git clone https://github.com/Shabthana123/genre-classifier-pyspark.git
cd Song-Genre-Classification-with-PySpark

```

### 🚀 Quick Start

To run the project:

```bash
.\run.bat
```

### 🧰 Manual Setup (Alternative to run.bat)

# Step 1: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows\
source venv/bin/activate   # On macOS/Linux

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py

