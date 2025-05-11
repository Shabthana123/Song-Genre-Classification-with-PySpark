import re
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer,
                                IDF, StringIndexer)
from pyspark.ml.feature import (StringIndexerModel, CountVectorizerModel, IDFModel)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import GBTClassifier


from pyspark.ml import PipelineModel
from functools import reduce
from pyspark.sql import DataFrame

import joblib
import numpy as np

import findspark
findspark.init()

import os
os.environ["PYSPARK_PYTHON"] = r"env_bigdata/Scripts/python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"env_bigdata/Scripts/python.exe"

# # Initialize Spark session
# spark = SparkSession.builder.appName("GenreClassifier").getOrCreate()

spark = SparkSession.builder \
    .appName("GenreClassifier") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.python.worker.memory", "1g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()
    
spark.catalog.clearCache()  # Clear the cache


# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# def preprocess(text):
#     text = text.lower()
#     text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
#     tokens = text.split()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
#     return " ".join(tokens)

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)
# def preprocess(text):
#     words = text.lower().split()
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#     return " ".join(words)


def run():
    # Load data
    df = spark.read.csv('Merged_dataset.csv', header=True, inferSchema=True)
    df = df.dropna(subset=["lyrics", "genre"])


    preprocess_udf = udf(preprocess, StringType())
    df = df.withColumn("cleaned_lyrics", preprocess_udf(col("lyrics")))

    # Label encoding
    label_indexer = StringIndexer(inputCol="genre", outputCol="label")

    # Tokenization and TF-IDF
    tokenizer = Tokenizer(inputCol="cleaned_lyrics", outputCol="tokens")
    stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features", vocabSize=15000, minDF=2)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # Classifier
    # lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

    # # Pipeline
    # pipeline = Pipeline(stages=[label_indexer, tokenizer, stopword_remover, vectorizer, idf, lr])

    # # CLassifier - Naive Bayes

    # nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
    # pipeline = Pipeline(stages=[label_indexer, tokenizer, stopword_remover, vectorizer, idf, nb])

    # Replace Naive Bayes with Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.3, elasticNetParam=0)
    pipeline = Pipeline(stages=[label_indexer, tokenizer, stopword_remover, vectorizer, idf, lr])
    
    # # Train-test split
    # train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Get unique genres
    genres = df.select("genre").distinct().rdd.flatMap(lambda x: x).collect()

    # Perform stratified split
    train_dfs = []
    test_dfs = []

    for genre in genres:
        genre_df = df.filter(df["genre"] == genre)
        train_part, test_part = genre_df.randomSplit([0.8, 0.2], seed=42)
        train_dfs.append(train_part)
        test_dfs.append(test_part)

    # Combine all genre-specific splits into final train and test sets
    train_df = reduce(DataFrame.unionAll, train_dfs)
    test_df = reduce(DataFrame.unionAll, test_dfs)

    # try:
    #     print("genre count: ",df.groupBy("genre").count().orderBy("count", ascending=False).show())
    # except Exception as e:
    #     print("Error in genre count: ", e)

    # Train model
    model = pipeline.fit(train_df)

    # Evaluate model
    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"\nFinal Test Accuracy: {accuracy:.2f}")

    # Save model
    model.write().overwrite().save("genre_classifier_model_logistic")

    # Save label indexer and vectorizer
    label_indexer_model = model.stages[0]
    vectorizer_model = model.stages[3]
    idf_model = model.stages[4]

    label_indexer_model.write().overwrite().save("label_indexer_model_logistic")
    vectorizer_model.write().overwrite().save("vectorizer_model_logistic")
    idf_model.write().overwrite().save("idf_model_logistic")



# Return dictionary of genre probabilities ---
def predict_genre(lyrics):
    # Load models for prediction
    loaded_model = PipelineModel.load("genre_classifier_model_logistic")
    loaded_label_indexer = StringIndexerModel.load("label_indexer_model_logistic")

    cleaned = preprocess(lyrics)
    input_df = spark.createDataFrame([(cleaned,)], ["cleaned_lyrics"])

    # Let the model handle all transformations
    predictions = loaded_model.transform(input_df)
    probabilities = predictions.select("probability").collect()[0][0]

    label_mapping = dict(zip(loaded_label_indexer.labels, range(len(loaded_label_indexer.labels))))
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    genre_probs = {inverse_label_mapping[i]: float(probabilities[i]) for i in range(len(probabilities))}
    return genre_probs

# Plot bar chart using the returned dictionary ---
def predict_and_plot(lyrics, show_plot=True):
    genre_probs = predict_genre(lyrics)
    # print('Genre Probabilities:', genre_probs)
    # Sort genres by probability
    sorted_genres = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
    genres_sorted = [item[0] for item in sorted_genres]
    probs_sorted = [item[1] for item in sorted_genres]

    most_probable_genre = genres_sorted[0]

    plt.figure(figsize=(10, 6))
    plt.barh(genres_sorted, probs_sorted, color='skyblue')
    plt.xlabel("Predicted Probability")
    plt.title("Genre Prediction Probabilities")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if show_plot:
        plt.show()

    return most_probable_genre, plt


# # Example usage
if __name__ == "__main__":
    run()
#     example_lyrics = """I got a feeling that tonight's gonna be a good night
#     That tonight's gonna be a good night
#     That tonight's gonna be a good, good night"""
    
#     most_probable_genre, plot = predict_and_plot(example_lyrics)
#     # plot.savefig("genre_prediction.png")
#     print(f"Most Probable Genre: {most_probable_genre}")
