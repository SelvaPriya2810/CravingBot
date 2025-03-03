# Craving-Based Food Recommendation System

## Overview
The Craving-Based Food Recommendation System is an AI-powered application that suggests dishes based on user cravings, ingredients, cuisine, and dish type. This system leverages Natural Language Processing (NLP) and machine learning techniques to provide personalized food recommendations.

## Features
- **User Input Processing**: Extracts craving type, ingredients, cuisine, and dish type from user input using spaCy NLP.
- **Data Preprocessing**: Cleans and processes a dataset containing 300+ dishes with associated ingredients, cuisine types, and craving classifications.
- **TF-IDF Vectorization**: Converts text features into numerical representations for similarity matching.
- **Cosine Similarity Matching**: Recommends the most relevant dishes based on user cravings and food features.
- **Duplicate Handling**: Identifies and removes duplicate entries from the dataset to ensure uniqueness.

## Technologies Used
- **Python**
- **Pandas** for data manipulation
- **spaCy** for NLP processing
- **scikit-learn** for TF-IDF vectorization and cosine similarity
- **Matplotlib** for data visualization (if needed)

## Installation
### Prerequisites:
- Python 3.x
- Required Libraries:
  ```sh
  pip install pandas numpy spacy scikit-learn matplotlib
  ```
- Download spaCy model:
  ```sh
  python -m spacy download en_core_web_sm
  ```

## Dataset
The dataset consists of 300+ dishes with the following columns:
- **Dish Name**: Name of the dish
- **Ingredients**: List of ingredients used
- **Cuisine**: Cuisine type (e.g., North Indian, South Indian, etc.)
- **Type**: Type of dish (e.g., curry, appetizer, dessert)
- **Craving Type**: Categorized as Spicy, Sweet, Savory, or Rich

## How It Works
1. **Load and Clean the Dataset**
   ```python
   df = pd.read_csv('/content/300_dishes.csv')
   df.drop_duplicates(inplace=True)
   ```
2. **Extract Features from User Input**
   ```python
   cravings, ingredients, types, cuisines = extract_features(user_input)
   ```
3. **TF-IDF Vectorization & Similarity Matching**
   ```python
   tfidf_matrix = tfidf.fit_transform(df['combined_features'])
   cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
   ```
4. **Retrieve Top Matching Dishes**
   ```python
   top_indices = cosine_sim.argsort()[0][-3:][::-1]
   recommendations = df.iloc[top_indices]
   ```
5. **Display Recommendations**
   ```python
   print(recommendations)
   ```

## Example Usage
```python
user_input = "I'm craving something sweet South Indian"
recommendations = recommend_food(user_input, df, tfidf_matrix)
print(recommendations)
```

## Future Enhancements
- Integrating a user feedback system for better recommendations.
- Expanding the dataset with more diverse dishes.
- Deploying the system as a web or mobile application.

## Author
Selvapriya Selvakumar

