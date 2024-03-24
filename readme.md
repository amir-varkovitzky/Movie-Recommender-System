# Movie Recommender System

This repository contains code for a movie recommender system implemented in Python. The system is capable of recommending movies to users based on their preferences and past ratings.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The movie recommender system employs collaborative filtering techniques to suggest movies to users based on their historical ratings and preferences. It utilizes the MovieLens dataset (ml-latest-small), which contains 100,836 ratings and 3,683 tag applications across 9,742 movies, collected from users on MovieLens, a movie recommendation service, and leverages machine learning algorithms to make personalized recommendations.

## Features

- Collaborative filtering approach for personalized movie recommendations.
- Implementation of a Bayesian average for improved rating predictions.
- Customizable search and sort functionality based on genres, release years, and number of ratings.
- Evaluation metrics including Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) for model performance assessment.
- Integration with TensorFlow for training the recommendation model.
- Interactive notebook interface for easy experimentation and usage.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/amir-varkovitzky/Recommender-System.git
   ```

2. Navigate to the project directory:

    ```bash
    cd recommender-system
    ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the notebook `recommender-notebook.ipynb` in your Jupyter Notebook environment.
2. Begin by reviewing the prewritten ratings for popular movies. You can either use these ratings as they are, modify them to match your preferences, or delete them entirely.
3. Use the interactive widget to search for any movie you like. Once you've found the desired movie, select it from the dropdown menu, rate it using the slider, and submit your rating.
4. Follow the instructions provided in the notebook to load the movie ratings data, train the recommendation model, and make movie predictions.
5. Experiment with different parameters, search queries, and evaluation metrics as needed.

## Examples

### Training the Recommendation Model

```python
# Train the recommendation model using a custom training loop
X, W, b, cost_history = train_model(X, W, b, Ynorm, R, lambda_=1, optimizer=optimizer, iterations=200, print_every=20)
```

### Making Movie Predictions

```python
# Make movie predictions using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
```

### Evaluating Model Performance

```python
# Calculate RMSE and MAE for rated items and all users
rmse_for_rated = calculate_rmse_for_my_rated(my_predictions, my_ratings, rated_indices)
rmse_for_all_users = calculate_rmse_for_all_users(predictions, targets, mask)
mae_for_rated = calculate_mae_for_my_rated(my_predictions, my_ratings, rated_indices)
mae_for_all_users = calculate_mae_for_all_users(predictions, targets, mask)
```

## Contributing

Contributions to the project are welcome. If you have suggestions for improvements, please open an issue or submit a pull request. Some ideas for contributions include updating the movie_recommender.py according to the notebook, writing the GUI, or anything else that comes to mind.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
