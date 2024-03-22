import pandas as pd
import numpy as np
import tensorflow as tf

def load_data(ratings_path, movies_path):
    """
    Loads the ratings and movies data from the ml-latest-small dataset

    Returns:
      ratings_df (DataFrame): DataFrame of ratings
      movies_df (DataFrame): DataFrame of movies
    """
    # Load ratings data
    ratings_df = pd.read_csv(ratings_path)

    # Load movies data
    movies_df = pd.read_csv(movies_path)

    return ratings_df, movies_df

def cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.

    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter

    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))

    return J

def train_model(X, W, b, Y, R, lambda_, optimizer, iterations=200, print_every=20):
    """
    Trains the collaborative filtering model using a custom training loop

    Args:
      X (ndarray (num_movies,num_features)): Matrix of item features
      W (ndarray (num_users,num_features)) : Matrix of user parameters
      b (ndarray (1, num_users)            : Vector of user parameters
      Y (ndarray (num_movies,num_users)    : Matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : Binary indicator matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): Regularization parameter
      optimizer (tf.keras.optimizers): Optimizer to use for training
      iterations (int): Number of iterations of gradient descent

    Returns:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
    """
    # Initialize the cost history
    cost_history = []

    # Train using a custom training loop
    for i in range(iterations):

        # Record the operations to compute the cost        
        with tf.GradientTape() as tape:
            J = cost_func(X, W, b, Y, R, lambda_)

        # Compute the gradients using automatic differentiation    
        grads = tape.gradient(J, [X, W, b])

        # Update the parameters using the Adam optimizer
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        # Append the cost to the history
        cost_history.append(J.numpy())

        # Print the cost every 100 iterations
        if i % print_every == 0:
            print(f'Cost at iteration {i} is {J.numpy()}')
    
    return X, W, b, cost_history

def normalize_ratings(Y, R):
    """
    Normalizes Y so that each movie has a rating of 0 on average, and returns the mean rating in Ymean.

    Args:
      Y (ndarray (num_movies,num_users)): matrix of user ratings of movies
      R (ndarray (num_movies,num_users)): matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user

    Returns:
      Ynorm (ndarray (num_movies,num_users)): normalized Y with mean ratings subtracted
      Ymean (ndarray (num_movies,1)): vector of mean ratings for each movie
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R)

    return Ynorm, Ymean

def movie_search(query, movies_df):
    """
    Returns a list of movies that match the keyword or ID.

    Args:
      query (str or int): keyword to search for or movie ID
      movies_df (DataFrame): DataFrame of movies

    Returns:
      results (DataFrame): DataFrame of movies that match the query
    """

    if isinstance(query, str):
        # Search by keyword
        results = movies_df.loc[movies_df['title'].str.contains(query, case=False, regex=True)]
    elif isinstance(query, int):
        # Search by movie ID
        results = movies_df.loc[movies_df['movieId'] == query]
    else:
        raise ValueError("Invalid query type. Please provide a string (keyword) or an integer (movie ID).")

    return results

def calculate_rmse_for_my_rated(predictions, targets, rated_indices):
    """
    Calculate Root Mean Squared Error for rated items between predictions and targets.

    Args:
      predictions (ndarray): Matrix of predicted ratings
      targets (ndarray): Matrix of actual ratings
      rated_indices (list): List of indices corresponding to rated items

    Returns:
      rmse (float): Root Mean Squared Error for rated items
    """
    rated_predictions = predictions[rated_indices]
    rated_targets = targets[rated_indices]

    mse = np.mean((rated_predictions - rated_targets) ** 2)
    rmse = np.sqrt(mse)

    return rmse

def calculate_rmse_for_all_users(predictions, targets, mask):
    """
    Calculate Root Mean Squared Error for all users between predictions and targets.

    Args:
      predictions (ndarray): Matrix of predicted ratings
      targets (ndarray): Matrix of actual ratings
      mask (ndarray): Binary indicator matrix (1 for rated items, 0 for unrated items)

    Returns:
      rmse (float): Root Mean Squared Error for all users
    """
    # Apply the mask to focus on rated items
    rated_predictions = predictions * mask
    rated_targets = targets * mask

    # Calculate RMSE only for rated items
    mse = np.sum((rated_predictions - rated_targets) ** 2) / np.sum(mask)
    rmse = np.sqrt(mse)

    return rmse

def calculate_mae_for_my_rated(predictions, targets, rated_indices):
    """
    Calculate Mean Absolute Error for rated items between predictions and targets.

    Args:
      predictions (ndarray): Matrix of predicted ratings
      targets (ndarray): Matrix of actual ratings
      rated_indices (list): List of indices corresponding to rated items

    Returns:
      mae (float): Mean Absolute Error for rated items
    """
    rated_predictions = predictions[rated_indices]
    rated_targets = targets[rated_indices]

    mae = np.mean(np.abs(rated_predictions - rated_targets))

    return mae

def calculate_mae_for_all_users(predictions, targets, mask):
    """
    Calculate Mean Absolute Error for all users between predictions and targets.

    Args:
      predictions (ndarray): Matrix of predicted ratings
      targets (ndarray): Matrix of actual ratings
      mask (ndarray): Binary indicator matrix (1 for rated items, 0 for unrated items)

    Returns:
      mae (float): Mean Absolute Error for all users
    """
    # Apply the mask to focus on rated items
    rated_predictions = predictions * mask
    rated_targets = targets * mask

    # Calculate MAE only for rated items
    mae = np.sum(np.abs(rated_predictions - rated_targets)) / np.sum(mask)
    
    return mae