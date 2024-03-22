import numpy as np
import tensorflow as tf
import re
from recommender_functions import *
import logging

# Configure the logging settings
log_file_path = 'my_log_file.log'
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)

# Load ratings and movies data
ratings_path, movies_path = 'ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv'
ratings_df, movies_df = load_data(ratings_path=ratings_path, movies_path=movies_path)

# Remove movies with no ratings from movies_df
movies_df = movies_df[movies_df.movieId.isin(ratings_df.movieId)].reset_index(drop=True)

# Create a new dataframe with movieId, numRatings, meanRating, and title columns
movieProperties_df = ratings_df.groupby('movieId').agg({'rating': ['size', 'mean']})
movieProperties_df.columns = ['numRatings', 'meanRating']
# Merge with movies_df to add movie title and genres
movieProperties_df = movieProperties_df.merge(movies_df, on='movieId')
movieProperties_df = movieProperties_df[['movieId', 'title', 'numRatings', 'meanRating', 'genres']]
# Log the updated dataframe
logging.info("\nMovie Properties DataFrame:")
logging.info("\n %s", movieProperties_df)

# Preprocess the data to create user-item interaction matrices
user_item_matrix = ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
Y = user_item_matrix

# Log the user-item matrix
logging.info("\nUser-Item Matrix:")
logging.info("\n %s", Y)
# Create a binary indicator matrix for Y
R = np.where(Y != 0, 1, 0)

# Log the binary indicator matrix
logging.info("\nBinary indicator shape: %s", R.shape)
logging.info("\nBinary Indicator:\n %s", R)


# Log a search result
search_results = movie_search('lord of the rings', movies_df)
logging.info("\n %s", search_results)

# Declare the number of movies and users
num_movies, num_users = Y.shape
# Initialize my_ratings based on the number of unique movies in movies_df
my_ratings = np.zeros(num_movies)

# Create a mapping between original movie IDs and their indices in my_ratings
unique_movie_ids = movies_df["movieId"].unique()
movie_id_to_index = {movie_id: index for index, movie_id in enumerate(unique_movie_ids)}

# Assign ratings at the correct positions using the mapping
my_ratings[movie_id_to_index[2571]] = 4.5 # The Matrix (1999)
my_ratings[movie_id_to_index[32]] = 4.0 # Twelve Monkeys (1995)
my_ratings[movie_id_to_index[260]] = 4.0 # Star Wars: Episode IV - A New Hope (1977)
my_ratings[movie_id_to_index[1196]] = 4.0 # Star Wars: Episode V - The Empire Strikes Back (1980)
my_ratings[movie_id_to_index[296]] = 4.5 # Pulp Fiction (1994)
my_ratings[movie_id_to_index[480]] = 3.5 # Jurassic Park (1993)
my_ratings[movie_id_to_index[356]] = 4.5 # Forrest Gump (1994)
my_ratings[movie_id_to_index[1]] = 4.0 # Toy Story (1995)
my_ratings[movie_id_to_index[527]] = 4.0 # Schindler's List (1993)
my_ratings[movie_id_to_index[4993]] = 5.0 # Lord of the Rings: The Fellowship of the Ring, The (2001)
my_ratings[movie_id_to_index[5952]] = 5.0 # Lord of the Rings: The Two Towers, The (2002)
my_ratings[movie_id_to_index[7153]] = 5.0 # Lord of the Rings: The Return of the King, The (2003)
my_ratings[movie_id_to_index[6539]] = 4.5 # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_ratings[movie_id_to_index[40815]] = 4.5 # Harry Potter and the Goblet of Fire (2005)
my_ratings[movie_id_to_index[1704]] = 4.5 # Good Will Hunting (1997)

# Log the updated my_ratings
logging.info("\nMy ratings shape: %s", my_ratings.shape)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

logging.info('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        logging.info(f'Rated {my_ratings[i]} for   {movies_df.loc[i,"title"]}')

logging.info("\nY shape:")
# Log Y shape before adding new user ratings
logging.info("before: %s", Y.shape)

# Add the new user ratings to the Y matrix
Y = np.c_[my_ratings, Y]
# Log Y shape after adding new user ratings
logging.info("after: %s", Y.shape)

logging.info("\nR shape:")
# Log R shape before adding new user ratings
logging.info("before: %s", R.shape)

# Update the binary indicator R matrix
R = np.c_[np.where(my_ratings != 0, 1, 0), R]
# Log R shape after adding new user ratings
logging.info("after: %s", R.shape)

# Normalize ratings
Ynorm, Ymean = normalize_ratings(Y, R)

num_movies, num_users = Y.shape 
num_features = 100 # number of features to learn

# Initialize parameters W, X, b, and use tf.Variable to track the variables
tf.random.set_seed(1234) # set the random seed so this always produces the same results
W = tf.Variable(tf.random.normal([num_users, num_features], dtype=tf.float64, name='W'))
X = tf.Variable(tf.random.normal([num_movies, num_features], dtype=tf.float64, name='X'))
b = tf.Variable(tf.random.normal([1, num_users], dtype=tf.float64, name='b'))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

# Train using a custom training loop
X, W, b, cost_history = train_model(X, W, b, Ynorm, R, lambda_=1, optimizer=optimizer, iterations=200, print_every=20)

# Log the cost history
logging.info("\nCost History: %s", cost_history)

# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# Save my predictions to a dataframe, and sort the predictions from highest to lowest
my_predictions_df = pd.DataFrame({'title': movies_df['title'], 'number of ratings': movieProperties_df["numRatings"], 'mean rating': movieProperties_df["meanRating"], 'prediction': my_predictions.round(2)})
my_predictions_df = my_predictions_df.sort_values(by='prediction', ascending=False)
# Define the path to the file where you want to save my_predictions
file_path = 'my_predictions.tsv'
# Save my_predictions_df to a file
my_predictions_df.to_csv(file_path, sep='\t', index=False)

# Sort predictions and get the indices
ix = np.argsort(my_predictions)[::-1]

# Set a cap for the number of movies to log
logging_cap = 50
# Set a cap for the release year
release_year_cap = 1970
# Set a cap for the number of ratings
num_ratings_cap = 50

# Initialize a counter for logged movies
logging_count = 0

logging.info('\n\nTop recommendations:\n')
for i in range(my_predictions.shape[0]):
    j = ix[i]

    try:
        # Extract the release year using a regular expression
        match = re.search(r'\((\d{4})\)', movies_df.loc[j, "title"])

        if match:
            release_year = int(match.group(1))

            # Check if the movie has over x ratings
            if movieProperties_df.loc[j, "numRatings"] > num_ratings_cap and release_year > release_year_cap:
                # Log the top movies
                logging.info(f'numRatings: {movieProperties_df.loc[j, "numRatings"]}, Predicted {my_predictions[j]:0.2f} for    {movies_df.loc[j, "title"]}')
                
                # Increment the counter
                logging_count += 1

                # Check if 20 movies have been printed
                if logging_count == logging_cap:
                    break
    except (ValueError, IndexError):
      # Handle errors in extracting the release year or accessing the list
      logging.info(f"\nError processing movie at index {j}. Skipping...")
      continue

# Log original ratings and compare with predicted ratings
logging.info('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        logging.info(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for    {movies_df.iloc[i]["title"]}')

# Define variables for RMSE and MAE all users calculations
predictions = pm
targets = Y
mask = R

# Get indices of rated items
rated_indices = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

# Calculate RMSE for rated items
rmse_for_rated = calculate_rmse_for_my_rated(my_predictions, my_ratings, rated_indices)
logging.info(f'\nRMSE for my rated items: {rmse_for_rated}')

# Calculate RMSE for all users
rmse_for_all_users = calculate_rmse_for_all_users(predictions, targets, mask)
logging.info(f'\nRMSE for all users: {rmse_for_all_users}')

# Calculate MAE for rated items
mae_for_rated = calculate_mae_for_my_rated(my_predictions, my_ratings, rated_indices)
logging.info(f'\nMAE for my rated items: {mae_for_rated}')

# Calculate MAE for all users
mae_for_all_users = calculate_mae_for_all_users(predictions, targets, mask)
logging.info(f'\nMAE for all users: {mae_for_all_users}')