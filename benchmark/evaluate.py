from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pandas as pd


class RecommendationSystem:
    def __init__(self, approach="hybrid"):
        self.approach = approach
        self.spark = SparkSession.builder.appName("MovieLensALS").getOrCreate()
        self.als_model = ALSModel.load("../models/als")
        self.rf_model = joblib.load("../models/random_forest.joblib")
        self.als_predictions_pandas = None
        self.rf_predictions_df = None
        self.final_evaluation_df = None

    def _get_als_recommendations(self):
        """
        Gets the ALS recommendations based on the users/movies ratings.
        Prints out the RMSE and MAE of the recommendations.
        """
        # Reading the data
        test = self.spark.read.parquet(
            "data/ratings_test.parquet", sep="\t", inferSchema=True
        )
        test = (
            test.withColumnRenamed("_c0", "user_id")
            .withColumnRenamed("_c1", "movie_id")
            .withColumnRenamed("_c2", "rating")
        )

        # Running predictions
        predictions = self.als_model.transform(test)

        # Evaluating the results
        rmse_evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )

        mae_evaluator = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol="prediction"
        )

        rmse = rmse_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)

        # Saving data for future use
        als_predictions_selected = predictions.select(
            "user_id", "movie_id", "prediction"
        )
        self.als_predictions_pandas = als_predictions_selected.toPandas()

        print(f"ALS RMSE = {rmse}")
        print(f"ALS MAE = {mae}")

    def _get_rf_recommendations(self):
        """
        Gets the Random Forest recommendations based on the users' demographic data.
        Prints out the RMSE and MAE of the recommendations.
        """
        # Reading the test data for evaluation
        ratings_df_test = pd.read_parquet("data/ratings_test.parquet")
        users_df = pd.read_parquet("data/users.parquet")

        # Preprocessing and merging the data to fit into the model
        combined_df_test = pd.merge(ratings_df_test, users_df, on="user_id")
        labels_test = combined_df_test["rating"]
        combined_df_test.drop(["rating"], axis=1, inplace=True)

        # Running inference
        rf_predictions = self.rf_model.predict(combined_df_test)

        # Calculating accuracy, RMSE, and MAE
        rmse = mean_squared_error(labels_test, rf_predictions, squared=False)
        mae = mean_absolute_error(labels_test, rf_predictions)

        # Saving outputs for future use
        self.rf_predictions_df = pd.DataFrame(
            {
                "user_id": ratings_df_test["user_id"],
                "movie_id": ratings_df_test["movie_id"],
                "prediction_rf": rf_predictions,
            }
        )

        print(f"RF RMSE = {rmse}")
        print(f"RF MAE = {mae}")

    def _get_hybrid_recommendations(self):
        """
        Gets the hybrid recommendations based on two methods: ALS and RF.
        First, runs both recommendations, and combines them into one using coefficients from a Linear Regression that was trained before.

        Returns:
        pd.DataFrame: A dataframe with predictions for all recommendation systems.
        """
        # Runs both recommendations, saving their outputs
        self._get_als_recommendations()
        self._get_rf_recommendations()

        # Combining both datasets into one
        ratings_df_test = pd.read_parquet("data/ratings_test.parquet")
        combined_predictions = pd.merge(
            self.als_predictions_pandas,
            self.rf_predictions_df,
            on=["user_id", "movie_id"],
        )
        self.final_evaluation_df = pd.merge(
            combined_predictions, ratings_df_test, on=["user_id", "movie_id"]
        )

        # Coming up with a final prediction based on the coefficients that were retrieved using Linear Regression
        combined_predictions["final_prediction"] = (
            combined_predictions["prediction"] * 1.005058707886241
            + combined_predictions["prediction_rf"] * 0.029824356446759744
        )
        self.final_evaluation_df = pd.merge(
            combined_predictions, ratings_df_test, on=["user_id", "movie_id"]
        )

        # Outputting the RMSE and MAE of the hybrid method
        final_rmse = mean_squared_error(
            self.final_evaluation_df["rating"],
            self.final_evaluation_df["final_prediction"],
            squared=False,
        )
        final_mae = mean_absolute_error(
            self.final_evaluation_df["rating"],
            self.final_evaluation_df["final_prediction"],
        )

        print(f"Final Combined RMSE = {final_rmse}")
        print(f"Final Combined MAE = {final_mae}")

        return self.final_evaluation_df

    def run_benchmark(self):
        """
        Runs the benchmark and prints RMSE and MAE scores.
        It is required to run benchmark, and then only then do recommendations.
        """
        return self._get_hybrid_recommendations()

    def get_recommendations_for_user_id(self, user_id, top_n=5):
        """
        Recommend top N movies for a given user.

        Parameters:
        user_id (int): The user ID for whom recommendations are to be made.
        top_n (int): The number of top recommendations to return.

        Returns:
        list: List of recommended movie IDs.
        list: List of top-rated movie IDs by the user.
        """
        # Filtering for the specific user
        user_df = self.final_evaluation_df[
            self.final_evaluation_df["user_id"] == user_id
        ]

        # Sorting by 'final_prediction' and 'rating' in descending order
        sorted_user_df = user_df.sort_values(by="final_prediction", ascending=False)
        rating_sorted_user_df = user_df.sort_values(by="rating", ascending=False)

        # Selecting top N movies
        top_movies = sorted_user_df.head(top_n)
        top_movies_real = rating_sorted_user_df.head(top_n)

        # Extracting movie IDs
        recommended_movies = top_movies["movie_id"].tolist()
        real_liked_movies = top_movies_real["movie_id"].tolist()

        return recommended_movies, real_liked_movies


def main():
    recommender = RecommendationSystem()
    recommender.run_benchmark()
    recommendations, real_preferences = recommender.get_recommendations_for_user_id(user_id=12, top_n=5)
    print(f"Hybrid recommendations: {recommendations}")
    print(f"Real user preferences: {real_preferences}")
    # As we can see, movie #318 is in the top rating!
    # To get more results, you can change the user id and top_n in the function above


if __name__ == "__main__":
    main()
