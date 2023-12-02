# Introduction
...
# Data analysis
First of all, I checked the u.data file. There, I decided to drop the timestamp column, because I can't use it in any way in the ALS approach, I just need the movies, users and their respective ratings. Furthermore, I decided not to clean data from the movies that have a small amount of ratings, because ALS is designed to handle sparse matrices, and it would be a waste of opportunity not to leverage recommendations of "long tail" items - those that may not be hugely popular but are highly relevant to a subset of users. Dropping movies with fewer ratings might exclude these niche but potentially valuable recommendations.<br>
After that, I checked the u.item dataset and cleared the `video_release_date` column since it had only NaNs in it. Also, I dropped the `IMDb_URL, title, release_date` columns, since they are unique values (apart from the release_date, which I just didn't use in the model). <br>
Finally, I opened the u.user dataset which contains the users' demographic data, and one-hot encoded `gender` and `occupation` columns. Furthermore, I dropped the zip code column, since my models don't really take your geographic location into account. Then, I decided to one-hot encode age as well, splitting the ages into groups: young people (age less than 25), adults (25-55), and seniors (55+). Initially, I wanted to split the age groups into real age groups (young people are people less than 20, and seniors are older than 60), but after looking at the age distribution, I realised that this would make this very unbalanced, so I shifted the groups a bit. <br>![Age distribution](figures/age_distribution.png)<br> However, I can't make them very balanced, as this would mean that adults young people are until 30, and seniors start at 40, which makes no sense in the real life. So, I have the following distribution:<br>![Still unbalanced](figures/imbalance.png)<br> It's still unbalanced, but that's better than the initial distribution.<br>
Also, it is worth noting that I saved all the files in .parquet format after preprocessing, saving disk space and making data retrieval more optimized.
<br>
For training and testing I used the [u1.base](../data/raw/ml-100k/u1.base) and the [u1.test](../data/raw/ml-100k/u1.test), as this is the split that is used to evaluate the model based on my findings online: [here is the leaderboard](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k)

# Model Implementation
For the model, I decided to go with a hybrid approach, using ALS for collaborative filtering, and Random Forest to handle the content-based predictions, which are the ones based on the demographics. So, my ALS and Random Forest algorithms predicted ratings, and after getting both ratings, I did the following math: `final_rating = als_coeff * als_ratings + rf_coeff * rf_ratings`, where `als_coeff` and `rf_coeff` are the coefficients I got using a Linear Regression. Initially, I tried just averaging the values, but it worsened the performance compared to just using ALS, as Random Forest gave worse predictions. Combining the values in such a smart way allows the RF model to partially fix the disadvantages of the ALS approach, such as cold-starting. <br>
Here are the reasons why I chose ALS in my collaborative filtering:
1) It is better than a user-based approach, because user-based doesn't scale well.
2) It is better than a item-based approach, because item-based handles sparse matrices worse than ALS.
3) It is better than other matrix techniques, such as SVD, because SVD doesn't "like" missing data and doesn't scale as well as ALS does.
4) It is better than Deep Learning, because DL is too complex and would require a lot of compute, while the ALS approach is much more intuitive and easier to implement (just using pyspark is easy!).
   
Here are the reasons why I chose the Random Forest approach in my content-based predictions:
1) It is robust to overfitting.
2) It can be computed quickly because it is easy to parallelize.
3) It can handle diverse data types, like categorical, numerical and binary.
# Model Advantages and Disadvantages
**Advantages:**
1) Leveraging strengths of both approaches: 
    - Collaborative Filtering (ALS): Captures user-item interactions and can recommend items based on the preferences of similar users.
    - Content-Based Filtering (Random Forest): Utilizes user demographic data, which helps in personalizing recommendations based on individual user characteristics.
2) Robustness to Sparse Data:
   - ALS is particularly effective in dealing with sparse datasets, which is a common issue in user-item matrices.
3) Mitigating Cold Start Problem:
   - The content-based component (Random Forest) can help in addressing the cold start problem for new users by leveraging their demographic data.

**Disadvantages:**
1) Complexity and Resource Intensity:
   - Implementing and maintaining a hybrid system is more complex than using a single approach. It may require more computational resources, especially in the training and tuning phases.
2) Data Dependencies:
   - The effectiveness of the content-based component heavily relies on the availability and quality of user demographic data. Inaccuracies or biases in this data can skew recommendations. For example, I didn't address the imbalance in the age distribution (mentioned in the Data analysis).
3) Potential Performance Increase:
   - I assume that trying a state-of-the-art solution like GNN would bring more accurate recommendations, and I didn't implement it.
# Training Process
Initially, I tried using the default hyperparameters for both models. 
# Evaluation
...
# Results
...

In the `reports` directory create a report about your work. In the report, describe in details the implementation of your system. Mention its advantages and disadvantages.




Creating a comprehensive recommendation system with these three components – ALS for collaborative filtering, user demographics-based recommendations, and genre-based recommendations – is an ambitious but highly effective approach. Let's break down the process:

1. ALS for Collaborative Filtering
You've already implemented this part. The ALS model in PySpark uses user-item interaction data (like ratings) to predict user preferences based on similar user behaviors.

2. User Demographics-Based Recommendations
To incorporate user demographics into the recommendation system, you'll typically use a content-based approach or a hybrid model that includes demographic data. Here's a high-level overview:

Feature Engineering: Process the demographic data (age, gender, occupation, zip code) and convert it into a format suitable for machine learning (e.g., one-hot encoding for categorical data).
Model Building: Use a machine learning model (like Random Forest, Logistic Regression, etc.) to predict user preferences based on their demographic data.
Combining with User Preferences: Use this model to predict movie preferences for each user based on their demographics. This could involve identifying demographic trends (e.g., certain age groups preferring specific genres) and using these insights to make recommendations.
3. Genre-Based Recommendations
This part involves recommending movies based on their similarity in genres:

Process Genre Data: Transform the genre information of each movie into a suitable format (like one-hot encoded vectors).
Calculate Similarities: For a given movie, calculate its similarity to all other movies based on genres (using cosine similarity, for example).
Recommendation Logic: For a user who likes a certain movie, find and recommend movies that are similar in genre to that movie.
Integrating the Three Components
Hybrid Recommendation System: Develop a system that combines recommendations from all three methods. This could be done by averaging the scores from each method or using a more complex approach like a machine learning model to integrate them.
Weighted Recommendations: Depending on the confidence in each method, assign different weights to each component's recommendations.
Example Workflow
Get Collaborative Filtering Recommendations: Use the ALS model to predict user preferences for movies.
Get Demographic-Based Recommendations: Predict user preferences based on their demographic profile.
Get Genre-Based Recommendations: Identify movies similar in genre to those the user has rated highly.
Combine Recommendations: Integrate these three sets of recommendations into a final recommendation list for each user.
Challenges and Considerations
Data Sparsity: Demographic and genre-based methods can help mitigate the cold start problem in collaborative filtering.
Model Complexity: Integrating multiple models increases complexity. Ensure that the final system is manageable and maintainable.
Evaluation: Evaluate the performance of each component and the integrated system using relevant metrics (RMSE, Precision@k, Recall@k, etc.).
User Feedback Loop: Consider incorporating a mechanism to capture user feedback, which can be used to continuously improve the recommendation system.
This approach leverages the strengths of collaborative filtering while augmenting it with the targeted focus of demographic and content-based methods, leading to a more rounded and effective recommendation system.