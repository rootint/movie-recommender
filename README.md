# Movie Recommendation

## Danil Timofeev, BS21-AI
## d.timofeev@innopolis.university

This is an implementation of movie recommendation using a hybrid approach, combining ALS and Random Forest methods. 
Notebooks in notebooks/testing don't have extensive comments as they were used as drafts to check hypotheses.

## How to use
To evaluate the model, just run the [evaluate.py](/benchmark/evaluate.py) file. If will run the models from the models folder and evaluate RMSE and MAE for the test dataset. Furthermore, it will print a list of movies that are recommended to a particular user from the database and compare them with the films that the user actually liked the most.