# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to Train a Model
To train a specific model, simply run the command from the shell:
```
python training_model.py
```

This command will parse the 3 configuration files specified in the *config* folder. Of particular relevance are `data_config.json`, for information on which data to use and how to process it, and `train_config.json`, for the configuration of model training.

### Structure of Configurations
The file structure is well-defined and must comply with the following schemes.

#### data_config.json
To use this configuration, ensure that the dataset source and preparation steps are correctly specified according to the requirements. Adjust the active flags and columns/categories as needed to fit the specific needs of your analysis or machine learning workflow.

1. **data** &rarr; This section contains information about the source and handling of the dataset.

    * **onlineSource** [*boolean*]: A boolean value indicating whether the dataset should be downloaded from an online source:
        * `true`: The dataset will be downloaded from the specified URL.
        * `false`: The dataset will be loaded from a local file path.

    * **url** [*string*]: The URL of the dataset if *onlineSource* is `true`.

    * **localPath** [*string*]: The local file path to the dataset if *onlineSource* is `false`.

    * **saveData** [*boolean*]: A boolean value indicating whether to save the downloaded or processed data locally.
        * `true`: The data will be saved locally.
        * `false`: The data will not be saved.

2. **preparation** &rarr; This section details the steps for preparing the dataset.

    > **cleanData**
    * **active** [*boolean*]: A boolean value indicating whether data cleaning should be performed.
        * `true`: Data cleaning will be performed.
        * `false`: Data cleaning will not be performed.

    > **dropColumns**
    * **active** [*boolean*]: A boolean value indicating whether certain columns should be dropped from the dataset.
        * `true`: The specified columns will be dropped.
        * `false`: The specified columns will not be dropped.

    * **columns** [*list*]: An array of column names to be dropped from the dataset. Example columns include `depth`, `table`, `y`, and `z`.

    > **toDummy**
    * **active** [*boolean*]: A boolean value indicating whether to convert categorical columns to dummy variables.
        * `true`: The specified categorical columns will be converted to dummy variables.
        * `false`: The specified categorical columns will not be converted to dummy variables.

    * **columns_categories** [*dict*]: An object specifying the categorical columns and their categories to be converted to dummy variables.
        * **cut** [*list*]: Categories include `Fair`, `Good`, `Very Good`, `Ideal`, and `Premium`.
        * **color** [*list*]: Categories include `D`, `E`, `F`, `G`, `H`, `I`, and `J`.
        * **clarity** [*list*]: Categories include `IF`, `VVS1`, `VVS2`, `VS1`, `VS2`, `SI1`, `SI2`, and `I1`.

    > **toOrdinal**
    * **active** [*boolean*]: A boolean value indicating whether to convert categorical columns to ordinal variables.
        * `true`: The specified categorical columns will be converted to ordinal variables.
        * `false`: The specified categorical columns will not be converted to ordinal variables.

    * **columns_categories** [*dict*]: An object specifying the categorical columns and their categories to be converted to ordinal variables.
        * **cut** [*list*]: Categories include `Fair`, `Good`, `Very Good`, `Ideal`, and `Premium`.
        * **color** [*list*]: Categories include `D`, `E`, `F`, `G`, `H`, `I`, and `J`.
        * **clarity** [*list*]: Categories include `IF`, `VVS1`, `VVS2`, `VS1`, `VS2`, `SI1`, `SI2`, and `I1`.

#### train_config.json
To use this configuration, ensure that the training parameters, model settings, and logging configurations are correctly specified according to the requirements. Adjust the active flags, hyperparameters, and logging details as needed to fit the specific needs of your machine learning workflow.

1. **Training** &rarr; This section contains parameters and settings related to the training process.

    * **testSize** [*float*]: The proportion of the dataset to include in the test split. For example, 0.2 indicates that 20% of the data will be used for testing.

    * **randomState** [*integer*]: The seed used by the random number generator to ensure reproducibility. For example, 42 is a commonly used seed value.

    * **transformation** [*string*]: Specifies any transformation to be applied to the data before training. In the default configuration it is set to null, indicating that there is no transformation. It may be chosen from one of:
        * `null`
        * `"log"`
        * `"standardisation"`
        * `"min_max_scaling"`
        * `"power"`
        * `"square_root"`
        * `"exponential"`
        * `"tanh"`
        * `"z_score"`
        * `"johnson"`

    * **processingData** [*boolean*]: A boolean value indicating whether data processing should be performed.
        * `true`: Data processing will be performed.
        * `false`: Data processing will not be performed.

    * **model**: This subsection defines the machine learning model to be used and its parameters.

        * **name** [*string*]: The name of the model to be used. It can be one of the following:
            * `"linear_regression"`
            * `"logistic_regression"`
            * `"lasso_regression"`
            * `"ridge_regression"`
            * `"decision_tree"`
            * `"svm"`
            * `"naive_bayes"`
            * `"knn"`
            * `"random_forest"`
            * `"gradient_boosting"`
            * `"xgb_regression"`
            * `"ada_boosting"`

        * **parameters** [*dict*]: A dictionary of hyperparameters for the specified model can be trained one-shot or will be optimised using Optuna hyperparameter tuning.\
        The parameters depend directly on the chosen model.

    * **metrics** [*string*]: A list of metrics to be evaluated during training. It can be one of the following:
        * `"mean_absolute_error"`
        * `"mean_squared_error"`
        * `"root_mean_squared_error"`
        * `"mean_absolute_percentage_error"`        
        * `"r2_score"`
        * `"explained_variance_score"`

    * **tuning**: This subsection contains settings related to hyperparameter tuning using `Optuna`.

        * **active** [*boolean*]: A boolean value indicating whether hyperparameter tuning should be performed.
            * `true`: Hyperparameter tuning will be performed.
            * `false`: Hyperparameter tuning will not be performed.

        * **trialsNumber** [*integer*]: The number of trials for the Optuna study, for example `150`.

        * **studyName** [*string*]: The name of the Optuna study, for example `"Diamonds XGBoost Regression"`.

2. logger &rarr; This section contains configurations for logging during the training process.

    * **configName** [*string*]: The name of the configuration file for the logger, for example `logger.ini`.

    * **name** [*string*]: The name of the log file, for example `train.log`.

    * **level** [*string*]: The logging level, for example `DEBUG`.

    * **maxFiles** [*string*]: The maximum number of log files to keep, for example `10`.

    * **rotateAtTime** [*string*]: The time at which to rotate the log files, for example `23:59`.

    * **formatMessage** [*string*]: The format string for log messages. Example: `"%(asctime)s - [%(name)s:%(filename)s:%(lineno)d] - %(levelname)s: %(message)s"`.

    * **formatDate** [*string*]: The format string for the date in log messages. For example: `"%Y-%m-%d %H:%M:%S"`.

### Default Configurations

1. **data_config.json**
```json
{
    "data": {
        "onlineSource": true,
        "url": "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv",
        "localPath": "diamonds.csv",
        "saveData": false
    },
    "preparation": {
        "cleanData": {
            "active": false
        },
        "dropColumns": {
            "active": false,
            "columns": [
                "depth", "table", "y", "z"
            ]
        },
        "toDummy": {
            "active": false,
            "columns_categories": {
                "cut": [
                    "Fair", "Good", "Very Good", "Ideal", "Premium"
                ],
                "color": [
                    "D", "E", "F", "G", "H", "I", "J"
                ],
                "clarity": [
                    "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"
                ]
            }
        },
        "toOrdinal": {
            "active": false,
            "columns_categories": {
                "cut": [
                    "Fair", "Good", "Very Good", "Ideal", "Premium"
                ],
                "color": [
                    "D", "E", "F", "G", "H", "I", "J"
                ],
                "clarity": [
                    "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"
                ]
            }
        }
    }
}
```

2. **train_config.json**
```json
{
    "training": {
        "testSize": 0.2,
        "randomState": null,
        "transformation": null,
        "processingData": true,
        "model": {
            "name": "linear_regression",
            "parameters": {}
        },
        "metrics":[
            "r2_score",
            "mean_absolute_error"
        ],
        "tuning": {
            "active": false,
            "trialsNumber": 100,
            "studyName": "Diamonds Linear Regression"
        }
    },
    "logger": {
        "configName": "logger.ini",
        "name": "train.log",
        "level": "DEBUG",
        "maxFiles": 10,
        "rotateAtTime": "23:59",
        "formatMessage": "%(asctime)s - [%(name)s:%(filename)s:%(lineno)d] - %(levelname)s: %(message)s",
        "formatDate": "%Y-%m-%d %H:%M:%S"
    }
}
```

## How to Use API

### POST `/predict`
The endpoint was developed with the aim of predicting the value of a diamond from its features. The prediction can be carried out through one of the trained models by specifying its identifier.

Its parameters are:
* **model** [*string*, Optional] &rarr; The id of the model to be used to determine the value of the diamond. It must be one of those in the `./train/models` folder. By default the best model is set.
* **carat** [*float*]
* **cut** [*string*]
* **color** [*string*]
* **clarity** [*string*]
* **depth** [*float*]
* **table** [*float*]
* **x** [*float*]
* **y** [*float*]
* **z** [*float*]

In the event that one of these parameters does not meet the required constraints, an X error is thrown with a status code of 400.

A json is returned as output with a single `predicted_value` key whose value corresponds to the float generated by the model.

### POST `/similar`
The endpoint was developed to find all those values with the same cut, colour and clarity, and with the most similar weight, with respect to the specified features. The weight difference can be determined via one of the implemented functions.

Its parameters are:
* **n** [*integer*, Optional] &rarr; The cardinality of the set of samples satisfying the search criterion. By default is 5.
* **method** [*string*, Optional] &rarr; The method for calculating the difference between the weights. It can be one of [`"absolute difference"`, `"relative difference"`, `"squared difference"`, `"z score"`, `"cosine similarity"`] and by default is `"cosine similarity"`.
* **dataset_name** [*string*, Optional] &rarr; The name of the saved dataset to the `./data` path from which to retrieve the set of samples. By default is `"diamonds.csv"`
* **carat** [*float*] 
* **cut** [*string*]
* **color** [*string*]
* **clarity** [*string*]

***!!!*** Errori e Output

All requests made to both endpoints are stored in the `api_logs.db` database at path `./log`. The database consists of 3 tables:
1. Qualcosa
2. Qualcosa
3. Qualcosa