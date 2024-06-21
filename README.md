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

## [How to Train a Model](train)
To train a specific model, simply run the command from the shell:
```
python main.py [options]
```
where options are:
* `-t`, `--training_config_file`: Path to the training configuration file. Default is `./config/train_config.json`.
* `-d`, `--data_config_file`: Path to the data configuration file. Default is `./config/data_config.json`.

This command will parse the 3 configuration files specified in the *config* folder. Of particular relevance are `data_config.json`, for information on which data to use and how to process it, and `train_config.json`, for the configuration of model training.

### Structure of Configurations
The file structure is well-defined and must comply with the following schemes.

#### [data_config.json](/config/data_config.json)
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
        * **cut** [*list*]: Categories include `FAIR`, `GOOD`, `VERY GOOD`, `IDEAL`, and `PREMIUM`.
        * **color** [*list*]: Categories include `D`, `E`, `F`, `G`, `H`, `I`, and `J`.
        * **clarity** [*list*]: Categories include `IF`, `VVS1`, `VVS2`, `VS1`, `VS2`, `SI1`, `SI2`, and `I1`.

    > **toOrdinal**
    * **active** [*boolean*]: A boolean value indicating whether to convert categorical columns to ordinal variables.
        * `true`: The specified categorical columns will be converted to ordinal variables.
        * `false`: The specified categorical columns will not be converted to ordinal variables.

    * **columns_categories** [*dict*]: An object specifying the categorical columns and their categories to be converted to ordinal variables.
        * **cut** [*list*]: Categories include `FAIR`, `GOOD`, `VERY GOOD`, `IDEAL`, and `PREMIUM`.
        * **color** [*list*]: Categories include `D`, `E`, `F`, `G`, `H`, `I`, and `J`.
        * **clarity** [*list*]: Categories include `IF`, `VVS1`, `VVS2`, `VS1`, `VS2`, `SI1`, `SI2`, and `I1`.

#### [train_config.json](/config/train_config.json)
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

## [How to Use API](api)

### POST `/predict`
The endpoint was developed with the aim of predicting the value of a diamond from its features. The prediction can be carried out through one of the trained models by specifying its identifier.

Its parameters are:
* **model** [*string*, Optional] &rarr; The id of the model to be used to determine the value of the diamond. It must be one of those in the `./output/models` folder. By default the best model is set.
* **carat** [*float*]
* **cut** [*string*]
* **color** [*string*]
* **clarity** [*string*]
* **depth** [*float*]
* **table** [*float*]
* **x** [*float*]
* **y** [*float*]
* **z** [*float*]

In the event that one of these parameters does not meet the required constraints, an error is thrown with a status code of 400.

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

A json is returned with a unique `samples` key that collects a list of all N samples that meet the similarity conditions.

### POST `/train`
The purpose of the endpoint is to launch the training of a model from the specified configuration files.\
It performs the same operation as could be achieved by launching the command: `python main.py -t <training_config_path> -d <data_config_path>`.

The object should contain the following optional fields:
* **training_config_name** [*string*, Optional]: The name of the training configuration stored in the `./config` folder. This field should be provided along with `data_config_name` if you choose to use named configurations.
* **data_config_name** [*string*, Optional]: The name of the data configuration stored in the `./config` folder. This field should be provided along with `training_config_name` if you choose to use named configurations.
* **training_config_path** [*string*, Optional]: The file path to the training configuration. This field should be provided along with `data_config_path` if you choose to use file paths for configurations.
* **data_config_path** [*string*, Optional]: The file path to the data configuration. This field should be provided along with `training_config_path` if you choose to use file paths for configurations.

You must specify either both `training_config_name` and `data_config_name` or both `training_config_path` and `data_config_path`. If neither pair is provided, the request will be considered invalid and an error will be returned.

### [API Logs Database](/log/api_logs.db)
All requests made to all the endpoints are stored in the `api_logs.db` database at path `./log`. The database consists of 4 tables:
1. `requests` &rarr; This table logs general information about each API request.

| Column | Type | Description |
|---|---|---|
| id* | INTEGER | Unique identifier for each request |
| timestamp | TEXT | The time when the request was made |
| method | TEXT | The HTTP method used for the request (e.g., GET, POST) |
| path | TEXT | The path of the API endpoint that was called |
| status code | INTEGER | The HTTP status code returned by the API |

2. `predict` &rarr; This table stores information about prediction requests and their responses.

| Column | Type | Description |
|---|---|---|
| id* | INTEGER | Unique identifier for each prediction request |
| request_id^ | INTEGER | References the `id` in the `requests` table |
| response_time | TEXT | The time taken to respond to the prediction request |
| carat | REAL | The carat of the diamond |
| cut | TEXT | The cut of the diamond |
| color | TEXT | The color grade of the diamond |
| clarity | TEXT | The clarity grade of the diamond |
| depth | REAL | The depth percentage of the diamond |
| table_val | REAL | The table percentage of the diamond |
| x | REAL | The length of the diamond in mm |
| y | REAL | The width of the diamond in mm |
| z | REAL | The depth of the diamond in mm |
| predicted_value | REAL | The predicted value of the diamond |
| note | TEXT | Additional notes, especially error messages if the status code is not 200 |

3. `similar` &rarr; This table stores information about similarity requests and their responses.

| Column | Type | Description |
|---|---|---|
| id* | INTEGER | Unique identifier for each similarity entry |
| request_id^ | INTEGER | References the id in the requests table |
| response_time | TEXT | The time taken to respond to the similarity request |
| carat | REAL | The carat of the diamond |
| cut | TEXT | The cut of the diamond |
| color | TEXT | The color grade of the diamond |
| clarity | TEXT | The clarity grade of the diamond |
| number_samples | INTEGER | The number of similar samples requested |
| method | TEXT | The method used for finding similarity (e.g., cosine similarity) |
| dataset_name | TEXT | The name of the dataset used |
| samples | TEXT | JSON string containing the similar samples |
| note | TEXT | Additional notes, especially error messages if the status code is not 200 |

4. `training` &rarr; This table stores information about training requests and their responses.

| Column | Type | Description |
|---|---|---|
| id* | TEXT | Unique identifier for each training session |
| request_id^ | INTEGER | References the id in the requests table |
| response_time | TEXT | The time taken to respond to the similarity request |
| type | TEXT | The type of model trained |
| parameters | TEXT | JSON string containing training parameters or "default" if no hyperparameters have been specified in configuration |
| dataset_name | TEXT | The name of the dataset used for training |
| preprocessing | TEXT | JSON string containing pre-processing operations or NULL if no operation was active |
| split_size | REAL | The size of the data split for training and testing |
| evaluation | TEXT | JSON string containing the evaluation metrics |
| training_type | TEXT | If trained through Bayesian optimisation, a JSON string with search parameters, otherwise "one-shot" |
| training_configuration | TEXT | File path or identifier for the training configuration |
| data_configuration | TEXT | File path or identifier for the data configuration |
| note | TEXT | Additional notes, especially error messages if the status code is not 200 |

## Project Structure

1. Inside the `./api` folder is the package of the FastAPI implementation:
    * `app.py`: The main API application script;
    * `core.py`: Core functionalities and logic for the API;
    * `database.py`: Database interaction logic;
    * `request_body.py`: Defines request body structures for the API;

2. The configuration files of the project, but especially of model training, are stored in `./config`. All those custom files that need to be generated for the various trainings must be placed in this folder. Otherwise, absolute paths to the configuration files must be specified.

    * `data_config.json`: Example of data processing configuration. If no alternative configuration name is specified, the system will always search for the current file;
    * `data_parameter_search_xgboost.json`: Example of data processing configuration with a customised name. In this case, it collects the data configuration for the hyperparameter search of the XGBoost model;
    * `logger.ini`: Configuration for logging;
    * `train_config.json`: Example of training process configuration. If no alternative configuration name is specified, the system will always search for the current file;
    * `train_parameter_search_xgboost.json`: Example of training process configuration with a customised name. In this case, it collects the training configuration for the hyperparameter search of the XGBoost model.

3. The `./data` folder contains the data required for training and possibly datasets saved during a process.

4. The log folder `./log` contains within it the database for the API (`api_logs.db`) and the logging file of the training processes (`train.log`) and possibly the *tar.gz* archive of the previous days' logs (`train-log.tar.gz`).

5. The `./output` folder is the directory for storing the output results and models trained in the various processes.

    * `results.json`: JSON file containing the metadata of each training conducted;
    * `./images`: subfolder in which all *GOF plots* of the various models are saved, identifiable by the associated UUID;
    * `./models`: subdirectory in which all serialised models are stored, identifiable by their associated UUID;
    * `./models`: subdirectory in which any serialised transformers for inverse transformations are stored. They are also identifiable by their associated UUID;

6. In the `./train` folder, the package for training models is collected.

    * `training_model.py`: It is the main file in which all functions and classes for training models are collected;
    * `utils.py`: Utility functions for training.

7. `main.py` is the python file with which one can directly launch the training of a model, as explained in [this section](#how-to-train-a-model).
