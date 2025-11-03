# ENPM611 Project Application Template

This is the template for the ENPM611 class project. Use this template in conjunction with the provided data to implement an application that analyzes GitHub issues for the [poetry](https://github.com/python-poetry/poetry/issues) Open Source project and generates interesting insights.

This application template implements some of the basic functions:

- `data_loader.py`: Utility to load the issues from the provided data file and returns the issues in a runtime data structure (e.g., objects)
- `model.py`: Implements the data model into which the data file is loaded. The data can then be accessed by accessing the fields of objects.
- `config.py`: Supports configuring the application via the `config.json` file. You can add other configuration paramters to the `config.json` file.
- `run.py`: This is the module that will be invoked to run your application. Based on the `--feature` command line parameter, one of the three analyses you implemented will be run. You need to extend this module to call other analyses.

With the utility functions provided, you should focus on implementing creative analyses that generate intersting and insightful insights.

In addition to the utility functions, an example analysis has also been implemented in `example_analysis.py`. It illustrates how to use the provided utility functions and how to produce output.

## Setup

To get started, your team should create a fork of this repository. Then, every team member should clone your repository to their local computer. 


### Install dependencies

In the root directory of the application, create a virtual environment, activate that environment, and install the dependencies like so:

```
pip install -r requirements.txt
```

### Download and configure the data file

Download the data file (in `json` format) from the project assignment in Canvas and update the `config.json` with the path to the file. Note, you can also specify an environment variable by the same name as the config setting (`ENPM611_PROJECT_DATA_PATH`) to avoid committing your personal path to the repository.


### Run an analysis

With everything set up, you should be able to run the existing example analysis:

```
python run.py --feature 0
```

That will output basic information about the issues to the command line.


## VSCode run configuration

To make the application easier to debug, runtime configurations are provided to run each of the analyses you are implementing. When you click on the run button in the left-hand side toolbar, you can select to run one of the three analyses or run the file you are currently viewing. That makes debugging a little easier. This run configuration is specified in the `.vscode/launch.json` if you want to modify it.

The `.vscode/settings.json` also customizes the VSCode user interface sligthly to make navigation and debugging easier. But that is a matter of preference and can be turned off by removing the appropriate settings.

## Feature 1: LABEL RESOLUTION TIME ANALYSIS AND PREDICTION

Feature 1 is basically predicting the approximate time to complete the open issues based on Machine Learning model which was trained on closed issues. Different features were used to train the model. 

Run below code to get analysis of feature 1
```
python run.py --feature 1
```

LABEL RESOLUTION TIME ANALYSIS AND PREDICTION:

Overall Statistics:
• Total closed issues analyzed: 5033
• Unique labels found: 54
• Overall median resolution: 9.69 days
• Overall average resolution: 162.31 days

📊 Top 10 Fastest Resolving Labels:

status/invalid - 0.04 days (n=22)
area/project/deps - 0.10 days (n=6)
kind/question - 0.20 days (n=263)
area/distribution - 0.21 days (n=1)
version/1.2.0 - 0.32 days (n=2)
status/duplicate - 0.41 days (n=318)
area/docs/faq - 1.13 days (n=29)
status/triage - 1.47 days (n=790)
status/external-issue - 1.56 days (n=143)
area/show - 4.44 days (n=1)
⏰ Top 10 Slowest Resolving Labels:

status/accepted - 788.46 days (n=3)
area/error-handling - 692.22 days (n=35)
area/ux - 503.40 days (n=32)
status/needs-consensus - 502.53 days (n=4)
area/publishing - 375.96 days (n=17)
status/wontfix - 358.98 days (n=10)
kind/enhancement - 323.88 days (n=30)
area/plugin-api - 323.20 days (n=8)
status/needs-reproduction - 290.73 days (n=59)
good first issue - 269.60 days (n=13)
Top Feature Importances:
1. month: 0.341
2. day_of_week: 0.304
3. num_labels: 0.134
4. has_feature_label: 0.083
5. has_area_label: 0.068

🔮 Sample Predictions for Open Issues (showing 5 of 317):
• Issue #9183: 0.8 days
Labels: area/docs, status/triage
• Issue #9146: 4.4 days
Labels: area/docs, status/triage
• Issue #7643: 21.5 days
Labels: kind/bug, status/triage, area/windows
• Issue #7610: 21.5 days
Labels: kind/bug, area/installer, status/triage
• Issue #9644: 25.7 days
Labels: area/docs

Some of the graphs and prediction time & statistics are as follows:
Different types of graphs and analysis are done based on the prediction time to complete the open issues.

Output Files:
• output/label_resolution_analysis.json - Complete analysis results
• output/label_statistics.json - Label-wise statistics
• output/open_issue_predictions.json - Predictions for open issues
• output/visualizations/ - All generated graphs

