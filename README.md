# ENPM611 Project Application Template

This is the template for the ENPM611 class project. Use this template in conjunction with the provided data to analyze GitHub issues for the [poetry](https://github.com/python-poetry/poetry/issues) Open Source project and generates interesting insights.

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

---

## Feature 1: LABEL RESOLUTION TIME ANALYSIS AND PREDICTION

Feature 1 is basically predicting the approximate time to complete the open issues based on Machine Learning model which was trained on closed issues. Different features were used to train the model. 

Run below code to get analysis of feature 1:
```
python run.py --feature 1
```

### LABEL RESOLUTION TIME ANALYSIS AND PREDICTION:

Overall Statistics:

• Total closed issues analyzed: 5033

• Unique labels found: 54

• Overall median resolution: 9.69 days

• Overall average resolution: 162.31 days



### 📊 Top 10 Fastest Resolving Labels:

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



### ⏰ Top 10 Slowest Resolving Labels:

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



### Top Feature Importances:

1. month: 0.341

2. day_of_week: 0.304

3. num_labels: 0.134

4. has_feature_label: 0.083

5. has_area_label: 0.068



### 🔮 Sample Predictions for Open Issues (showing 5 of 317):

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



### Graphs and prediction time & statistics

Different types of graphs and analysis are done based on the prediction time to complete the open issues.

### Output Files:

• output/label_resolution_analysis.json - Complete analysis results

• output/label_statistics.json - Label-wise statistics

• output/open_issue_predictions.json - Predictions for open issues

• output/visualizations/ - All generated graphs

---

## Feature 3: ML-BASED PRIORITY AND COMPLEXITY PREDICTION

Feature 3 uses machine learning to predict both the **priority** and **complexity** of open issues. Unlike simple time-based predictions, this feature separates business urgency from technical complexity, providing actionable insights for project maintainers.

Run below code to get analysis of feature 3:
```
python run.py --feature 3
```

### ML-BASED PRIORITY AND COMPLEXITY PREDICTION:

**Key Capabilities:**

• **Priority Classification**: Categorizes issues as Critical/High/Medium/Low based on:
  - Labels (bug, critical, security)
  - Community engagement (comments, participants)
  - Maintainer response time
  - Historical resolution patterns

• **Complexity Scoring**: Calculates technical complexity (0-100) based on:
  - Code depth and length
  - Technical indicators (stack traces, code blocks)
  - Multiple component involvement
  - Technical scope (architecture, refactoring, performance)

• **Independent Metrics**: Priority and complexity are calculated separately, allowing identification of:
  - 🔴 High Priority, Low Complexity: Simple urgent bugs
  - 🟡 Low Priority, High Complexity: Technical refactors
  - 🔵 High Priority, High Complexity: Critical architectural issues
  - 🟢 Low Priority, Low Complexity: Minor fixes

### Sample Output Statistics:

**Training Data:**
- Total closed issues analyzed: 5,256
- Valid training samples: 5,256
- Open issues predicted: 317

**Resolution Time Statistics:**
- Median: 13.8 days
- Mean: 210.2 days
- 75th percentile: 261.2 days
- 95th percentile: 1003.3 days

**Priority Distribution:**
- Critical: ~5-8%
- High: ~15-20%
- Medium: ~35-40%
- Low: ~35-45%

### Model Performance:

**Priority Classification:**
- Overall Accuracy: 80%
- Top Feature: Number of comments (10.3% importance)
- Second Feature: Bug label (8.3% importance)
- Third Feature: Number of events (6.5% importance)

### 🔮 Sample Predictions for Open Issues:

**Top 5 Issues by Priority and Complexity:**

1. **[Medium] #9780** - Complexity: 75/100
   - Unable to install PyTorch version 2.5.0 with CUDA 12.4
   - Confidence: 89.0%
   - Current activity: 1 comment

2. **[Medium] #9682** - Complexity: 75/100
   - Cannot install Monorepo deps without sourcecode for Dockerfile caching
   - Confidence: 65.0%
   - Current activity: 1 comment

3. **[Medium] #9634** - Complexity: 75/100
   - Poetry forgetting some dependencies (mix of extras, groups and version markers)
   - Confidence: 76.5%
   - Current activity: 6 comments

4. **[Low] #5138** - Complexity: 75/100
   - Poetry debugging with PyCharm not possible?
   - Confidence: 67.0%
   - *Example of Low Priority but High Complexity*

5. **[Low] #9161** - Complexity: 5/100
   - Add test coverage for tests/helpers.py
   - Confidence: 78.5%
   - *Example of Low Priority and Low Complexity*

### Output Files:

• `output/priority_predictions.json` - Complete priority and complexity predictions for all 317 open issues

**JSON Output Format:**
```json
{
  "predicted_priority": "Medium",
  "priority_confidence": 89.0,
  "complexity_score": 75,
  "number": 9780,
  "title": "Issue title...",
  "url": "https://github.com/...",
  "labels": ["kind/bug", "status/triage"],
  "num_comments": 1
}
```

### Use Cases:

1. **Triage Automation**: Quickly identify which issues need immediate attention
2. **Resource Allocation**: Match developers to issues based on complexity
3. **Sprint Planning**: Balance high-priority items with complexity estimates
4. **Maintainer Insights**: Understand which types of issues are most urgent vs. most complex
5. **Trend Analysis**: Track how priority and complexity correlate over time

---

## Project Structure

```
project-application-template/
├── analysis/
│   ├── contributors_analyzer.py     # Feature 2 analysis logic
│   └── priority_analyzer.py         # Feature 3 analysis logic
├── controllers/
│   ├── contributors_controller.py   # Feature 2 controller
│   ├── priority_controller.py       # Feature 3 controller
│   └── label_resolution_controller.py  # Feature 1 controller
├── visualization/
│   ├── visualizer.py                # Feature 2 visualizations
│   └── label_resolution_visualizer.py  # Feature 1 visualizations
├── app/
│   └── feature_runner.py            # Main feature orchestrator
├── data/
│   └── poetry_issues_all.json       # Issue data
├── output/                          # Generated outputs
├── model.py                         # Data models & ML models
├── data_loader.py                   # Data loading utilities
├── config.py                        # Configuration management
├── run.py                           # Application entry point
└── requirements.txt                 # Python dependencies
```

---

## Team Contributions

- **Feature 1**: Label Resolution Time Analysis - Neel Patel
- **Feature 2**: Contributors Dashboard - [Team Member Name]
- **Feature 3**: Priority & Complexity Prediction - Subiksha Jegadish
