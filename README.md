#Fake_News_Detection
## Overview
The goal of this project is to classify news articles as fake or real using a machine learning model. The Passive Aggressive Classifier is trained on a dataset of news articles with corresponding labels.
## Dataset
The dataset used in this project is split into two files:
- `Train.csv`: Contains the training data
- `Test.csv`: Contains the test data

Both files are expected to have the following columns:
- `text`: The text of the news article
- `label`: The label indicating whether the news is `FAKE` or `REAL`
## Libraries
The following Python libraries are required for this project:
- `numpy`: For numerical operations
- `pandas`: For data manipulation and analysis
- `scikit-learn`: For machine learning algorithms and tools
- `matplotlib`: For plotting and visualization
You can install the required libraries using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
