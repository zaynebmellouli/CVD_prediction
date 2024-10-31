# Machine Learning Project 1 - CVD Prediction

## Authors : 
- Anas Himmi (anas.himmi@epfl.ch)
- Zeineb Mellouli (zeineb.mellouli@epfl.ch)
- Antoine Cornaz (antoine.cornaz@epfl.ch)

## Description : 
The objective of this project is to develop a machine learning model capable of predicting the presence of cardiovascular disease (CVD) in patients based on their medical information. We explore and compare different algorithms to determine the best approach for this classification task.

## Dependencies :

Before running the solution, make sure to have the following Python library installed:

- `numpy`

You can install them using the following command:

```sh
pip install numpy
```
For visualization purposes in the preprocessing and feature selection phases, we also use `matplotlib`, `seaborn` but they are not needed to run the models and the final training.

## How to Run the Project (Minimal) :
Run the command: 
```sh
python run.py
```

The train and test csv files should be located in a folder named "data" one directory above the location of the run.py file. The predictions will be saved in a file named 'final_submission.csv'

## Code Organization :
- `implementations.py`contains the mandatory functions for the project

- **`run.ipynb` contains all the code and steps we followed to implement the project. It is the most complete and we invite the reader to follow the steps we took to implement the project in this file.**

- `run.py`contains the minimal code to generate the submission (might take 3 minutes of loading the data and 3 minutes of training)

- `helpers.py` contains various helper functions, including data loading, standardization, and utility functions for handling CSV submissions and batch iterations.
