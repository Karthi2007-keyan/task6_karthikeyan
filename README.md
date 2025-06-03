# task6_karthikeyan
task 6

# Iris Classification without scikit-learn

This project implements a simple **Iris flower classification** system **without using external libraries** like `scikit-learn`, `numpy`, or `pandas`. The goal is to manually load and process data, split datasets, normalize features, apply a basic KNN (K-Nearest Neighbors) algorithm, and generate a confusion matrix.

---

## 📦 Packages Used

This script uses only **built-in Python modules**, so no need to install third-party packages:

- `csv` – to read data from `iris.csv`
- `random` – to shuffle and split the dataset
- `math` – to calculate Euclidean distance
- `collections` – to help with majority vote and counting
- `os` – optional, for file checking

---

## 📁 Dataset

The dataset used is `iris.csv` and should be placed in the same directory as `task6.py`. It must follow this format:

```csv
5.1,3.5,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
...

                                                                        ## OUTPUT ######## OUTPUT

PS D:\karthikeyan_elivatelab>  & 'c:\Users\shiva\AppData\Local\Programs\Python\Python313\python.exe' 'c:\Users\shiva\.vscode\extensions\ms-python.debugpy-2025.8.0-win32-x64\bundled\libs\debugpy\launcher' '54183' '--' 'D:\karthikeyan_elivatelab\task6.py' 

Accuracy: 33.33%

Confusion Matrix:
               Iris-versicolorIris-virginica 
Iris-versicolor0              0
Iris-virginica 2              1
PS D:\karthikeyan_elivatelab> 




