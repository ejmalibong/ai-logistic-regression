# Logistic Regression – Customer Churn

## Overview

Logistic Regression implemented **from scratch with NumPy** to predict customer churn (`Exited = 0/1`).
Learning-focused project, not production.

---

## Setup

```bash
git clone https://github.com/coderanger428/ai-logistic-regression.git
cd ai-logistic-regression
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

Notebook:

```bash
jupyter notebook notebooks/logistic_regression_demo.ipynb
```

Training:

```bash
python src/train.py
```

---

## Structure

```bash
logistic-regression-churn/
├── README.md
├── data/churn.csv
├── src/
│   ├── logistic_regression.py
│   └── train.py
├── notebooks/logistic_regression_demo.ipynb
├── requirements.txt
└── .gitignore
```
