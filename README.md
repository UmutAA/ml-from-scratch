# Machine Learning From Scratch

Welcome to my **ML From Scratch** repository! The goal of this project is to implement foundational machine learning and deep learning algorithms completely from scratch using Python, without relying on black-box libraries like Scikit-learn. 

By stripping away the abstractions, this repository focuses on understanding the underlying mathematics, including forward propagation, loss functions, partial derivatives, gradients, and gradient descent optimization.

---

## Project Structure

```text
ml-from-scratch/
│
├── linear_regression/      # Single-variable Linear Regression from scratch
│   └── linear_regression.py
│
├── .gitignore              # Python & project-specific ignore rules
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

---

## Implemented Algorithms

### 1. Linear Regression (with Gradient Descent)
* **Mathematical Model:** $y = w \cdot x + b$
* **Loss Function:** Mean Squared Error (MSE)
* **Optimization:** Gradient Descent (iterative partial derivatives update)

---

## Getting Started & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/UmutAA/ml-from-scratch.git
   cd ml-from-scratch
   ```

2. **Set up a virtual environment**
    ```bash
    python -m venv env #your environment's name
    source env/bin/activate  # "venv\Scripts\activate.bat" on Windows
    ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run a model script:**
   Navigate into the respective algorithm folder and execute the Python file:
   ```bash
   cd linear_regression
   python linear_regression.py
   ```


## Why From Scratch?
* **Mathematical Intuition:** To truly understand how parameters ($\omega$ and $b$) update and minimize loss.
* **Engineering Mindset:** Moving beyond "using libraries" to understanding the mechanics under the hood.
* **Debugging Capability:** Knowing what actually happens when models fail or gradients explode/vanish.

---