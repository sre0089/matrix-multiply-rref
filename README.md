# matrix-multiply-rref
A local Python GUI for matrix multiplication and Gaussian row reduction (RREF), with exact fraction support and optional NumPy acceleration.

# Matrix Multiply & RREF Tool (Python GUI)

A lightweight, local-first **Python GUI application** for **matrix multiplication** and **Gaussian row reduction (RREF)**.

Designed for speed, clarity, and ease of use — paste matrices as text, compute instantly, and copy results with one click.

---

## ✨ Features

### Matrix Multiplication
- Multiply matrices **A × B**
- Live **shape preview**
- Fast pure-Python backend
- Optional **NumPy acceleration** (auto-detected)

### Row Reduction (RREF)
- Full **Gauss–Jordan elimination**
- Two modes:
  - **Float mode** (with rounding + epsilon control)
  - **Exact Fraction mode** (perfect for math correctness)
- Displays **pivot columns**
- Works with augmented matrices

### Usability
- Paste-friendly input (`rows = newlines`, `values = spaces/commas`)
- Copy output to clipboard
- Helpful error messages (row mismatch, invalid values, etc.)
- Keyboard shortcut: **Ctrl / ⌘ + Enter** to compute

---

## 📸 Input Format

Paste matrices directly into the text boxes:

1 2 3

4, 5, 6


- Rows → new lines  
- Values → spaces and/or commas  

Augmented matrices work the same way.

---

## 🚀 Getting Started

### 1) Clone the repository
```bash
git clone https://github.com/sre0089/matrix-multiply-rref.git
cd matrix-multiply-rref

```
### 2) (Optional) Create a virtual environment

macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```bash

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install numpy
```

### 4) Run the application

```bash
python app.py
```






