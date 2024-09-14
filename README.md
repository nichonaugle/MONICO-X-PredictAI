# MONICO-X-PredictAI

## Table of Contents
1. [Python Enviornment Setup and API](#python-enviornment-setup)
2. [Machine Learning Model](#machine-learning-model)
3. [Data Management and Sorting for ML Training](#data-management-and-sorting-for-ml-training)

---

## Python Enviornment Setup

### 1. Cloning the Repository

To clone the project repository, run the following commands in your terminal:

```bash
git clone https://github.com/nichonaugle/MONICO-X-PredictAI
cd MONICO-X-PREDICTAI
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Setting Up the Virtual Environment
After navigating to the predictai folder, you need to set up a Python virtual environment.

Ensure your Python version is 3.11:
```bash
python --version
```

Install virtualenv if it's not already installed:
```bash
pip install virtualenv
```

Then create and activate the virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Installing Requirements
Once the virtual environment is activated, install the required Python libraries:
```bash
pip install -r requirements.txt
```

NOTE: Before running "pip install (your-library)", please ensure you are inside of the virtual enviornment