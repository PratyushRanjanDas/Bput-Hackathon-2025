# Solar Panel Optimizer - Backend Documentation

## 1. Overview

This backend is designed to power the Solar Panel Optimizer application. It has two primary functions, supporting two different architectural approaches:

1.  **Offline-First Model Generation:** The core purpose is to train a machine learning model that predicts energy loss in solar panels and convert it into a format (`.onnx`) that can be run directly in the user's browser. This enables the application to work in remote areas with no internet connection.

2.  **Online API Server:** It also includes a fully functional Flask web server that can serve the same recommendations via a REST API. This is useful for testing, development, or for application versions that require a live server connection.

---

## 2. Project Structure

The backend is organized into the following key directories:

-   **/data/**: Stores the generated CSV datasets used for training the models.
-   **/ml_training/**: Contains all scripts and saved models related to the machine learning workflow.
    -   **/scripts/**: Python scripts for data simulation, model training, and ONNX conversion.
    -   **/saved_model/**: The final, trained model files (`.pkl` and `.onnx`).
-   **/services/**: Contains the business logic, such as the `RecommendationService` that interprets model predictions and generates user-friendly advice.
-   **/api/**: Defines the Flask API endpoints (routes) for the online server.
-   **app.py**: The main entry point to start the Flask web server.
-   **requirements.txt**: A list of all Python dependencies required for the project.

---

## 3. Machine Learning Workflow (Offline-First)

The primary goal is to produce `loss_prediction_model.onnx`. This is achieved through a sequence of scripts in the `/ml_training/scripts/` folder.

### Step A: `1b_simulate_loss_data.py`

-   **Purpose:** Generates a realistic dataset (`historical_loss_data.csv`) that simulates solar panel performance over two years.
-   **Key Features:** It introduces factors that cause energy loss, such as:
    -   **Soiling:** Simulates the accumulation of dirt over time and its removal by "cleaning events" (rain).
    -   **Degradation:** Simulates the gradual loss of efficiency as the panel ages.
-   **Output:** `data/historical_loss_data.csv`

### Step B: `2b_train_loss_model.py`

-   **Purpose:** Trains a machine learning model to predict energy loss.
-   **Process:**
    1.  Loads the `historical_loss_data.csv` dataset.
    2.  Uses features like `temperature`, `cloud_cover`, `panel_age_in_days`, and `days_since_cleaning` to predict the target variable: `energy_loss_kw`.
    3.  Uses a `RandomForestRegressor` model, which is effective for this type of problem.
-   **Output:** `ml_training/saved_model/loss_prediction_model.pkl` (a Python-specific model file).

### Step C: `3_convert_model_to_onnx.py`

-   **Purpose:** Converts the Python-specific `.pkl` model into the universal ONNX format.
-   **Why?** The `.onnx` format can be run by many platforms, including directly in a web browser using JavaScript. This is the key to making the application work offline.
-   **Output:** `ml_training/saved_model/loss_prediction_model.onnx`. **This is the final artifact that gets passed to the frontend team.**

---

## 4. Setup and Usage

Follow these steps to set up and run the backend from a fresh start.

### Prerequisites

-   Python 3.10 or higher.
-   `pip` for package installation.

### Setup Instructions

1.  **Navigate to the Backend Directory:**
    ```bash
    cd /path/to/solar-panel-optimizer/backend
    ```

2.  **(Recommended) Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

3.  **Install Dependencies:**
    Install all required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### How to Run the Full ML Pipeline

To generate the final `loss_prediction_model.onnx` file from scratch, run the scripts in order:

```bash
# Navigate to the scripts folder
cd ml_training/scripts

# Run data simulation
python 3_convert_model_to_onnx.py

# Run model training
python 2b_train_loss_model.py

# Run ONNX conversion
python 3_convert_model_to_onnx.py
```

### How to Run the Online API Server

If you want to test the online version of the application:

1.  **Navigate to the Backend Root:**
    ```bash
    cd /path/to/solar-panel-optimizer/backend
    ```

2.  **(Optional) Create a `.env` file** in the `backend` directory to use the live weather feature:
    ```
    # backend/.env
    WEATHER_API_KEY=your_openweathermap_api_key
    ```

3.  **Start the Flask Server:**
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5001`. You can then test the `/api/recommend` endpoint using a tool like Postman.

---

You've helped build a robust and flexible system. I hope this documentation is helpful for you and your team!