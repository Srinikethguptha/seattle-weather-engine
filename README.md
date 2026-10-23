# Seattle Weather Engine - Rainfall Prediction System ⛅ (Project 11)

A Machine Learning web application predicting rainfall occurrence (Classification) and precipitation amounts (Regression) using the official **Seattle Weather Dataset** from the Vega repository.

Both **Classification** and **Regression** models execute simultaneously on the same web interface.

---

## 📌 Dataset & Feature Engineering
- **Source**: [`https://raw.githubusercontent.com/vega/vega-datasets/next/data/seattle-weather.csv`](https://raw.githubusercontent.com/vega/vega-datasets/next/data/seattle-weather.csv)
- **Features (`X`)**:
  - `temp_max`: Maximum daily temperature (°C)
  - `temp_min`: Minimum daily temperature (°C)
  - `wind`: Wind speed (m/s)
  - `year`: Extracted observation year
  - `month`: Extracted observation month (1–12)
  - `day`: Extracted calendar day of month (1–31)
- **Targets (`y`)**:
  - **Regression**: `precipitation` (continuous rainfall volume in mm)
  - **Classification**: Binary target `1` for rain/drizzle (`['rain', 'drizzle']`), `0` otherwise (`['sun', 'snow', 'fog']`)

---

## 🤖 Algorithms Supported

### Classification (Predicting Rain Occurrence & Probability)
- **Random Forest Classifier** (`RandomForestClassifier`)
- **XGBoost Classifier** (`XGBClassifier`)
- **Logistic Regression** (`LogisticRegression`)
- **Decision Tree Classifier** (`DecisionTreeClassifier`)

### Regression (Predicting Rainfall Amount in mm)
- **Random Forest Regressor** (`RandomForestRegressor`)
- **XGBoost Regressor** (`XGBRegressor`)
- **Linear Regression** (`LinearRegression`)
- **Ridge Regressor** (`Ridge`)

---

## 🗂️ Project Directory Layout
```
rainfall_prediction/
├── data/
│   └── seattle-weather.csv     # Official Seattle weather dataset
├── static/
│   └── css/
│       └── style.css           # Modern responsive design & UI styles
├── templates/
│   └── index.html              # Dual-model interactive UI with presets & metrics
├── app.py                      # Flask server (serving Web UI & REST API)
├── train.py                    # Training & evaluation pipeline
├── models_bundle.pkl           # Saved trained models & feature definitions
├── model_metrics.json          # Benchmark evaluation metrics (Accuracy, F1, MAE, R²)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Running the Application

### 1. (Optional) Retrain Models
```bash
python train.py
```
This downloads the official Seattle dataset, trains all classification and regression models, and outputs evaluation benchmarks.

### 2. Start the Flask Web App
```bash
python app.py
```
Visit: **`http://localhost:5000`** in your browser.

---

## ⚡ Highlights
1. **Simultaneous Dual Execution**:
   - Computes both **Rain Status / Category / Probability** and **Rainfall Volume (mm)** together on the same screen.
2. **Algorithm Selectors**:
   - Choose any combination of Classification and Regression algorithms dynamically from dropdowns.
3. **Seattle Presets**:
   - 1-click test scenarios: *Typical Rainy Fall (Nov)*, *Warm Sunny Summer (Jul)*, *Chilly Winter (Jan)*, *Stormy Wind Day*.
4. **Built-in Benchmark Drawer**:
   - Expandable section displaying test-split Accuracy, Precision, Recall, F1-Score, MAE, RMSE, and $R^2$ Score across all models.
"# seattle-weather-engine" 
