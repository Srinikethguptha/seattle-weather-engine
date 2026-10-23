import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, 'models_bundle.pkl')

# Algorithm label display mappings
CLF_NAMES = {
    'random_forest': 'Random Forest Classifier',
    'xgboost': 'XGBoost Classifier',
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree Classifier'
}

REG_NAMES = {
    'random_forest': 'Random Forest Regressor',
    'xgboost': 'XGBoost Regressor',
    'linear_regression': 'Linear Regression',
    'ridge': 'Ridge Regressor'
}

# Auto-train if model bundle does not exist
if not os.path.exists(MODEL_BUNDLE_PATH):
    print("Model bundle not found. Training models on Seattle dataset...")
    from train import train_and_evaluate
    train_and_evaluate()

bundle = joblib.load(MODEL_BUNDLE_PATH)
clf_models = bundle['clf_models']
reg_models = bundle['reg_models']
clf_metrics = bundle.get('clf_metrics', {})
reg_metrics = bundle.get('reg_metrics', {})
feature_cols = bundle.get('feature_cols', ['temp_max', 'temp_min', 'wind', 'year', 'month', 'day'])

def get_rain_category(volume_mm, is_rain):
    if not is_rain or volume_mm < 0.1:
        return 'No Rain (Dry / Clear)'
    elif volume_mm <= 2.5:
        return 'Light Rain / Drizzle'
    elif volume_mm <= 10.0:
        return 'Moderate Rain'
    else:
        return 'Heavy Rain / Storm'

@app.route('/')
def home():
    default_values = {
        'temp_max': 14.5,
        'temp_min': 6.2,
        'wind': 4.1,
        'year': 2025,
        'month': 11,
        'day': 18,
        'selected_clf': 'random_forest',
        'selected_reg': 'random_forest'
    }
    return render_template(
        'index.html',
        has_prediction=False,
        form_data=default_values,
        clf_names=CLF_NAMES,
        reg_names=REG_NAMES,
        clf_metrics=clf_metrics,
        reg_metrics=reg_metrics
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Parse weather observation inputs
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        selected_clf = request.form.get('clf_algo', 'random_forest')
        selected_reg = request.form.get('reg_algo', 'random_forest')

        if selected_clf not in clf_models:
            selected_clf = 'random_forest'
        if selected_reg not in reg_models:
            selected_reg = 'random_forest'

        form_data = {
            'temp_max': temp_max,
            'temp_min': temp_min,
            'wind': wind,
            'year': year,
            'month': month,
            'day': day,
            'selected_clf': selected_clf,
            'selected_reg': selected_reg
        }

        # 2. Build feature DataFrame
        input_data = pd.DataFrame([[temp_max, temp_min, wind, year, month, day]],
                                  columns=feature_cols)

        # 3. Classification Prediction (Rain status & probability)
        chosen_clf_model = clf_models[selected_clf]
        clf_pred = int(chosen_clf_model.predict(input_data)[0])
        rain_status = 'Yes' if clf_pred == 1 else 'No'

        rain_prob = None
        if hasattr(chosen_clf_model, 'predict_proba'):
            probs = chosen_clf_model.predict_proba(input_data)[0]
            rain_prob = round(float(probs[1]) * 100, 1)
        elif hasattr(chosen_clf_model, 'decision_function'):
            df_val = float(chosen_clf_model.decision_function(input_data)[0])
            rain_prob = round(100.0 / (1.0 + np.exp(-df_val)), 1)
        else:
            rain_prob = 85.0 if clf_pred == 1 else 15.0

        # 4. Regression Prediction (Rainfall volume in mm)
        chosen_reg_model = reg_models[selected_reg]
        predicted_volume = float(chosen_reg_model.predict(input_data)[0])
        rain_amount = max(0.0, predicted_volume)

        # Consistency smoothing if classification is No and residual is tiny
        if rain_status == 'No' and rain_amount < 1.0:
            rain_amount = 0.0

        rain_category = get_rain_category(rain_amount, is_rain=(rain_status == 'Yes'))

        return render_template(
            'index.html',
            has_prediction=True,
            form_data=form_data,
            clf_names=CLF_NAMES,
            reg_names=REG_NAMES,
            selected_clf=selected_clf,
            selected_reg=selected_reg,
            selected_clf_name=CLF_NAMES[selected_clf],
            selected_reg_name=REG_NAMES[selected_reg],
            rain_status=rain_status,
            rain_prob=rain_prob,
            rain_amount=rain_amount,
            rain_category=rain_category,
            clf_metrics=clf_metrics,
            reg_metrics=reg_metrics
        )

    except Exception as e:
        return render_template(
            'index.html',
            has_prediction=False,
            error_message=f"Error executing predictions: {str(e)}",
            form_data=request.form.to_dict(),
            clf_names=CLF_NAMES,
            reg_names=REG_NAMES,
            clf_metrics=clf_metrics,
            reg_metrics=reg_metrics
        )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Programmatic JSON endpoint for dual prediction."""
    try:
        data = request.get_json(force=True)
        temp_max = float(data.get('temp_max', 14.5))
        temp_min = float(data.get('temp_min', 6.2))
        wind = float(data.get('wind', 4.1))
        year = int(data.get('year', 2025))
        month = int(data.get('month', 11))
        day = int(data.get('day', 18))

        selected_clf = data.get('clf_algo', 'random_forest')
        selected_reg = data.get('reg_algo', 'random_forest')

        input_data = pd.DataFrame([[temp_max, temp_min, wind, year, month, day]],
                                  columns=feature_cols)

        chosen_clf = clf_models.get(selected_clf, clf_models['random_forest'])
        chosen_reg = reg_models.get(selected_reg, reg_models['random_forest'])

        clf_pred = int(chosen_clf.predict(input_data)[0])
        pred_volume = float(chosen_reg.predict(input_data)[0])
        rain_amount = max(0.0, pred_volume)
        if clf_pred == 0 and rain_amount < 1.0:
            rain_amount = 0.0

        return jsonify({
            'status': 'success',
            'classification': {
                'algorithm': CLF_NAMES.get(selected_clf, selected_clf),
                'will_rain': bool(clf_pred == 1),
                'status': 'Yes' if clf_pred == 1 else 'No'
            },
            'regression': {
                'algorithm': REG_NAMES.get(selected_reg, selected_reg),
                'expected_rainfall_mm': round(rain_amount, 2),
                'category': get_rain_category(rain_amount, clf_pred == 1)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Seattle Weather & Rainfall Prediction System running on http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
