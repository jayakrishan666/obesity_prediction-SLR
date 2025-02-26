from django.shortcuts import render
from .train import model, scaler, model_accuracy

def predict_obesity(request):
    """Handles obesity prediction and categorizes risk as 'Low' or 'High'."""
    if request.method == 'GET':
        bmi_value = request.GET.get('bmi', None)

        if not bmi_value:
            return render(request, 'prediction/predict.html', {'error': 'BMI value is required'})

        try:
            bmi_value = float(bmi_value)
            bmi_scaled = scaler.transform([[bmi_value]])  # Apply same scaling as training
            
            # Predict obesity probability
            predicted_probability = round(model.predict(bmi_scaled)[0], 2) if model else "N/A"

            # Categorize risk
            risk_level = "High" if predicted_probability > 0.5 else "Low"
        
        except ValueError:
            return render(request, 'prediction/predict.html', {'error': 'Invalid BMI value'})

        return render(request, 'prediction/predict.html', {
            'bmi': bmi_value,
            'risk_level': risk_level,
            'model_accuracy': model_accuracy  # Include accuracy score
        })
