import joblib
import pandas as pd
import os


class RecommendationService:
    _loss_model = None
    # Let's define a cost for cleaning and a value for energy
    CLEANING_COST = 2000  # Example cost in currency (e.g., INR)
    ENERGY_VALUE_PER_KWH = 7 # Example value of 1 kWh of energy

    @classmethod
    def _get_model(cls):
        """Loads the loss prediction model."""
        if cls._loss_model is None:
            # UPDATED PATH: This path will work once the file is moved to the 'services' folder.
            script_dir = os.path.dirname(__file__)
            model_path = os.path.join(script_dir, '..', 'ml_training', 'saved_model', 'loss_prediction_model.pkl')
            try:
                print(f"Attempting to load model from: {os.path.abspath(model_path)}")
                cls._loss_model = joblib.load(model_path)
                print("Model loaded successfully.")
            except FileNotFoundError:
                print("Error: Model file not found at the specified path.")
                return None
        return cls._loss_model

    @classmethod
    def generate_recommendations(cls, current_conditions):
        """
        Predicts energy loss and generates a maintenance recommendation.

        Args:
            current_conditions (dict): A dictionary with current panel status.

            Example: {
                                        'temperature_celsius': 22,
                                        'cloud_cover_percentage': 20,
                                        'panel_age_in_days': 365,
                                        'days_since_cleaning': 45,
                                        'hour': 12,
                                        'day_of_year': 150
                                    }
        
        Returns:
            dict: A dictionary containing the prediction and a recommendation.
        """
        model = cls._get_model()
        if model is None:
            return {"error": "Loss prediction model not found."}

        try:
            # Prepare data for prediction
            df = pd.DataFrame([current_conditions])
            features_order = [
                'temperature_celsius', 'cloud_cover_percentage', 'panel_age_in_days',
                'days_since_cleaning', 'hour', 'day_of_year'
            ]
            df = df[features_order]

            # Predict the current hourly loss
            predicted_hourly_loss_kw = model.predict(df)[0]

            # --- Recommendation Logic ---
            # Estimate the financial loss over a full day (e.g., 8 peak sun hours)
            estimated_daily_loss_kwh = predicted_hourly_loss_kw * 8
            daily_financial_loss = estimated_daily_loss_kwh * cls.ENERGY_VALUE_PER_KWH

            recommendation = "No immediate action required. System performing within expected parameters."
            action_required = False

            # CORRECTED RULE: Lower the threshold to make it more sensitive.
            # Let's trigger a recommendation if the daily loss is greater than ₹20
            if daily_financial_loss > 20:
                action_required = True
                recommendation = (
                    f"High energy loss detected due to soiling. "
                    f"Estimated daily financial loss: ₹{daily_financial_loss:.2f}. "
                    f"Recommend scheduling panel cleaning. The cost of cleaning (₹{cls.CLEANING_COST}) "
                    f"could be recovered in approximately {cls.CLEANING_COST / daily_financial_loss:.1f} days."
                )

            return {
                "predicted_hourly_loss_kw": round(predicted_hourly_loss_kw, 4),
                "estimated_daily_financial_loss": round(daily_financial_loss, 2),
                "action_required": action_required,
                "recommendation_message": recommendation
            }

        except Exception as e:
            return {"error": f"An error occurred during recommendation generation: {e}"}


# Example of how to use the service
if __name__ == '__main__':
    # Scenario 1: Panels are clean
    clean_panel_conditions = {
        'temperature_celsius': 25, 'cloud_cover_percentage': 10,
        'panel_age_in_days': 180, 'days_since_cleaning': 5,
        'hour': 13, 'day_of_year': 185
    }
    print("--- Scenario 1: Clean Panels ---")
    recommendation1 = RecommendationService.generate_recommendations(
        clean_panel_conditions)
    print(recommendation1)

    # Scenario 2: Panels are dirty
    dirty_panel_conditions = {
        'temperature_celsius': 25, 'cloud_cover_percentage': 10,
        'panel_age_in_days': 180, 'days_since_cleaning': 60,  # 60 days since last clean
        'hour': 13, 'day_of_year': 185
    }
    print("\n--- Scenario 2: Dirty Panels ---")
    recommendation2 = RecommendationService.generate_recommendations(
        dirty_panel_conditions)
    print(recommendation2)