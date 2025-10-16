from flask import Blueprint, request, jsonify
from datetime import datetime
# CORRECTED IMPORT: Import from the 'services' directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.recommendation_service import RecommendationService

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/recommend', methods=['POST'])
# ... (the rest of the file is the same as before) ...
def get_recommendation():
    """
    API endpoint to get a maintenance recommendation.
    Expects a JSON payload with current conditions.
    """
    # Get data from the frontend request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input. JSON payload required."}), 400

    # Prepare the data for the recommendation service
    # In a real app, you'd get this from sensors or user input
    # For now, we'll use the data sent from the frontend
    try:
        now = datetime.now()
        current_conditions = {
            'temperature_celsius': float(data.get('temperature_celsius', 25)),
            'cloud_cover_percentage': float(data.get('cloud_cover_percentage', 20)),
            'panel_age_in_days': int(data.get('panel_age_in_days', 365)),
            'days_since_cleaning': int(data.get('days_since_cleaning', 10)),
            'hour': now.hour,
            'day_of_year': now.timetuple().tm_yday
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data type in input: {e}"}), 400

    # Generate the recommendation
    recommendation = RecommendationService.generate_recommendations(current_conditions)

    # The service returns numpy types, which need to be converted for JSON
    if 'predicted_hourly_loss_kw' in recommendation:
        recommendation['predicted_hourly_loss_kw'] = float(recommendation['predicted_hourly_loss_kw'])
    if 'estimated_daily_financial_loss' in recommendation:
        recommendation['estimated_daily_financial_loss'] = float(recommendation['estimated_daily_financial_loss'])

    return jsonify(recommendation)