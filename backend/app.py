from flask import Flask
from flask_cors import CORS
from api.routes import api_blueprint

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS to allow requests from the frontend
    CORS(app)

    # Register the blueprint that contains our API routes
    app.register_blueprint(api_blueprint, url_prefix='/api')

    @app.route('/')
    def index():
        return "Solar Panel Optimizer Backend is running!"

    return app

if __name__ == '__main__':
    app = create_app()
    # Running on port 5001 to avoid conflicts with frontend dev server
    app.run(debug=True, port=5001)