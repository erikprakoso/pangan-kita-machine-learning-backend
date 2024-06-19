from flask import Flask
from config import Config
from routes import ml_bp
from flask_cors import CORS

app = Flask(__name__)
app.config.from_object(Config)

CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

app.register_blueprint(ml_bp, url_prefix='/api/ml')

if __name__ == '__main__':
    app.run(debug=True)
