from flask import Flask
from config import Config
from routes import ml_bp

app = Flask(__name__)
app.config.from_object(Config)

app.register_blueprint(ml_bp, url_prefix='/api/ml')

if __name__ == '__main__':
    app.run(debug=True)
