# utils/response.py

from flask import jsonify

def standard_response(code, message, data=None, status_code=200):
    """
    Returns a standard JSON response with status code
    """
    response = {
        "code": code,
        "message": message
    }
    if data is not None:
        response["data"] = data
    return jsonify(response), status_code
