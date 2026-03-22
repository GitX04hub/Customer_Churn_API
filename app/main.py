"""
app/main.py - Flask REST API for customer churn prediction.

Run from project root:
    python -m app.main
"""
import logging
from flask import Flask, request, jsonify
from app.utils import load_artifacts, preprocess

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App & global artifacts ──────────────────────────────────────────────────
app = Flask(__name__)
model, artifact = load_artifacts()
logger.info("Model and transformer loaded successfully.")


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body  : {"customer": { <raw customer fields> }}
    Return: {"churn_probability": 0.83, "churn_prediction": "Yes"}
    """
    try:
        body          = request.get_json(force=True)
        customer_dict = body.get("customer", {})

        if not customer_dict:
            return jsonify({"error": "Missing 'customer' key in request body"}), 400

        X_transformed     = preprocess(customer_dict, artifact)
        churn_prob        = float(model.predict_proba(X_transformed)[0][1])
        churn_pred        = "Yes" if churn_prob >= 0.5 else "No"

        logger.info("Prediction made: prob=%.4f  pred=%s", churn_prob, churn_pred)
        return jsonify({"churn_probability": round(churn_prob, 4),
                        "churn_prediction":  churn_pred})

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
