"""
batch.py - Batch scoring script for the churn prediction API.

Usage:
    python batch.py --input test_data/all_customers.csv
"""
import argparse
import csv
import json
import logging
import os
import time

import pandas as pd
import requests

# ── Logging setup ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/batch_log.txt"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PREDICT_URL = "http://localhost:8000/predict"


def score_customers(input_path: str, output_path: str = "scored_customers.csv"):
    df = pd.read_csv(input_path)

    # Drop columns not needed by the model (same as training)
    drop_cols = [c for c in ["customerID", "Unnamed: 0", "Churn"] if c in df.columns]
    customer_df = df.drop(columns=drop_cols)

    results      = []
    total        = len(customer_df)
    failures     = 0
    probabilities = []

    logger.info("Starting batch scoring — %d records from '%s'", total, input_path)

    for idx, row in customer_df.iterrows():
        customer_dict = row.to_dict()
        payload       = {"customer": customer_dict}

        try:
            resp = requests.post(PREDICT_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                probabilities.append(result["churn_probability"])
                results.append({
                    **customer_dict,
                    "churn_probability": result["churn_probability"],
                    "churn_prediction":  result["churn_prediction"],
                })
            else:
                logger.warning("Row %d — non-200 status: %d  body: %s",
                               idx, resp.status_code, resp.text[:200])
                failures += 1
                results.append({**customer_dict,
                                 "churn_probability": None,
                                 "churn_prediction":  "ERROR"})
        except Exception as exc:
            logger.error("Row %d — exception: %s", idx, exc)
            failures += 1
            results.append({**customer_dict,
                             "churn_probability": None,
                             "churn_prediction":  "ERROR"})

    # ── Write output CSV ────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(output_path, index=False)

    avg_prob = sum(probabilities) / len(probabilities) if probabilities else 0.0

    # ── Summary log ────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Batch scoring complete.")
    logger.info("Total requests   : %d", total)
    logger.info("Successful       : %d", total - failures)
    logger.info("Failed           : %d", failures)
    logger.info("Average churn probability: %.4f", avg_prob)
    logger.info("Results written to: %s", output_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn scorer")
    parser.add_argument("--input",  required=True, help="Path to input CSV")
    parser.add_argument("--output", default="scored_customers.csv",
                        help="Path to output CSV (default: scored_customers.csv)")
    args = parser.parse_args()

    score_customers(args.input, args.output)
