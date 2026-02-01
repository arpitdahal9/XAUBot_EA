"""
ML Trade Filter Training Script (Institutional Grade)

Usage:
    python train_ml_model.py --data backtest_trades.csv
    python train_ml_model.py --data trades.csv --model xgboost --output models/trade_filter.pkl
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml_trainer import MLDatasetBuilder, WalkForwardTrainer, save_model_artifact


def main():
    parser = argparse.ArgumentParser(description='Train ML Trade Filter')
    parser.add_argument('--data', type=str, required=True, help='Path to trades CSV')
    parser.add_argument('--model', type=str, default='xgboost', choices=['logistic', 'xgboost'])
    parser.add_argument('--output', type=str, default='models/trade_filter.pkl')
    parser.add_argument('--train-months', type=int, default=24)
    parser.add_argument('--test-months', type=int, default=6)
    parser.add_argument('--min-auc', type=float, default=0.55)
    parser.add_argument('--max-cal-error', type=float, default=0.10)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    print("=" * 60)
    print("ML TRADE FILTER TRAINING (Institutional Grade)")
    print("=" * 60)

    # Build dataset
    builder = MLDatasetBuilder()
    count = builder.load_from_csv(args.data)

    if count < 50:
        logger.error(f"Insufficient data: {count} trades. Need at least 50.")
        sys.exit(1)

    X, y, timestamps = builder.get_arrays()
    feature_names = builder.get_feature_names()

    print(f"\nDataset: {args.data}")
    print(f"  Trades: {len(y)}")
    print(f"  Win rate: {100*sum(y)/len(y):.1f}%")
    print(f"  Features: {len(feature_names)}")

    # Run walk-forward validation
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    trainer = WalkForwardTrainer(
        train_months=args.train_months,
        test_months=args.test_months,
        min_auc=args.min_auc,
        max_calibration_error=args.max_cal_error
    )

    result, model, scaler = trainer.run(X, y, timestamps, args.model)

    if not result.folds:
        logger.warning("No walk-forward folds created - insufficient data for rolling validation")
        logger.info("Training on full dataset instead...")

    # Print results
    print("\n" + "-" * 60)
    print("WALK-FORWARD RESULTS")
    print("-" * 60)

    for fold in result.folds:
        print(f"  Fold {fold.fold_number}: AUC={fold.auc:.3f}, Brier={fold.brier_score:.3f}, "
              f"CalErr={fold.calibration_error:.3f}, Thresh={fold.optimal_threshold:.2f}")

    print(f"\n  Overall AUC: {result.overall_auc:.3f} +/- {result.auc_stability:.3f}")
    print(f"  Overall Brier: {result.overall_brier:.3f}")
    print(f"  Overall Cal Error: {result.overall_calibration_error:.3f}")
    print(f"  Recommended Threshold: {result.recommended_threshold:.2f}")
    print(f"  Model Valid: {result.model_valid}")

    # Save model
    if model is not None:
        artifact = save_model_artifact(model, scaler, result, feature_names, args.output)
        print(f"\nModel saved to: {args.output}")

    # Quality checks
    print("\n" + "-" * 60)
    print("QUALITY CHECKS")
    print("-" * 60)

    if result.overall_auc >= args.min_auc:
        print(f"  [PASS] AUC >= {args.min_auc}")
    else:
        print(f"  [FAIL] AUC < {args.min_auc} - model may not add value")

    if result.overall_calibration_error <= args.max_cal_error:
        print(f"  [PASS] Calibration error <= {args.max_cal_error}")
    else:
        print(f"  [FAIL] Calibration error > {args.max_cal_error}")

    if result.model_valid:
        print("\n  >>> Model PASSED quality checks - ready for production")
    else:
        print("\n  >>> Model FAILED quality checks - will auto-disable in production")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
