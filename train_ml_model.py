"""
ML Trade Filter Training Script

This script:
1. Loads trade data from backtest CSVs
2. Extracts ML features from TPU signals
3. Trains Logistic Regression (baseline) and XGBoost (production) models
4. Runs walk-forward validation to prevent overfitting
5. Generates calibration curves and reports
6. Saves the trained model for live trading

Usage:
    python train_ml_model.py --data backtest_5year_trades.csv
    python train_ml_model.py --data trades.csv --model xgboost --output models/trade_filter.pkl
"""

import os
import sys
import csv
import json
import pickle
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load settings from config file."""
    try:
        with open("config/settings.json", "r") as f:
            return json.load(f)
    except:
        return {}


def load_trades_from_csv(csv_path: str) -> Tuple[List[Dict], List[int]]:
    """
    Load trade data from backtest CSV.
    
    Returns:
        (features_list, labels)
        Label: 1 = trade reached TP1, 0 = stopped out
    """
    logger.info(f"Loading trades from {csv_path}")
    
    features_list = []
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Only use FINAL rows (one per trade)
            if row.get('row_type') != 'FINAL':
                continue
            
            # Determine label
            exit_reason = row.get('exit_reason', '')
            tp1_closed = float(row.get('tp1_closed_lots', 0) or 0)
            
            # Win if TP1 was hit at any point
            label = 1 if (tp1_closed > 0 or 'TP1' in exit_reason) else 0
            
            try:
                # Parse timestamp
                entry_time_str = row.get('entry_time', '')
                try:
                    timestamp = datetime.fromisoformat(entry_time_str.replace('Z', ''))
                except:
                    timestamp = datetime.now()
                
                # Extract features from row
                features = {
                    'timestamp': timestamp,
                    'pair': row.get('pair', 'EUR/USD'),
                    'direction': row.get('side', 'BUY'),
                    'entry_price': float(row.get('entry_price', 0) or 0),
                    'stop_loss': float(row.get('stop_loss', 0) or 0),
                    'lot_size': float(row.get('lot_size', 0.01) or 0.01),
                    
                    # TPU features (from CSV if available, else defaults)
                    'tpu_confidence': float(row.get('tpu_confidence', 0.75) or 0.75),
                    'tda_alignment': row.get('tda_alignment', 'GOOD'),
                    'fib_distance_pips': float(row.get('fib_distance_pips', 10) or 10),
                    'has_divergence': 1 if row.get('divergence_type') else 0,
                    'has_pattern': 1 if row.get('pattern_type') else 0,
                    
                    # Computed features
                    'hour_of_day': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    
                    # Result
                    'label': label,
                    'pnl': float(row.get('pnl_total_trade', 0) or 0)
                }
                
                features_list.append(features)
                labels.append(label)
                
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping row due to error: {e}")
                continue
    
    logger.info(f"Loaded {len(labels)} trades: {sum(labels)} wins ({100*sum(labels)/len(labels):.1f}%), {len(labels)-sum(labels)} losses")
    return features_list, labels


def features_to_array(features_list: List[Dict]) -> np.ndarray:
    """
    Convert feature dictionaries to numpy array.
    
    This creates a simplified feature set from available CSV data.
    In production, full TPU features would be extracted.
    """
    X = []
    
    for f in features_list:
        row = [
            # TPU confidence (proxy for overall quality)
            f.get('tpu_confidence', 0.75),
            
            # TDA alignment (encoded)
            1.0 if f.get('tda_alignment') == 'PERFECT' else (0.5 if f.get('tda_alignment') == 'GOOD' else 0.0),
            
            # Fibonacci proximity
            f.get('fib_distance_pips', 10) / 20.0,  # Normalized
            
            # Divergence
            f.get('has_divergence', 0),
            
            # Pattern
            f.get('has_pattern', 0),
            
            # Time features
            f.get('hour_of_day', 12) / 24.0,
            f.get('day_of_week', 2) / 7.0,
            
            # Session indicators (simplified)
            1.0 if 8 <= f.get('hour_of_day', 12) < 16 else 0.0,  # London
            1.0 if 13 <= f.get('hour_of_day', 12) < 21 else 0.0,  # NY
            1.0 if 13 <= f.get('hour_of_day', 12) < 16 else 0.0,  # Overlap
            
            # Stop distance (normalized to typical range)
            min(abs(f.get('entry_price', 1.1) - f.get('stop_loss', 1.095)) * 10000 / 50, 2.0),
        ]
        X.append(row)
    
    return np.array(X, dtype=np.float32)


def get_feature_names() -> List[str]:
    """Get ordered list of feature names."""
    return [
        'tpu_confidence',
        'tda_alignment_score',
        'fib_distance_normalized',
        'has_divergence',
        'has_pattern',
        'hour_normalized',
        'day_normalized',
        'is_london',
        'is_ny',
        'is_overlap',
        'stop_distance_normalized'
    ]


def train_logistic_model(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train Logistic Regression baseline model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    
    logger.info("Training Logistic Regression...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(model, cv=5, method='isotonic')
    calibrated.fit(X_scaled, y)
    
    # Cross-validation
    cv_scores = cross_val_score(calibrated, X_scaled, y, cv=5, scoring='roc_auc')
    
    # Feature importances
    feature_names = get_feature_names()
    importances = dict(zip(feature_names, model.coef_[0]))
    
    logger.info(f"Logistic Regression CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    
    return {
        'model': calibrated,
        'model_type': 'logistic',
        'scaler': scaler,
        'feature_names': feature_names,
        'cv_auc': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'feature_importances': importances
    }


def train_xgboost_model(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train XGBoost production model."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed, falling back to Logistic Regression")
        return train_logistic_model(X, y)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    
    logger.info("Training XGBoost...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate class weight
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(X_scaled, y)
    
    # Calibrate
    calibrated = CalibratedClassifierCV(model, cv=5, method='isotonic')
    calibrated.fit(X_scaled, y)
    
    # Cross-validation
    cv_scores = cross_val_score(calibrated, X_scaled, y, cv=5, scoring='roc_auc')
    
    # Feature importances
    feature_names = get_feature_names()
    importances = dict(zip(feature_names, model.feature_importances_))
    
    logger.info(f"XGBoost CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    
    return {
        'model': calibrated,
        'model_type': 'xgboost',
        'scaler': scaler,
        'feature_names': feature_names,
        'cv_auc': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'feature_importances': importances
    }


def run_walk_forward_validation(
    features_list: List[Dict],
    labels: List[int],
    model_type: str = 'xgboost',
    train_months: int = 24,
    test_months: int = 6
) -> Dict:
    """
    Run walk-forward validation.
    
    This is the proper way to validate ML models for trading:
    1. Train on N months
    2. Test on next M months
    3. Roll forward
    4. Repeat
    
    No peeking at future data!
    """
    logger.info(f"Running walk-forward validation (train={train_months}mo, test={test_months}mo)")
    
    # Sort by timestamp
    sorted_data = sorted(zip(features_list, labels), key=lambda x: x[0]['timestamp'])
    features_list = [d[0] for d in sorted_data]
    labels = [d[1] for d in sorted_data]
    
    X = features_to_array(features_list)
    y = np.array(labels)
    
    # Determine fold boundaries
    if len(features_list) == 0:
        return {'error': 'No data'}
    
    start_date = features_list[0]['timestamp']
    end_date = features_list[-1]['timestamp']
    
    results = {
        'folds': [],
        'overall_auc': 0,
        'overall_pf': 0,
        'calibration_errors': [],
        'predictions_all': [],
        'actuals_all': []
    }
    
    current_start = start_date
    fold_num = 0
    
    while current_start + timedelta(days=(train_months + test_months) * 30) <= end_date:
        train_end = current_start + timedelta(days=train_months * 30)
        test_end = train_end + timedelta(days=test_months * 30)
        
        # Split data
        train_idx = [i for i, f in enumerate(features_list) if current_start <= f['timestamp'] < train_end]
        test_idx = [i for i, f in enumerate(features_list) if train_end <= f['timestamp'] < test_end]
        
        if len(train_idx) < 30 or len(test_idx) < 5:
            current_start += timedelta(days=test_months * 30)
            continue
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # Train
        if model_type == 'xgboost':
            result = train_xgboost_model(X_train, y_train)
        else:
            result = train_logistic_model(X_train, y_train)
        
        model = result['model']
        scaler = result['scaler']
        
        # Test
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.58).astype(int)
        
        # Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        # Calibration
        avg_pred = np.mean(y_pred_proba)
        actual_rate = np.mean(y_test)
        cal_error = abs(avg_pred - actual_rate)
        
        # Profit factor (simplified: wins=2R, losses=1R)
        wins = sum((y_pred == 1) & (y_test == 1))
        losses = sum((y_pred == 1) & (y_test == 0))
        pf = (wins * 2) / max(losses, 1)
        
        fold_result = {
            'fold': fold_num,
            'train_start': current_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': train_end.isoformat(),
            'test_end': test_end.isoformat(),
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
            'accuracy': accuracy,
            'auc': auc,
            'profit_factor': pf,
            'avg_predicted': avg_pred,
            'actual_win_rate': actual_rate,
            'calibration_error': cal_error
        }
        
        results['folds'].append(fold_result)
        results['calibration_errors'].append(cal_error)
        results['predictions_all'].extend(y_pred_proba.tolist())
        results['actuals_all'].extend(y_test.tolist())
        
        logger.info(
            f"  Fold {fold_num}: AUC={auc:.3f}, PF={pf:.2f}, "
            f"Cal.Err={cal_error:.3f}, Samples={len(test_idx)}"
        )
        
        fold_num += 1
        current_start += timedelta(days=test_months * 30)
    
    if results['folds']:
        results['overall_auc'] = np.mean([f['auc'] for f in results['folds']])
        results['overall_pf'] = np.mean([f['profit_factor'] for f in results['folds']])
        results['auc_stability'] = np.std([f['auc'] for f in results['folds']])
        results['avg_calibration_error'] = np.mean(results['calibration_errors'])
        
        logger.info(f"\n=== Walk-Forward Summary ===")
        logger.info(f"  Folds: {len(results['folds'])}")
        logger.info(f"  Overall AUC: {results['overall_auc']:.3f} +/- {results['auc_stability']:.3f}")
        logger.info(f"  Overall PF: {results['overall_pf']:.2f}")
        logger.info(f"  Avg Calibration Error: {results['avg_calibration_error']:.3f}")
    
    return results


def save_model(result: Dict, output_path: str, version: str = "1.0.0"):
    """Save trained model to file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    data = {
        'model': result['model'],
        'model_type': result['model_type'],
        'version': version,
        'feature_names': result['feature_names'],
        'scaler': result['scaler'],
        'cv_auc': result['cv_auc'],
        'saved_at': datetime.now().isoformat()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Model saved to {output_path}")


def generate_report(wf_results: Dict, output_path: str):
    """Generate walk-forward validation report."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ML TRADE FILTER - WALK-FORWARD VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Folds: {len(wf_results.get('folds', []))}\n")
        f.write(f"Overall AUC: {wf_results.get('overall_auc', 0):.3f} +/- {wf_results.get('auc_stability', 0):.3f}\n")
        f.write(f"Overall Profit Factor: {wf_results.get('overall_pf', 0):.2f}\n")
        f.write(f"Avg Calibration Error: {wf_results.get('avg_calibration_error', 0):.3f}\n\n")
        
        f.write("FOLD-BY-FOLD RESULTS\n")
        f.write("-" * 40 + "\n")
        
        for fold in wf_results.get('folds', []):
            f.write(f"\nFold {fold['fold']}:\n")
            f.write(f"  Period: {fold['train_start'][:10]} to {fold['test_end'][:10]}\n")
            f.write(f"  Train: {fold['train_samples']} | Test: {fold['test_samples']}\n")
            f.write(f"  AUC: {fold['auc']:.3f}\n")
            f.write(f"  Profit Factor: {fold['profit_factor']:.2f}\n")
            f.write(f"  Calibration Error: {fold['calibration_error']:.3f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ML Trade Filter')
    parser.add_argument('--data', type=str, required=True, help='Path to trades CSV')
    parser.add_argument('--model', type=str, default='xgboost', choices=['logistic', 'xgboost'])
    parser.add_argument('--output', type=str, default='models/trade_filter.pkl')
    parser.add_argument('--report', type=str, default='reports/wf_validation_report.txt')
    parser.add_argument('--train-months', type=int, default=24)
    parser.add_argument('--test-months', type=int, default=6)
    
    args = parser.parse_args()
    
    # Check data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    # Load trades
    features_list, labels = load_trades_from_csv(args.data)
    
    if len(labels) < 50:
        logger.error(f"Insufficient data: {len(labels)} trades. Need at least 50.")
        sys.exit(1)
    
    # Convert to arrays
    X = features_to_array(features_list)
    y = np.array(labels)
    
    print("\n" + "=" * 60)
    print("ML TRADE FILTER TRAINING")
    print("=" * 60)
    print(f"\nData: {args.data}")
    print(f"Trades: {len(labels)}")
    print(f"Win Rate: {100*sum(labels)/len(labels):.1f}%")
    print(f"Model Type: {args.model}")
    print()
    
    # Train final model on all data
    print("Training final model on all data...")
    if args.model == 'xgboost':
        result = train_xgboost_model(X, y)
    else:
        result = train_logistic_model(X, y)
    
    # Show feature importances
    print("\nFeature Importances:")
    print("-" * 40)
    sorted_imp = sorted(result['feature_importances'].items(), key=lambda x: abs(x[1]), reverse=True)
    for name, imp in sorted_imp:
        print(f"  {name:30s}: {imp:+.4f}")
    
    # Run walk-forward validation
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    wf_results = run_walk_forward_validation(
        features_list, labels,
        model_type=args.model,
        train_months=args.train_months,
        test_months=args.test_months
    )
    
    # Save model
    save_model(result, args.output)
    
    # Generate report
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    generate_report(wf_results, args.report)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {args.output}")
    print(f"Report saved to: {args.report}")
    
    # Final recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)
    
    if wf_results.get('overall_auc', 0) >= 0.55:
        print("[OK] AUC >= 0.55 - Model has predictive power")
    else:
        print("[!] AUC < 0.55 - Model may not add value, consider more data")
    
    if wf_results.get('avg_calibration_error', 1) <= 0.10:
        print("[OK] Calibration error <= 10% - Probabilities are reliable")
    else:
        print("[!] Calibration error > 10% - Probabilities may be unreliable")
    
    if wf_results.get('auc_stability', 1) <= 0.10:
        print("[OK] AUC stability good - Model is consistent across time")
    else:
        print("[!] AUC varies by fold - Model may be unstable")
    
    print()


if __name__ == "__main__":
    main()
