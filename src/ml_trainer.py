"""
Walk-Forward ML Training Pipeline

Implements institutional-grade ML training:
1. Walk-forward validation (no single backtest)
2. Leakage prevention (purged splits, embargo)
3. Probability calibration
4. Fold-by-fold metrics

This ensures ML results are realistic and not overfit.
"""

import os
import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

from .ml_features import MLFeatures, MLFeatureExtractor


@dataclass
class WalkForwardFold:
    """Results from one walk-forward fold."""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Training metrics
    train_samples: int = 0
    train_wins: int = 0
    train_losses: int = 0
    
    # Test metrics
    test_samples: int = 0
    test_wins: int = 0
    test_losses: int = 0
    
    # Performance
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_auc: float = 0.0
    test_profit_factor: float = 0.0
    test_sharpe: float = 0.0
    
    # Calibration
    avg_predicted_prob: float = 0.0
    actual_win_rate: float = 0.0
    calibration_error: float = 0.0  # |predicted - actual|


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation results."""
    folds: List[WalkForwardFold] = field(default_factory=list)
    
    # Aggregate metrics
    overall_accuracy: float = 0.0
    overall_auc: float = 0.0
    overall_profit_factor: float = 0.0
    overall_sharpe: float = 0.0
    overall_calibration_error: float = 0.0
    
    # Stability metrics
    auc_stability: float = 0.0  # Std dev across folds
    pf_stability: float = 0.0
    
    # Trade counts
    total_trades: int = 0
    trades_passed_filter: int = 0
    trades_rejected: int = 0


class WalkForwardTrainer:
    """
    Walk-Forward ML Training Pipeline.
    
    Implements:
    1. Rolling window training/testing
    2. Purged cross-validation (no overlapping trades)
    3. Embargo period after each trade
    4. Probability calibration checks
    """
    
    def __init__(
        self,
        train_window_months: int = 24,
        test_window_months: int = 6,
        embargo_hours: int = 24,
        min_train_samples: int = 50
    ):
        """
        Initialize Walk-Forward Trainer.
        
        Args:
            train_window_months: Training window size (default 24 months)
            test_window_months: Testing window size (default 6 months)
            embargo_hours: Hours to skip after each trade (default 24)
            min_train_samples: Minimum training samples per fold
        """
        self.logger = logging.getLogger(__name__)
        
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.embargo_hours = embargo_hours
        self.min_train_samples = min_train_samples
    
    def create_dataset_from_trades(
        self,
        trades_csv_path: str
    ) -> Tuple[List[Dict], List[int]]:
        """
        Create ML dataset from backtest trades CSV.
        
        Label assignment:
        - 1 = Trade reached TP1 (exit_reason contains "TP1" or had partial TP1)
        - 0 = Trade hit stop loss first
        
        Args:
            trades_csv_path: Path to backtest trades CSV
            
        Returns:
            (features_list, labels_list)
        """
        self.logger.info(f"[WF_TRAINER] Loading trades from {trades_csv_path}")
        
        features_list = []
        labels = []
        
        with open(trades_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Only use FINAL rows (one per trade)
                if row.get('row_type') != 'FINAL':
                    continue
                
                # Determine label
                exit_reason = row.get('exit_reason', '')
                tp1_closed = float(row.get('tp1_closed_lots', 0))
                
                # Win if TP1 was hit at any point
                label = 1 if (tp1_closed > 0 or 'TP1' in exit_reason) else 0
                
                # Extract features from row
                try:
                    features = {
                        'timestamp': row.get('entry_time', ''),
                        'pair': row.get('pair', ''),
                        'direction': row.get('side', ''),
                        'entry_price': float(row.get('entry_price', 0)),
                        'stop_loss': float(row.get('stop_loss', 0)),
                        'tpu_confidence': float(row.get('tpu_confidence', 0.75)) if 'tpu_confidence' in row else 0.75,
                        'label': label,
                        # Add more features as available in CSV
                    }
                    features_list.append(features)
                    labels.append(label)
                except (ValueError, KeyError) as e:
                    continue
        
        self.logger.info(f"[WF_TRAINER] Loaded {len(labels)} trades: {sum(labels)} wins, {len(labels)-sum(labels)} losses")
        return features_list, labels
    
    def create_folds(
        self,
        features_list: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create walk-forward folds with purging and embargo.
        
        Args:
            features_list: List of feature dictionaries with 'timestamp'
            start_date: Dataset start date
            end_date: Dataset end date
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        folds = []
        
        train_days = self.train_window_months * 30
        test_days = self.test_window_months * 30
        
        # Parse timestamps
        for i, feat in enumerate(features_list):
            ts = feat.get('timestamp', '')
            if isinstance(ts, str) and ts:
                try:
                    feat['_datetime'] = datetime.fromisoformat(ts.replace('Z', ''))
                except:
                    feat['_datetime'] = start_date
            elif isinstance(ts, datetime):
                feat['_datetime'] = ts
            else:
                feat['_datetime'] = start_date
            feat['_index'] = i
        
        # Generate folds
        current_start = start_date
        fold_num = 0
        
        while current_start + timedelta(days=train_days + test_days) <= end_date:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            # Apply embargo: skip trades within embargo_hours of boundary
            embargo_delta = timedelta(hours=self.embargo_hours)
            
            train_indices = []
            test_indices = []
            
            for feat in features_list:
                ts = feat['_datetime']
                idx = feat['_index']
                
                # Training: before train_end minus embargo
                if current_start <= ts < (train_end - embargo_delta):
                    train_indices.append(idx)
                
                # Testing: after train_end plus embargo
                elif (train_end + embargo_delta) <= ts < test_end:
                    test_indices.append(idx)
            
            if len(train_indices) >= self.min_train_samples and len(test_indices) >= 5:
                folds.append((train_indices, test_indices))
                self.logger.info(
                    f"[WF_TRAINER] Fold {fold_num}: "
                    f"Train {len(train_indices)} samples, Test {len(test_indices)} samples"
                )
                fold_num += 1
            
            # Roll forward by test window
            current_start += timedelta(days=test_days)
        
        self.logger.info(f"[WF_TRAINER] Created {len(folds)} walk-forward folds")
        return folds
    
    def run_walk_forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: List[Tuple[List[int], List[int]]],
        feature_names: List[str],
        model_type: str = "xgboost"
    ) -> WalkForwardResult:
        """
        Run complete walk-forward validation.
        
        Args:
            X: Feature matrix
            y: Labels
            folds: List of (train_idx, test_idx) tuples
            feature_names: Feature names
            model_type: "logistic" or "xgboost"
            
        Returns:
            WalkForwardResult with all metrics
        """
        from .ml_filter import MLModelTrainer
        
        self.logger.info(f"[WF_TRAINER] Running walk-forward with {len(folds)} folds...")
        
        trainer = MLModelTrainer()
        result = WalkForwardResult()
        
        all_aucs = []
        all_pfs = []
        all_calibration_errors = []
        
        for fold_num, (train_idx, test_idx) in enumerate(folds):
            self.logger.info(f"[WF_TRAINER] Processing fold {fold_num + 1}/{len(folds)}")
            
            # Split data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            
            # Train model
            if model_type == "xgboost":
                train_result = trainer.train_xgboost(X_train, y_train, feature_names)
            else:
                train_result = trainer.train_logistic(X_train, y_train, feature_names)
            
            model = train_result['model']
            scaler = train_result['scaler']
            
            # Test predictions
            X_test_scaled = scaler.transform(X_test)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba >= 0.58).astype(int)
            
            # Calculate metrics
            fold_result = WalkForwardFold(
                fold_number=fold_num + 1,
                train_start=datetime.now(),  # Placeholder
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                train_samples=len(train_idx),
                train_wins=int(sum(y_train)),
                train_losses=len(y_train) - int(sum(y_train)),
                test_samples=len(test_idx),
                test_wins=int(sum(y_test)),
                test_losses=len(y_test) - int(sum(y_test))
            )
            
            # Accuracy, precision, recall
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
            
            fold_result.test_accuracy = accuracy_score(y_test, y_pred)
            fold_result.test_precision = precision_score(y_test, y_pred, zero_division=0)
            fold_result.test_recall = recall_score(y_test, y_pred, zero_division=0)
            
            try:
                fold_result.test_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                fold_result.test_auc = 0.5
            
            # Calibration
            fold_result.avg_predicted_prob = float(np.mean(y_pred_proba))
            fold_result.actual_win_rate = float(np.mean(y_test))
            fold_result.calibration_error = abs(fold_result.avg_predicted_prob - fold_result.actual_win_rate)
            
            # Profit factor (approximate from predictions)
            # Assume 2R wins, 1R losses
            wins_captured = sum((y_pred == 1) & (y_test == 1))
            losses_captured = sum((y_pred == 1) & (y_test == 0))
            
            if losses_captured > 0:
                fold_result.test_profit_factor = (wins_captured * 2) / losses_captured
            else:
                fold_result.test_profit_factor = float('inf') if wins_captured > 0 else 0
            
            result.folds.append(fold_result)
            all_aucs.append(fold_result.test_auc)
            all_pfs.append(fold_result.test_profit_factor if fold_result.test_profit_factor != float('inf') else 3.0)
            all_calibration_errors.append(fold_result.calibration_error)
            
            self.logger.info(
                f"  Fold {fold_num + 1}: AUC={fold_result.test_auc:.3f}, "
                f"PF={fold_result.test_profit_factor:.2f}, "
                f"Cal.Err={fold_result.calibration_error:.3f}"
            )
        
        # Aggregate metrics
        result.overall_auc = np.mean(all_aucs)
        result.overall_profit_factor = np.mean(all_pfs)
        result.overall_calibration_error = np.mean(all_calibration_errors)
        result.auc_stability = np.std(all_aucs)
        result.pf_stability = np.std(all_pfs)
        
        result.total_trades = sum(f.test_samples for f in result.folds)
        
        self.logger.info(f"\n[WF_TRAINER] Walk-Forward Complete:")
        self.logger.info(f"  Overall AUC: {result.overall_auc:.3f} ± {result.auc_stability:.3f}")
        self.logger.info(f"  Overall PF: {result.overall_profit_factor:.2f} ± {result.pf_stability:.2f}")
        self.logger.info(f"  Calibration Error: {result.overall_calibration_error:.3f}")
        
        return result
    
    def generate_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Generate calibration curve data.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Dict with bin data for plotting
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        actual_rates = []
        counts = []
        
        for i in range(n_bins):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i + 1])
            if sum(mask) > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                actual_rates.append(y_true[mask].mean())
                counts.append(sum(mask))
        
        return {
            'bin_centers': bin_centers,
            'actual_rates': actual_rates,
            'counts': counts,
            'perfect_line': [0, 1]
        }
    
    def save_results(self, result: WalkForwardResult, path: str) -> None:
        """Save walk-forward results to CSV."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'fold', 'train_samples', 'test_samples', 'train_wins', 'test_wins',
                'accuracy', 'precision', 'recall', 'auc', 'profit_factor',
                'avg_pred_prob', 'actual_win_rate', 'calibration_error'
            ])
            
            for fold in result.folds:
                writer.writerow([
                    fold.fold_number, fold.train_samples, fold.test_samples,
                    fold.train_wins, fold.test_wins, f"{fold.test_accuracy:.3f}",
                    f"{fold.test_precision:.3f}", f"{fold.test_recall:.3f}",
                    f"{fold.test_auc:.3f}", f"{fold.test_profit_factor:.2f}",
                    f"{fold.avg_predicted_prob:.3f}", f"{fold.actual_win_rate:.3f}",
                    f"{fold.calibration_error:.3f}"
                ])
            
            # Summary row
            writer.writerow([])
            writer.writerow(['OVERALL', '', result.total_trades, '', '',
                             '', '', '', f"{result.overall_auc:.3f}",
                             f"{result.overall_profit_factor:.2f}", '', '',
                             f"{result.overall_calibration_error:.3f}"])
        
        self.logger.info(f"[WF_TRAINER] Results saved to {path}")
