"""
Walk-Forward ML Training Pipeline (Institutional Grade)

Features:
- Proper dataset construction from backtest trades
- Walk-forward validation with embargo windows
- Threshold optimization for expectancy (not accuracy)
- Probability calibration with safeguards
- Brier score and calibration error metrics
"""

import os
import csv
import pickle
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingRow:
    """Single row in ML training dataset."""
    timestamp: datetime
    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    features: Dict[str, float]
    label: int  # 1=TP1 hit, 0=SL hit
    pnl: float = 0.0
    trade_id: str = ""


@dataclass
class FoldResult:
    """Results from one walk-forward fold."""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    auc: float = 0.5
    brier_score: float = 0.25
    calibration_error: float = 0.1
    precision: float = 0.0
    recall: float = 0.0
    optimal_threshold: float = 0.58
    expected_value_at_threshold: float = 0.0
    trades_passed: int = 0
    trades_total: int = 0


@dataclass
class WalkForwardResult:
    folds: List[FoldResult] = field(default_factory=list)
    overall_auc: float = 0.5
    overall_brier: float = 0.25
    overall_calibration_error: float = 0.1
    auc_stability: float = 0.0
    recommended_threshold: float = 0.58
    model_valid: bool = False


class MLDatasetBuilder:
    """Build ML training dataset from backtest trades."""
    
    def __init__(self):
        self.rows: List[TrainingRow] = []
    
    def load_from_csv(self, csv_path: str) -> int:
        """
        Load training data from backtest CSV.
        
        Label definition:
        - y=1 if TP1 was hit before SL (tp1_closed_lots > 0)
        - y=0 if SL was hit before TP1
        """
        if not os.path.exists(csv_path):
            logger.error(f"CSV not found: {csv_path}")
            return 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Only use FINAL rows
                if row.get('row_type') != 'FINAL':
                    continue
                
                try:
                    # Parse timestamp
                    ts_str = row.get('entry_time', '')
                    try:
                        timestamp = datetime.fromisoformat(ts_str.replace('Z', ''))
                    except:
                        continue
                    
                    # Determine label
                    tp1_closed = float(row.get('tp1_closed_lots', 0) or 0)
                    exit_reason = row.get('exit_reason', '')
                    label = 1 if (tp1_closed > 0 or 'TP1' in exit_reason) else 0
                    
                    # Extract features available in CSV
                    features = self._extract_csv_features(row, timestamp)
                    
                    training_row = TrainingRow(
                        timestamp=timestamp,
                        pair=row.get('pair', 'EUR/USD'),
                        direction=row.get('side', 'BUY'),
                        entry_price=float(row.get('entry_price', 0) or 0),
                        stop_loss=float(row.get('stop_loss', 0) or 0),
                        features=features,
                        label=label,
                        pnl=float(row.get('pnl_total_trade', 0) or 0),
                        trade_id=row.get('trade_id', '')
                    )
                    self.rows.append(training_row)
                    
                except Exception as e:
                    logger.debug(f"Skipping row: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.rows)} training rows from {csv_path}")
        win_count = sum(1 for r in self.rows if r.label == 1)
        logger.info(f"  Win rate: {100*win_count/len(self.rows):.1f}%")
        return len(self.rows)
    
    def _extract_csv_features(self, row: dict, timestamp: datetime) -> Dict[str, float]:
        """Extract features from CSV row at ENTRY TIME only."""
        hour = timestamp.hour
        
        return {
            'tpu_confidence': float(row.get('tpu_confidence', 0.75) or 0.75),
            'tda_alignment_score': self._parse_alignment(row.get('tda_alignment', 'GOOD')),
            'fib_distance_normalized': float(row.get('fib_distance_pips', 10) or 10) / 20.0,
            'has_divergence': 1 if row.get('divergence_type') else 0,
            'has_pattern': 1 if row.get('pattern_type') else 0,
            'hour_normalized': hour / 24.0,
            'day_normalized': timestamp.weekday() / 7.0,
            'is_london': 1.0 if 8 <= hour < 16 else 0.0,
            'is_ny': 1.0 if 13 <= hour < 21 else 0.0,
            'is_overlap': 1.0 if 13 <= hour < 16 else 0.0,
            'stop_distance_normalized': min(
                abs(float(row.get('entry_price', 0) or 0) - float(row.get('stop_loss', 0) or 0)) * 10000 / 50, 
                2.0
            ),
        }
    
    def _parse_alignment(self, alignment: str) -> float:
        mapping = {'PERFECT': 1.0, 'GOOD': 0.5, 'WEAK': 0.0}
        return mapping.get(str(alignment).upper(), 0.0)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """Get feature array, labels, and timestamps."""
        if not self.rows:
            return np.array([]), np.array([]), []
        
        # Sort by timestamp
        self.rows.sort(key=lambda r: r.timestamp)
        
        X = []
        y = []
        timestamps = []
        
        for row in self.rows:
            X.append(list(row.features.values()))
            y.append(row.label)
            timestamps.append(row.timestamp)
        
        return np.array(X, dtype=np.float32), np.array(y), timestamps
    
    def get_feature_names(self) -> List[str]:
        if self.rows:
            return list(self.rows[0].features.keys())
        return []


class WalkForwardTrainer:
    """
    Walk-forward ML training with threshold optimization.
    
    Features:
    - Rolling train/test windows with embargo
    - Threshold optimization for expectancy (not accuracy)
    - Probability calibration and safeguards
    - Auto-disable if model fails quality checks
    """
    
    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 6,
        embargo_hours: int = 24,
        min_samples: int = 50,
        min_auc: float = 0.55,
        max_calibration_error: float = 0.10
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.embargo_hours = embargo_hours
        self.min_samples = min_samples
        self.min_auc = min_auc
        self.max_calibration_error = max_calibration_error
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: List[datetime],
        model_type: str = 'xgboost'
    ) -> Tuple[WalkForwardResult, Any, Any]:
        """
        Run walk-forward validation.
        
        Returns (result, trained_model, scaler)
        """
        if len(y) < self.min_samples:
            logger.warning(f"Insufficient samples: {len(y)} < {self.min_samples}")
            return WalkForwardResult(), None, None
        
        start_date = min(timestamps)
        end_date = max(timestamps)
        
        result = WalkForwardResult()
        all_thresholds = []
        
        train_days = self.train_months * 30
        test_days = self.test_months * 30
        embargo = timedelta(hours=self.embargo_hours)
        
        current_start = start_date
        fold_num = 0
        
        while current_start + timedelta(days=train_days + test_days) <= end_date:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            # Get indices with embargo
            train_idx = [i for i, t in enumerate(timestamps) if current_start <= t < (train_end - embargo)]
            test_idx = [i for i, t in enumerate(timestamps) if (train_end + embargo) <= t < test_end]
            
            if len(train_idx) < 30 or len(test_idx) < 5:
                current_start += timedelta(days=test_days)
                continue
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train model
            model, scaler = self._train_model(X_train, y_train, model_type)
            
            # Test
            X_test_scaled = scaler.transform(X_test)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Find optimal threshold
            optimal_thresh, ev = self._optimize_threshold(y_test, y_pred_proba)
            all_thresholds.append(optimal_thresh)
            
            # Calculate metrics
            fold_result = self._calculate_fold_metrics(
                fold_num, current_start, train_end, test_end,
                len(train_idx), len(test_idx),
                y_test, y_pred_proba, optimal_thresh, ev
            )
            result.folds.append(fold_result)
            
            logger.info(f"Fold {fold_num}: AUC={fold_result.auc:.3f}, Thresh={optimal_thresh:.2f}, EV={ev:.3f}")
            
            fold_num += 1
            current_start += timedelta(days=test_days)
        
        # Aggregate results
        if result.folds:
            result.overall_auc = np.mean([f.auc for f in result.folds])
            result.overall_brier = np.mean([f.brier_score for f in result.folds])
            result.overall_calibration_error = np.mean([f.calibration_error for f in result.folds])
            result.auc_stability = np.std([f.auc for f in result.folds])
            result.recommended_threshold = np.median(all_thresholds) if all_thresholds else 0.58
            
            # Validate model quality
            result.model_valid = (
                result.overall_auc >= self.min_auc and
                result.overall_calibration_error <= self.max_calibration_error
            )
        
        # Train final model on all data
        final_model, final_scaler = self._train_model(X, y, model_type)
        
        return result, final_model, final_scaler
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, model_type: str):
        """Train a single model."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                n_pos = sum(y)
                n_neg = len(y) - n_pos
                scale_pos_weight = n_neg / max(n_pos, 1)
                base_model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='auc'
                )
            except ImportError:
                from sklearn.linear_model import LogisticRegression
                base_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            base_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        
        base_model.fit(X_scaled, y)
        
        # Calibrate probabilities (isotonic)
        calibrated = CalibratedClassifierCV(base_model, cv=min(5, len(y)//10), method='isotonic')
        calibrated.fit(X_scaled, y)
        
        return calibrated, scaler
    
    def _optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """
        Find threshold that maximizes expected value.
        
        Uses Model A payoff structure: Win = +2R (TP1), Loss = -1R
        """
        best_thresh = 0.58
        best_ev = -999
        
        for thresh in np.arange(0.50, 0.76, 0.02):
            # Trades that would pass this threshold
            mask = y_pred_proba >= thresh
            if sum(mask) < 3:
                continue
            
            wins = sum((y_true == 1) & mask)
            losses = sum((y_true == 0) & mask)
            total = wins + losses
            
            if total == 0:
                continue
            
            # Expected value per trade (Model A: +2R win, -1R loss)
            ev = (wins * 2 - losses * 1) / total
            
            if ev > best_ev:
                best_ev = ev
                best_thresh = thresh
        
        return best_thresh, best_ev
    
    def _calculate_fold_metrics(
        self, fold_num, train_start, train_end, test_end,
        train_samples, test_samples, y_true, y_pred_proba, threshold, ev
    ) -> FoldResult:
        """Calculate all metrics for a fold."""
        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.5
        
        brier = brier_score_loss(y_true, y_pred_proba)
        calibration_error = abs(np.mean(y_pred_proba) - np.mean(y_true))
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return FoldResult(
            fold_number=fold_num,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            train_samples=train_samples,
            test_samples=test_samples,
            auc=auc,
            brier_score=brier,
            calibration_error=calibration_error,
            precision=precision,
            recall=recall,
            optimal_threshold=threshold,
            expected_value_at_threshold=ev,
            trades_passed=sum(y_pred),
            trades_total=len(y_true)
        )


def save_model_artifact(model, scaler, result: WalkForwardResult, feature_names: List[str], path: str):
    """Save trained model with metadata."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    artifact = {
        'model': model,
        'scaler': scaler,
        'model_type': 'xgboost',
        'version': datetime.now().strftime('%Y%m%d_%H%M'),
        'feature_names': feature_names,
        'threshold': result.recommended_threshold,
        'overall_auc': result.overall_auc,
        'overall_calibration_error': result.overall_calibration_error,
        'model_valid': result.model_valid,
        'created_at': datetime.now().isoformat()
    }
    
    with open(path, 'wb') as f:
        pickle.dump(artifact, f)
    
    logger.info(f"Model artifact saved to {path}")
    return artifact
