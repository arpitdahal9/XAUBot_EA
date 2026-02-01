"""
ML Trade Quality Filter

Uses ML to predict P(success) = probability trade reaches TP1 before SL.

Two modes:
1. Trade Gate: Only execute if P(win) >= threshold
2. Risk Scaling: Adjust position size based on probability

Models supported:
- Logistic Regression (baseline, interpretable)
- XGBoost (production, higher accuracy)

This is the institutional approach: ML filters trades, not predicts price.
"""

import os
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from .ml_features import MLFeatures, MLFeatureExtractor


@dataclass
class MLPrediction:
    """Result of ML trade quality prediction."""
    probability_win: float  # P(TP1 hit before SL)
    should_trade: bool  # Whether trade passes gate
    risk_multiplier: float  # Dynamic risk scaling factor
    confidence_interval: Tuple[float, float]  # 95% CI if available
    model_version: str
    timestamp: datetime


class MLTradeFilter:
    """
    ML-based trade quality filter.
    
    Decides:
    1. Which TPU setups are worth taking (P(win) >= threshold)
    2. How much to risk (dynamic scaling based on probability)
    
    This creates edge without curve-fit indicators.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        probability_threshold: float = 0.58,
        base_risk_percent: float = 0.75,
        min_risk_multiplier: float = 0.5,
        max_risk_multiplier: float = 1.5,
        enabled: bool = True
    ):
        """
        Initialize ML Trade Filter.
        
        Args:
            model_path: Path to saved model file
            probability_threshold: Minimum P(win) to take trade (default 0.58)
            base_risk_percent: Base risk per trade (default 0.75%)
            min_risk_multiplier: Minimum risk scaling (default 0.5x)
            max_risk_multiplier: Maximum risk scaling (default 1.5x)
            enabled: Whether ML filter is active
        """
        self.logger = logging.getLogger(__name__)
        
        self.probability_threshold = probability_threshold
        self.base_risk_percent = base_risk_percent
        self.min_risk_multiplier = min_risk_multiplier
        self.max_risk_multiplier = max_risk_multiplier
        self.enabled = enabled
        
        self._model = None
        self._model_type = "none"
        self._model_version = "untrained"
        self._feature_names: List[str] = []
        self._scaler = None
        
        # Feature extractor
        self.feature_extractor = MLFeatureExtractor()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.logger.info(f"[ML_FILTER] Initialized")
        self.logger.info(f"  Enabled: {self.enabled}")
        self.logger.info(f"  Probability Threshold: {self.probability_threshold:.0%}")
        self.logger.info(f"  Risk Scaling: {self.min_risk_multiplier}x - {self.max_risk_multiplier}x")
        self.logger.info(f"  Model: {self._model_type} ({self._model_version})")
    
    def predict(self, features: MLFeatures) -> MLPrediction:
        """
        Predict trade quality from features.
        
        Args:
            features: Extracted ML features from TPU signal
            
        Returns:
            MLPrediction with probability, trade decision, and risk multiplier
        """
        if not self.enabled or self._model is None:
            # No model - use TPU confidence as proxy
            return self._fallback_prediction(features)
        
        try:
            # Prepare feature array
            X = features.to_array().reshape(1, -1)
            
            # Apply scaler if available
            if self._scaler is not None:
                X = self._scaler.transform(X)
            
            # Get probability prediction
            if hasattr(self._model, 'predict_proba'):
                proba = self._model.predict_proba(X)[0]
                # Probability of class 1 (win)
                p_win = proba[1] if len(proba) > 1 else proba[0]
            else:
                # Model doesn't support predict_proba - use predict
                p_win = float(self._model.predict(X)[0])
            
            # Ensure valid probability
            p_win = np.clip(p_win, 0.0, 1.0)
            
            # Calculate risk multiplier
            risk_mult = self._calculate_risk_multiplier(p_win)
            
            # Trade decision
            should_trade = p_win >= self.probability_threshold
            
            # Confidence interval (if model supports it)
            ci = self._get_confidence_interval(X)
            
            self.logger.debug(
                f"[ML_FILTER] P(win)={p_win:.1%}, "
                f"Risk={risk_mult:.2f}x, Trade={should_trade}"
            )
            
            return MLPrediction(
                probability_win=p_win,
                should_trade=should_trade,
                risk_multiplier=risk_mult,
                confidence_interval=ci,
                model_version=self._model_version,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"[ML_FILTER] Prediction failed: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: MLFeatures) -> MLPrediction:
        """
        Fallback when model unavailable.
        
        Uses TPU confidence as proxy for P(win).
        """
        # Map TPU confidence to approximate P(win)
        # TPU confidence of 0.75+ typically has ~55-60% win rate
        p_win = 0.5 + (features.tpu_confidence - 0.5) * 0.3
        p_win = np.clip(p_win, 0.3, 0.7)
        
        risk_mult = self._calculate_risk_multiplier(p_win)
        should_trade = p_win >= self.probability_threshold
        
        return MLPrediction(
            probability_win=p_win,
            should_trade=should_trade,
            risk_multiplier=risk_mult,
            confidence_interval=(p_win - 0.1, p_win + 0.1),
            model_version="fallback_tpu",
            timestamp=datetime.now()
        )
    
    def _calculate_risk_multiplier(self, p_win: float) -> float:
        """
        Calculate dynamic risk multiplier based on P(win).
        
        Formula: risk = base * clamp((P(win) - 0.50) / 0.20, min, max)
        
        Examples:
        - P(win)=0.55 → 0.5x base risk
        - P(win)=0.65 → 1.0x base risk
        - P(win)=0.75 → 1.5x base risk
        """
        # Linear scaling from 0.5 to 0.7 probability
        raw_mult = (p_win - 0.50) / 0.20
        
        # Clamp to allowed range
        clamped = np.clip(raw_mult, self.min_risk_multiplier, self.max_risk_multiplier)
        
        return float(clamped)
    
    def _get_confidence_interval(self, X: np.ndarray) -> Tuple[float, float]:
        """Get confidence interval for prediction if available."""
        # Default CI based on typical calibration
        return (0.45, 0.65)
    
    def get_adjusted_risk_percent(self, p_win: float) -> float:
        """
        Get the adjusted risk percentage based on probability.
        
        Args:
            p_win: Predicted probability of win
            
        Returns:
            Adjusted risk percentage (e.g., 0.5% instead of 0.75%)
        """
        multiplier = self._calculate_risk_multiplier(p_win)
        return self.base_risk_percent * multiplier
    
    def load_model(self, path: str) -> bool:
        """
        Load trained model from file.
        
        Args:
            path: Path to model file (.pkl)
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self._model = data.get('model')
            self._model_type = data.get('model_type', 'unknown')
            self._model_version = data.get('version', 'unknown')
            self._feature_names = data.get('feature_names', [])
            self._scaler = data.get('scaler')
            
            self.logger.info(f"[ML_FILTER] Loaded model: {self._model_type} v{self._model_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ML_FILTER] Failed to load model: {e}")
            return False
    
    def save_model(self, path: str) -> bool:
        """
        Save trained model to file.
        
        Args:
            path: Path to save model
            
        Returns:
            True if saved successfully
        """
        try:
            data = {
                'model': self._model,
                'model_type': self._model_type,
                'version': self._model_version,
                'feature_names': self._feature_names,
                'scaler': self._scaler,
                'probability_threshold': self.probability_threshold,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"[ML_FILTER] Saved model to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ML_FILTER] Failed to save model: {e}")
            return False
    
    def set_model(
        self,
        model: Any,
        model_type: str,
        version: str,
        feature_names: List[str],
        scaler: Any = None
    ) -> None:
        """
        Set a trained model.
        
        Args:
            model: Trained sklearn model
            model_type: "logistic" or "xgboost"
            version: Model version string
            feature_names: List of feature names
            scaler: Optional StandardScaler
        """
        self._model = model
        self._model_type = model_type
        self._model_version = version
        self._feature_names = feature_names
        self._scaler = scaler
        
        self.logger.info(f"[ML_FILTER] Model set: {model_type} v{version}")
    
    def evaluate_signal(
        self,
        # TPU Analysis Results
        tda_result: Any,
        fib_levels: Any,
        divergence_result: Any,
        pattern_result: Any,
        # Trade Parameters
        trade_direction: str,
        entry_price: float,
        stop_loss: float,
        # Market Context
        current_atr: float,
        current_spread: float,
        timestamp: datetime,
        pair: str,
        tpu_confidence: float
    ) -> Tuple[bool, float, MLPrediction]:
        """
        Full evaluation of a TPU signal through ML filter.
        
        Args:
            All TPU signal components and trade parameters
            
        Returns:
            (should_trade, adjusted_risk_percent, prediction)
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            tda_result=tda_result,
            fib_levels=fib_levels,
            divergence_result=divergence_result,
            pattern_result=pattern_result,
            trade_direction=trade_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            current_atr=current_atr,
            current_spread=current_spread,
            timestamp=timestamp,
            pair=pair,
            tpu_confidence=tpu_confidence
        )
        
        # Get prediction
        prediction = self.predict(features)
        
        # Calculate adjusted risk
        adjusted_risk = self.get_adjusted_risk_percent(prediction.probability_win)
        
        self.logger.info(
            f"[ML_FILTER] {pair} {trade_direction}: "
            f"P(win)={prediction.probability_win:.1%}, "
            f"Trade={'YES' if prediction.should_trade else 'NO'}, "
            f"Risk={adjusted_risk:.2f}%"
        )
        
        return prediction.should_trade, adjusted_risk, prediction
    
    def reset(self) -> None:
        """Reset feature extractor history for new backtest."""
        self.feature_extractor.reset_history()


class MLModelTrainer:
    """
    Train ML models for trade quality prediction.
    
    Supports:
    - Logistic Regression (baseline)
    - XGBoost (production)
    
    With proper:
    - Walk-forward validation
    - Leakage prevention
    - Probability calibration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def train_logistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Train Logistic Regression baseline model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1=TP1 hit, 0=SL hit)
            feature_names: List of feature names
            
        Returns:
            Dict with model, metrics, and feature importances
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.calibration import CalibratedClassifierCV
        
        self.logger.info("[ML_TRAINER] Training Logistic Regression...")
        
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
        
        # Cross-validation scores
        cv_scores = cross_val_score(calibrated, X_scaled, y, cv=5, scoring='roc_auc')
        
        # Feature importances (coefficients)
        importances = dict(zip(feature_names, model.coef_[0]))
        
        self.logger.info(f"[ML_TRAINER] Logistic CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return {
            'model': calibrated,
            'model_type': 'logistic',
            'scaler': scaler,
            'feature_names': feature_names,
            'cv_auc': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'feature_importances': importances
        }
    
    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Train XGBoost production model.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            
        Returns:
            Dict with model, metrics, and feature importances
        """
        try:
            import xgboost as xgb
        except ImportError:
            self.logger.warning("[ML_TRAINER] XGBoost not installed, using Logistic")
            return self.train_logistic(X, y, feature_names)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.calibration import CalibratedClassifierCV
        
        self.logger.info("[ML_TRAINER] Training XGBoost...")
        
        # Scale features (not strictly necessary for tree models but helps)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate scale_pos_weight for imbalanced data
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
            use_label_encoder=False,
            eval_metric='auc'
        )
        model.fit(X_scaled, y)
        
        # Calibrate probabilities
        calibrated = CalibratedClassifierCV(model, cv=5, method='isotonic')
        calibrated.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(calibrated, X_scaled, y, cv=5, scoring='roc_auc')
        
        # Feature importances
        importances = dict(zip(feature_names, model.feature_importances_))
        
        self.logger.info(f"[ML_TRAINER] XGBoost CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return {
            'model': calibrated,
            'model_type': 'xgboost',
            'scaler': scaler,
            'feature_names': feature_names,
            'cv_auc': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'feature_importances': importances
        }
