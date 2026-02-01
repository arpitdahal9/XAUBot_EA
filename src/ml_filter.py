"""
ML Trade Quality Filter (Institutional Grade)

Predicts P(TP1 hit before SL) and uses it for:
1. Trade gate: Only execute if P(win) >= threshold
2. Risk scaling: Adjust position size based on probability

Features:
- Smooth sigmoid risk scaling (no noise whipsaw)
- Dead-zone around threshold (stabilizes behavior)
- Auto-disable on model quality failure
- Calibrated probability thresholds
"""

import os
import pickle
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Any, List
import numpy as np

from .ml_features import MLFeatures, MLFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    probability_win: float
    should_trade: bool
    risk_multiplier: float
    threshold_used: float
    model_version: str
    model_valid: bool


class MLTradeFilter:
    """
    ML-based trade quality filter.
    
    Improved features:
    - Sigmoid risk scaling (smoother than linear)
    - Dead-zone: no risk change if p_win within +/-0.02 of threshold
    - Auto-disable if model fails quality checks
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        probability_threshold: float = 0.58,
        base_risk_percent: float = 0.75,
        min_risk_multiplier: float = 0.5,
        max_risk_multiplier: float = 1.5,
        dead_zone: float = 0.02,
        enabled: bool = True
    ):
        self.probability_threshold = probability_threshold
        self.base_risk_percent = base_risk_percent
        self.min_risk_mult = min_risk_multiplier
        self.max_risk_mult = max_risk_multiplier
        self.dead_zone = dead_zone
        self.enabled = enabled
        
        self._model = None
        self._scaler = None
        self._model_version = "none"
        self._model_valid = False
        self._feature_names: List[str] = []
        
        self.feature_extractor = MLFeatureExtractor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, path: str) -> bool:
        """Load trained model artifact."""
        try:
            with open(path, 'rb') as f:
                artifact = pickle.load(f)
            
            self._model = artifact.get('model')
            self._scaler = artifact.get('scaler')
            self._model_version = artifact.get('version', 'unknown')
            self._model_valid = artifact.get('model_valid', False)
            self._feature_names = artifact.get('feature_names', [])
            
            # Use trained threshold if available
            trained_thresh = artifact.get('threshold')
            if trained_thresh:
                self.probability_threshold = trained_thresh
            
            # Auto-disable if model failed quality checks
            if not self._model_valid:
                logger.warning(f"ML model failed quality checks - reverting to fallback")
                self.enabled = False
            else:
                logger.info(f"Loaded ML model v{self._model_version}, threshold={self.probability_threshold:.2f}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False
    
    def predict(self, features: MLFeatures) -> MLPrediction:
        """Predict trade quality."""
        if not self.enabled or self._model is None:
            return self._fallback_prediction(features)
        
        try:
            X = features.to_array().reshape(1, -1)
            
            if self._scaler is not None:
                X = self._scaler.transform(X)
            
            proba = self._model.predict_proba(X)[0]
            p_win = float(proba[1] if len(proba) > 1 else proba[0])
            p_win = np.clip(p_win, 0.0, 1.0)
            
            # Trade decision
            should_trade = p_win >= self.probability_threshold
            
            # Risk scaling with dead-zone and smooth sigmoid
            risk_mult = self._calculate_risk_multiplier(p_win)
            
            return MLPrediction(
                probability_win=p_win,
                should_trade=should_trade,
                risk_multiplier=risk_mult,
                threshold_used=self.probability_threshold,
                model_version=self._model_version,
                model_valid=self._model_valid
            )
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: MLFeatures) -> MLPrediction:
        """Fallback when model unavailable."""
        # Use TPU confidence as proxy
        p_win = 0.5 + (features.tpu_confidence - 0.5) * 0.3
        p_win = np.clip(p_win, 0.3, 0.7)
        
        return MLPrediction(
            probability_win=p_win,
            should_trade=p_win >= self.probability_threshold,
            risk_multiplier=1.0,  # No scaling in fallback
            threshold_used=self.probability_threshold,
            model_version="fallback",
            model_valid=False
        )
    
    def _calculate_risk_multiplier(self, p_win: float) -> float:
        """
        Calculate risk multiplier with dead-zone and smooth sigmoid.
        
        Dead-zone: If p_win is within +/-0.02 of threshold, return 1.0
        Sigmoid: Smooth transition instead of linear (reduces noise sensitivity)
        """
        # Dead-zone check
        if abs(p_win - self.probability_threshold) <= self.dead_zone:
            return 1.0
        
        # Sigmoid mapping from p_win to risk multiplier
        # Center at threshold, map [0.5, 0.75] -> [min_mult, max_mult]
        x = (p_win - 0.5) / 0.25  # Normalize to [-2, 2] range roughly
        sigmoid = 1 / (1 + np.exp(-2 * x))  # Smoother sigmoid
        
        # Map sigmoid output to [min_mult, max_mult]
        mult = self.min_risk_mult + sigmoid * (self.max_risk_mult - self.min_risk_mult)
        
        return float(np.clip(mult, self.min_risk_mult, self.max_risk_mult))
    
    def get_adjusted_risk_percent(self, p_win: float) -> float:
        """Get adjusted risk percentage based on probability."""
        mult = self._calculate_risk_multiplier(p_win)
        return self.base_risk_percent * mult
    
    def evaluate_signal(
        self,
        tda_result: Any,
        fib_levels: Any,
        divergence_result: Any,
        pattern_result: Any,
        trade_direction: str,
        entry_price: float,
        stop_loss: float,
        current_atr: float,
        current_spread: float,
        timestamp: datetime,
        pair: str,
        tpu_confidence: float
    ) -> Tuple[bool, float, MLPrediction]:
        """Full evaluation of a TPU signal through ML filter."""
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
        
        prediction = self.predict(features)
        adjusted_risk = self.get_adjusted_risk_percent(prediction.probability_win)
        
        return prediction.should_trade, adjusted_risk, prediction
    
    def reset(self):
        """Reset feature extractor for new backtest."""
        self.feature_extractor.reset_history()
