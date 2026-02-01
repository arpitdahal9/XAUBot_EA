"""
ML Trade Quality Filter (Institutional Grade)

Predicts P(TP1 hit before SL) and uses it for:
1. Trade gate: Only execute if P(win) >= threshold
2. Risk scaling: Adjust position size based on probability

UPGRADE 2: ML as Soft Filter (not brick wall)
- Zone A (≥0.60): Take trade at normal risk (1.0x)
- Zone B (0.55-0.60): Take trade at 0.5x risk (reduced size)
- Zone C (<0.55): Take trade at 0.25x risk (very small) - NO REJECTION

Features:
- 3-zone risk scaling (soft filter, ranks trades)
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
    ML-based trade quality filter (UPGRADE 2: Soft Filter).
    
    3-Zone Behavior:
    - Zone A (p >= zone_a_threshold): Take at normal risk (1.0x)
    - Zone B (zone_b_threshold <= p < zone_a_threshold): Take at reduced risk (0.5x)
    - Zone C (p < zone_b_threshold): Take at very small risk (0.25x) - NO REJECTION
    
    ML ranks trades and scales risk dynamically, doesn't reject everything near threshold.
    Auto-disable if model fails quality checks.
    """
    
    # 3-Zone thresholds (configurable)
    ZONE_A_THRESHOLD = 0.60  # Normal risk zone
    ZONE_B_THRESHOLD = 0.55  # Reduced risk zone
    ZONE_B_RISK_MULT = 0.5   # Risk multiplier for Zone B
    ZONE_C_RISK_MULT = 0.25  # Risk multiplier for Zone C (soft filter - no rejection)
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        probability_threshold: float = 0.55,  # Lowered: Zone C cutoff
        base_risk_percent: float = 0.75,
        min_risk_multiplier: float = 0.5,
        max_risk_multiplier: float = 1.5,
        dead_zone: float = 0.02,
        enabled: bool = True,
        zone_a_threshold: float = 0.60,
        zone_b_threshold: float = 0.55
    ):
        self.probability_threshold = probability_threshold
        self.base_risk_percent = base_risk_percent
        self.min_risk_mult = min_risk_multiplier
        self.max_risk_mult = max_risk_multiplier
        self.dead_zone = dead_zone
        self.enabled = enabled
        
        # 3-Zone configuration
        self.zone_a_threshold = zone_a_threshold
        self.zone_b_threshold = zone_b_threshold
        
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
            
            # 3-Zone Trade Decision
            should_trade, risk_mult, zone = self._apply_3zone_logic(p_win)
            
            logger.debug(f"ML 3-Zone: p_win={p_win:.2%} -> Zone {zone}, risk={risk_mult}x, trade={should_trade}")
            
            return MLPrediction(
                probability_win=p_win,
                should_trade=should_trade,
                risk_multiplier=risk_mult,
                threshold_used=self.zone_b_threshold,  # Rejection threshold
                model_version=self._model_version,
                model_valid=self._model_valid
            )
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: MLFeatures) -> MLPrediction:
        """Fallback when model unavailable - use TPU confidence for 3-zone."""
        # Use TPU confidence as proxy for p_win
        p_win = 0.5 + (features.tpu_confidence - 0.5) * 0.3
        p_win = np.clip(p_win, 0.3, 0.7)
        
        # Apply 3-zone logic even in fallback
        should_trade, risk_mult, zone = self._apply_3zone_logic(p_win)
        
        return MLPrediction(
            probability_win=p_win,
            should_trade=should_trade,
            risk_multiplier=risk_mult,
            threshold_used=self.zone_b_threshold,
            model_version="fallback",
            model_valid=False
        )
    
    def _apply_3zone_logic(self, p_win: float) -> Tuple[bool, float, str]:
        """
        Apply 3-Zone trading logic (UPGRADE 2: Soft Filter).
        
        Returns: (should_trade, risk_multiplier, zone_name)
        
        Zone A (p >= 0.60): Take at normal risk (1.0x)
        Zone B (0.55 <= p < 0.60): Take at reduced risk (0.5x)
        Zone C (p < 0.55): Take at very small risk (0.25x) - NO REJECTION
        
        ML now ranks and scales risk, doesn't reject trades near threshold.
        """
        if p_win >= self.zone_a_threshold:
            # Zone A: High confidence - take at normal risk
            return True, 1.0, "A"
        elif p_win >= self.zone_b_threshold:
            # Zone B: Medium confidence - take at reduced risk
            return True, self.ZONE_B_RISK_MULT, "B"
        else:
            # Zone C: Low confidence - take at very small risk (soft filter, not brick wall)
            return True, self.ZONE_C_RISK_MULT, "C"
    
    def _calculate_risk_multiplier(self, p_win: float) -> float:
        """
        Calculate risk multiplier using 3-zone logic.
        (Legacy method for backward compatibility)
        """
        _, mult, _ = self._apply_3zone_logic(p_win)
        return mult
    
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
