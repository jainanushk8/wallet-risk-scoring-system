"""
Wallet Risk Scoring Module
=========================

Calculates comprehensive risk scores (0-1000) based on engineered features
from DeFi wallet transaction analysis.

Optimized for Windows systems with lightweight, interpretable scoring models.

Author: AI Development Team
Date: July 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json

# Setup logging
logger = logging.getLogger(__name__)

class WalletRiskScorer:
    """
    Calculates wallet risk scores using weighted composite scoring methodology.
    
    Features:
    - Lightweight, CPU-optimized scoring (no GPU required)
    - Interpretable scoring logic with clear methodology
    - Robust handling of missing/incomplete data
    - Configurable risk weights and thresholds
    """
    
    def __init__(self, config: Dict = None): #type:ignore
        """
        Initialize the wallet risk scoring system.
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config or {}
        self.setup_config()
        self.setup_scoring_parameters()
        
        logger.info("WalletRiskScorer initialized successfully")
    
    def setup_config(self):
        """Setup default configuration parameters"""
        self.config.setdefault('min_risk_score', 0)
        self.config.setdefault('max_risk_score', 1000)
        self.config.setdefault('risk_score_precision', 0)  # Integer scores
        
    def setup_scoring_parameters(self):
        """Setup scoring weights, thresholds, and parameters"""
        
        # Main category weights (must sum to 1.0)
        self.category_weights = {
            'behavioral': 0.40,   # Transaction patterns and activity
            'liquidity': 0.35,    # Liquidation and leverage risk
            'market': 0.25        # Market exposure and diversification
        }
        
        # Behavioral scoring weights
        self.behavioral_weights = {
            'transaction_frequency': 0.25,
            'transaction_size_patterns': 0.20,
            'compound_activity_level': 0.30,
            'temporal_patterns': 0.25
        }
        
        # Liquidity scoring weights  
        self.liquidity_weights = {
            'collateral_ratio': 0.40,
            'liquidation_risk': 0.35,
            'leverage_usage': 0.25
        }
        
        # Market scoring weights
        self.market_weights = {
            'portfolio_risk': 0.35,
            'concentration_risk': 0.30,
            'diversification': 0.35
        }
        
        # Risk thresholds for scoring
        self.risk_thresholds = {
            # Behavioral thresholds
            'high_activity_tx_per_day': 10.0,
            'very_high_activity_tx_per_day': 50.0,
            'large_transaction_threshold': 1e18,  # 1 ETH in wei
            'burst_activity_threshold': 0.3,
            
            # Liquidity thresholds
            'safe_collateral_ratio': 2.0,     # 200%
            'warning_collateral_ratio': 1.5,   # 150%
            'danger_collateral_ratio': 1.2,    # 120%
            'high_utilization_threshold': 0.8,  # 80%
            
            # Market thresholds
            'high_risk_portfolio_score': 0.7,
            'high_concentration_threshold': 0.8,
            'low_diversification_threshold': 0.3
        }
        
        # Score scaling factors
        self.scaling_factors = {
            'behavioral_max': 1.0,
            'liquidity_max': 1.0,
            'market_max': 1.0
        }
    
    def calculate_score(self, features: Dict[str, Any]) -> int:
        """
        Calculate comprehensive risk score for a wallet.
        
        Args:
            features (Dict[str, Any]): Extracted features from feature engineer
            
        Returns:
            int: Risk score (0-1000)
        """
        try:
            wallet_address = features.get('wallet_address', 'unknown')
            logger.info(f"Calculating risk score for wallet: {wallet_address}")
            
            # Check data quality
            data_quality = features.get('data_quality', {})
            if data_quality.get('data_completeness') == 'none':
                logger.warning(f"No data available for {wallet_address}, returning minimal risk score")
                return self._get_minimal_risk_score()
            
            # Calculate component scores
            behavioral_score = self._calculate_behavioral_score(features.get('behavioral_features', {}))
            liquidity_score = self._calculate_liquidity_score(features.get('liquidity_features', {}))
            market_score = self._calculate_market_score(features.get('market_features', {}))
            
            # Apply category weights for composite score
            composite_score = (
                behavioral_score * self.category_weights['behavioral'] +
                liquidity_score * self.category_weights['liquidity'] +
                market_score * self.category_weights['market']
            )
            
            # Apply data quality adjustment
            quality_adjustment = self._get_quality_adjustment(data_quality)
            adjusted_score = composite_score * quality_adjustment
            
            # Scale to 0-1000 range
            final_score = int(np.clip(
                adjusted_score * self.config['max_risk_score'],
                self.config['min_risk_score'],
                self.config['max_risk_score']
            ))
            
            logger.info(f"Risk score calculated for {wallet_address}: {final_score}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return self._get_default_risk_score()
    
    def _calculate_behavioral_score(self, behavioral_features: Dict[str, float]) -> float:
        """Calculate behavioral risk score (0-1)"""
        try:
            # 1. Transaction Frequency Score
            freq_score = self._score_transaction_frequency(behavioral_features)
            
            # 2. Transaction Size Pattern Score
            size_score = self._score_transaction_sizes(behavioral_features)
            
            # 3. Compound Activity Score
            compound_score = self._score_compound_activity(behavioral_features)
            
            # 4. Temporal Pattern Score
            temporal_score = self._score_temporal_patterns(behavioral_features)
            
            # Weighted combination
            behavioral_score = (
                freq_score * self.behavioral_weights['transaction_frequency'] +
                size_score * self.behavioral_weights['transaction_size_patterns'] +
                compound_score * self.behavioral_weights['compound_activity_level'] +
                temporal_score * self.behavioral_weights['temporal_patterns']
            )
            
            return np.clip(behavioral_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in behavioral scoring: {str(e)}")
            return 0.3  # Default medium risk
    
    def _score_transaction_frequency(self, features: Dict[str, float]) -> float:
        """Score transaction frequency patterns"""
        try:
            avg_tx_per_day = features.get('avg_tx_per_day', 0)
            recent_activity_ratio = features.get('recent_activity_ratio', 0)
            
            # Frequency risk scoring
            if avg_tx_per_day >= self.risk_thresholds['very_high_activity_tx_per_day']:
                freq_risk = 0.9  # Very high activity = high risk
            elif avg_tx_per_day >= self.risk_thresholds['high_activity_tx_per_day']:
                freq_risk = 0.7  # High activity = elevated risk
            elif avg_tx_per_day >= 1.0:
                freq_risk = 0.4  # Normal activity = medium risk
            else:
                freq_risk = 0.1  # Low activity = low risk
            
            # Recent activity adjustment
            if recent_activity_ratio > 0.5:
                freq_risk *= 1.2  # Boost risk for recent high activity
            elif recent_activity_ratio < 0.1:
                freq_risk *= 0.8  # Reduce risk for inactive wallets
            
            return np.clip(freq_risk, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring frequency: {str(e)}")
            return 0.3
    
    def _score_transaction_sizes(self, features: Dict[str, float]) -> float:
        """Score transaction size patterns"""
        try:
            avg_tx_value = features.get('avg_tx_value', 0)
            large_tx_ratio = features.get('large_tx_ratio', 0)
            tx_value_std = features.get('tx_value_std', 0)
            
            # Size-based risk
            size_risk = 0.3  # Base risk
            
            # Large transaction penalty
            if avg_tx_value > self.risk_thresholds['large_transaction_threshold']:
                size_risk += 0.3
            
            # High variance in transaction sizes
            if large_tx_ratio > 0.3:  # >30% large transactions
                size_risk += 0.2
            
            # Transaction size volatility
            if tx_value_std > avg_tx_value:  # High volatility
                size_risk += 0.2
            
            return np.clip(size_risk, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring transaction sizes: {str(e)}")
            return 0.3
    
    def _score_compound_activity(self, features: Dict[str, float]) -> float:
        """Score Compound protocol activity patterns"""
        try:
            compound_activity_ratio = features.get('compound_activity_ratio', 0)
            compound_tx_count = features.get('compound_tx_count', 0)
            high_gas_tx_ratio = features.get('high_gas_tx_ratio', 0)
            
            # Activity level risk
            activity_risk = compound_activity_ratio * 0.7  # Base activity risk
            
            # Transaction count adjustment
            if compound_tx_count > 50:
                activity_risk += 0.3  # High activity = higher risk
            elif compound_tx_count > 10:
                activity_risk += 0.1
            
            # Gas usage patterns (complex transactions)
            if high_gas_tx_ratio > 0.5:
                activity_risk += 0.2  # Complex transactions = higher risk
            
            return np.clip(activity_risk, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring Compound activity: {str(e)}")
            return 0.3
    
    def _score_temporal_patterns(self, features: Dict[str, float]) -> float:
        """Score temporal transaction patterns"""
        try:
            burst_activity_score = features.get('burst_activity_score', 0)
            tx_regularity = features.get('tx_regularity', 0.5)
            
            # Burst activity risk
            temporal_risk = burst_activity_score * 0.6
            
            # Irregularity penalty
            if tx_regularity < 0.3:  # Very irregular
                temporal_risk += 0.4
            elif tx_regularity > 0.8:  # Very regular (bot-like)
                temporal_risk += 0.2
            
            return np.clip(temporal_risk, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring temporal patterns: {str(e)}")
            return 0.3
    
    def _calculate_liquidity_score(self, liquidity_features: Dict[str, float]) -> float:
        """Calculate liquidity risk score (0-1)"""
        try:
            # 1. Collateral Ratio Score
            collateral_score = self._score_collateral_ratio(liquidity_features)
            
            # 2. Liquidation Risk Score
            liquidation_score = self._score_liquidation_risk(liquidity_features)
            
            # 3. Leverage Usage Score
            leverage_score = self._score_leverage_usage(liquidity_features)
            
            # Weighted combination
            liquidity_score = (
                collateral_score * self.liquidity_weights['collateral_ratio'] +
                liquidation_score * self.liquidity_weights['liquidation_risk'] +
                leverage_score * self.liquidity_weights['leverage_usage']
            )
            
            return np.clip(liquidity_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in liquidity scoring: {str(e)}")
            return 0.3
    
    def _score_collateral_ratio(self, features: Dict[str, float]) -> float:
        """Score collateral ratio safety"""
        try:
            collateral_ratio = features.get('collateral_ratio', 0)
            
            if collateral_ratio == 0:
                return 0.1  # No borrowing = low risk
            elif collateral_ratio >= self.risk_thresholds['safe_collateral_ratio']:
                return 0.1  # Safe collateralization
            elif collateral_ratio >= self.risk_thresholds['warning_collateral_ratio']:
                return 0.5  # Warning level
            elif collateral_ratio >= self.risk_thresholds['danger_collateral_ratio']:
                return 0.8  # Dangerous level
            else:
                return 0.95  # Critical risk
                
        except Exception as e:
            logger.warning(f"Error scoring collateral ratio: {str(e)}")
            return 0.3
    
    def _score_liquidation_risk(self, features: Dict[str, float]) -> float:
        """Score liquidation risk"""
        try:
            liquidation_risk_score = features.get('liquidation_risk_score', 0.5)
            return np.clip(liquidation_risk_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring liquidation risk: {str(e)}")
            return 0.3
    
    def _score_leverage_usage(self, features: Dict[str, float]) -> float:
        """Score leverage usage patterns"""
        try:
            utilization_ratio = features.get('utilization_ratio', 0)
            leverage_score = features.get('leverage_score', 0)
            
            # High utilization = higher risk
            if utilization_ratio >= self.risk_thresholds['high_utilization_threshold']:
                usage_risk = 0.8
            elif utilization_ratio >= 0.5:
                usage_risk = 0.5
            else:
                usage_risk = 0.2
            
            # Combine with existing leverage score
            combined_score = (usage_risk + leverage_score) / 2
            
            return np.clip(combined_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring leverage usage: {str(e)}")
            return 0.3
    
    def _calculate_market_score(self, market_features: Dict[str, float]) -> float:
        """Calculate market risk score (0-1)"""
        try:
            # 1. Portfolio Risk Score
            portfolio_score = self._score_portfolio_risk(market_features)
            
            # 2. Concentration Risk Score
            concentration_score = self._score_concentration_risk(market_features)
            
            # 3. Diversification Score
            diversification_score = self._score_diversification(market_features)
            
            # Weighted combination
            market_score = (
                portfolio_score * self.market_weights['portfolio_risk'] +
                concentration_score * self.market_weights['concentration_risk'] +
                diversification_score * self.market_weights['diversification']
            )
            
            return np.clip(market_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in market scoring: {str(e)}")
            return 0.3
    
    def _score_portfolio_risk(self, features: Dict[str, float]) -> float:
        """Score overall portfolio risk"""
        try:
            portfolio_risk_score = features.get('portfolio_risk_score', 0.5)
            
            # Apply threshold-based scoring
            if portfolio_risk_score >= self.risk_thresholds['high_risk_portfolio_score']:
                return 0.8  # High risk portfolio
            else:
                return portfolio_risk_score  # Use calculated score
                
        except Exception as e:
            logger.warning(f"Error scoring portfolio risk: {str(e)}")
            return 0.5
    
    def _score_concentration_risk(self, features: Dict[str, float]) -> float:
        """Score asset concentration risk"""
        try:
            concentration_risk = features.get('concentration_risk', 0)
            
            if concentration_risk >= self.risk_thresholds['high_concentration_threshold']:
                return 0.9  # Very concentrated = high risk
            elif concentration_risk >= 0.5:
                return 0.6  # Moderate concentration
            else:
                return 0.2  # Well distributed
                
        except Exception as e:
            logger.warning(f"Error scoring concentration risk: {str(e)}")
            return 0.3
    
    def _score_diversification(self, features: Dict[str, float]) -> float:
        """Score portfolio diversification (inverted - less diversification = higher risk)"""
        try:
            diversification_score = features.get('diversification_score', 0)
            asset_diversification = features.get('asset_diversification', 0)
            
            # Average diversification measures
            avg_diversification = (diversification_score + asset_diversification) / 2
            
            # Invert score (low diversification = high risk)
            risk_score = 1.0 - avg_diversification
            
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error scoring diversification: {str(e)}")
            return 0.5
    
    def _get_quality_adjustment(self, data_quality: Dict[str, Any]) -> float:
        """Calculate data quality adjustment factor"""
        try:
            completeness = data_quality.get('data_completeness', 'medium')
            
            # Quality adjustment factors
            quality_adjustments = {
                'high': 1.0,      # No adjustment for high quality data
                'medium': 0.9,    # Slight reduction for medium quality
                'low': 0.7,       # More significant reduction for low quality
                'none': 0.5       # Major reduction for no data
            }
            
            return quality_adjustments.get(completeness, 0.8)
            
        except Exception as e:
            logger.warning(f"Error calculating quality adjustment: {str(e)}")
            return 0.8
    
    def _get_minimal_risk_score(self) -> int:
        """Return minimal risk score for wallets with no data"""
        return 50  # Low risk for inactive wallets
    
    def _get_default_risk_score(self) -> int:
        """Return default risk score for error cases"""
        return 500  # Medium risk for unknown cases
    
    def get_score_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed explanation of risk score calculation.
        
        Args:
            features (Dict[str, Any]): Extracted features
            
        Returns:
            Dict[str, Any]: Detailed score breakdown and explanation
        """
        try:
            # Calculate component scores
            behavioral_features = features.get('behavioral_features', {})
            liquidity_features = features.get('liquidity_features', {})
            market_features = features.get('market_features', {})
            
            behavioral_score = self._calculate_behavioral_score(behavioral_features)
            liquidity_score = self._calculate_liquidity_score(liquidity_features)
            market_score = self._calculate_market_score(market_features)
            
            # Calculate final score
            final_score = self.calculate_score(features)
            
            explanation = {
                'wallet_address': features.get('wallet_address', 'unknown'),
                'final_risk_score': final_score,
                'risk_category': self._get_risk_category(final_score),
                'component_scores': {
                    'behavioral': {
                        'score': round(behavioral_score, 3),
                        'weight': self.category_weights['behavioral'],
                        'contribution': round(behavioral_score * self.category_weights['behavioral'], 3)
                    },
                    'liquidity': {
                        'score': round(liquidity_score, 3),
                        'weight': self.category_weights['liquidity'],
                        'contribution': round(liquidity_score * self.category_weights['liquidity'], 3)
                    },
                    'market': {
                        'score': round(market_score, 3),
                        'weight': self.category_weights['market'],
                        'contribution': round(market_score * self.category_weights['market'], 3)
                    }
                },
                'data_quality': features.get('data_quality', {}),
                'key_risk_factors': self._identify_key_risk_factors(features),
                'methodology': self._get_methodology_summary()
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating score explanation: {str(e)}")
            return {'error': str(e)}
    
    def _get_risk_category(self, score: int) -> str:
        """Convert numerical score to risk category"""
        if score <= 250:
            return 'Low Risk'
        elif score <= 750:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def _identify_key_risk_factors(self, features: Dict[str, Any]) -> List[str]:
        """Identify the key risk factors for this wallet"""
        risk_factors = []
        
        try:
            behavioral = features.get('behavioral_features', {})
            liquidity = features.get('liquidity_features', {})
            market = features.get('market_features', {})
            
            # Behavioral risk factors
            if behavioral.get('avg_tx_per_day', 0) > 10:
                risk_factors.append('High transaction frequency')
            
            if behavioral.get('compound_activity_ratio', 0) > 0.5:
                risk_factors.append('Very active in DeFi protocols')
            
            # Liquidity risk factors
            collateral_ratio = liquidity.get('collateral_ratio', 0)
            if 0 < collateral_ratio < 1.5:
                risk_factors.append('Low collateral ratio - liquidation risk')
            
            if liquidity.get('utilization_ratio', 0) > 0.8:
                risk_factors.append('High leverage usage')
            
            # Market risk factors
            if market.get('concentration_risk', 0) > 0.8:
                risk_factors.append('Concentrated portfolio - single asset dependency')
            
            if market.get('portfolio_risk_score', 0) > 0.7:
                risk_factors.append('High exposure to volatile assets')
            
            if not risk_factors:
                risk_factors.append('No significant risk factors identified')
                
        except Exception as e:
            logger.warning(f"Error identifying risk factors: {str(e)}")
            risk_factors.append('Error analyzing risk factors')
        
        return risk_factors
    
    def _get_methodology_summary(self) -> Dict[str, str]:
        """Return methodology summary for transparency"""
        return {
            'approach': 'Weighted Composite Scoring',
            'categories': 'Behavioral (40%), Liquidity (35%), Market (25%)',
            'scale': '0-1000 (Low: 0-250, Medium: 251-750, High: 751-1000)',
            'model_type': 'Rule-based statistical scoring (no ML training required)',
            'optimization': 'CPU-optimized for Windows systems'
        }


# Batch scoring utilities
def batch_score_wallets(wallets_features: Dict[str, Dict], config: Dict = None) -> Dict[str, int]: #type:ignore
    """
    Score multiple wallets in batch.
    
    Args:
        wallets_features (Dict[str, Dict]): Dictionary mapping wallet addresses to features
        config (Dict): Scoring configuration
        
    Returns:
        Dict[str, int]: Dictionary mapping wallet addresses to risk scores
    """
    scorer = WalletRiskScorer(config)
    scores = {}
    
    for wallet_address, features in wallets_features.items():
        try:
            score = scorer.calculate_score(features)
            scores[wallet_address] = score
        except Exception as e:
            logger.error(f"Error scoring wallet {wallet_address}: {str(e)}")
            scores[wallet_address] = scorer._get_default_risk_score()
    
    return scores


def generate_risk_distribution_analysis(scores: Dict[str, int]) -> Dict[str, Any]:
    """
    Generate statistical analysis of risk score distribution.
    
    Args:
        scores (Dict[str, int]): Dictionary of wallet scores
        
    Returns:
        Dict[str, Any]: Distribution analysis
    """
    try:
        score_values = list(scores.values())
        
        if not score_values:
            return {'error': 'No scores to analyze'}
        
        analysis = {
            'total_wallets': len(score_values),
            'score_statistics': {
                'mean': round(np.mean(score_values), 1),
                'median': round(np.median(score_values), 1),
                'std': round(np.std(score_values), 1),
                'min': int(np.min(score_values)),
                'max': int(np.max(score_values))
            },
            'risk_distribution': {
                'low_risk (0-250)': len([s for s in score_values if s <= 250]),
                'medium_risk (251-750)': len([s for s in score_values if 251 <= s <= 750]),
                'high_risk (751-1000)': len([s for s in score_values if s >= 751])
            },
            'percentiles': {
                '25th': round(np.percentile(score_values, 25), 1),
                '50th': round(np.percentile(score_values, 50), 1),
                '75th': round(np.percentile(score_values, 75), 1),
                '90th': round(np.percentile(score_values, 90), 1),
                '95th': round(np.percentile(score_values, 95), 1)
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in distribution analysis: {str(e)}")
        return {'error': str(e)}


def main():
    """Test the risk scoring module"""
    print("Testing Wallet Risk Scorer...")
    
    # Create sample features for testing
    sample_features = {
        'wallet_address': '0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7',
        'data_quality': {'data_completeness': 'high'},
        'behavioral_features': {
            'avg_tx_per_day': 5.2,
            'recent_activity_ratio': 0.3,
            'avg_tx_value': 2e18,
            'large_tx_ratio': 0.2,
            'compound_activity_ratio': 0.4,
            'compound_tx_count': 15,
            'burst_activity_score': 0.1,
            'tx_regularity': 0.6
        },
        'liquidity_features': {
            'collateral_ratio': 1.8,
            'liquidation_risk_score': 0.3,
            'utilization_ratio': 0.6,
            'leverage_score': 0.5
        },
        'market_features': {
            'portfolio_risk_score': 0.6,
            'concentration_risk': 0.4,
            'diversification_score': 0.7,
            'asset_diversification': 0.6
        }
    }
    
    # Initialize scorer
    scorer = WalletRiskScorer()
    
    # Calculate score
    risk_score = scorer.calculate_score(sample_features)
    
    # Get explanation
    explanation = scorer.get_score_explanation(sample_features)
    
    print(f"Risk Score: {risk_score}")
    print(f"Risk Category: {explanation.get('risk_category')}")
    print(f"Component Scores: {explanation.get('component_scores')}")
    print(f"Key Risk Factors: {explanation.get('key_risk_factors')}")


if __name__ == "__main__":
    main()
