"""
Risk Feature Engineering Module
==============================

Extracts behavioral, liquidity, and market risk features from Compound protocol
transaction data for wallet risk analysis.

Optimized for Windows systems with lightweight, CPU-efficient processing.

Author: AI Development Team
Date: July 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class RiskFeatureEngineer:
    """
    Extracts comprehensive risk features from DeFi transaction data.
    
    Features extracted:
    - Behavioral risk indicators (transaction patterns, frequency)
    - Liquidity risk features (collateral ratios, liquidation risk)
    - Market risk indicators (volatility exposure, concentration)
    """
    
    def __init__(self, config: Dict = None): #type:ignore
        """
        Initialize the feature engineering system.
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config or {}
        self.setup_config()
        
        # Feature weights for risk calculation
        self.feature_weights = {
            'behavioral': 0.40,
            'liquidity': 0.35,
            'market': 0.25
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'high_frequency_tx': 50,      # > 50 transactions = high activity
            'low_collateral_ratio': 1.5,  # < 150% collateral = risky
            'high_volatility_exposure': 0.7,  # > 70% in volatile assets
            'concentration_risk': 0.8,    # > 80% in single asset
            'liquidation_proximity': 0.1  # Within 10% of liquidation
        }
        
        logger.info("RiskFeatureEngineer initialized successfully")
    
    def setup_config(self):
        """Setup default configuration parameters"""
        self.config.setdefault('lookback_days', 365)
        self.config.setdefault('min_transactions', 5)
        self.config.setdefault('volatility_window', 30)
        
    def extract_features(self, wallet_data: Dict) -> Dict[str, Any]:
        """
        Main feature extraction function that processes wallet data.
        
        Args:
            wallet_data (Dict): Complete wallet data from data collector
            
        Returns:
            Dict[str, Any]: Extracted features for risk scoring
        """
        logger.info(f"Extracting features for wallet: {wallet_data.get('address', 'unknown')}")
        
        features = {
            'wallet_address': wallet_data.get('address', ''),
            'extraction_timestamp': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(wallet_data),
            'behavioral_features': {},
            'liquidity_features': {},
            'market_features': {},
            'summary_features': {}
        }
        
        try:
            # Extract behavioral features
            features['behavioral_features'] = self._extract_behavioral_features(wallet_data)
            
            # Extract liquidity features  
            features['liquidity_features'] = self._extract_liquidity_features(wallet_data)
            
            # Extract market features
            features['market_features'] = self._extract_market_features(wallet_data)
            
            # Create summary features
            features['summary_features'] = self._create_summary_features(features)
            
            logger.info(f"Feature extraction completed for {features['wallet_address']}")
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            features['error'] = str(e)
        
        return features
    
    def _assess_data_quality(self, wallet_data: Dict) -> Dict[str, Any]:
        """Assess the quality of input data"""
        quality = {
            'has_transactions': len(wallet_data.get('etherscan_transactions', [])) > 0,
            'has_compound_data': bool(wallet_data.get('compound_data', {})),
            'has_compound_transactions': len(wallet_data.get('compound_transactions', [])) > 0,
            'transaction_count': len(wallet_data.get('etherscan_transactions', [])),
            'compound_transaction_count': len(wallet_data.get('compound_transactions', [])),
            'data_completeness': 'high'
        }
        
        # Determine data completeness
        if not quality['has_transactions']:
            quality['data_completeness'] = 'none'
        elif not quality['has_compound_transactions'] and not quality['has_compound_data']:
            quality['data_completeness'] = 'low'
        elif quality['compound_transaction_count'] < self.config['min_transactions']:
            quality['data_completeness'] = 'medium'
        
        return quality
    
    def _extract_behavioral_features(self, wallet_data: Dict) -> Dict[str, float]:
        """Extract behavioral risk indicators"""
        features = {}
        
        # Get transaction data
        all_txs = wallet_data.get('etherscan_transactions', [])
        compound_txs = wallet_data.get('compound_transactions', [])
        
        if not all_txs:
            return self._get_default_behavioral_features()
        
        # Convert to DataFrame for easier analysis
        df_all = pd.DataFrame(all_txs)
        
        if len(df_all) > 0:
            # Convert timestamp if it's string
            if 'timeStamp' in df_all.columns:
                df_all['timestamp'] = pd.to_numeric(df_all['timeStamp'], errors='coerce')
                df_all['datetime'] = pd.to_datetime(df_all['timestamp'], unit='s', errors='coerce')
            
            # 1. Transaction Frequency Analysis
            features.update(self._analyze_transaction_frequency(df_all))
            
            # 2. Transaction Size Analysis  
            features.update(self._analyze_transaction_sizes(df_all))
            
            # 3. Compound-specific behavior
            if compound_txs:
                df_compound = pd.DataFrame(compound_txs)
                features.update(self._analyze_compound_behavior(df_compound))
            
            # 4. Temporal patterns
            features.update(self._analyze_temporal_patterns(df_all))
        
        return features
    
    def _analyze_transaction_frequency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze transaction frequency patterns"""
        features = {}
        
        try:
            # Basic frequency metrics
            features['total_transactions'] = len(df)
            
            # Calculate time span
            if 'datetime' in df.columns and len(df) > 1:
                df_sorted = df.sort_values('datetime')
                time_span_days = (df_sorted['datetime'].iloc[-1] - df_sorted['datetime'].iloc[0]).days
                time_span_days = max(1, time_span_days)  # Avoid division by zero
                
                features['time_span_days'] = time_span_days
                features['avg_tx_per_day'] = features['total_transactions'] / time_span_days
                features['tx_frequency_score'] = min(1.0, features['avg_tx_per_day'] / 5.0)  # Normalize to 0-1
                
                # Recent activity (last 30 days)
                recent_cutoff = datetime.now() - timedelta(days=30)
                recent_txs = df[df['datetime'] > recent_cutoff] if 'datetime' in df.columns else df.iloc[-30:]
                features['recent_tx_count'] = len(recent_txs)
                features['recent_activity_ratio'] = features['recent_tx_count'] / max(1, features['total_transactions'])
            else:
                features['time_span_days'] = 1
                features['avg_tx_per_day'] = features['total_transactions']
                features['tx_frequency_score'] = 0.0
                features['recent_tx_count'] = features['total_transactions']
                features['recent_activity_ratio'] = 1.0
                
        except Exception as e:
            logger.warning(f"Error in frequency analysis: {str(e)}")
            features.update(self._get_default_frequency_features())
        
        return features
    
    def _analyze_transaction_sizes(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze transaction size patterns"""
        features = {}
        
        try:
            if 'value' in df.columns:
                df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
                values = df['value_numeric'][df['value_numeric'] > 0]
                
                if len(values) > 0:
                    features['avg_tx_value'] = float(values.mean())
                    features['median_tx_value'] = float(values.median())
                    features['max_tx_value'] = float(values.max())
                    features['tx_value_std'] = float(values.std())
                    features['large_tx_ratio'] = float((values > values.quantile(0.9)).sum() / len(values))
                else:
                    features.update(self._get_default_size_features())
            else:
                features.update(self._get_default_size_features())
                
        except Exception as e:
            logger.warning(f"Error in size analysis: {str(e)}")
            features.update(self._get_default_size_features())
        
        return features
    
    def _analyze_compound_behavior(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze Compound-specific behavioral patterns"""
        features = {}
        
        try:
            features['compound_tx_count'] = len(df)
            features['compound_activity_ratio'] = min(1.0, features['compound_tx_count'] / 20.0)
            
            # Analyze transaction types based on input data
            if 'input' in df.columns:
                # Simple heuristic for transaction type classification
                supply_like = df['input'].str.len() > 10  # Transactions with data
                features['supply_tx_ratio'] = float(supply_like.sum() / len(df)) if len(df) > 0 else 0.0
            else:
                features['supply_tx_ratio'] = 0.5  # Default assumption
            
            # Gas usage patterns (risk indicator)
            if 'gasUsed' in df.columns:
                df['gas_numeric'] = pd.to_numeric(df['gasUsed'], errors='coerce').fillna(0)
                features['avg_gas_used'] = float(df['gas_numeric'].mean())
                features['high_gas_tx_ratio'] = float((df['gas_numeric'] > df['gas_numeric'].quantile(0.8)).sum() / len(df))
            else:
                features['avg_gas_used'] = 100000.0  # Default gas usage
                features['high_gas_tx_ratio'] = 0.2
                
        except Exception as e:
            logger.warning(f"Error in Compound behavior analysis: {str(e)}")
            features.update(self._get_default_compound_features())
        
        return features
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze temporal transaction patterns"""
        features = {}
        
        try:
            if 'datetime' in df.columns and len(df) > 1:
                df_sorted = df.sort_values('datetime')
                
                # Calculate time gaps between transactions
                time_diffs = df_sorted['datetime'].diff().dt.total_seconds() / 3600  # Hours
                time_diffs = time_diffs.dropna()
                
                if len(time_diffs) > 0:
                    features['avg_time_between_tx'] = float(time_diffs.mean())
                    features['tx_regularity'] = float(1.0 / (1.0 + time_diffs.std() / max(time_diffs.mean(), 1)))
                    features['burst_activity_score'] = float((time_diffs < 1.0).sum() / len(time_diffs))  # Transactions within 1 hour
                else:
                    features.update(self._get_default_temporal_features())
            else:
                features.update(self._get_default_temporal_features())
                
        except Exception as e:
            logger.warning(f"Error in temporal analysis: {str(e)}")
            features.update(self._get_default_temporal_features())
        
        return features
    
    def _extract_liquidity_features(self, wallet_data: Dict) -> Dict[str, float]:
        """Extract liquidity risk features"""
        features = {}
        
        compound_data = wallet_data.get('compound_data', {})
        compound_txs = wallet_data.get('compound_transactions', [])
        
        if compound_data and 'tokens' in compound_data:
            # Analyze position data from The Graph
            features.update(self._analyze_compound_positions(compound_data))
        else:
            # Fallback to transaction-based analysis
            features.update(self._analyze_liquidity_from_transactions(compound_txs))
        
        return features
    
    def _analyze_compound_positions(self, compound_data: Dict) -> Dict[str, float]:
        """Analyze liquidity risk from Compound position data"""
        features = {}
        
        try:
            tokens = compound_data.get('tokens', [])
            
            if tokens:
                total_supplied = 0
                total_borrowed = 0
                position_count = 0
                
                for token in tokens:
                    supplied = float(token.get('totalUnderlyingSupplied', 0))
                    borrowed = float(token.get('totalUnderlyingBorrowed', 0))
                    
                    total_supplied += supplied
                    total_borrowed += borrowed
                    
                    if supplied > 0 or borrowed > 0:
                        position_count += 1
                
                # Calculate key liquidity metrics
                features['total_supplied_value'] = total_supplied
                features['total_borrowed_value'] = total_borrowed
                features['net_position'] = total_supplied - total_borrowed
                features['position_count'] = position_count
                
                # Utilization ratio
                if total_supplied > 0:
                    features['utilization_ratio'] = total_borrowed / total_supplied
                    features['collateral_ratio'] = total_supplied / max(total_borrowed, 1)
                else:
                    features['utilization_ratio'] = 0.0
                    features['collateral_ratio'] = 0.0
                
                # Risk scoring
                features['liquidation_risk_score'] = self._calculate_liquidation_risk(features['collateral_ratio'])
                features['leverage_score'] = min(1.0, features['utilization_ratio'])
                
            else:
                features.update(self._get_default_liquidity_features())
                
        except Exception as e:
            logger.warning(f"Error in position analysis: {str(e)}")
            features.update(self._get_default_liquidity_features())
        
        return features
    
    def _analyze_liquidity_from_transactions(self, compound_txs: List[Dict]) -> Dict[str, float]:
        """Fallback liquidity analysis from transactions"""
        features = {}
        
        try:
            if compound_txs:
                # Estimate activity level
                tx_count = len(compound_txs)
                features['estimated_activity_level'] = min(1.0, tx_count / 20.0)
                features['estimated_risk_level'] = 0.5  # Medium risk assumption
                
                # Analyze transaction values if available
                if any('value' in tx for tx in compound_txs):
                    values = [float(tx.get('value', 0)) for tx in compound_txs]
                    avg_value = np.mean(values) if values else 0
                    features['avg_compound_tx_value'] = avg_value
                    features['estimated_position_size'] = avg_value * tx_count
                else:
                    features['avg_compound_tx_value'] = 0.0
                    features['estimated_position_size'] = 0.0
            else:
                features.update(self._get_default_liquidity_features())
                
        except Exception as e:
            logger.warning(f"Error in transaction-based liquidity analysis: {str(e)}")
            features.update(self._get_default_liquidity_features())
        
        return features
    
    def _calculate_liquidation_risk(self, collateral_ratio: float) -> float:
        """Calculate liquidation risk score based on collateral ratio"""
        if collateral_ratio == 0:
            return 1.0  # Maximum risk
        elif collateral_ratio < 1.2:  # Less than 120% collateralized
            return 0.9
        elif collateral_ratio < 1.5:  # Less than 150% collateralized
            return 0.7
        elif collateral_ratio < 2.0:  # Less than 200% collateralized
            return 0.4
        else:
            return 0.1  # Low risk
    
    def _extract_market_features(self, wallet_data: Dict) -> Dict[str, float]:
        """Extract market risk features"""
        features = {}
        
        try:
            compound_data = wallet_data.get('compound_data', {})
            
            if compound_data and 'tokens' in compound_data:
                features.update(self._analyze_asset_exposure(compound_data['tokens']))
            
            # Analyze transaction timing patterns
            all_txs = wallet_data.get('etherscan_transactions', [])
            if all_txs:
                features.update(self._analyze_market_timing(all_txs))
            
            # Portfolio diversification
            features.update(self._analyze_diversification(compound_data))
            
        except Exception as e:
            logger.warning(f"Error in market feature extraction: {str(e)}")
            features.update(self._get_default_market_features())
        
        return features
    
    def _analyze_asset_exposure(self, tokens: List[Dict]) -> Dict[str, float]:
        """Analyze exposure to different asset types"""
        features = {}
        
        try:
            # Define asset categories and their risk levels
            asset_risk_mapping = {
                'ETH': 0.6, 'WETH': 0.6,
                'DAI': 0.2, 'USDC': 0.2, 'USDT': 0.3,
                'WBTC': 0.7, 'BTC': 0.7,
                'COMP': 0.8, 'UNI': 0.8,
                'LINK': 0.7, 'AAVE': 0.8
            }
            
            total_exposure = 0
            weighted_risk = 0
            asset_count = 0
            max_single_exposure = 0
            
            for token in tokens:
                symbol = token.get('symbol', '').upper()
                supplied = float(token.get('totalUnderlyingSupplied', 0))
                borrowed = float(token.get('totalUnderlyingBorrowed', 0))
                exposure = supplied + borrowed
                
                if exposure > 0:
                    total_exposure += exposure
                    asset_count += 1
                    max_single_exposure = max(max_single_exposure, exposure)
                    
                    # Apply risk weighting
                    risk_weight = asset_risk_mapping.get(symbol, 0.5)  # Default medium risk
                    weighted_risk += exposure * risk_weight
            
            if total_exposure > 0:
                features['portfolio_risk_score'] = weighted_risk / total_exposure
                features['concentration_risk'] = max_single_exposure / total_exposure
                features['asset_diversification'] = min(1.0, asset_count / 5.0)  # Normalize to 0-1
            else:
                features['portfolio_risk_score'] = 0.0
                features['concentration_risk'] = 0.0
                features['asset_diversification'] = 0.0
            
            features['active_asset_count'] = asset_count
            
        except Exception as e:
            logger.warning(f"Error in asset exposure analysis: {str(e)}")
            features.update(self._get_default_exposure_features())
        
        return features
    
    def _analyze_market_timing(self, transactions: List[Dict]) -> Dict[str, float]:
        """Analyze market timing patterns"""
        features = {}
        
        try:
            if len(transactions) > 1:
                # Convert to DataFrame
                df = pd.DataFrame(transactions)
                if 'timeStamp' in df.columns:
                    df['timestamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    
                    # Analyze activity during different time periods
                    # This is a simplified analysis - in production, you'd correlate with market data
                    df['hour'] = df['datetime'].dt.hour
                    df['day_of_week'] = df['datetime'].dt.dayofweek
                    
                    # Market hours activity (rough proxy for market timing)
                    market_hours = (df['hour'] >= 9) & (df['hour'] <= 16)  # 9 AM - 4 PM
                    features['market_hours_activity'] = float(market_hours.sum() / len(df))
                    
                    # Weekend activity
                    weekend_activity = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
                    features['weekend_activity_ratio'] = float(weekend_activity.sum() / len(df))
                    
                    # Recent activity trend (last 30 days vs previous 30 days)
                    recent_cutoff = datetime.now() - timedelta(days=30)
                    older_cutoff = datetime.now() - timedelta(days=60)
                    
                    recent_txs = df[df['datetime'] > recent_cutoff]
                    older_txs = df[(df['datetime'] > older_cutoff) & (df['datetime'] <= recent_cutoff)]
                    
                    if len(older_txs) > 0:
                        features['activity_trend'] = len(recent_txs) / max(len(older_txs), 1)
                    else:
                        features['activity_trend'] = 1.0
                else:
                    features.update(self._get_default_timing_features())
            else:
                features.update(self._get_default_timing_features())
                
        except Exception as e:
            logger.warning(f"Error in market timing analysis: {str(e)}")
            features.update(self._get_default_timing_features())
        
        return features
    
    def _analyze_diversification(self, compound_data: Dict) -> Dict[str, float]:
        """Analyze portfolio diversification"""
        features = {}
        
        try:
            if compound_data and 'tokens' in compound_data:
                tokens = compound_data['tokens']
                active_tokens = [t for t in tokens 
                               if float(t.get('totalUnderlyingSupplied', 0)) > 0 
                               or float(t.get('totalUnderlyingBorrowed', 0)) > 0]
                
                features['diversification_score'] = min(1.0, len(active_tokens) / 5.0)
                features['single_asset_dependency'] = 1.0 / max(len(active_tokens), 1)
            else:
                features['diversification_score'] = 0.0
                features['single_asset_dependency'] = 1.0
                
        except Exception as e:
            logger.warning(f"Error in diversification analysis: {str(e)}")
            features['diversification_score'] = 0.0
            features['single_asset_dependency'] = 1.0
        
        return features
    
    def _create_summary_features(self, features: Dict) -> Dict[str, float]:
        """Create high-level summary features for risk scoring"""
        summary = {}
        
        try:
            behavioral = features.get('behavioral_features', {})
            liquidity = features.get('liquidity_features', {})
            market = features.get('market_features', {})
            
            # Behavioral risk summary
            tx_frequency_risk = min(1.0, behavioral.get('avg_tx_per_day', 0) / 10.0)
            activity_risk = behavioral.get('compound_activity_ratio', 0)
            summary['behavioral_risk_score'] = (tx_frequency_risk + activity_risk) / 2
            
            # Liquidity risk summary
            liquidation_risk = liquidity.get('liquidation_risk_score', 0.5)
            leverage_risk = liquidity.get('leverage_score', 0)
            summary['liquidity_risk_score'] = (liquidation_risk + leverage_risk) / 2
            
            # Market risk summary  
            portfolio_risk = market.get('portfolio_risk_score', 0.5)
            concentration_risk = market.get('concentration_risk', 0)
            summary['market_risk_score'] = (portfolio_risk + concentration_risk) / 2
            
            # Overall risk score (0-1 scale)
            summary['overall_risk_score'] = (
                summary['behavioral_risk_score'] * self.feature_weights['behavioral'] +
                summary['liquidity_risk_score'] * self.feature_weights['liquidity'] +
                summary['market_risk_score'] * self.feature_weights['market']
            )
            
            # Risk category
            if summary['overall_risk_score'] < 0.3:
                summary['risk_category'] = 'low'
            elif summary['overall_risk_score'] < 0.7:
                summary['risk_category'] = 'medium'
            else:
                summary['risk_category'] = 'high'
                
        except Exception as e:
            logger.warning(f"Error creating summary features: {str(e)}")
            summary = self._get_default_summary_features()
        
        return summary
    
    # Default feature methods for error handling
    def _get_default_behavioral_features(self) -> Dict[str, float]:
        """Return default behavioral features when data is insufficient"""
        return {
            'total_transactions': 0, 'avg_tx_per_day': 0, 'tx_frequency_score': 0,
            'recent_tx_count': 0, 'recent_activity_ratio': 0, 'avg_tx_value': 0,
            'median_tx_value': 0, 'max_tx_value': 0, 'tx_value_std': 0,
            'large_tx_ratio': 0, 'compound_tx_count': 0, 'compound_activity_ratio': 0,
            'supply_tx_ratio': 0.5, 'avg_gas_used': 100000, 'high_gas_tx_ratio': 0.2,
            'avg_time_between_tx': 24, 'tx_regularity': 0.5, 'burst_activity_score': 0
        }
    
    def _get_default_frequency_features(self) -> Dict[str, float]:
        return {'total_transactions': 0, 'avg_tx_per_day': 0, 'tx_frequency_score': 0,
                'recent_tx_count': 0, 'recent_activity_ratio': 0}
    
    def _get_default_size_features(self) -> Dict[str, float]:
        return {'avg_tx_value': 0, 'median_tx_value': 0, 'max_tx_value': 0,
                'tx_value_std': 0, 'large_tx_ratio': 0}
    
    def _get_default_compound_features(self) -> Dict[str, float]:
        return {'compound_tx_count': 0, 'compound_activity_ratio': 0, 'supply_tx_ratio': 0.5,
                'avg_gas_used': 100000, 'high_gas_tx_ratio': 0.2}
    
    def _get_default_temporal_features(self) -> Dict[str, float]:
        return {'avg_time_between_tx': 24, 'tx_regularity': 0.5, 'burst_activity_score': 0}
    
    def _get_default_liquidity_features(self) -> Dict[str, float]:
        return {'utilization_ratio': 0, 'collateral_ratio': 0, 'liquidation_risk_score': 0.5,
                'leverage_score': 0, 'total_supplied_value': 0, 'total_borrowed_value': 0}
    
    def _get_default_market_features(self) -> Dict[str, float]:
        return {'portfolio_risk_score': 0.5, 'concentration_risk': 0, 'asset_diversification': 0,
                'market_hours_activity': 0.5, 'weekend_activity_ratio': 0.3, 'activity_trend': 1.0,
                'diversification_score': 0, 'single_asset_dependency': 1.0}
    
    def _get_default_exposure_features(self) -> Dict[str, float]:
        return {'portfolio_risk_score': 0.5, 'concentration_risk': 0,
                'asset_diversification': 0, 'active_asset_count': 0}
    
    def _get_default_timing_features(self) -> Dict[str, float]:
        return {'market_hours_activity': 0.5, 'weekend_activity_ratio': 0.3, 'activity_trend': 1.0}
    
    def _get_default_summary_features(self) -> Dict[str, float]:
        return {'behavioral_risk_score': 0.3, 'liquidity_risk_score': 0.3,
                'market_risk_score': 0.3, 'overall_risk_score': 0.3, 'risk_category': 'medium'} #type: ignore


def main():
    """Test the feature engineering module"""
    print("Testing Risk Feature Engineering...")
    
    # Create sample wallet data for testing
    sample_wallet_data = {
        'address': '0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7',
        'etherscan_transactions': [
            {'timeStamp': '1640995200', 'value': '1000000000000000000', 'gasUsed': '21000'},
            {'timeStamp': '1641081600', 'value': '2000000000000000000', 'gasUsed': '25000'}
        ],
        'compound_transactions': [
            {'timeStamp': '1640995200', 'value': '500000000000000000', 'gasUsed': '150000'},
        ],
        'compound_data': {
            'tokens': [
                {'symbol': 'cDAI', 'totalUnderlyingSupplied': '1000', 'totalUnderlyingBorrowed': '500'}
            ]
        }
    }
    
    # Initialize feature engineer
    engineer = RiskFeatureEngineer()
    
    # Extract features
    features = engineer.extract_features(sample_wallet_data)
    
    print("Feature extraction completed!")
    print(f"Behavioral features: {len(features.get('behavioral_features', {}))}")
    print(f"Liquidity features: {len(features.get('liquidity_features', {}))}")
    print(f"Market features: {len(features.get('market_features', {}))}")
    print(f"Overall risk score: {features.get('summary_features', {}).get('overall_risk_score', 'N/A')}")


if __name__ == "__main__":
    main()
