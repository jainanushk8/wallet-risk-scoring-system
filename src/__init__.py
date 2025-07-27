"""
Wallet Risk Scoring System
==========================

A comprehensive system for analyzing DeFi wallet risk based on 
Compound protocol transaction history and behavioral patterns.

Author: AI Development Team
Version: 1.0.0
Date: July 2025
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"
__email__ = "development@walletrisks.com"

# Core modules
from .data_collector import CompoundDataCollector
from .feature_engineer import RiskFeatureEngineer  
from .risk_scorer import WalletRiskScorer
from .utils import (
    setup_logging,
    validate_wallet_address,
    load_config,
    save_results
)

# Package-level constants
MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 1000
SUPPORTED_PROTOCOLS = ['compound-v2', 'compound-v3']

# Default configuration
DEFAULT_CONFIG = {
    'api_rate_limit': 5,
    'max_retries': 3,
    'timeout_seconds': 30,
    'lookback_days': 365,
    'min_transactions': 5
}

def get_version():
    """Return the current version of the package."""
    return __version__

def get_supported_protocols():
    """Return list of supported DeFi protocols."""
    return SUPPORTED_PROTOCOLS.copy()

class WalletRiskScoringSystem:
    """
    Main system class that orchestrates the entire wallet risk scoring process.
    
    This class provides a unified interface for:
    - Data collection from Compound protocol
    - Feature engineering from transaction data
    - Risk score calculation and validation
    - Result export and reporting
    """
    
    def __init__(self, config=None):
        """
        Initialize the wallet risk scoring system.
        
        Args:
            config (dict, optional): Configuration parameters. 
                                   Uses DEFAULT_CONFIG if not provided.
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.data_collector = None
        self.feature_engineer = None
        self.risk_scorer = None
        
    def initialize_components(self):
        """Initialize all system components with current configuration."""
        self.data_collector = CompoundDataCollector(self.config)
        self.feature_engineer = RiskFeatureEngineer(self.config)
        self.risk_scorer = WalletRiskScorer(self.config)
        
    def process_wallets(self, wallet_addresses):
        """
        Process a list of wallet addresses and return risk scores.
        
        Args:
            wallet_addresses (list): List of Ethereum wallet addresses
            
        Returns:
            dict: Dictionary mapping wallet addresses to risk scores
        """
        if not self.data_collector:
            self.initialize_components()
            
        results = {}
        
        for address in wallet_addresses:
            try:
                # Collect transaction data
                tx_data = self.data_collector.fetch_wallet_data(address)
                
                # Extract features
                features = self.feature_engineer.extract_features(tx_data)
                
                # Calculate risk score
                risk_score = self.risk_scorer.calculate_score(features)
                
                results[address] = risk_score
                
            except Exception as e:
                print(f"Error processing wallet {address}: {str(e)}")
                results[address] = None
                
        return results

# Module-level convenience functions
def quick_score_wallet(wallet_address, config=None):
    """
    Quick function to score a single wallet address.
    
    Args:
        wallet_address (str): Ethereum wallet address
        config (dict, optional): Configuration parameters
        
    Returns:
        int: Risk score (0-1000) or None if error
    """
    system = WalletRiskScoringSystem(config)
    results = system.process_wallets([wallet_address])
    return results.get(wallet_address)

def batch_score_wallets(wallet_addresses, config=None):
    """
    Batch function to score multiple wallet addresses.
    
    Args:
        wallet_addresses (list): List of Ethereum wallet addresses
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Dictionary mapping addresses to risk scores
    """
    system = WalletRiskScoringSystem(config)
    return system.process_wallets(wallet_addresses)

# Package metadata for introspection
__all__ = [
    'WalletRiskScoringSystem',
    'CompoundDataCollector',
    'RiskFeatureEngineer',
    'WalletRiskScorer',
    'quick_score_wallet',
    'batch_score_wallets',
    'get_version',
    'get_supported_protocols',
    'MIN_RISK_SCORE',
    'MAX_RISK_SCORE',
    'DEFAULT_CONFIG'