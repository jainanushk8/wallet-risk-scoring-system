"""
Utility Functions for Wallet Risk Scoring System
===============================================

Common utilities for data validation, configuration, logging, and file operations.
Optimized for Windows systems and production use.

Author: AI Development Team
Date: July 2025
"""

import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import hashlib

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger: #type: ignore
    """
    Setup logging configuration for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup logging configuration
    logging_config = {
        'level': getattr(logging, log_level.upper(), logging.INFO),
        'format': log_format,
        'datefmt': date_format
    }
    
    if log_file:
        log_path = log_dir / log_file
        logging_config['filename'] = str(log_path)
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)
    
    # Create and return logger
    logger = logging.getLogger('WalletRiskScoring')
    logger.info("Logging initialized successfully")
    
    return logger

def validate_wallet_address(address: str) -> bool:
    """
    Validate Ethereum wallet address format.
    
    Args:
        address (str): Ethereum address to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(address, str):
        return False
    
    # Remove '0x' prefix if present
    if address.startswith('0x'):
        address = address[2:]
    
    # Check if address is 40 characters long and contains only hex characters
    if len(address) != 40:
        return False
    
    # Check if all characters are valid hexadecimal
    try:
        int(address, 16)
        return True
    except ValueError:
        return False

def validate_wallet_list(wallet_addresses: List[str]) -> Dict[str, Any]:
    """
    Validate a list of wallet addresses and return validation results.
    
    Args:
        wallet_addresses (List[str]): List of wallet addresses to validate
        
    Returns:
        Dict[str, Any]: Validation results including valid/invalid addresses
    """
    valid_addresses = []
    invalid_addresses = []
    
    for address in wallet_addresses:
        if validate_wallet_address(address):
            # Normalize address format (add 0x prefix, lowercase)
            normalized = f"0x{address.lower().replace('0x', '')}"
            valid_addresses.append(normalized)
        else:
            invalid_addresses.append(address)
    
    return {
        'total_addresses': len(wallet_addresses),
        'valid_addresses': valid_addresses,
        'invalid_addresses': invalid_addresses,
        'valid_count': len(valid_addresses),
        'invalid_count': len(invalid_addresses),
        'validation_rate': len(valid_addresses) / len(wallet_addresses) if wallet_addresses else 0
    }

def load_config(config_path: str = ".env") -> Dict[str, Any]:
    """
    Load configuration from environment file and environment variables.
    
    Args:
        config_path (str): Path to environment configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    from dotenv import load_dotenv
    
    # Load environment variables from file
    if Path(config_path).exists():
        load_dotenv(config_path)
    
    # Build configuration dictionary
    config = {
        # API Configuration
        'etherscan_api_key': os.getenv('ETHERSCAN_API_KEY', ''),
        'compound_api_url': os.getenv('COMPOUND_API_URL', 'https://api.compound.finance/api/v2'),
        'the_graph_url': os.getenv('THE_GRAPH_URL', 'https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2'),
        
        # Rate Limiting
        'api_rate_limit': int(os.getenv('API_RATE_LIMIT', '5')),
        'max_retries': int(os.getenv('MAX_RETRIES', '3')),
        'timeout_seconds': int(os.getenv('TIMEOUT_SECONDS', '30')),
        
        # Risk Scoring
        'min_risk_score': int(os.getenv('MIN_RISK_SCORE', '0')),
        'max_risk_score': int(os.getenv('MAX_RISK_SCORE', '1000')),
        
        # Data Collection
        'lookback_days': int(os.getenv('LOOKBACK_DAYS', '365')),
        'min_transactions': int(os.getenv('MIN_TRANSACTIONS', '5'))
    }
    
    return config

def save_results(data: Union[Dict, pd.DataFrame], filename: str, format: str = 'csv') -> bool:
    """
    Save results to file in specified format.
    
    Args:
        data: Data to save (Dictionary or DataFrame)
        filename (str): Output filename
        format (str): Output format ('csv', 'json', 'excel')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        if format.lower() == 'csv':
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            df.to_csv(filepath, index=False)
            
        elif format.lower() == 'json':
            if isinstance(data, pd.DataFrame):
                data_to_save = data.to_dict('records')
            else:
                data_to_save = data
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str)
                
        elif format.lower() == 'excel':
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            df.to_excel(filepath, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved successfully to: {filepath}")
        return True
        
    except Exception as e:
        print(f"Failed to save results: {str(e)}")
        return False

def calculate_file_hash(filepath: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of a file for integrity checking.
    
    Args:
        filepath: Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {str(e)}")
        return ""

def normalize_transaction_data(transactions: List[Dict]) -> pd.DataFrame:
    """
    Normalize transaction data into a consistent DataFrame format.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries
        
    Returns:
        pd.DataFrame: Normalized transaction data
    """
    if not transactions:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Standardize column names
    column_mapping = {
        'timeStamp': 'timestamp',
        'blockNumber': 'block_number',
        'hash': 'tx_hash',
        'from': 'from_address',
        'to': 'to_address',
        'value': 'value_wei',
        'gas': 'gas_limit',
        'gasUsed': 'gas_used',
        'gasPrice': 'gas_price',
        'isError': 'is_error'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert data types
    numeric_columns = ['timestamp', 'block_number', 'value_wei', 'gas_limit', 'gas_used', 'gas_price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Add derived columns
    if 'value_wei' in df.columns:
        df['value_eth'] = df['value_wei'] / 1e18
    
    if 'gas_used' in df.columns and 'gas_price' in df.columns:
        df['gas_cost_wei'] = df['gas_used'] * df['gas_price']
        df['gas_cost_eth'] = df['gas_cost_wei'] / 1e18
    
    return df

def create_data_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive summary of collected data.
    
    Args:
        data (Dict[str, Any]): Raw data dictionary
        
    Returns:
        Dict[str, Any]: Data summary statistics
    """
    summary = {
        'collection_timestamp': datetime.now().isoformat(),
        'total_wallets': 0,
        'wallets_with_data': 0,
        'wallets_with_compound_activity': 0,
        'total_transactions': 0,
        'total_compound_transactions': 0,
        'data_quality_metrics': {},
        'error_analysis': {}
    }
    
    if 'wallets_data' in data:
        wallet_data = data['wallets_data']
        summary['total_wallets'] = len(wallet_data)
        
        errors = []
        for wallet_address, wallet_info in wallet_data.items():
            if isinstance(wallet_info, dict):
                if 'error' not in wallet_info:
                    summary['wallets_with_data'] += 1
                    
                    # Count transactions
                    tx_count = len(wallet_info.get('etherscan_transactions', []))
                    compound_tx_count = len(wallet_info.get('compound_transactions', []))
                    
                    summary['total_transactions'] += tx_count
                    summary['total_compound_transactions'] += compound_tx_count
                    
                    # Check for Compound activity
                    has_compound = (compound_tx_count > 0 or 
                                  bool(wallet_info.get('compound_data', {})))
                    if has_compound:
                        summary['wallets_with_compound_activity'] += 1
                else:
                    errors.append(wallet_info['error'])
        
        # Data quality metrics
        if summary['total_wallets'] > 0:
            summary['data_quality_metrics'] = {
                'data_collection_rate': summary['wallets_with_data'] / summary['total_wallets'],
                'compound_activity_rate': summary['wallets_with_compound_activity'] / summary['total_wallets'],
                'avg_transactions_per_wallet': summary['total_transactions'] / max(summary['wallets_with_data'], 1),
                'avg_compound_transactions_per_wallet': summary['total_compound_transactions'] / max(summary['wallets_with_compound_activity'], 1)
            }
        
        # Error analysis
        if errors:
            from collections import Counter
            error_counts = Counter(errors)
            summary['error_analysis'] = {
                'total_errors': len(errors),
                'unique_errors': len(error_counts),
                'most_common_errors': error_counts.most_common(5)
            }
    
    return summary

def format_large_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B).
    
    Args:
        number: Number to format
        precision (int): Decimal precision
        
    Returns:
        str: Formatted number string
    """
    if abs(number) >= 1e9:
        return f"{number / 1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number / 1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number / 1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"

def validate_api_response(response: Dict, required_fields: List[str]) -> bool:
    """
    Validate API response contains required fields.
    
    Args:
        response (Dict): API response dictionary
        required_fields (List[str]): List of required field names
        
    Returns:
        bool: True if all required fields are present
    """
    if not isinstance(response, dict):
        return False
    
    for field in required_fields:
        if field not in response:
            return False
    
    return True

def clean_wallet_address(address: str) -> str:
    """
    Clean and normalize wallet address format.
    
    Args:
        address (str): Raw wallet address
        
    Returns:
        str: Cleaned and normalized address
    """
    if not address:
        return ""
    
    # Remove whitespace and convert to lowercase
    address = address.strip().lower()
    
    # Add 0x prefix if missing
    if not address.startswith('0x'):
        address = '0x' + address
    
    return address

def get_project_info() -> Dict[str, str]:
    """
    Get project information and metadata.
    
    Returns:
        Dict[str, str]: Project information dictionary
    """
    return {
        'project_name': 'Wallet Risk Scoring System',
        'version': '1.0.0',
        'author': 'AI Development Team',
        'description': 'DeFi wallet risk analysis based on Compound protocol interactions',
        'supported_protocols': 'Compound V2/V3',
        'created_date': '2025-07-27',
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}" #type: ignore
    }

def check_system_requirements() -> Dict[str, Any]:
    """
    Check system requirements and dependencies.
    
    Returns:
        Dict[str, Any]: System requirements check results
    """
    import sys
    import platform
    
    requirements = {
        'system_info': {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'python_version': sys.version,
            'python_executable': sys.executable
        },
        'directory_structure': {},
        'dependencies': {},
        'recommendations': []
    }
    
    # Check directory structure
    required_dirs = ['src', 'data', 'results', 'logs']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        requirements['directory_structure'][dir_name] = {
            'exists': dir_path.exists(),
            'is_directory': dir_path.is_dir() if dir_path.exists() else False,
            'path': str(dir_path.absolute())
        }
        
        if not dir_path.exists():
            requirements['recommendations'].append(f"Create {dir_name}/ directory")
    
    # Check key files
    key_files = ['.env', 'requirements.txt', 'data/wallet_list.csv']
    requirements['key_files'] = {}
    for file_name in key_files:
        file_path = Path(file_name)
        requirements['key_files'][file_name] = {
            'exists': file_path.exists(),
            'size_bytes': file_path.stat().st_size if file_path.exists() else 0
        }
        
        if not file_path.exists():
            requirements['recommendations'].append(f"Create {file_name} file")
    
    # Check Python dependencies - FIXED VERSION
    required_packages = [
        ('pandas', 'pandas'),
        ('requests', 'requests'), 
        ('web3', 'web3'),
        ('python-dotenv', 'dotenv'),  # Fixed: import name is 'dotenv', not 'python_dotenv'
        ('tqdm', 'tqdm'),
        ('numpy', 'numpy')
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            requirements['dependencies'][package_name] = {'installed': True, 'version': 'unknown'}
        except ImportError:
            requirements['dependencies'][package_name] = {'installed': False, 'version': None}
            requirements['recommendations'].append(f"Install {package_name}: pip install {package_name}")
    
    return requirements


# Windows-specific utilities
def get_windows_safe_filename(filename: str) -> str:
    """
    Convert filename to Windows-safe format.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Windows-safe filename
    """
    # Remove invalid characters for Windows
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed_file'
    
    return filename

def create_progress_callback(total_items: int, description: str = "Processing"):
    """
    Create a progress callback function for long-running operations.
    
    Args:
        total_items (int): Total number of items to process
        description (str): Description for progress bar
        
    Returns:
        callable: Progress callback function
    """
    from tqdm import tqdm
    
    pbar = tqdm(total=total_items, desc=description)
    
    def callback(current_item: int = None, message: str = None): # type: ignore
        if current_item is not None:
            pbar.n = current_item
        else:
            pbar.update(1)
        
        if message:
            pbar.set_postfix_str(message)
        
        pbar.refresh()

    callback.close = pbar.close #type: ignore
    return callback

# Error handling utilities
class WalletRiskScoringError(Exception):
    """Custom exception for wallet risk scoring errors."""
    pass

class DataCollectionError(WalletRiskScoringError):
    """Exception raised during data collection."""
    pass

class FeatureEngineeringError(WalletRiskScoringError):
    """Exception raised during feature engineering."""
    pass

class RiskScoringError(WalletRiskScoringError):
    """Exception raised during risk scoring."""
    pass

def handle_error(func):
    """
    Decorator for consistent error handling.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        callable: Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger('WalletRiskScoring')
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise WalletRiskScoringError(f"Error in {func.__name__}: {str(e)}") from e
    
    return wrapper

# Configuration validation
def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Required configuration keys
    required_keys = ['api_rate_limit', 'max_retries', 'timeout_seconds', 'lookback_days']
    
    for key in required_keys:
        if key not in config:
            validation_results['errors'].append(f"Missing required configuration: {key}")
            validation_results['is_valid'] = False
    
    # Validate API key
    if not config.get('etherscan_api_key'):
        validation_results['warnings'].append("Etherscan API key not provided - some features may not work")
    
    # Validate numeric ranges
    if config.get('api_rate_limit', 0) <= 0:
        validation_results['errors'].append("API rate limit must be positive")
        validation_results['is_valid'] = False
    
    if config.get('max_retries', 0) < 0:
        validation_results['errors'].append("Max retries cannot be negative")
        validation_results['is_valid'] = False
    
    if config.get('lookback_days', 0) <= 0:
        validation_results['errors'].append("Lookback days must be positive")
        validation_results['is_valid'] = False
    
    # Performance recommendations
    if config.get('api_rate_limit', 5) > 10:
        validation_results['recommendations'].append("Consider lowering API rate limit to avoid being rate-limited")
    
    if config.get('lookback_days', 365) > 730:
        validation_results['recommendations'].append("Very long lookback period may slow down data collection")
    
    return validation_results

def main():
    """Test utility functions"""
    print("Testing Wallet Risk Scoring Utilities...")
    
    # Test wallet address validation
    test_addresses = [
        "0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7",  # Valid
        "742d35cc6408c532c32Cf7D26E5DE5925F59D2b7",   # Valid without 0x
        "0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b",    # Invalid (too short)
        "not_an_address",                               # Invalid
    ]
    
    print("\nTesting wallet address validation:")
    for addr in test_addresses:
        is_valid = validate_wallet_address(addr)
        print(f"  {addr}: {'Valid' if is_valid else 'Invalid'}")
    
    # Test system requirements
    print("\nChecking system requirements:")
    requirements = check_system_requirements()
    print(f"  Platform: {requirements['system_info']['platform']}")
    print(f"  Python: {requirements['system_info']['python_version'].split()[0]}")
    
    # Test configuration loading
    print("\nTesting configuration:")
    config = load_config()
    validation = validate_config(config)
    print(f"  Configuration valid: {validation['is_valid']}")
    if validation['errors']:
        print(f"  Errors: {len(validation['errors'])}")
    
    print("\nUtility functions test completed!")

if __name__ == "__main__":
    main()

