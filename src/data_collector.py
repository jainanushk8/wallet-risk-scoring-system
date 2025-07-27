"""
Compound Protocol Data Collector
================================

Fetches transaction data from Compound V2/V3 protocol for risk analysis.
Optimized for Windows systems with rate limiting and error handling.

Author: AI Development Team
Date: July 2025
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompoundDataCollector:
    """
    Collects transaction data from Compound protocol for wallet risk analysis.
    
    Features:
    - Rate limiting for API calls
    - Multiple data sources (Etherscan, The Graph, Compound API)
    - Robust error handling and retries
    - Windows-compatible file handling
    - Progress tracking
    """
    
    def __init__(self, config: Dict = None): #type:ignore
        """
        Initialize the Compound data collector.
        
        Args:
            config (Dict): Configuration parameters including API keys and settings
        """
        self.config = config or {}
        self.setup_config()
        self.setup_apis()
        
        # Data storage
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Rate limiting
        self.last_api_call = 0
        self.api_delay = 1.0 / self.config.get('api_rate_limit', 5)  # Default 5 calls/sec
        
        logger.info("CompoundDataCollector initialized successfully")
    
    def setup_config(self):
        """Load configuration from environment variables and defaults"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # API Configuration
        self.config.setdefault('etherscan_api_key', os.getenv('ETHERSCAN_API_KEY', ''))
        self.config.setdefault('api_rate_limit', int(os.getenv('API_RATE_LIMIT', '5')))
        self.config.setdefault('max_retries', int(os.getenv('MAX_RETRIES', '3')))
        self.config.setdefault('timeout_seconds', int(os.getenv('TIMEOUT_SECONDS', '30')))
        
        # Data Collection Settings
        self.config.setdefault('lookback_days', int(os.getenv('LOOKBACK_DAYS', '365')))
        self.config.setdefault('min_transactions', int(os.getenv('MIN_TRANSACTIONS', '5')))
        
    def setup_apis(self):
        """Setup API endpoints and headers"""
        self.api_endpoints = {
            'etherscan': 'https://api.etherscan.io/api',
            'compound_api': 'https://api.compound.finance/api/v2',
            'the_graph': 'https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2'
        }
        
        self.headers = {
            'User-Agent': 'WalletRiskScoring/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.api_delay:
            sleep_time = self.api_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def make_api_request(self, url: str, params: Dict = None, retries: int = None) -> Optional[Dict]: #type:ignore
        """
        Make API request with rate limiting and error handling.
        
        Args:
            url (str): API endpoint URL
            params (Dict): Request parameters
            retries (int): Number of retry attempts
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        if retries is None:
            retries = self.config['max_retries']
            
        self.rate_limit()
        
        for attempt in range(retries + 1):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=self.headers,
                    timeout=self.config['timeout_seconds']
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"API request failed after {retries + 1} attempts")
                    return None
    
    def fetch_etherscan_transactions(self, wallet_address: str) -> List[Dict]:
        """
        Fetch transaction history from Etherscan API.
        
        Args:
            wallet_address (str): Ethereum wallet address
            
        Returns:
            List[Dict]: List of transaction data
        """
        if not self.config['etherscan_api_key']:
            logger.warning("Etherscan API key not provided, skipping Etherscan data")
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['lookback_days'])
        
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': wallet_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'desc',
            'apikey': self.config['etherscan_api_key']
        }
        
        logger.info(f"Fetching Etherscan transactions for {wallet_address}")
        response = self.make_api_request(self.api_endpoints['etherscan'], params)
        
        if response and response.get('status') == '1':
            transactions = response.get('result', [])
            logger.info(f"Found {len(transactions)} transactions from Etherscan")
            return transactions
        else:
            logger.warning(f"Failed to fetch Etherscan data: {response}")
            return []
    
    def fetch_compound_data_from_graph(self, wallet_address: str) -> List[Dict]:
        """
        Fetch Compound-specific data from The Graph Protocol.
        
        Args:
            wallet_address (str): Ethereum wallet address
            
        Returns:
            List[Dict]: List of Compound transaction data
        """
        query = """
        {
          account(id: "%s") {
            id
            hasBorrowed
            tokens {
              id
              symbol
              cTokenBalance
              totalUnderlyingSupplied
              totalUnderlyingBorrowed
              accountBorrowIndex
              totalUnderlyingRedeemed
              totalUnderlyingRepaid
              storedBorrowBalance
            }
            transactions {
              id
              timestamp
              blockNumber
              from
              to
              amount
              market {
                symbol
                underlyingSymbol
              }
            }
          }
        }
        """ % wallet_address.lower()
        
        payload = {'query': query}
        
        logger.info(f"Fetching Compound data from The Graph for {wallet_address}")
        response = requests.post(
            self.api_endpoints['the_graph'],
            json=payload,
            headers=self.headers,
            timeout=self.config['timeout_seconds']
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'account' in data['data'] and data['data']['account']:
                account_data = data['data']['account']
                logger.info(f"Found Compound account data for {wallet_address}")
                return account_data
            else:
                logger.info(f"No Compound data found for {wallet_address}")
                return {} #type:ignore
        else:
            logger.warning(f"Failed to fetch The Graph data: {response.status_code}")
            return {} #type:ignore
    
    def extract_compound_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """
        Filter and extract Compound-related transactions from general transaction list.
        
        Args:
            transactions (List[Dict]): List of all transactions
            
        Returns:
            List[Dict]: Filtered Compound transactions
        """
        # Compound V2 contract addresses (cTokens)
        compound_addresses = {
            '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',  # cDAI
            '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5',  # cETH
            '0x39aa39c021dfbaecaad6fabf7aa1b4a6ea1b0dfa',  # cUSDC
            '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',  # cUSDT
            '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4',  # cWBTC
            # Add more cToken addresses as needed
        }
        
        compound_txs = []
        for tx in transactions:
            if tx.get('to', '').lower() in [addr.lower() for addr in compound_addresses]:
                compound_txs.append({
                    'hash': tx.get('hash'),
                    'timestamp': int(tx.get('timeStamp', 0)),
                    'from': tx.get('from'),
                    'to': tx.get('to'),
                    'value': tx.get('value', '0'),
                    'gas': tx.get('gas', '0'),
                    'gasUsed': tx.get('gasUsed', '0'),
                    'gasPrice': tx.get('gasPrice', '0'),
                    'input': tx.get('input', ''),
                    'isError': tx.get('isError', '0'),
                    'contract_type': 'compound_v2'
                })
        
        logger.info(f"Extracted {len(compound_txs)} Compound transactions from {len(transactions)} total")
        return compound_txs
    
    def fetch_wallet_data(self, wallet_address: str) -> Dict:
        """
        Comprehensive data collection for a single wallet address.
        
        Args:
            wallet_address (str): Ethereum wallet address
            
        Returns:
            Dict: Complete wallet data including transactions and Compound interactions
        """
        logger.info(f"Starting data collection for wallet: {wallet_address}")
        
        wallet_data = {
            'address': wallet_address,
            'collection_timestamp': datetime.now().isoformat(),
            'etherscan_transactions': [],
            'compound_data': {},
            'compound_transactions': [],
            'summary': {}
        }
        
        # Fetch data from multiple sources
        try:
            # 1. Get all transactions from Etherscan
            all_transactions = self.fetch_etherscan_transactions(wallet_address)
            wallet_data['etherscan_transactions'] = all_transactions
            
            # 2. Extract Compound-specific transactions
            compound_txs = self.extract_compound_transactions(all_transactions)
            wallet_data['compound_transactions'] = compound_txs
            
            # 3. Get detailed Compound data from The Graph
            compound_data = self.fetch_compound_data_from_graph(wallet_address)
            wallet_data['compound_data'] = compound_data
            
            # 4. Create summary
            wallet_data['summary'] = {
                'total_transactions': len(all_transactions),
                'compound_transactions': len(compound_txs),
                'has_compound_activity': len(compound_txs) > 0 or bool(compound_data),
                'data_quality': 'complete' if all_transactions and (compound_txs or compound_data) else 'partial'
            }
            
            logger.info(f"Data collection completed for {wallet_address}")
            
        except Exception as e:
            logger.error(f"Error collecting data for {wallet_address}: {str(e)}")
            wallet_data['error'] = str(e)
        
        return wallet_data
    
    def save_wallet_data(self, wallet_data: Dict):
        """
        Save wallet data to JSON file.
        
        Args:
            wallet_data (Dict): Wallet data to save
        """
        wallet_address = wallet_data['address']
        filename = f"{wallet_address}.json"
        filepath = self.raw_data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(wallet_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved wallet data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save wallet data: {str(e)}")
    
    def batch_collect_data(self, wallet_addresses: List[str], save_individual: bool = True) -> Dict:
        """
        Collect data for multiple wallet addresses with progress tracking.
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses to process
            save_individual (bool): Whether to save individual wallet files
            
        Returns:
            Dict: Aggregated results for all wallets
        """
        logger.info(f"Starting batch data collection for {len(wallet_addresses)} wallets")
        
        results = {
            'total_wallets': len(wallet_addresses),
            'successful_collections': 0,
            'failed_collections': 0,
            'wallets_data': {},
            'collection_summary': {}
        }
        
        # Progress bar for batch processing
        with tqdm(total=len(wallet_addresses), desc="Collecting wallet data") as pbar:
            for wallet_address in wallet_addresses:
                try:
                    # Collect data for individual wallet
                    wallet_data = self.fetch_wallet_data(wallet_address)
                    
                    # Save individual file if requested
                    if save_individual:
                        self.save_wallet_data(wallet_data)
                    
                    # Add to results
                    results['wallets_data'][wallet_address] = wallet_data
                    results['successful_collections'] += 1
                    
                    pbar.set_postfix({
                        'Success': results['successful_collections'],
                        'Failed': results['failed_collections']
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to collect data for {wallet_address}: {str(e)}")
                    results['failed_collections'] += 1
                    results['wallets_data'][wallet_address] = {'error': str(e)}
                
                pbar.update(1)
        
        # Create collection summary
        results['collection_summary'] = {
            'success_rate': results['successful_collections'] / len(wallet_addresses),
            'total_compound_wallets': sum(1 for data in results['wallets_data'].values() 
                                        if isinstance(data, dict) and 
                                        data.get('summary', {}).get('has_compound_activity', False)),
            'completion_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch collection completed: {results['successful_collections']}/{len(wallet_addresses)} successful")
        return results

def load_wallet_list(filepath: str = "data/wallet_list.csv") -> List[str]:
    """
    Load wallet addresses from CSV file.
    
    Args:
        filepath (str): Path to wallet list CSV file
        
    Returns:
        List[str]: List of wallet addresses
    """
    try:
        df = pd.read_csv(filepath)
        wallet_addresses = df['wallet_id'].tolist()
        logger.info(f"Loaded {len(wallet_addresses)} wallet addresses from {filepath}")
        return wallet_addresses
    except Exception as e:
        logger.error(f"Failed to load wallet list: {str(e)}")
        return []

def main():
    """Main function for testing the data collector"""
    print("Testing Compound Data Collector...")
    
    # Initialize collector
    collector = CompoundDataCollector()
    
    # Load wallet addresses
    wallet_addresses = load_wallet_list()
    
    if wallet_addresses:
        # Test with first 3 wallets
        test_wallets = wallet_addresses[:3]
        print(f"Testing with {len(test_wallets)} wallets...")
        
        results = collector.batch_collect_data(test_wallets)
        print(f"Collection completed: {results['collection_summary']}")
    else:
        print("No wallet addresses found. Please check data/wallet_list.csv")

if __name__ == "__main__":
    main()