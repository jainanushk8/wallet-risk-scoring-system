"""
Main Execution Script for Wallet Risk Scoring System
===================================================

Orchestrates the complete workflow:
1. Load wallet addresses
2. Collect transaction data from Compound protocol
3. Engineer risk features
4. Calculate risk scores
5. Generate final results

Author: AI Development Team
Date: July 2025
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from data_collector import CompoundDataCollector, load_wallet_list
from utils import (
    setup_logging, 
    load_config, 
    validate_config,
    validate_wallet_list,
    save_results,
    create_data_summary,
    check_system_requirements
)

class WalletRiskScoringPipeline:
    """
    Main pipeline class that orchestrates the entire wallet risk scoring process.
    """
    
    def __init__(self, config_path: str = ".env"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.data_collector = None
        
        # Results storage
        self.wallet_addresses = []
        self.collected_data = {}
        self.risk_scores = {}
        
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize all pipeline components."""
        print("🚀 Initializing Wallet Risk Scoring Pipeline...")
        
        # Setup logging
        self.logger = setup_logging("INFO", "wallet_risk_scoring.log")
        
        # Load and validate configuration
        self.config = load_config(self.config_path)
        config_validation = validate_config(self.config)
        
        if not config_validation['is_valid']:
            self.logger.error("Configuration validation failed:")
            for error in config_validation['errors']:
                self.logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        # Log warnings and recommendations
        for warning in config_validation['warnings']:
            self.logger.warning(warning)
        
        for recommendation in config_validation['recommendations']:
            self.logger.info(f"Recommendation: {recommendation}")
        
        # Initialize data collector
        self.data_collector = CompoundDataCollector(self.config)
        
        self.logger.info("Pipeline initialization completed successfully")
    
    def run_system_check(self):
        """Run comprehensive system requirements check."""
        print("🔍 Running system requirements check...")
        
        requirements = check_system_requirements()
        
        print(f"System: {requirements['system_info']['platform']}")
        print(f"Python: {requirements['system_info']['python_version'].split()[0]}")
        
        # Check directory structure
        missing_dirs = [name for name, info in requirements['directory_structure'].items() 
                       if not info['exists']]
        if missing_dirs:
            print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")
        else:
            print("✅ Directory structure OK")
        
        # Check dependencies
        missing_deps = [name for name, info in requirements['dependencies'].items() 
                       if not info['installed']]
        if missing_deps:
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            print("Please run: pip install -r requirements.txt")
            return False
        else:
            print("✅ Dependencies OK")
        
        return True
    
    def load_wallet_addresses(self, wallet_file: str = "data/wallet_list.csv"):
        """
        Load and validate wallet addresses from file.
        
        Args:
            wallet_file (str): Path to wallet addresses file
        """
        print(f"📄 Loading wallet addresses from {wallet_file}...")
        
        # Load addresses
        self.wallet_addresses = load_wallet_list(wallet_file)
        
        if not self.wallet_addresses:
            raise ValueError(f"No wallet addresses found in {wallet_file}")
        
        # Validate addresses
        validation_results = validate_wallet_list(self.wallet_addresses)
        
        print(f"📊 Wallet validation results:")
        print(f"   Total addresses: {validation_results['total_addresses']}")
        print(f"   Valid addresses: {validation_results['valid_count']}")
        print(f"   Invalid addresses: {validation_results['invalid_count']}")
        print(f"   Validation rate: {validation_results['validation_rate']:.1%}")
        
        if validation_results['invalid_addresses']:
            self.logger.warning(f"Found {len(validation_results['invalid_addresses'])} invalid addresses")
            for addr in validation_results['invalid_addresses'][:5]:  # Show first 5
                self.logger.warning(f"  Invalid: {addr}")
        
        # Use only valid addresses
        self.wallet_addresses = validation_results['valid_addresses']
        
        self.logger.info(f"Loaded {len(self.wallet_addresses)} valid wallet addresses")
    
    def collect_data(self, test_mode: bool = False, test_count: int = 5):
        """
        Collect transaction data for all wallet addresses.
        
        Args:
            test_mode (bool): If True, only process a subset of wallets for testing
            test_count (int): Number of wallets to process in test mode
        """
        if test_mode:
            wallets_to_process = self.wallet_addresses[:test_count]
            print(f"🧪 Test mode: Processing {len(wallets_to_process)} wallets")
        else:
            wallets_to_process = self.wallet_addresses
            print(f"🔄 Production mode: Processing {len(wallets_to_process)} wallets")
        
        print("📡 Starting data collection from Compound protocol...")
        
        # Collect data using batch processing
        self.collected_data = self.data_collector.batch_collect_data(
            wallets_to_process, 
            save_individual=True
        )
        
        # Log collection results
        success_rate = self.collected_data['collection_summary']['success_rate']
        compound_wallets = self.collected_data['collection_summary']['total_compound_wallets']
        
        print(f"📈 Data collection completed:")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Wallets with Compound activity: {compound_wallets}")
        
        self.logger.info(f"Data collection completed with {success_rate:.1%} success rate")
        
        # Save collection summary
        summary = create_data_summary(self.collected_data)
        save_results(summary, 'data_collection_summary.json', 'json')
    
    def engineer_features(self):
        """
        Engineer risk features from collected transaction data.
        """
        print("🔧 Engineering features from transaction data...")
        
        from feature_engineer import RiskFeatureEngineer
        
        engineer = RiskFeatureEngineer(self.config)
        feature_data = {}
        
        successful_extractions = 0
        failed_extractions = 0
        
        for wallet_address, wallet_data in self.collected_data.get('wallets_data', {}).items():
            try:
                if isinstance(wallet_data, dict) and 'error' not in wallet_data:
                    # Extract features for this wallet
                    features = engineer.extract_features(wallet_data)
                    feature_data[wallet_address] = features
                    successful_extractions += 1
                else:
                    self.logger.warning(f"Skipping feature extraction for {wallet_address} due to data collection error")
                    failed_extractions += 1
                    
            except Exception as e:
                self.logger.error(f"Feature extraction failed for {wallet_address}: {str(e)}")
                failed_extractions += 1
        
        print(f"✅ Feature engineering completed:")
        print(f"   Successful extractions: {successful_extractions}")
        print(f"   Failed extractions: {failed_extractions}")
        
        self.logger.info(f"Feature engineering completed for {successful_extractions} wallets")
        return feature_data
    
    def calculate_risk_scores(self, feature_data: dict):
        """
        Calculate risk scores based on engineered features.
        """
        print("📊 Calculating risk scores using weighted composite model...")
        
        from risk_scorer import WalletRiskScorer
        
        scorer = WalletRiskScorer(self.config)
        risk_scores = {}
        
        successful_scores = 0
        failed_scores = 0
        
        for wallet_address, features in feature_data.items():
            try:
                # Calculate risk score
                score = scorer.calculate_score(features)
                risk_scores[wallet_address] = score
                successful_scores += 1
                
            except Exception as e:
                self.logger.error(f"Risk scoring failed for {wallet_address}: {str(e)}")
                risk_scores[wallet_address] = 500  # Default medium risk
                failed_scores += 1
        
        print(f"✅ Risk scoring completed:")
        print(f"   Successful scores: {successful_scores}")
        print(f"   Failed scores: {failed_scores}")
        
        if successful_scores > 0:
            scores_list = list(risk_scores.values())
            print(f"   Score statistics:")
            print(f"     • Mean: {sum(scores_list) / len(scores_list):.1f}")
            print(f"     • Range: {min(scores_list)} - {max(scores_list)}")
            
            # Risk distribution
            low_risk = len([s for s in scores_list if s <= 250])
            medium_risk = len([s for s in scores_list if 251 <= s <= 750])
            high_risk = len([s for s in scores_list if s >= 751])
            
            print(f"   Risk distribution:")
            print(f"     • Low risk (0-250): {low_risk} wallets")
            print(f"     • Medium risk (251-750): {medium_risk} wallets")
            print(f"     • High risk (751-1000): {high_risk} wallets")
        
        self.risk_scores = risk_scores
        self.logger.info(f"Risk scoring completed for {successful_scores} wallets")
        return risk_scores
    
    def generate_final_results(self):
        """Generate final results CSV and summary report."""
        print("📋 Generating final results...")
        
        # Create final results DataFrame
        results_data = []
        for wallet_address in self.wallet_addresses:
            score = self.risk_scores.get(wallet_address, 0)
            results_data.append({
                'wallet_id': wallet_address,
                'score': score
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Save main results CSV
        save_results(results_df, 'wallet_risk_scores.csv', 'csv')
        
        # Create detailed summary
        summary_report = {
            'execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_wallets_processed': len(self.wallet_addresses),
                'successful_collections': self.collected_data.get('successful_collections', 0),
                'average_score': results_df['score'].mean(),
                'score_distribution': {
                    'min': results_df['score'].min(),
                    'max': results_df['score'].max(),
                    'median': results_df['score'].median(),
                    'std': results_df['score'].std()
                }
            },
            'risk_distribution': {
                'low_risk (0-250)': len(results_df[results_df['score'] <= 250]),
                'medium_risk (251-750)': len(results_df[(results_df['score'] > 250) & (results_df['score'] <= 750)]),
                'high_risk (751-1000)': len(results_df[results_df['score'] > 750])
            },
            'data_quality': self.collected_data.get('collection_summary', {})
        }
        
        save_results(summary_report, 'execution_summary.json', 'json')
        
        print("✅ Results generated successfully:")
        print(f"   Main results: results/wallet_risk_scores.csv")
        print(f"   Summary report: results/execution_summary.json")
        
        return results_df, summary_report


def main():
    """Main execution function."""
    print("=" * 60)
    print("🎯 WALLET RISK SCORING SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = WalletRiskScoringPipeline()
        
        # Run system check
        if not pipeline.run_system_check():
            print("❌ System requirements check failed. Please fix issues and try again.")
            return 1
        
        # Load wallet addresses
        pipeline.load_wallet_addresses()
        
        # Check if we should run in test mode
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            test_mode = True
            test_count = 3
            print("🧪 Running in TEST MODE")
        else:
            test_mode = False
            test_count = 0
            print("🚀 Running in PRODUCTION MODE")
        
        # Execute main pipeline
        print("\n" + "=" * 60)
        print("PHASE 1: DATA COLLECTION")
        print("=" * 60)
        pipeline.collect_data(test_mode=test_mode, test_count=test_count)
        
        print("\n" + "=" * 60)
        print("PHASE 2: FEATURE ENGINEERING")
        print("=" * 60)
        feature_data = pipeline.engineer_features()
        
        print("\n" + "=" * 60)
        print("PHASE 3: RISK SCORING")
        print("=" * 60)
        risk_scores = pipeline.calculate_risk_scores(feature_data)
        
        print("\n" + "=" * 60)
        print("PHASE 4: RESULTS GENERATION")
        print("=" * 60)
        results_df, summary = pipeline.generate_final_results()
        
        print("\n" + "=" * 60)
        print("🎉 EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Processed {len(results_df)} wallets")
        print(f"Average risk score: {results_df['score'].mean():.1f}")
        print("Check the results/ folder for detailed outputs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
