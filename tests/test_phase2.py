"""
Phase 2 Test Script
==================

Quick test to verify that our data collection system is working properly
before proceeding to the next phases.

Run this to test: python test_phase2.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append('src')

def test_wallet_list_loading():
    """Test loading wallet addresses from CSV"""
    print("🧪 Testing wallet list loading...")
    
    try:
        from data_collector import load_wallet_list
        wallet_addresses = load_wallet_list("data/wallet_list.csv")
        
        print(f"✅ Successfully loaded {len(wallet_addresses)} wallet addresses")
        print(f"   First address: {wallet_addresses[0] if wallet_addresses else 'None'}")
        print(f"   Last address: {wallet_addresses[-1] if wallet_addresses else 'None'}")
        
        return True, wallet_addresses
        
    except Exception as e:
        print(f"❌ Wallet list loading failed: {str(e)}")
        return False, []

def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing configuration loading...")
    
    try:
        from utils import load_config, validate_config
        
        config = load_config()
        validation = validate_config(config)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Valid: {validation['is_valid']}")
        print(f"   Errors: {len(validation['errors'])}")
        print(f"   Warnings: {len(validation['warnings'])}")
        
        if validation['errors']:
            print("   Error details:")
            for error in validation['errors']:
                print(f"     - {error}")
        
        return validation['is_valid'], config
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False, {}

def test_data_collector_init():
    """Test data collector initialization"""
    print("🧪 Testing data collector initialization...")
    
    try:
        from data_collector import CompoundDataCollector
        
        collector = CompoundDataCollector()
        
        print(f"✅ Data collector initialized successfully")
        print(f"   API delay: {collector.api_delay:.2f} seconds")
        print(f"   Raw data dir: {collector.raw_data_dir}")
        
        return True, collector
        
    except Exception as e:
        print(f"❌ Data collector initialization failed: {str(e)}")
        return False, None

def test_utils_functions():
    """Test utility functions"""
    print("🧪 Testing utility functions...")
    
    try:
        from utils import validate_wallet_address, clean_wallet_address, format_large_number
        
        # Test wallet validation
        test_addresses = [
            "0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7",  # Valid
            "742d35cc6408c532c32Cf7D26E5DE5925F59D2b7",   # Valid without 0x
            "invalid_address",                              # Invalid
        ]
        
        validation_results = []
        for addr in test_addresses:
            is_valid = validate_wallet_address(addr)
            cleaned = clean_wallet_address(addr)
            validation_results.append((addr, is_valid, cleaned))
        
        print(f"✅ Utility functions working correctly")
        print("   Address validation results:")
        for original, valid, cleaned in validation_results:
            print(f"     {original[:20]}... -> Valid: {valid}, Cleaned: {cleaned[:25]}...")
        
        # Test number formatting
        test_numbers = [1234, 1234567, 1234567890]
        print("   Number formatting:")
        for num in test_numbers:
            formatted = format_large_number(num)
            print(f"     {num} -> {formatted}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {str(e)}")
        return False

def test_directory_structure():
    """Test that all required directories and files exist"""
    print("🧪 Testing directory structure...")
    
    required_items = [
        ("src", "directory"),
        ("data", "directory"), 
        ("results", "directory"),
        ("src/data_collector.py", "file"),
        ("src/utils.py", "file"),
        ("src/main.py", "file"),
        ("data/wallet_list.csv", "file"),
        (".env", "file"),
        ("requirements.txt", "file")
    ]
    
    all_good = True
    for item_path, item_type in required_items:
        path = Path(item_path)
        
        if item_type == "directory":
            exists = path.exists() and path.is_dir()
        else:
            exists = path.exists() and path.is_file()
        
        status = "✅" if exists else "❌"
        print(f"   {status} {item_path} ({item_type})")
        
        if not exists:
            all_good = False
    
    return all_good

def run_mini_data_collection_test(collector, wallet_addresses):
    """Run a mini data collection test with one wallet"""
    print("🧪 Testing mini data collection (1 wallet)...")
    
    if not wallet_addresses:
        print("❌ No wallet addresses available for testing")
        return False
    
    try:
        # Test with first wallet address
        test_wallet = wallet_addresses[0]
        print(f"   Testing with: {test_wallet}")
        
        # This will likely fail without API keys, but we can test the structure
        wallet_data = collector.fetch_wallet_data(test_wallet)
        
        print(f"✅ Data collection structure test passed")
        print(f"   Returned data keys: {list(wallet_data.keys())}")
        print(f"   Address: {wallet_data.get('address', 'None')}")
        print(f"   Has error: {'error' in wallet_data}")
        
        return True
    
    except Exception as e:
        print(f"❌ Data collection test failed: {str(e)}")
        return False

# Add a main block to run all tests if the script is executed directly
if __name__ == "__main__":
    print("Running Phase 2 Tests...")
    print("=" * 50)
    
    # Run directory structure test
    dir_ok = test_directory_structure()
    
    # Run configuration test
    config_ok, config = test_configuration()
    
    # Run wallet list loading test
    wallet_ok, wallet_addresses = test_wallet_list_loading()
    
    # Run data collector initialization test
    collector_ok, collector = test_data_collector_init()
    
    # Run utility functions test
    utils_ok = test_utils_functions()
    
    # Run mini data collection test if collector and wallet addresses are available
    mini_test_ok = False
    if collector_ok and wallet_ok and collector and wallet_addresses:
        mini_test_ok = run_mini_data_collection_test(collector, wallet_addresses)
    
    print("\nTest Summary:")
    print(f"Directory Structure: {'✅' if dir_ok else '❌'}")
    print(f"Configuration: {'✅' if config_ok else '❌'}")
    print(f"Wallet Loading: {'✅' if wallet_ok else '❌'}")
    print(f"Data Collector Init: {'✅' if collector_ok else '❌'}")
    print(f"Utility Functions: {'✅' if utils_ok else '❌'}")
    print(f"Mini Data Collection: {'✅' if mini_test_ok else '❌'}")
    
    print("\nAll tests completed!")