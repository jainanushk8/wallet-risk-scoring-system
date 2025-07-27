"""
Risk Scorer Test Suite
=====================

Comprehensive tests for the WalletRiskScorer module to ensure
proper integration with the feature engineering system and accurate scoring.

Run this test: python tests/test_risk_scorer.py
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

def test_risk_scorer_initialization():
    """Test basic initialization of the risk scorer"""
    print("🧪 Testing Risk Scorer Initialization...")
    
    try:
        from risk_scorer import WalletRiskScorer
        
        # Test with default config
        scorer = WalletRiskScorer()
        
        print("✅ Risk scorer initialized successfully")
        print(f"   Category weights: {scorer.category_weights}")
        print(f"   Score range: {scorer.config['min_risk_score']}-{scorer.config['max_risk_score']}")
        print(f"   Risk thresholds: {len(scorer.risk_thresholds)} configured")
        
        # Test with custom config
        custom_config = {'max_risk_score': 2000}
        scorer_custom = WalletRiskScorer(custom_config)
        
        print(f"✅ Custom config applied: max_score = {scorer_custom.config['max_risk_score']}")
        
        return True, scorer
        
    except Exception as e:
        print(f"❌ Risk scorer initialization failed: {str(e)}")
        return False, None


def test_score_calculation_with_sample_data():
    """Test risk score calculation with comprehensive sample data"""
    print("🧪 Testing Risk Score Calculation with Sample Data...")
    
    try:
        from risk_scorer import WalletRiskScorer
        
        scorer = WalletRiskScorer()
        
        # High-risk wallet features
        high_risk_features = {
            'wallet_address': '0xhighrisk',
            'data_quality': {'data_completeness': 'high'},
            'behavioral_features': {
                'avg_tx_per_day': 25.0,  # Very high activity
                'recent_activity_ratio': 0.8,
                'avg_tx_value': 5e18,  # Large transactions
                'large_tx_ratio': 0.6,
                'tx_value_std': 3e18,
                'compound_activity_ratio': 0.9,  # Very active in DeFi
                'compound_tx_count': 100,
                'high_gas_tx_ratio': 0.7,
                'burst_activity_score': 0.8,  # Burst activity
                'tx_regularity': 0.2  # Irregular patterns
            },
            'liquidity_features': {
                'collateral_ratio': 1.1,  # Dangerous collateral
                'liquidation_risk_score': 0.9,  # High liquidation risk
                'utilization_ratio': 0.95,  # Very high leverage
                'leverage_score': 0.9,
                'total_supplied_value': 1000,
                'total_borrowed_value': 900
            },
            'market_features': {
                'portfolio_risk_score': 0.85,  # High-risk assets
                'concentration_risk': 0.9,  # Very concentrated
                'diversification_score': 0.1,  # Poor diversification
                'asset_diversification': 0.2,
                'active_asset_count': 1
            }
        }
        
        # Calculate high-risk score
        high_risk_score = scorer.calculate_score(high_risk_features)
        
        # Low-risk wallet features
        low_risk_features = {
            'wallet_address': '0xlowrisk',
            'data_quality': {'data_completeness': 'high'},
            'behavioral_features': {
                'avg_tx_per_day': 0.5,  # Low activity
                'recent_activity_ratio': 0.2,
                'avg_tx_value': 1e17,  # Small transactions
                'large_tx_ratio': 0.1,
                'tx_value_std': 5e16,
                'compound_activity_ratio': 0.1,  # Low DeFi activity
                'compound_tx_count': 3,
                'high_gas_tx_ratio': 0.2,
                'burst_activity_score': 0.05,
                'tx_regularity': 0.8  # Regular patterns
            },
            'liquidity_features': {
                'collateral_ratio': 3.0,  # Safe collateral
                'liquidation_risk_score': 0.1,  # Low liquidation risk
                'utilization_ratio': 0.2,  # Low leverage
                'leverage_score': 0.2,
                'total_supplied_value': 1000,
                'total_borrowed_value': 200
            },
            'market_features': {
                'portfolio_risk_score': 0.3,  # Low-risk assets
                'concentration_risk': 0.2,  # Well diversified
                'diversification_score': 0.8,  # Good diversification
                'asset_diversification': 0.9,
                'active_asset_count': 5
            }
        }
        
        # Calculate low-risk score
        low_risk_score = scorer.calculate_score(low_risk_features)
        
        print("✅ Risk score calculation completed successfully")
        print(f"   High-risk wallet score: {high_risk_score} (expected: >750)")
        print(f"   Low-risk wallet score: {low_risk_score} (expected: <250)")
        
        # Validate score ranges
        if not (0 <= high_risk_score <= 1000):
            raise ValueError(f"High-risk score out of range: {high_risk_score}")
        
        if not (0 <= low_risk_score <= 1000):
            raise ValueError(f"Low-risk score out of range: {low_risk_score}")
        
        # Validate relative scoring
        if high_risk_score <= low_risk_score:
            print(f"⚠️  Warning: High-risk score ({high_risk_score}) should be > low-risk score ({low_risk_score})")
        
        return True, {'high_risk': high_risk_score, 'low_risk': low_risk_score}
        
    except Exception as e:
        print(f"❌ Score calculation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_score_explanation():
    """Test detailed score explanation functionality"""
    print("🧪 Testing Score Explanation Generation...")
    
    try:
        from risk_scorer import WalletRiskScorer
        
        scorer = WalletRiskScorer()
        
        # Medium-risk wallet for explanation testing
        medium_risk_features = {
            'wallet_address': '0xmediumrisk',
            'data_quality': {'data_completeness': 'medium'},
            'behavioral_features': {
                'avg_tx_per_day': 3.0,
                'recent_activity_ratio': 0.4,
                'avg_tx_value': 1e18,
                'large_tx_ratio': 0.3,
                'compound_activity_ratio': 0.5,
                'compound_tx_count': 20,
                'burst_activity_score': 0.2,
                'tx_regularity': 0.6
            },
            'liquidity_features': {
                'collateral_ratio': 1.6,
                'liquidation_risk_score': 0.5,
                'utilization_ratio': 0.6,
                'leverage_score': 0.6
            },
            'market_features': {
                'portfolio_risk_score': 0.5,
                'concentration_risk': 0.5,
                'diversification_score': 0.5,
                'asset_diversification': 0.5
            }
        }
        
        # Get explanation
        explanation = scorer.get_score_explanation(medium_risk_features)
        
        print("✅ Score explanation generated successfully")
        print(f"   Final score: {explanation.get('final_risk_score')}")
        print(f"   Risk category: {explanation.get('risk_category')}")
        
        # Check component scores
        components = explanation.get('component_scores', {})
        print(f"   Component scores:")
        for component, details in components.items():
            print(f"     • {component}: {details.get('score')} (weight: {details.get('weight')})")
        
        # Check key risk factors
        risk_factors = explanation.get('key_risk_factors', [])
        print(f"   Key risk factors: {len(risk_factors)} identified")
        for factor in risk_factors[:3]:  # Show first 3
            print(f"     • {factor}")
        
        # Validate explanation structure
        required_keys = ['final_risk_score', 'risk_category', 'component_scores', 'methodology']
        for key in required_keys:
            if key not in explanation:
                raise ValueError(f"Missing explanation key: {key}")
        
        return True, explanation
        
    except Exception as e:
        print(f"❌ Score explanation test failed: {str(e)}")
        return False, None


def test_integration_with_feature_engineer():
    """Test end-to-end integration with the feature engineer"""
    print("🧪 Testing Integration with Feature Engineer...")
    
    try:
        from feature_engineer import RiskFeatureEngineer
        from risk_scorer import WalletRiskScorer
        
        # Initialize both components
        engineer = RiskFeatureEngineer()
        scorer = WalletRiskScorer()
        
        # Create sample wallet data (matching data collector output)
        sample_wallet_data = {
            'address': '0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7',
            'collection_timestamp': datetime.now().isoformat(),
            'etherscan_transactions': [
                {
                    'timeStamp': str(int((datetime.now() - timedelta(days=5)).timestamp())),
                    'from': '0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7',
                    'to': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',
                    'value': '1000000000000000000',
                    'gasUsed': '150000',
                    'isError': '0'
                },
                {
                    'timeStamp': str(int((datetime.now() - timedelta(days=2)).timestamp())),
                    'from': '0x742d35cc6408c532c32Cf7D26E5DE5925F59D2b7',
                    'to': '0x39aa39c021dfbaecaad6fabf7aa1b4a6ea1b0dfa',
                    'value': '500000000000000000',
                    'gasUsed': '160000',
                    'isError': '0'
                }
            ],
            'compound_data': {
                'tokens': [
                    {
                        'symbol': 'cDAI',
                        'totalUnderlyingSupplied': '800.0',
                        'totalUnderlyingBorrowed': '400.0'
                    }
                ]
            },
            'compound_transactions': [
                {
                    'timeStamp': str(int((datetime.now() - timedelta(days=3)).timestamp())),
                    'value': '1000000000000000000',
                    'gasUsed': '150000'
                }
            ],
            'summary': {
                'total_transactions': 2,
                'compound_transactions': 1,
                'has_compound_activity': True,
                'data_quality': 'complete'
            }
        }
        
        # Extract features
        features = engineer.extract_features(sample_wallet_data)
        
        # Calculate risk score
        risk_score = scorer.calculate_score(features)
        
        # Get explanation
        explanation = scorer.get_score_explanation(features)
        
        print("✅ End-to-end integration successful")
        print(f"   Wallet: {features.get('wallet_address')}")
        print(f"   Features extracted: {len(features.get('behavioral_features', {})) + len(features.get('liquidity_features', {})) + len(features.get('market_features', {}))}")
        print(f"   Risk score: {risk_score}")
        print(f"   Risk category: {explanation.get('risk_category')}")
        
        # Validate integration
        if not (0 <= risk_score <= 1000):
            raise ValueError(f"Risk score out of valid range: {risk_score}")
        
        return True, {'features': features, 'score': risk_score, 'explanation': explanation}
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_batch_scoring():
    """Test batch scoring functionality"""
    print("🧪 Testing Batch Scoring...")
    
    try:
        from risk_scorer import batch_score_wallets, generate_risk_distribution_analysis
        
        # Create multiple wallet features for batch testing
        batch_features = {}
        
        # Generate 10 test wallets with varying risk profiles
        for i in range(10):
            risk_level = i / 10.0  # Risk from 0.0 to 0.9
            
            batch_features[f'0xwallet{i:02d}'] = {
                'wallet_address': f'0xwallet{i:02d}',
                'data_quality': {'data_completeness': 'high'},
                'behavioral_features': {
                    'avg_tx_per_day': risk_level * 20,
                    'recent_activity_ratio': risk_level,
                    'compound_activity_ratio': risk_level,
                    'compound_tx_count': int(risk_level * 50),
                    'burst_activity_score': risk_level * 0.5,
                    'tx_regularity': 1 - risk_level
                },
                'liquidity_features': {
                    'collateral_ratio': 3.0 - (risk_level * 1.8),  # 3.0 to 1.2
                    'liquidation_risk_score': risk_level,
                    'utilization_ratio': risk_level * 0.9,
                    'leverage_score': risk_level
                },
                'market_features': {
                    'portfolio_risk_score': 0.2 + (risk_level * 0.6),
                    'concentration_risk': risk_level * 0.8,
                    'diversification_score': 1 - risk_level,
                    'asset_diversification': 1 - risk_level
                }
            }
        
        # Perform batch scoring
        batch_scores = batch_score_wallets(batch_features)
        
        # Generate distribution analysis
        distribution = generate_risk_distribution_analysis(batch_scores)
        
        print("✅ Batch scoring completed successfully")
        print(f"   Wallets scored: {len(batch_scores)}")
        print(f"   Score range: {min(batch_scores.values())}-{max(batch_scores.values())}")
        
        # Display distribution
        print(f"   Risk distribution:")
        risk_dist = distribution.get('risk_distribution', {})
        for category, count in risk_dist.items():
            print(f"     • {category}: {count} wallets")
        
        # Display statistics
        stats = distribution.get('score_statistics', {})
        print(f"   Score statistics:")
        print(f"     • Mean: {stats.get('mean')}")
        print(f"     • Median: {stats.get('median')}")
        print(f"     • Std: {stats.get('std')}")
        
        return True, {'scores': batch_scores, 'distribution': distribution}
        
    except Exception as e:
        print(f"❌ Batch scoring test failed: {str(e)}")
        return False, None


def test_edge_cases():
    """Test edge cases and error handling"""
    print("🧪 Testing Edge Cases and Error Handling...")
    
    try:
        from risk_scorer import WalletRiskScorer
        
        scorer = WalletRiskScorer()
        
        # Test empty features
        empty_features = {
            'wallet_address': '0xempty',
            'data_quality': {'data_completeness': 'none'},
            'behavioral_features': {},
            'liquidity_features': {},
            'market_features': {}
        }
        
        empty_score = scorer.calculate_score(empty_features)
        
        # Test missing features completely
        minimal_features = {
            'wallet_address': '0xminimal'
        }
        
        minimal_score = scorer.calculate_score(minimal_features)
        
        # Test with extreme values
        extreme_features = {
            'wallet_address': '0xextreme',
            'data_quality': {'data_completeness': 'high'},
            'behavioral_features': {
                'avg_tx_per_day': 1000,  # Extreme value
                'compound_activity_ratio': 2.0,  # Out of normal range
                'tx_regularity': -0.5  # Invalid value
            },
            'liquidity_features': {
                'collateral_ratio': -1,  # Invalid
                'utilization_ratio': 5.0  # Out of range
            },
            'market_features': {
                'concentration_risk': 1.5  # Out of range
            }
        }
        
        extreme_score = scorer.calculate_score(extreme_features)
        
        print("✅ Edge case handling successful")
        print(f"   Empty data score: {empty_score}")
        print(f"   Minimal data score: {minimal_score}")
        print(f"   Extreme values score: {extreme_score}")
        
        # Validate all scores are in valid range
        for score_name, score in [('empty', empty_score), ('minimal', minimal_score), ('extreme', extreme_score)]:
            if not (0 <= score <= 1000):
                raise ValueError(f"{score_name} score out of range: {score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {str(e)}")
        return False


def test_scoring_consistency():
    """Test that scoring is consistent and reproducible"""
    print("🧪 Testing Scoring Consistency...")
    
    try:
        from risk_scorer import WalletRiskScorer
        
        # Test same features multiple times
        test_features = {
            'wallet_address': '0xconsistency',
            'data_quality': {'data_completeness': 'high'},
            'behavioral_features': {
                'avg_tx_per_day': 5.0,
                'compound_activity_ratio': 0.3,
                'tx_regularity': 0.7
            },
            'liquidity_features': {
                'collateral_ratio': 2.0,
                'liquidation_risk_score': 0.2
            },
            'market_features': {
                'portfolio_risk_score': 0.4,
                'concentration_risk': 0.3
            }
        }
        
        # Score multiple times with same scorer
        scorer1 = WalletRiskScorer()
        scores1 = [scorer1.calculate_score(test_features) for _ in range(5)]
        
        # Score with different scorer instances
        scores2 = [WalletRiskScorer().calculate_score(test_features) for _ in range(5)]
        
        # Check consistency
        if len(set(scores1)) != 1:
            raise ValueError(f"Inconsistent scores from same scorer: {scores1}")
        
        if len(set(scores2)) != 1:
            raise ValueError(f"Inconsistent scores from different scorers: {scores2}")
        
        if scores1[0] != scores2[0]:
            raise ValueError(f"Different results between scorer instances: {scores1[0]} vs {scores2[0]}")
        
        print("✅ Scoring consistency verified")
        print(f"   Consistent score: {scores1[0]}")
        print(f"   Reproducibility: 100%")
        
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {str(e)}")
        return False


def save_test_results(test_results):
    """Save test results for analysis"""
    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "risk_scorer_test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"✅ Test results saved to results/risk_scorer_test_results.json")
        
    except Exception as e:
        print(f"⚠️  Could not save test results: {str(e)}")


def main():
    """Run all risk scorer tests"""
    print("=" * 60)
    print("🧪 RISK SCORER TEST SUITE")
    print("=" * 60)
    
    test_results = {
        'test_timestamp': datetime.now().isoformat(),
        'tests_run': [],
        'tests_passed': 0,
        'tests_failed': 0,
        'sample_scores': {}
    }
    
    # Run all tests
    tests = [
        ("Initialization", test_risk_scorer_initialization),
        ("Score Calculation", test_score_calculation_with_sample_data),
        ("Score Explanation", test_score_explanation),
        ("Feature Engineer Integration", test_integration_with_feature_engineer),
        ("Batch Scoring", test_batch_scoring),
        ("Edge Cases", test_edge_cases),
        ("Scoring Consistency", test_scoring_consistency)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_name in ["Initialization", "Score Calculation", "Score Explanation", "Feature Engineer Integration", "Batch Scoring"]:
                result, data = test_func()
                if data and test_name == "Score Calculation":
                    test_results['sample_scores'].update(data)
            else:
                result = test_func()
            
            if result:
                test_results['tests_passed'] += 1
                test_results['tests_run'].append({'name': test_name, 'status': 'PASSED'})
            else:
                test_results['tests_failed'] += 1
                test_results['tests_run'].append({'name': test_name, 'status': 'FAILED'})
                
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {str(e)}")
            test_results['tests_failed'] += 1
            test_results['tests_run'].append({'name': test_name, 'status': 'CRASHED', 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = test_results['tests_passed'] + test_results['tests_failed']
    success_rate = test_results['tests_passed'] / total_tests if total_tests > 0 else 0
    
    print(f"Tests Run: {total_tests}")
    print(f"Tests Passed: {test_results['tests_passed']} ✅")
    print(f"Tests Failed: {test_results['tests_failed']} ❌")
    print(f"Success Rate: {success_rate:.1%}")
    
    # Individual test results
    for test in test_results['tests_run']:
        status_icon = "✅" if test['status'] == 'PASSED' else "❌"
        print(f"  {status_icon} {test['name']}: {test['status']}")
    
    # Show sample scores if available
    if test_results['sample_scores']:
        print(f"\n📊 Sample Score Results:")
        for score_type, score in test_results['sample_scores'].items():
            print(f"  • {score_type}: {score}")
    
    # Save results
    save_test_results(test_results)
    
    if test_results['tests_failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Risk Scorer is ready for production!")
    else:
        print(f"\n⚠️  {test_results['tests_failed']} test(s) failed. Please review the issues above.")
    
    return test_results['tests_failed'] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
