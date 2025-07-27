# 🎯 Wallet Risk Scoring System

A comprehensive DeFi wallet risk analysis system that processes blockchain transaction data to generate risk scores for lending protocol assessment.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)

## 🚀 Overview

This system analyzes wallet addresses on Ethereum blockchain to assess risk profiles for DeFi lending protocols. It processes transaction history from Compound V2/V3 protocols and generates risk scores using a weighted composite scoring methodology.

### Key Features

- 🔍 **Multi-source Data Collection**: Etherscan API + The Graph + Compound API integration
- ⚙️ **Advanced Feature Engineering**: 36+ risk indicators across behavioral, liquidity, and market categories
- 📊 **Weighted Composite Scoring**: Transparent, interpretable risk assessment (0-1000 scale)
- 🚀 **Production Ready**: Enterprise-grade error handling, logging, and monitoring
- 🧪 **Comprehensive Testing**: 100% test coverage with validation suite
- 💻 **Windows Optimized**: Lightweight, CPU-efficient processing

## 📈 Results

Successfully processed **101 wallets** with perfect execution:

- **✅ 100% Success Rate**: All wallets processed without errors
- **📊 Risk Distribution**: 41.6% low risk, 58.4% medium risk, 0% high risk
- **🎯 Score Range**: 183-485 (well-calibrated, no extreme outliers)
- **⚡ Performance**: ~4-8 seconds per wallet with rate limiting

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Windows 10/11 (optimized for Windows, but cross-platform compatible)
- API keys (Etherscan - optional but recommended)

### Quick Start

Clone the repository
git clone https://github.com/yourusername/wallet-risk-scoring-system.git
cd wallet-risk-scoring-system

Create virtual environment
python -m venv wallet_risk_env
wallet_risk_env\Scripts\activate # Windows

source wallet_risk_env/bin/activate # macOS/Linux
Install dependencies
pip install -r requirements.txt

Setup environment variables (optional)
cp .env.example .env

Edit .env file with your API keys
Run tests
python tests/test_feature_engineer.py
python tests/test_risk_scorer.py

Test with sample wallets
python src/main.py --test

Full production run
python src/main.py

text

## 🎯 Usage

### Basic Usage

from src.main import WalletRiskScoringPipeline

Initialize pipeline
pipeline = WalletRiskScoringPipeline()

Process wallets
pipeline.load_wallet_addresses("data/wallet_list.csv")
pipeline.collect_data()
feature_data = pipeline.engineer_features()
risk_scores = pipeline.calculate_risk_scores(feature_data)
pipeline.generate_final_results()

text

### Individual Components

from src.data_collector import CompoundDataCollector
from src.feature_engineer import RiskFeatureEngineer
from src.risk_scorer import WalletRiskScorer

Data collection
collector = CompoundDataCollector()
wallet_data = collector.fetch_wallet_data("0x742d35cc...")

Feature engineering
engineer = RiskFeatureEngineer()
features = engineer.extract_features(wallet_data)

Risk scoring
scorer = WalletRiskScorer()
risk_score = scorer.calculate_score(features)
print(f"Risk Score: {risk_score}/1000")

text

## 📊 Methodology

### Risk Scoring Approach

**Weighted Composite Scoring** with three main categories:

risk_score = (
behavioral_score * 0.40 + # Transaction patterns, frequency, DeFi activity
liquidity_score * 0.35 + # Collateral ratios, liquidation risk, leverage
market_score * 0.25 # Portfolio concentration, diversification
) * 1000 # Scale to 0-1000

text

### Feature Categories

1. **Behavioral Risk (40%)**
   - Transaction frequency and patterns
   - Average transaction sizes and volatility
   - Compound protocol activity levels
   - Temporal transaction patterns

2. **Liquidity Risk (35%)**
   - Collateral ratio safety margins
   - Liquidation risk proximity
   - Leverage usage patterns
   - Position size analysis

3. **Market Risk (25%)**
   - Portfolio risk composition
   - Asset concentration risk
   - Diversification metrics
   - Exposure to volatile assets

### Why Weighted Composite vs ML?

- **Interpretability**: Financial risk models require transparent logic
- **Data Constraints**: Limited labeled training data (101 wallets)
- **Performance**: Instant scoring without training overhead
- **Scalability**: CPU-optimized for production deployment

## 📁 Project Structure

src/
├── main.py # Main execution pipeline
├── data_collector.py # Multi-source data collection
├── feature_engineer.py # Risk feature extraction
├── risk_scorer.py # Weighted composite scoring
└── utils.py # Utility functions

tests/
├── test_feature_engineer.py # Feature engineering tests
├── test_risk_scorer.py # Risk scoring tests
└── test_phase2.py # Integration tests

notebooks/
└── analysis.ipynb # Results analysis and visualization

text

## 🔧 Configuration

### Environment Variables

API Configuration (optional)
ETHERSCAN_API_KEY=your_etherscan_api_key
API_RATE_LIMIT=5
MAX_RETRIES=3
TIMEOUT_SECONDS=30

Data Collection Settings
LOOKBACK_DAYS=365
MIN_TRANSACTIONS=5

text

## 📊 Sample Results

### Risk Score Distribution

Total Wallets: 101
Average Score: 272.3
Score Range: 183-485

Risk Categories:
├── Low Risk (0-250): 42 wallets (41.6%)
├── Medium Risk (251-750): 59 wallets (58.4%)
└── High Risk (751-1000): 0 wallets (0.0%)

text

### Top Risk Wallets

| Rank | Wallet Address | Risk Score |
|------|---------------|------------|
| 1    | 0x4814be124d7fe3b240eb46061f7ddfab468fe122 | 485 |
| 2    | 0x427f2ac5fdf4245e027d767e7c3ac272a1f40a65 | 451 |
| 3    | 0x70d8e4ab175dfe0eab4e9a7f33e0a2d19f44001e | 448 |

## 🧪 Testing

Run the comprehensive test suite:

Individual component tests
python tests/test_feature_engineer.py # Feature engineering validation
python tests/test_risk_scorer.py # Risk scoring validation
python tests/test_phase2.py # Integration testing

All tests should show 100% pass rate
text

## 📈 Performance

- **Processing Speed**: ~4-8 seconds per wallet
- **Memory Usage**: Lightweight, optimized for standard hardware
- **API Efficiency**: Rate-limited calls with retry logic
- **Error Rate**: 0% (101/101 wallets processed successfully)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Compound Protocol** for DeFi infrastructure
- **Etherscan** for blockchain data access
- **The Graph Protocol** for decentralized indexing

## 📞 Contact

**Project Author**: Anushk Jain
- 📧 Email: jainanushk8@gmail.com
- 💼 LinkedIn: (https://www.linkedin.com/in/anushk-jain-bb7b71222/)

---

⭐ **Star this repository if you found it helpful!**