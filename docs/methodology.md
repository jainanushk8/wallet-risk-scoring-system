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