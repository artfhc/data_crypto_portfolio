# Crypto Portfolio Analyzer

A comprehensive Streamlit application for analyzing cryptocurrency portfolio performance, tracking gains/losses, and visualizing asset price movements over time.

## Features

- **Portfolio Analysis**: Calculate realized and unrealized gains/losses for each asset
- **Price Visualization**: Interactive charts showing asset prices over time with transaction markers
- **Portfolio Tracking**: Monitor total portfolio value changes over time
- **Transaction Management**: View and filter complete transaction history
- **Real-time Prices**: Fetch current cryptocurrency prices using Yahoo Finance
- **Asset Breakdown**: Detailed analysis of individual cryptocurrency holdings

## Setup

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data_crypto_portfolio
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Option 1 - Using the provided script:
```bash
./run_app.sh
```

Option 2 - Manual execution:
```bash
source venv/bin/activate
streamlit run crypto_portfolio_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Data Format

Place your Coinbase transaction CSV exports in the `data/` directory. The application expects files with the naming pattern:
```
YYYYMMDD_coinbase_transactions.csv
```

The CSV should contain the following columns:
- ID, Timestamp, Transaction Type, Asset, Quantity Transacted
- Price Currency, Price at Transaction, Subtotal
- Total (inclusive of fees and/or spread), Fees and/or Spread, Notes

## Application Sections

1. **Portfolio Summary**: Overview of all holdings, gains/losses, and portfolio allocation
2. **Asset Analysis**: Detailed view of individual cryptocurrency performance
3. **Price Charts**: Historical price data with buy/sell transaction markers
4. **Transaction History**: Complete filterable transaction log

## Supported Assets

The application supports major cryptocurrencies including:
- Bitcoin (BTC)
- Solana (SOL)
- Ethereum (ETH)
- USD Coin (USDC)
- Cardano (ADA)
- Polkadot (DOT)

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- requests
- yfinance

## License

MIT License - see LICENSE file for details.
