import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
from pathlib import Path
import os
from dotenv import load_dotenv
import requests

load_dotenv()

# Translation dictionaries
translations = {
    'en': {
        'page_title': 'Small Crypto Portfolio Analyzer',
        'portfolio_overview': 'Portfolio Overview',
        'total_portfolio_value': 'Total Portfolio Value',
        'total_realized_gains': 'Total Realized Gains',
        'total_unrealized_gains': 'Total Unrealized Gains',
        'total_gains_losses': 'Total Gains/Losses',
        'portfolio_summary': 'Portfolio Summary',
        'asset_analysis': 'Asset Analysis',
        'price_charts': 'Price Charts',
        'transaction_history': 'Transaction History',
        'portfolio_holdings_gains': 'Portfolio Holdings & Gains/Losses',
        'asset': 'Asset',
        'quantity': 'Quantity',
        'avg_cost': 'Avg Cost',
        'current_price': 'Current Price',
        'current_value': 'Current Value',
        'realized_gain_loss': 'Realized Gain/Loss',
        'unrealized_gain_loss': 'Unrealized Gain/Loss',
        'total_gain_loss': 'Total Gain/Loss',
        'portfolio_allocation': 'Portfolio Allocation',
        'gains_losses_by_asset': 'Gains/Losses by Asset',
        'detailed_asset_analysis': 'Detailed Asset Analysis',
        'select_asset_analysis': 'Select Asset for Detailed Analysis',
        'current_holdings': 'Current Holdings',
        'transaction_history_asset': 'Transaction History',
        'asset_price_charts': 'Asset Price Charts',
        'select_asset_chart': 'Select Asset for Price Chart',
        'price_over_time': 'Price Over Time with Transactions',
        'buy_transactions': 'Buy Transactions',
        'sell_transactions': 'Sell Transactions',
        'complete_transaction_history': 'Complete Transaction History',
        'filter_by_asset': 'Filter by Asset',
        'filter_by_type': 'Filter by Transaction Type',
        'portfolio_value_over_time': 'Portfolio Value Over Time',
        'date': 'Date',
        'type': 'Type',
        'price': 'Price',
        'total': 'Total',
        'no_historical_data': 'No historical data available for',
        'error_fetching_data': 'Error fetching price data:',
        'language': 'Language'
    },
    'zh_HK': {
        'page_title': '小型加密貨幣投資組合分析器',
        'portfolio_overview': '投資組合概覽',
        'total_portfolio_value': '投資組合總價值',
        'total_realized_gains': '已實現收益總額',
        'total_unrealized_gains': '未實現收益總額',
        'total_gains_losses': '總收益/虧損',
        'portfolio_summary': '投資組合摘要',
        'asset_analysis': '資產分析',
        'price_charts': '價格圖表',
        'transaction_history': '交易記錄',
        'portfolio_holdings_gains': '投資組合持股及收益/虧損',
        'asset': '資產',
        'quantity': '數量',
        'avg_cost': '平均成本',
        'current_price': '現時價格',
        'current_value': '現時價值',
        'realized_gain_loss': '已實現收益/虧損',
        'unrealized_gain_loss': '未實現收益/虧損',
        'total_gain_loss': '總收益/虧損',
        'portfolio_allocation': '投資組合配置',
        'gains_losses_by_asset': '按資產劃分的收益/虧損',
        'detailed_asset_analysis': '詳細資產分析',
        'select_asset_analysis': '選擇資產進行詳細分析',
        'current_holdings': '目前持股',
        'transaction_history_asset': '交易記錄',
        'asset_price_charts': '資產價格圖表',
        'select_asset_chart': '選擇資產查看價格圖表',
        'price_over_time': '價格走勢及交易記錄',
        'buy_transactions': '買入交易',
        'sell_transactions': '賣出交易',
        'complete_transaction_history': '完整交易記錄',
        'filter_by_asset': '按資產篩選',
        'filter_by_type': '按交易類型篩選',
        'portfolio_value_over_time': '投資組合價值走勢',
        'date': '日期',
        'type': '類型',
        'price': '價格',
        'total': '總計',
        'no_historical_data': '沒有歷史數據可供查看：',
        'error_fetching_data': '獲取價格數據時出錯：',
        'language': '語言'
    }
}

st.set_page_config(page_title="Crypto Portfolio Analyzer", layout="wide")

def get_theme_colors():
    """Get dark mode color scheme"""
    return {
        'bg_color': '#0E1117',
        'paper_bg': '#262730',
        'text_color': '#FAFAFA',
        'grid_color': '#444444',
        'line_color': '#00D4FF',  # Bright blue for dark mode
        'positive_color': '#00FF88',  # Bright green
        'negative_color': '#FF4B4B',  # Bright red
    }

def apply_theme_to_fig(fig, theme_colors):
    """Apply theme colors to a plotly figure"""
    fig.update_layout(
        plot_bgcolor=theme_colors['bg_color'],
        paper_bgcolor=theme_colors['paper_bg'],
        font_color=theme_colors['text_color'],
        xaxis=dict(
            gridcolor=theme_colors['grid_color'],
            color=theme_colors['text_color']
        ),
        yaxis=dict(
            gridcolor=theme_colors['grid_color'],
            color=theme_colors['text_color']
        ),
        legend=dict(
            font_color=theme_colors['text_color']
        )
    )
    return fig

@st.cache_data
def load_transaction_data():
    """Load and clean transaction data from CSV"""
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*coinbase_transactions.csv"))
    
    if not csv_files:
        st.error("No transaction CSV files found in data directory")
        return None
    
    # Use the most recent file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    # Read CSV, skipping the first 3 rows (header info)
    df = pd.read_csv(latest_file, skiprows=3)
    
    # Get allowed assets from environment variable
    allowed_assets = os.getenv('ALLOWED_ASSETS', 'BTC,LINK,SOL').split(',')
    allowed_assets = [asset.strip() for asset in allowed_assets]
    
    # Filter data to only include allowed assets
    df = df[df['Asset'].isin(allowed_assets)]
    
    if df.empty:
        st.error(f"No transactions found for allowed assets: {', '.join(allowed_assets)}")
        return None
    
    # Clean and process the data
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Price at Transaction'] = df['Price at Transaction'].str.replace('$', '').astype(float)
    df['Quantity Transacted'] = df['Quantity Transacted'].astype(float)
    df['Total (inclusive of fees and/or spread)'] = df['Total (inclusive of fees and/or spread)'].str.replace('$', '').astype(float)
    df['Fees and/or Spread'] = df['Fees and/or Spread'].str.replace('$', '').astype(float)
    
    return df.sort_values('Timestamp')

@st.cache_data
def get_historical_prices(assets, start_date, end_date=None):
    """Get historical prices for assets using yfinance"""
    historical_prices = {}
    
    # Mapping of crypto symbols to Yahoo Finance tickers
    ticker_map = {
        'BTC': 'BTC-USD',
        'SOL': 'SOL-USD',
        'LINK': 'LINK-USD',
        'USDC': 'USDC-USD',
        'ETH': 'ETH-USD',
        'ADA': 'ADA-USD',
        'DOT': 'DOT-USD'
    }
    
    for asset in assets:
        if asset == 'USDC':
            # USDC is a stablecoin, always $1
            date_range = pd.date_range(start=start_date, end=end_date or pd.Timestamp.now().date(), freq='D')
            usdc_series = pd.Series(1.0, index=date_range.date)
            historical_prices[asset] = usdc_series
        elif asset in ticker_map:
            try:
                ticker = yf.Ticker(ticker_map[asset])
                if end_date:
                    hist = ticker.history(start=start_date, end=end_date)
                else:
                    hist = ticker.history(start=start_date)
                
                if not hist.empty:
                    # Create a dictionary with date -> price mapping
                    price_series = hist['Close'].copy()
                    price_series.index = price_series.index.date  # Convert to date for easier matching
                    historical_prices[asset] = price_series
                else:
                    historical_prices[asset] = pd.Series(dtype=float)
            except Exception as e:
                st.warning(f"Could not fetch historical data for {asset}: {e}")
                historical_prices[asset] = pd.Series(dtype=float)
        else:
            # For unknown assets, create empty series
            historical_prices[asset] = pd.Series(dtype=float)
    
    return historical_prices

@st.cache_data
def get_current_prices(assets):
    """Get current prices for assets using cryptoprices.cc API"""
    current_prices = {}
    
    for asset in assets:
        if asset == 'USDC':
            # USDC is a stablecoin, always $1
            current_prices[asset] = 1.0
        else:
            try:
                # Fetch from cryptoprices.cc API
                response = requests.get(f"https://cryptoprices.cc/{asset}", timeout=10)
                if response.status_code == 200:
                    price = float(response.text.strip())
                    current_prices[asset] = price
                else:
                    # Fallback to yfinance if cryptoprices.cc fails
                    current_prices[asset] = get_fallback_price(asset)
            except Exception as e:
                st.warning(f"Failed to fetch price for {asset} from cryptoprices.cc: {e}")
                # Fallback to yfinance
                current_prices[asset] = get_fallback_price(asset)
    
    return current_prices

def get_fallback_price(asset):
    """Fallback price fetching using yfinance"""
    ticker_map = {
        'BTC': 'BTC-USD',
        'SOL': 'SOL-USD',
        'LINK': 'LINK-USD',
        'USDC': 'USDC-USD',
        'ETH': 'ETH-USD',
        'ADA': 'ADA-USD',
        'DOT': 'DOT-USD'
    }
    
    if asset in ticker_map:
        try:
            ticker = yf.Ticker(ticker_map[asset])
            hist = ticker.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            pass
    
    return 1.0 if asset == 'USDC' else 0.0

def calculate_portfolio_holdings(df):
    """Calculate current holdings for each asset"""
    holdings = {}
    
    for _, row in df.iterrows():
        asset = row['Asset']
        quantity = row['Quantity Transacted']
        transaction_type = row['Transaction Type']
        
        if asset not in holdings:
            holdings[asset] = {
                'quantity': 0,
                'total_cost': 0,
                'transactions': []
            }
        
        # Handle different transaction types
        if 'Sell' in transaction_type:
            quantity = -abs(quantity)  # Ensure sells are negative
        
        holdings[asset]['quantity'] += quantity
        holdings[asset]['total_cost'] += row['Total (inclusive of fees and/or spread)']
        holdings[asset]['transactions'].append({
            'date': row['Timestamp'],
            'quantity': quantity,
            'price': row['Price at Transaction'],
            'total': row['Total (inclusive of fees and/or spread)'],
            'type': transaction_type
        })
    
    return holdings

def calculate_gains_losses(holdings, current_prices):
    """Calculate realized and unrealized gains/losses"""
    results = {}
    
    for asset, data in holdings.items():
        current_quantity = data['quantity']
        current_price = current_prices.get(asset, 0)
        
        # Calculate average cost basis
        total_bought = 0
        total_cost = 0
        realized_gain = 0
        
        for tx in data['transactions']:
            if tx['quantity'] > 0:  # Buy transactions
                total_bought += tx['quantity']
                total_cost += tx['total']
            else:  # Sell transactions
                # Calculate realized gain for this sale
                if total_bought > 0:
                    avg_cost = total_cost / total_bought
                    sale_quantity = abs(tx['quantity'])
                    cost_basis = avg_cost * sale_quantity
                    sale_proceeds = abs(tx['total'])
                    realized_gain += sale_proceeds - cost_basis
                    
                    # Update remaining cost basis
                    remaining_ratio = max(0, (total_bought - sale_quantity) / total_bought)
                    total_cost *= remaining_ratio
                    total_bought -= sale_quantity
        
        # Calculate unrealized gain/loss on remaining holdings
        if current_quantity > 0 and total_bought > 0:
            avg_cost = total_cost / total_bought if total_bought > 0 else 0
            current_value = current_quantity * current_price
            cost_basis_remaining = avg_cost * current_quantity
            unrealized_gain = current_value - cost_basis_remaining
        else:
            unrealized_gain = 0
            current_value = 0
            avg_cost = 0
        
        results[asset] = {
            'quantity': current_quantity,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'current_value': current_value,
            'realized_gain': realized_gain,
            'unrealized_gain': unrealized_gain,
            'total_gain': realized_gain + unrealized_gain
        }
    
    return results

def main():
    # Language selection in sidebar
    language = st.sidebar.selectbox(
        "Language / 語言",
        options=['en', 'zh_HK'],
        format_func=lambda x: "English" if x == 'en' else "繁體中文 (香港)"
    )
    
    # Get translations for selected language
    t = translations[language]
    
    # Get theme colors (dark mode only)
    theme_colors = get_theme_colors()
    
    # Apply custom CSS for dark mode
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #444444;
    }
    .stDataFrame {
        background-color: #262730;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_transaction_data()
    if df is None:
        return
    
    # Get date range from the data
    date_range = f"{df['Timestamp'].min().strftime('%Y-%m-%d')} to {df['Timestamp'].max().strftime('%Y-%m-%d')}"
    
    st.title(f"🚀 {t['page_title']} ({date_range})")
    
    st.sidebar.header(t['portfolio_overview'])
    
    # Calculate holdings and gains
    holdings = calculate_portfolio_holdings(df)
    assets = list(holdings.keys())
    current_prices = get_current_prices(assets)
    gains_losses = calculate_gains_losses(holdings, current_prices)
    
    # Portfolio summary
    total_value = sum(data['current_value'] for data in gains_losses.values())
    total_realized = sum(data['realized_gain'] for data in gains_losses.values())
    total_unrealized = sum(data['unrealized_gain'] for data in gains_losses.values())
    total_gain = total_realized + total_unrealized
    
    # Calculate total cost basis for percentage calculations
    total_cost_basis = total_value - total_gain if total_gain != 0 else total_value
    
    # Calculate percentages
    if total_cost_basis > 0:
        total_realized_pct = (total_realized / total_cost_basis) * 100
        total_unrealized_pct = (total_unrealized / total_cost_basis) * 100
        total_gain_pct = (total_gain / total_cost_basis) * 100
    else:
        total_realized_pct = total_unrealized_pct = total_gain_pct = 0
    
    st.sidebar.metric(t['total_portfolio_value'], f"${total_value:,.2f}")
    st.sidebar.metric(t['total_realized_gains'], f"${total_realized:,.2f}", f"{total_realized_pct:.1f}%")
    st.sidebar.metric(t['total_unrealized_gains'], f"${total_unrealized:,.2f}", f"{total_unrealized_pct:.1f}%")
    st.sidebar.metric(t['total_gains_losses'], f"${total_gain:,.2f}", f"{total_gain_pct:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([t['portfolio_summary'], t['asset_analysis'], t['price_charts'], t['transaction_history']])
    
    with tab1:
        st.header(t['portfolio_holdings_gains'])
        
        # Create portfolio summary table
        portfolio_data = []
        for asset, data in gains_losses.items():
            # Calculate percentage gains/losses
            total_cost = data['current_value'] - data['total_gain'] if data['total_gain'] != 0 else data['current_value']
            
            if total_cost > 0:
                realized_pct = (data['realized_gain'] / total_cost) * 100
                unrealized_pct = (data['unrealized_gain'] / total_cost) * 100
                total_pct = (data['total_gain'] / total_cost) * 100
            else:
                realized_pct = unrealized_pct = total_pct = 0
            
            portfolio_data.append({
                t['asset']: asset,
                t['quantity']: f"{data['quantity']:.8f}",
                t['avg_cost']: f"${data['avg_cost']:.2f}",
                t['current_price']: f"${data['current_price']:.2f}",
                t['current_value']: f"${data['current_value']:.2f}",
                t['realized_gain_loss']: f"${data['realized_gain']:.2f} ({realized_pct:.1f}%)",
                t['unrealized_gain_loss']: f"${data['unrealized_gain']:.2f} ({unrealized_pct:.1f}%)",
                t['total_gain_loss']: f"${data['total_gain']:.2f} ({total_pct:.1f}%)",
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Portfolio allocation pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            values = [data['current_value'] for data in gains_losses.values() if data['current_value'] > 0]
            labels = [asset for asset, data in gains_losses.items() if data['current_value'] > 0]
            
            if values:
                fig_pie = px.pie(values=values, names=labels, title=t['portfolio_allocation'])
                fig_pie = apply_theme_to_fig(fig_pie, theme_colors)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gains/Losses bar chart
            gains_data = [(asset, data['total_gain']) for asset, data in gains_losses.items()]
            gains_df = pd.DataFrame(gains_data, columns=[t['asset'], t['total_gain_loss']])
            
            fig_bar = px.bar(gains_df, x=t['asset'], y=t['total_gain_loss'], 
                           title=t['gains_losses_by_asset'],
                           color=t['total_gain_loss'],
                           color_continuous_scale=[theme_colors['negative_color'], theme_colors['positive_color']])
            fig_bar = apply_theme_to_fig(fig_bar, theme_colors)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Portfolio value over time calculation
        st.header(t['portfolio_value_over_time'])
        
        # Get historical prices for all assets
        df_sorted = df.sort_values('Timestamp')
        start_date = df_sorted['Timestamp'].min().strftime('%Y-%m-%d')
        end_date = df_sorted['Timestamp'].max().strftime('%Y-%m-%d')
        
        with st.spinner("Fetching historical price data..."):
            historical_prices = get_historical_prices(assets, start_date, end_date)
        
        # Calculate cumulative portfolio value using historical prices
        portfolio_values = []
        running_holdings = {}
        
        for _, row in df_sorted.iterrows():
            asset = row['Asset']
            quantity = row['Quantity Transacted']
            transaction_date = row['Timestamp'].date()
            
            if 'Sell' in row['Transaction Type']:
                quantity = -abs(quantity)
            
            if asset not in running_holdings:
                running_holdings[asset] = 0
            running_holdings[asset] += quantity
            
            # Calculate portfolio value at this point in time using historical prices
            total_value = 0
            for held_asset, held_quantity in running_holdings.items():
                if held_quantity > 0:
                    # Try to get historical price for this date
                    if held_asset in historical_prices and len(historical_prices[held_asset]) > 0:
                        # Find the closest price date
                        price_series = historical_prices[held_asset]
                        if transaction_date in price_series.index:
                            price = price_series[transaction_date]
                        else:
                            # Find the closest available price
                            available_dates = price_series.index
                            if len(available_dates) > 0:
                                closest_date = min(available_dates, key=lambda x: abs((x - transaction_date).days))
                                price = price_series[closest_date]
                            else:
                                price = current_prices.get(held_asset, 0)  # Fallback to current price
                    else:
                        price = current_prices.get(held_asset, 0)  # Fallback to current price
                    
                    total_value += held_quantity * price
            
            portfolio_values.append({
                t['date']: row['Timestamp'],
                t['portfolio_value_over_time']: total_value
            })
        
        if portfolio_values:
            portfolio_df = pd.DataFrame(portfolio_values)
            
            fig_portfolio = px.line(portfolio_df, x=t['date'], y=t['portfolio_value_over_time'],
                                  title=t['portfolio_value_over_time'])
            fig_portfolio.update_layout(
                xaxis_title=t['date'],
                yaxis_title=f"{t['portfolio_value_over_time']} (USD)"
            )
            fig_portfolio.update_traces(line_color=theme_colors['line_color'])
            fig_portfolio = apply_theme_to_fig(fig_portfolio, theme_colors)
            st.plotly_chart(fig_portfolio, use_container_width=True)
    
    with tab2:
        st.header(t['detailed_asset_analysis'])
        
        selected_asset = st.selectbox(t['select_asset_analysis'], assets)
        
        if selected_asset:
            asset_data = gains_losses[selected_asset]
            asset_holdings = holdings[selected_asset]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t['current_holdings'], f"{asset_data['quantity']:.8f} {selected_asset}")
            col2.metric(t['avg_cost'], f"${asset_data['avg_cost']:.2f}")
            col3.metric(t['current_price'], f"${asset_data['current_price']:.2f}")
            col4.metric(t['current_value'], f"${asset_data['current_value']:.2f}")
            
            # Transaction history for selected asset
            asset_transactions = []
            for tx in asset_holdings['transactions']:
                asset_transactions.append({
                    t['date']: tx['date'].strftime('%Y-%m-%d %H:%M'),
                    t['type']: tx['type'],
                    t['quantity']: tx['quantity'],
                    t['price']: f"${tx['price']:.2f}",
                    t['total']: f"${tx['total']:.2f}"
                })
            
            st.subheader(f"{selected_asset} {t['transaction_history_asset']}")
            st.dataframe(pd.DataFrame(asset_transactions), use_container_width=True)
    
    with tab3:
        st.header(t['asset_price_charts'])
        
        # Get historical price data for visualization
        selected_chart_asset = st.selectbox(t['select_asset_chart'], 
                                          [asset for asset in assets if asset != 'USDC'])
        
        if selected_chart_asset:
            ticker_map = {
                'BTC': 'BTC-USD',
                'SOL': 'SOL-USD',
                'LINK': 'LINK-USD',
                'ETH': 'ETH-USD',
                'ADA': 'ADA-USD',
                'DOT': 'DOT-USD'
            }
            
            if selected_chart_asset in ticker_map:
                try:
                    ticker = yf.Ticker(ticker_map[selected_chart_asset])
                    
                    # Get the date range from transaction data to fetch appropriate historical data
                    asset_transactions = holdings[selected_chart_asset]['transactions']
                    if asset_transactions:
                        earliest_date = min(tx['date'] for tx in asset_transactions)
                        # Fetch data from a bit before the earliest transaction to show context
                        start_date = (earliest_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                        hist_data = ticker.history(start=start_date)
                    else:
                        # Fallback to 2 years if no transactions found
                        hist_data = ticker.history(period='2y')
                    
                    if not hist_data.empty:
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(
                            x=hist_data.index, 
                            y=hist_data['Close'],
                            mode='lines',
                            name=f'{selected_chart_asset} Price',
                            line=dict(color=theme_colors['line_color'])
                        ))
                        
                        # Add transaction points
                        asset_txs = holdings[selected_chart_asset]['transactions']
                        buy_dates = [tx['date'] for tx in asset_txs if tx['quantity'] > 0]
                        buy_prices = [tx['price'] for tx in asset_txs if tx['quantity'] > 0]
                        sell_dates = [tx['date'] for tx in asset_txs if tx['quantity'] < 0]
                        sell_prices = [tx['price'] for tx in asset_txs if tx['quantity'] < 0]
                        
                        if buy_dates:
                            fig_price.add_trace(go.Scatter(
                                x=buy_dates, y=buy_prices,
                                mode='markers',
                                name=t['buy_transactions'],
                                marker=dict(color=theme_colors['positive_color'], size=12, symbol='triangle-up', 
                                          line=dict(width=2, color='darkgreen'))
                            ))
                        
                        if sell_dates:
                            fig_price.add_trace(go.Scatter(
                                x=sell_dates, y=sell_prices,
                                mode='markers',
                                name=t['sell_transactions'],
                                marker=dict(color=theme_colors['negative_color'], size=12, symbol='triangle-down', 
                                          line=dict(width=2, color='darkred'))
                            ))
                        
                        fig_price.update_layout(
                            title=f'{selected_chart_asset} {t["price_over_time"]}',
                            xaxis_title=t['date'],
                            yaxis_title=f'{t["price"]} (USD)',
                            hovermode='x unified'
                        )
                        
                        fig_price = apply_theme_to_fig(fig_price, theme_colors)
                        st.plotly_chart(fig_price, use_container_width=True)
                    else:
                        st.warning(f"{t['no_historical_data']} {selected_chart_asset}")
                        
                except Exception as e:
                    st.error(f"{t['error_fetching_data']} {e}")
    
    with tab4:
        st.header(t['complete_transaction_history'])
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            asset_filter = st.multiselect(t['filter_by_asset'], assets, default=assets)
        with col2:
            transaction_types = df['Transaction Type'].unique()
            type_filter = st.multiselect(t['filter_by_type'], 
                                       transaction_types, default=transaction_types)
        
        # Apply filters
        filtered_df = df[
            (df['Asset'].isin(asset_filter)) & 
            (df['Transaction Type'].isin(type_filter))
        ].copy()
        
        # Display filtered data
        display_df = filtered_df[['Timestamp', 'Transaction Type', 'Asset', 
                                'Quantity Transacted', 'Price at Transaction',
                                'Total (inclusive of fees and/or spread)', 'Notes']]
        
        st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()