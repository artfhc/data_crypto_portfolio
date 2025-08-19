# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency portfolio tracking project that analyzes crypto transactions. The project currently contains transaction data from Coinbase exports in CSV format.

## Data Structure

### Transaction Data Format
The primary data source is Coinbase transaction exports stored in the `data/` directory. CSV files contain the following key fields:
- ID: Unique transaction identifier
- Timestamp: Transaction date/time in UTC
- Transaction Type: (Reward Income, Staking Income, Buy, Sell, etc.)
- Asset: Cryptocurrency symbol (USDC, SOL, etc.)
- Quantity Transacted: Amount of crypto
- Price Currency: Usually USD
- Price at Transaction: Price per unit at time of transaction
- Total (inclusive of fees and/or spread): Final transaction amount
- Fees and/or Spread: Transaction costs

### File Naming Convention
CSV files follow the pattern: `YYYYMMDD_coinbase_transactions.csv`

## Development Notes

- This is an early-stage project focused on data analysis and portfolio tracking
- No build tools, test frameworks, or dependency management files are currently present
- The project appears to be data-centric rather than application-centric
- Transaction data includes various types: rewards, staking income, and trading activities