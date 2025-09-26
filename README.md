# Portfolio Optimization App

A full-stack system for portfolio optimization built with Python, applying **Modern Portfolio Theory (MPT)**.

# Goal  
The goal of this app is to help **beginner investors** build smarter portfolios by achieving the **best possible risk-to-reward ratio** based on their individual **risk tolerance** — whether they prefer a **conservative**, **moderate**, or **aggressive** investment strategy.

## Features

- Collects and processes 5+ years of historical stock data via **Yahoo Finance API (yfinance)**.
- Cleans and transforms financial data for portfolio analysis.
- Runs **backtesting workflows** with automated performance analytics.
- Compares optimized portfolios against baseline benchmarks.
- Interactive **Streamlit dashboard** with:
  - Real-time efficient frontier visualization
  - Investment recommendations
  - Portfolio analytics

## Tech Stack

- **Python**: NumPy, pandas, SciPy
- **Data Source**: Yahoo Finance API (yfinance)
- **Dashboard**: Streamlit

## Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/portfolio_optimization_app.git

   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```

3. Launch the app:

   ```bash
   streamlit run streamlit-portfolio-optimizer.py

   ```
