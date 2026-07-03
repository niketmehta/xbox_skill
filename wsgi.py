"""Gunicorn entry point for the trading dashboard."""

import logging

import app as dashboard
from trading_agent import TradingAgent


logging.basicConfig(level=logging.INFO)

if dashboard.trading_agent is None:
    dashboard.trading_agent = TradingAgent()

app = dashboard.app
