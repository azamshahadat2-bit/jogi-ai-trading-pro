# JOGI AI TRADING PRO

Free Streamlit website for educational BTCUSDT / ETHUSDT / SOLUSDT signal analysis.

## Files
- `app.py` — Streamlit dashboard and model code
- `requirements.txt` — Python packages

## Deploy free
1. Create a GitHub account.
2. Create a **public** repository named `jogi-ai-trading-pro`.
3. Upload `app.py` and `requirements.txt` to the repository root.
4. Open https://share.streamlit.io/ and sign in.
5. Choose **Create app**, select your repository, branch `main`, and file `app.py`.
6. Click **Deploy**.

## Notes
- Uses public Binance market candles; no Binance API key is required.
- The app does not connect to Binarium or place trades.
- First model training may take several minutes and free hosting resources are limited.
- ICT/SMC Order Block and Breaker features are simplified approximations.
- Scores are model outputs, not calibrated win probabilities or profit guarantees.
- Use demo/paper testing. Binance prices can differ from the broker's quote.
