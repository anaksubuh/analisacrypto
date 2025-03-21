import requests
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import json
from sklearn.linear_model import LinearRegression

'''

Menganalisa semua coin dan memasukan ke dalam analisa.txt dan membuat file json

'''

# 🔹 Baca daftar koin dari file coingacor.txt
try:
    with open("coingacor.txt", "r", encoding="utf-8") as file:
        coin_list = [line.strip() for line in file.readlines() if line.strip()]
except FileNotFoundError:
    print("File coingacor.txt tidak ditemukan!")
    coin_list = []

# 🔹 Simpan hasil analisis ke dalam analisa.txt dan analisa.json
analysis_results = {}
with open("analisa.txt", "w", encoding="utf-8") as output_file:
    for crypto in coin_list:
        try:
            print(f"🔍 Menganalisis {crypto}...")
            
            # 🔹 Ambil data harga dari Yahoo Finance
            df = yf.download(crypto, period="6mo", interval="1h")
            if df.empty:
                print(f"Gagal mengambil data {crypto}, mungkin tidak tersedia di Yahoo Finance.")
                continue

            # Resample ke 4 jam (ambil harga terakhir tiap 4 jam)
            df = df.resample("4h").last()

            # 🔹 Hitung indikator teknikal (Moving Average, RSI, Bollinger Bands)
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
            bb = ta.volatility.BollingerBands(df['Close'].squeeze())
            df['UpperBB'] = bb.bollinger_hband().squeeze()
            df['LowerBB'] = bb.bollinger_lband().squeeze()

            # 🔹 Tentukan Support & Resistance
            support = df['Low'].rolling(window=20).min()
            resistance = df['High'].rolling(window=20).max()
            current_price = df['Close'].iloc[-1]
            stop_loss = support.iloc[-1] * 0.98  # SL di bawah support
            take_profit = resistance.iloc[-1] * 1.02  # TP di atas resistance

            # 🔹 Konversi ke IDR (asumsi 1 USD = 15,000 IDR)
            kurs = 15000
            support *= kurs
            resistance *= kurs
            current_price *= kurs
            stop_loss *= kurs
            take_profit *= kurs
            df['Close'] *= kurs

            # 🔹 Tentukan harga beli ideal
            buying_in = (support.iloc[-1] + resistance.iloc[-1]) / 2  # Rata-rata dari support dan resistance

            # 🔹 Simpan hasil analisis ke JSON
            analysis_results[crypto] = {
                "Support Level": f"Rp{support.iloc[-1]:,.0f}",
                "Resistance Level": f"Rp{resistance.iloc[-1]:,.0f}",
                "Buying In": f"Rp{buying_in:,.0f}",
                "Take Profit": f"Rp{take_profit:,.0f}",
                "Stop Loss": f"Rp{stop_loss:,.0f}",
                "Current Level": f"Rp{current_price:,.0f}"
            }

            # 🔹 Simpan hasil analisis ke file teks
            output_file.write(f"\n=== {crypto} ===\n")
            for key, value in analysis_results[crypto].items():
                output_file.write(f"{key}: {value}\n")
            output_file.write("=" * 30 + "\n")

            print(f"✅ {crypto} selesai dianalisis!")

        except Exception as e:
            print(f"⚠️ Gagal menganalisis {crypto}: {str(e)}")

# 🔹 Simpan hasil analisis ke file JSON
with open("analisa.json", "w", encoding="utf-8") as json_file:
    json.dump(analysis_results, json_file, indent=4)

print("🎯 Semua hasil analisis telah disimpan di analisa.txt dan analisa.json!")
