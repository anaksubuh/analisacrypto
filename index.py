import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import ta
from sklearn.linear_model import LinearRegression

import requests
import pandas as pd
import time
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

while True:
    # 🔹 Baca daftar koin dari file coingacor.txt
    try:
        with open("coingacor.txt", "r", encoding="utf-8") as file:
            coin_list = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        print("File coingacor.txt tidak ditemukan!")
        coin_list = []

    # 🔹 Simpan hasil analisis ke dalam analisa.txt
    with open("analisa.txt", "w", encoding="utf-8") as output_file:
        for crypto in coin_list:
            try:
                print(f"🔍 Menganalisis {crypto}...")
                
                timeframe   = str('4h')
                # Menunjukan seberapa jauh kamu mengambil data ke masa lampau , contoh 1mo(1bulan) berarti kamu hanya mengambil data 1 bulan terahkir
                scrapselama = str('6mo')

                usd_rate     = 16300
                modal        = 100000
                risk_ratio   = 1
                reward_ratio = 2

                # Resample ke 4 jam (ambil harga terakhir tiap 4 jam)

                log_text = ('Sedang menganalisa coin = '+crypto)

                with open("log.txt", "w") as log_file:
                    log_file.write(log_text)

                if timeframe == '4h':
                    df = yf.download(crypto, period=scrapselama, interval='1h')
                    df = df.resample("4h").last()
                else:
                    df = yf.download(crypto, period=scrapselama, interval=timeframe)

                # 🔹 Hilangkan MultiIndex jika ada
                df.columns = df.columns.get_level_values(0)

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

                # 🔹 Pastikan semua fitur tersedia untuk prediksi harga
                valid_features = ['RSI', 'UpperBB', 'LowerBB', 'SMA20', 'SMA50', 'SMA200']
                available_features = [col for col in valid_features if col in df.columns]
                X = df[available_features].dropna()
                y = df['Close'].shift(-1).dropna()
                
                # Pastikan X dan y punya jumlah data yang sama
                X, y = X.align(y, join='inner', axis=0)

                # 🔹 Inisialisasi dan latih model
                model = LinearRegression()
                if not X.empty and not y.empty:
                    model.fit(X, y)
                    df['Predicted'] = np.nan
                    df.loc[X.index, 'Predicted'] = model.predict(X)
                else:
                    df['Predicted'] = np.nan

                # 🔹 Hitung potensi akurasi analisa (MAPE)
                if 'Predicted' in df.columns and not df['Predicted'].isna().all():
                    y_true = df.loc[X.index, 'Close']  # Harga asli
                    y_pred = df.loc[X.index, 'Predicted']  # Harga prediksi
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    potensi_akurasi = 100 - mape
                else:
                    potensi_akurasi = "Data tidak cukup"

                # 🔹 Simpan hasil analisis
                output_file.write(f"\n✨ Hasil Analisis Trading = {crypto} ✨\n")
                output_file.write(f"📈 Resistance Level : Rp{float(resistance.iloc[-1]):,.0f} / ${resistance.iloc[-1] / kurs:.3f}\n")
                output_file.write(f"📉 Support Level    : Rp{float(support.iloc[-1]):,.0f} / ${support.iloc[-1] / kurs:.3f}\n")
                output_file.write(f"💵 Current Level    : Rp{float(current_price):,.0f} / ${current_price / kurs:.3f}\n")
                output_file.write(f"💰 Buying In        : Rp{float(buying_in):,.0f} / ${buying_in / kurs:.3f}\n")
                output_file.write(f"🎯 Take Profit      : Rp{float(take_profit):,.0f} / ${take_profit / kurs:.3f}\n")
                output_file.write(f"🚯 Stop Loss        : Rp{float(stop_loss):,.0f} / ${stop_loss / kurs:.3f}\n")

                if isinstance(potensi_akurasi, (int, float)):
                    output_file.write(f"🔎 Potensi Akurasi Analisa: {potensi_akurasi:.2f}%\n")
                else:
                    output_file.write(f"🔎 Potensi Akurasi Analisa: {potensi_akurasi}\n")

                if 'Predicted' in df.columns and not df['Predicted'].isna().all():
                    prediksi = df['Predicted'].dropna().iloc[-1]
                    posisi = "LONG CUY 🚀" if current_price < prediksi else "SHORT CUY 📉"
                    output_file.write(f"🧙‍♂️ Prediksi Harga dalam 5 hari: Rp{prediksi:,.0f} / ${prediksi / kurs:.3f}\n")
                    output_file.write(f"💰 Posisi: {posisi}\n")

                    # 🔹 Perbandingan Harga dan Potensi Profit/Loss
                    potensi_profit = prediksi - current_price
                    potensi_loss = current_price - stop_loss
                    persen_profit = (potensi_profit / current_price) * 100
                    persen_loss = (potensi_loss / current_price) * 100


                    # 🔹 Risk to Reward Calculation
                    lot_size = 100000 / current_price
                    keuntungan = lot_size * (take_profit - current_price)
                    kerugian = lot_size * (current_price - stop_loss)


                    # Risk Ratio 1:2
                    stop_loss = support.iloc[-1]
                    risk_amount = buying_in - stop_loss
                    take_profit = buying_in + (risk_amount * 2)

                    keuntungan_risk = lot_size * (take_profit - buying_in)
                    kerugian_risk = lot_size * (buying_in - stop_loss)

                    output_file.write("\n🔹 Perbandingan Harga:\n")
                    output_file.write(f"- Harga Sekarang     : Rp{current_price:,.0f} / ${current_price / kurs:.3f}\n")
                    output_file.write(f"- Prediksi Harga     : Rp{prediksi:,.0f} / ${prediksi / kurs:.3f}\n")
                    output_file.write(f"- Selisih            : Rp{abs(potensi_profit):,.0f} / ${abs(potensi_profit) / kurs:.3f}\n")
                    output_file.write(f"📈 Potensi Profit    : {persen_profit:.2f}%\n")
                    output_file.write(f"📉 Potensi Loss      : {persen_loss:.2f}%\n")
                    output_file.write('\nKesimpulannya       :\n')
                    output_file.write(f"💰 Buying In         : Rp{buying_in:,.0f} / ${buying_in / kurs:.3f}\n")
                    output_file.write(f"🎯 Take Profit       : Rp{take_profit:,.0f} / ${take_profit / kurs:.3f}\n")
                    output_file.write(f"🚩 Stop Loss         : Rp{stop_loss:,.0f} / ${stop_loss / kurs:.3f}\n")
                    output_file.write(f"Jika modal 100 ribu maka\n")
                    output_file.write(f"Keuntungan            : Rp{keuntungan:,.0f}\n")
                    output_file.write(f"Kerugian              : Rp{kerugian:,.0f}\n")
                    output_file.write(f'\nRisk Rasio 1:2\n')
                    output_file.write(f"💰 Buying In          : Rp{buying_in:,.0f} / ${buying_in / kurs:.3f}\n")
                    output_file.write(f"🚩 Stop Loss          : Rp{stop_loss:,.0f} / ${stop_loss / kurs:.3f}\n")
                    output_file.write(f"🎯 Take Profit        : Rp{take_profit:,.0f} / ${take_profit / kurs:.3f}\n")
                    output_file.write(f"Jika modal 100000 ribu maka\n")
                    output_file.write(f"Keuntungan            : Rp{keuntungan_risk:,.0f}\n")
                    output_file.write(f"Kerugian              : Rp{kerugian_risk:,.0f}\n")

                output_file.write("=" * 30 + "\n")
                print(f"✅ {crypto} selesai dianalisis!")

            except Exception as e:
                print(f"⚠️ Gagal menganalisis {crypto}: {str(e)}")

    print("🎯 Semua hasil analisis telah disimpan di analisa.txt!")
    time.sleep(5)

    import streamlit as st

    # Fungsi untuk membaca isi file dan menampilkan di Streamlit
    def tampilkan_analisa():
        try:
            with open("analisa.txt", "r", encoding="utf-8") as file:
                # Membaca seluruh isi file
                content = file.readlines()
            
            # Menampilkan setiap baris isi file menggunakan st.write
            for line in content:
                st.write(line)  # Menampilkan setiap baris dalam file
            
        except FileNotFoundError:
            st.error("File 'analisa.txt' tidak ditemukan!")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

    # Menambahkan judul untuk aplikasi Streamlit
    st.title("Hasil Analisa Trading")

    # Menampilkan hasil analisa dari file analisa.txt
    tampilkan_analisa()

    time.sleep(3600*4)