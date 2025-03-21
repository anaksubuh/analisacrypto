import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta
from sklearn.linear_model import LinearRegression

'''
Rentang dalam satuan waktu menit jam hari minggu bulan dan tahun

valid_periods = {
    '1m': ["1d", "2d", "3d", "4d", "5d", "6d", "7d"],
    '2m': ["1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "60d"],
    '5m': ["1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "30d", "60d"],
    '15m': ["1d", "2d", "3d", "4d", "5d", "6d", "7d", "30d", "60d"],
    '30m': ["1d", "2d", "3d", "4d", "6d", "12d", "30d", "60d"],
    '1h': ["1d", "2d", "3d", "4d", "6d", "12d", "1y", "2y", "3y"],
    '1d': ["1d", "2d", "3d", "5d", "7d", "10d", "1mo", "2mo", "3mo", "6mo", "1y", "5y", "max"],
    '1wk': ["1wk", "2wk", "3wk", "1mo", "2mo", "5mo", "1y", "2y", "5y", "max"],
    '1mo': ["30d", "60d", "90d", "180d", "1y", "2y", "5y", "10y", "max"],
    '3mo': ["90d", "180d", "270d", "1y", "2y", "5y", "10y", "max"]
}
'''

# Menunjukan 1 candle dalam satuan waktu menit jam hari minggu bulan tahun
timeframe   = str('4h')
# Menunjukan seberapa jauh kamu mengambil data ke masa lampau , contoh 1mo(1bulan) berarti kamu hanya mengambil data 1 bulan terahkir
scrapselama = str('6mo')

usd_rate     = 16300
modal        = 100000
risk_ratio   = 1
reward_ratio = 2

# 🔹 Ambil data harga DOGE dari Yahoo Finance
crypto = 'TRX-USD'

# Resample ke 4 jam (ambil harga terakhir tiap 4 jam)
if timeframe == '4h':
    df = yf.download(crypto, period=scrapselama, interval='1h')
    df = df.resample("4h").last()
else:
    df = yf.download(crypto, period=scrapselama, interval=timeframe)

# 🔹 Hilangkan MultiIndex jika ada
df.columns = df.columns.get_level_values(0)

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
support *= usd_rate
resistance *= usd_rate
current_price *= usd_rate
stop_loss *= usd_rate
take_profit *= usd_rate

df['Close'] *= usd_rate

# 🔹 Tentukan harga beli ideal
buying_in = (support.iloc[-1] + resistance.iloc[-1]) / 2  # Rata-rata dari support dan resistance

# 🔹 Pastikan semua fitur tersedia
valid_features = ['RSI', 'UpperBB', 'LowerBB', 'SMA20', 'SMA50', 'SMA200']
available_features = [col for col in valid_features if col in df.columns]
X = df[available_features].dropna()
y = df['Close'].shift(-1).dropna()

# Pastikan X dan y punya jumlah data yang sama
X, y = X.align(y, join='inner', axis=0)

# 🔹 Inisialisasi dan latih model
model = LinearRegression()
if X.empty or y.empty:
    print("Data tidak cukup untuk melatih model!")
else:
    model.fit(X, y)
    df['Predicted'] = np.nan
    df.loc[X.index, 'Predicted'] = model.predict(X)

# 🔹 Tampilkan Hasil
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Harga Asli (IDR)')
if 'Predicted' in df.columns:
    plt.plot(df.index, df['Predicted'], label='Prediksi Harga (IDR)', linestyle='dashed')
plt.axhline(y=support.iloc[-1], color='green', linestyle='dotted', label='Support (IDR)')
plt.axhline(y=resistance.iloc[-1], color='red', linestyle='dotted', label='Resistance (IDR)')
plt.axhline(y=stop_loss, color='blue', linestyle='dotted', label='Stop Loss (SL) (IDR)')
plt.axhline(y=take_profit, color='purple', linestyle='dotted', label='Take Profit (TP) (IDR)')
plt.axhline(y=current_price, color='orange', linestyle='dotted', label='Current Level (CL) (IDR)')

# Tambahkan teks di grafik
plt.text(df.index[-1], support.iloc[-1], f"Support: Rp{support.iloc[-1]:,.0f}", verticalalignment='bottom', fontsize=10, color='green')
plt.text(df.index[-1], resistance.iloc[-1], f"Resistance: Rp{resistance.iloc[-1]:,.0f}", verticalalignment='bottom', fontsize=10, color='red')
plt.text(df.index[-1], stop_loss, f"SL: Rp{stop_loss:,.0f}", verticalalignment='bottom', fontsize=10, color='blue')
plt.text(df.index[-1], take_profit, f"TP: Rp{take_profit:,.0f}", verticalalignment='bottom', fontsize=10, color='purple')
plt.text(df.index[-1], current_price, f"CL: Rp{current_price:,.0f}", verticalalignment='bottom', fontsize=10, color='orange')
plt.text(df.index[-1], buying_in, f"Buy: Rp{buying_in:,.0f}", verticalalignment='bottom', fontsize=10, color='black')

################################################################################################################################################################################

from sklearn.metrics import mean_absolute_percentage_error

# Hitung akurasi prediksi (MAPE)
if 'Predicted' in df.columns and not df['Predicted'].dropna().empty:
    y_true = df.loc[X.index, 'Close']  # Harga asli
    y_pred = df.loc[X.index, 'Predicted']  # Harga prediksi
    
    if len(y_true) > 0 and len(y_pred) > 0:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        potensi_akurasi = 100 - mape
    else:
        potensi_akurasi = "Data tidak cukup"
else:
    potensi_akurasi = "Prediksi harga tidak tersedia"

# Cetak hasil analisis trading dengan format IDR/USD
print(f"\n✨ Hasil Analisis Trading = {crypto}✨")
print(f"📈 Resistance Level : Rp{resistance.iloc[-1]:,.0f} / ${resistance.iloc[-1] / usd_rate:.3f}")
print(f"📉 Support Level    : Rp{support.iloc[-1]:,.0f} / ${support.iloc[-1] / usd_rate:.3f}")
print(f"💵 Current Level    : Rp{current_price:,.0f} / ${current_price / usd_rate:.3f}")
print(f"💰 Buying In        : Rp{buying_in:,.0f} / ${buying_in / usd_rate:.3f}")
print(f"🎯 Take Profit      : Rp{take_profit:,.0f} / ${take_profit / usd_rate:.3f}")
print(f"🚯 Stop Loss        : Rp{stop_loss:,.0f} / ${stop_loss / usd_rate:.3f}")

if isinstance(potensi_akurasi, (int, float)):
    print(f"🔎 Potensi Akurasi Analisa: {potensi_akurasi:.2f}%")
else:
    print(f"🔎 Potensi Akurasi Analisa: {potensi_akurasi}")

# Pastikan harga terakhir adalah angka
hargaterakhir = int(current_price)
if 'Predicted' in df.columns and not df['Predicted'].dropna().empty:
    prediksi = int(df['Predicted'].dropna().iloc[-1])
    potensi_profit = prediksi - hargaterakhir

    if potensi_profit > 0:
        persen_profit = (potensi_profit / hargaterakhir) * 100
        persen_loss = ((hargaterakhir - stop_loss) / hargaterakhir) * 100
    else:
        persen_profit = ((take_profit - hargaterakhir) / hargaterakhir) * 100
        persen_loss = (abs(potensi_profit) / hargaterakhir) * 100

    posisi = "LONG CUY 🚀" if hargaterakhir < prediksi else "SHORT CUY 📉"

    print(f"\n🧙‍♂️ Prediksi Harga dalam 5 hari: Rp{prediksi:,.0f} / ${prediksi / usd_rate:.3f}")
    print(f"💰 Posisi: {posisi}")
    
    print("\n🔹 Perbandingan Harga:")
    print(f"- Harga Sekarang     : Rp{hargaterakhir:,.0f} / ${hargaterakhir / usd_rate:.3f}")
    print(f"- Prediksi Harga     : Rp{prediksi:,.0f} / ${prediksi / usd_rate:.3f}")
    print(f"- Selisih            : Rp{abs(potensi_profit):,.0f} / ${abs(potensi_profit) / usd_rate:.3f}")
    print(f"📈 Potensi Profit    : {persen_profit:.2f}%")
    print(f"📉 Potensi Loss      : {persen_loss:.2f}%")

    # Hitung keuntungan dan kerugian berdasarkan modal awal Rp100.000
    lot_size = modal / hargaterakhir
    keuntungan = lot_size * (take_profit - hargaterakhir)
    kerugian = lot_size * (hargaterakhir - stop_loss)

    print('\nKesimpulannya        :')
    print(f"💰 Buying In         : Rp{buying_in:,.0f} / ${buying_in / usd_rate:.3f}")
    print(f"🎯 Take Profit       : Rp{take_profit:,.0f} / ${take_profit / usd_rate:.3f}")
    print(f"🚩 Stop Loss         : Rp{stop_loss:,.0f} / ${stop_loss / usd_rate:.3f}")
    print(f"Jika modal 100 ribu maka")
    print(f"Keuntungan           : Rp{keuntungan:,.0f}")
    print(f"Kerugian             : Rp{kerugian:,.0f}")

    # Hitung Risk to Reward Ratio 1:2
    stop_loss = support.iloc[-1]  # SL di area support
    risk_amount = buying_in - stop_loss  # Selisih harga beli ke SL (kerugian maksimal)
    take_profit = buying_in + (risk_amount * reward_ratio)  # TP = reward_ratio x risk_amount

    keuntungan_risk = lot_size * (take_profit - buying_in)
    kerugian_risk = lot_size * (buying_in - stop_loss)

    print(f'\nRisk Rasio {risk_ratio}:{reward_ratio}')
    print(f"💰 Buying In         : Rp{buying_in:,.0f} / ${buying_in / usd_rate:.3f}")
    print(f"🚩 Stop Loss         : Rp{stop_loss:,.0f} / ${stop_loss / usd_rate:.3f}")
    print(f"🎯 Take Profit       : Rp{take_profit:,.0f} / ${take_profit / usd_rate:.3f}")
    print(f"Jika modal {modal} ribu maka")
    print(f"Keuntungan           : Rp{keuntungan_risk:,.0f}")
    print(f"Kerugian             : Rp{kerugian_risk:,.0f}")
else:
    print("Prediksi harga tidak tersedia.")

plt.legend()
#plt.show()