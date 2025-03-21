import requests
from bs4 import BeautifulSoup

# Header agar terdeteksi sebagai browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

# Baca file kalimat.txt jika sudah ada
try:
    with open("kalimat.txt", "r", encoding="utf-8") as file:
        existing_coins = set(file.read().splitlines())  # Simpan ke dalam set untuk pencarian cepat
except FileNotFoundError:
    existing_coins = set()

# Loop untuk mengambil data dari halaman 0 hingga 9999 dengan kelipatan 100
for start in range(0, 10000, 100):
    yahoo_url = f"https://finance.yahoo.com/markets/crypto/all/?start={start}&count=100"
    
    # Request halaman
    response = requests.get(yahoo_url, headers=headers)
    
    # Jika gagal mendapatkan response, lanjutkan ke iterasi berikutnya
    if response.status_code != 200:
        print(f"Gagal mengambil data dari {yahoo_url}")
        continue
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Ambil semua elemen <span> dengan class "symbol yf-1fqyif7"
    coin_elements = soup.find_all("span", class_="symbol yf-1fqyif7")
    
    # Ambil teks dari setiap elemen
    coin_names = [coin.text for coin in coin_elements]
    
    # Filter hanya koin yang belum ada dalam file
    new_coins = [coin for coin in coin_names if coin not in existing_coins]
    
    # Jika ada koin baru, tambahkan ke file
    if new_coins:
        with open("coingacor.txt", "a", encoding="utf-8") as file:
            file.write("\n".join(new_coins) + "\n")  # Tambah data baru dengan newline
            
        # Perbarui set existing_coins agar tidak ada duplikasi dalam iterasi berikutnya
        existing_coins.update(new_coins)
    
    print(f"Scraped {len(new_coins)} new coins from {yahoo_url}")

print("Semua data berhasil diperbarui di kalimat.txt!")
