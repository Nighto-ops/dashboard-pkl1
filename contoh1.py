import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import openrouteservice
from folium.features import DivIcon # Penting untuk menampilkan teks huruf

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(layout="wide", page_title="Peta Sebaran Venue")

# ==========================================
# 2. LOAD DATA (TIDAK DIUBAH DARI VERSI SEBELUMNYA)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Membaca format CSV (Pemisah titik koma, Desimal koma)
        df = pd.read_csv("data/Sleman_Kode.csv", sep=";", decimal=",", dtype=str)
        
        # Bersihkan nama kolom
        df.columns = df.columns.str.strip().str.lower()
        
        # Bersihkan & Konversi Koordinat
        for col in ['latitude', 'longitude']:
            # Ganti koma jadi titik, lalu ubah ke angka
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Hapus data tanpa koordinat
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Rapikan kolom teks
        for col in ['title', 'kecamatan', 'kode_venue', 'klasifikasi_venue', 'address']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '-')
                
        return df
    except Exception as e:
        st.error(f"Error membaca data: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.error("Data kosong. Cek file CSV Anda.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTER)
# ==========================================
st.sidebar.title("Kontrol Peta")

# API Key
with st.sidebar.expander("API Key ORS (Untuk Isochrone)", expanded=False):
    ORS_API_KEY = st.text_input("Paste API Key", type="password")

st.sidebar.divider()

# Filter Kecamatan
list_kecamatan = sorted(df_raw['kecamatan'].unique())
list_kecamatan.insert(0, "TAMPILKAN SEMUA")
pilih_kecamatan = st.sidebar.selectbox("1. Pilih Kecamatan", list_kecamatan)

if pilih_kecamatan == "TAMPILKAN SEMUA":
    df_step1 = df_raw.copy()
else:
    df_step1 = df_raw[df_raw['kecamatan'] == pilih_kecamatan]

# Filter Kode Venue
list_kode = sorted(df_step1['kode_venue'].unique())
list_kode.insert(0, "SEMUA KODE")
pilih_kode = st.sidebar.selectbox("2. Pilih Kode Venue", list_kode)

if pilih_kode == "SEMUA KODE":
    df_final = df_step1.copy()
else:
    df_final = df_step1[df_step1['kode_venue'] == pilih_kode]

st.sidebar.markdown(f"**Total Data:** {len(df_final)}")
st.sidebar.divider()

# Mode
mode = st.sidebar.radio("Mode:", ["Sebaran Titik", "Analisis Jangkauan (Isochrone)"])

selected_center = None
speed = 30

if mode == "Analisis Jangkauan (Isochrone)":
    # Dropdown venue (Hanya yang ada di filter)
    list_venue = sorted(df_final['title'].unique())
    if len(list_venue) > 0:
        pilih_pusat = st.sidebar.selectbox("Pilih Titik Pusat", list_venue)
        selected_center = df_final[df_final['title'] == pilih_pusat].iloc[0]
        speed = st.sidebar.slider("Kecepatan (km/jam)", 10, 60, 30)
    else:
        st.warning("Tidak ada venue yang tersedia.")

# ==========================================
# 4. RENDER PETA
# ==========================================
# Tentukan pusat peta
if selected_center is not None:
    lat_c, lon_c = selected_center['latitude'], selected_center['longitude']
    zoom_start = 14
elif not df_final.empty:
    lat_c, lon_c = df_final['latitude'].mean(), df_final['longitude'].mean()
    zoom_start = 12
else:
    lat_c, lon_c = -7.7956, 110.3695
    zoom_start = 11

m = folium.Map(location=[lat_c, lon_c], zoom_start=zoom_start, tiles="CartoDB positron")

# --- LAYER 1: ISOCHRONE ---
if mode == "Analisis Jangkauan (Isochrone)" and selected_center is not None and ORS_API_KEY:
    try:
        client = openrouteservice.Client(key=ORS_API_KEY)
        ranges_m = [(t/60)*speed*1000 for t in [20, 15, 10, 5]]
        colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641']
        labels = ['15-20 mnt', '10-15 mnt', '5-10 mnt', '< 5 mnt']
        
        iso = client.isochrones(
            locations=[[selected_center['longitude'], selected_center['latitude']]],
            profile='driving-car', range_type='distance', range=ranges_m, units='m'
        )
        for i, feature in enumerate(iso['features']):
            col = colors[i] if i < len(colors) else 'gray'
            folium.GeoJson(
                feature,
                style_function=lambda x, col=col: {'fillColor': col, 'color': 'black', 'weight': 1, 'fillOpacity': 0.4},
                tooltip=labels[i] if i < len(labels) else ''
            ).add_to(m)
        
        # Bintang Pusat
        folium.Marker(
            [selected_center['latitude'], selected_center['longitude']],
            icon=folium.Icon(color='red', icon='star'),
            tooltip="PUSAT"
        ).add_to(m)
    except Exception as e:
        st.error(f"Gagal Isochrone: {e}")

# --- LAYER 2: TITIK + LABEL KODE (Sesuai Request) ---
for _, row in df_final.iterrows():
    # Jika ini titik pusat, skip (sudah digambar bintang di atas)
    if selected_center is not None and row['title'] == selected_center['title']:
        continue
    
    # Ambil Kode Venue
    kode_text = str(row['kode_venue'])
    
    # 1. Gambar Titik Biru Kecil (Scatter)
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color='blue',
        fill=True,
        fill_color='cyan',
        fill_opacity=0.8,
        popup=folium.Popup(f"<b>{row['title']}</b><br>Kode: {kode_text}", max_width=200),
        tooltip=f"{row['title']} ({kode_text})"
    ).add_to(m)
    
    # 2. Gambar Huruf Kode di Atas Titik
    # Menggunakan DivIcon HTML sederhana
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(5, 14), # Mengatur posisi teks agar pas di tengah titik
            html=f'<div style="font-size: 10pt; font-weight: bold; color: black; text-shadow: 1px 1px 0 #fff;">{kode_text}</div>'
        )
    ).add_to(m)

st.title(f"Peta Venue: {pilih_kecamatan}")
st_folium(m, width="100%", height=700)