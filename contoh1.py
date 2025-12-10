import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import openrouteservice
from folium.features import DivIcon # Penting untuk menampilkan teks huruf

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(layout="wide", page_title="Test Peta Standalone")

# ==========================================
# 2. LOAD DATA
# ==========================================
st.sidebar.title("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload File (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

@st.cache_data
def load_data(file):
    try:
        # Deteksi tipe file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=";", dtype=str) # Coba separator titik koma dulu
            if len(df.columns) < 2: # Kalau gagal, coba koma
                file.seek(0)
                df = pd.read_csv(file, sep=",", dtype=str)
        else:
            df = pd.read_excel(file, dtype=str)
        
        # Bersihkan nama kolom
        df.columns = df.columns.str.strip().str.lower()
        
        # Bersihkan & Konversi Koordinat
        # Pastikan kolom latitude/longitude ada
        cols_coord = [c for c in df.columns if 'lat' in c or 'lon' in c]
        
        if len(cols_coord) >= 2:
            lat_col = [c for c in cols_coord if 'lat' in c][0]
            lon_col = [c for c in cols_coord if 'lon' in c][0]
            
            for col in [lat_col, lon_col]:
                # Ganti koma jadi titik, lalu ubah ke angka
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Hapus data tanpa koordinat
            df = df.dropna(subset=[lat_col, lon_col])
            
            # Rename biar standar
            df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
        else:
            st.error("Kolom Latitude/Longitude tidak ditemukan.")
            return pd.DataFrame()
            
        # Rapikan kolom teks penting
        for col in ['title', 'kecamatan', 'kode_venue', 'klasifikasi_venue', 'address']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '-')
                
        return df
    except Exception as e:
        st.error(f"Error membaca data: {e}")
        return pd.DataFrame()

# Logika Load
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
else:
    st.info("Silakan upload file 'Sleman_Kode' di sidebar kiri untuk memulai.")
    st.stop()

if df_raw.empty:
    st.error("Data kosong atau format salah.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTER)
# ==========================================
st.sidebar.divider()
st.sidebar.subheader("Kontrol Peta")

# API Key
with st.sidebar.expander("API Key ORS (Wajib untuk Isochrone)", expanded=True):
    ORS_API_KEY = st.text_input("Paste API Key", type="password")

# Filter Kecamatan
if 'kecamatan' in df_raw.columns:
    list_kecamatan = sorted(df_raw['kecamatan'].unique())
    list_kecamatan.insert(0, "TAMPILKAN SEMUA")
    pilih_kecamatan = st.sidebar.selectbox("1. Pilih Kecamatan", list_kecamatan)

    if pilih_kecamatan == "TAMPILKAN SEMUA":
        df_step1 = df_raw.copy()
    else:
        df_step1 = df_raw[df_raw['kecamatan'] == pilih_kecamatan]
else:
    df_step1 = df_raw.copy()

# Filter Kode Venue
if 'kode_venue' in df_raw.columns:
    list_kode = sorted(df_step1['kode_venue'].unique())
    list_kode.insert(0, "SEMUA KODE")
    pilih_kode = st.sidebar.selectbox("2. Pilih Kode Venue", list_kode)

    if pilih_kode == "SEMUA KODE":
        df_final = df_step1.copy()
    else:
        df_final = df_step1[df_step1['kode_venue'] == pilih_kode]
else:
    df_final = df_step1.copy()

st.sidebar.markdown(f"**Total Data Ditampilkan:** {len(df_final)}")
st.sidebar.divider()

# Mode
mode = st.sidebar.radio("Mode Tampilan:", ["Sebaran Titik Biasa", "Analisis Jangkauan (Isochrone)"])

selected_center = None
speed = 30

if mode == "Analisis Jangkauan (Isochrone)":
    # Dropdown venue (Hanya yang ada di filter)
    if 'title' in df_final.columns:
        list_venue = sorted(df_final['title'].unique())
        if len(list_venue) > 0:
            pilih_pusat = st.sidebar.selectbox("Pilih Titik Pusat", list_venue)
            selected_center = df_final[df_final['title'] == pilih_pusat].iloc[0]
            
            st.sidebar.markdown("---")
            st.sidebar.write("**Parameter Kendaraan**")
            speed = st.sidebar.slider("Kecepatan Rata-rata (km/jam)", 10, 80, 30)
            
            st.sidebar.info(f"Analisis: Seberapa jauh kendaraan melaju dengan kecepatan {speed} km/jam dalam waktu tertentu.")
        else:
            st.warning("Tidak ada data venue untuk dipilih.")
    else:
        st.error("Kolom 'title' tidak ditemukan di data.")

# ==========================================
# 4. RENDER PETA
# ==========================================
# Tentukan pusat peta awal
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

# --- LAYER 1: ISOCHRONE (Jika Mode Aktif) ---
if mode == "Analisis Jangkauan (Isochrone)" and selected_center is not None:
    if ORS_API_KEY:
        try:
            client = openrouteservice.Client(key=ORS_API_KEY)
            
            # Konversi waktu (menit) ke Jarak tempuh (meter) berdasarkan kecepatan
            # Rumus: (Menit / 60) * Speed(km/h) * 1000 = Meter
            times_min = [20, 15, 10, 5]
            ranges_m = [(t/60)*speed*1000 for t in times_min]
            
            colors = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'] # Merah (Jauh) -> Hijau (Dekat)
            labels = ['15-20 mnt', '10-15 mnt', '5-10 mnt', '< 5 mnt']
            
            with st.spinner("Menghitung jangkauan..."):
                iso = client.isochrones(
                    locations=[[selected_center['longitude'], selected_center['latitude']]],
                    profile='driving-car', 
                    range_type='distance', 
                    range=ranges_m, 
                    units='m'
                )
                
                for i, feature in enumerate(iso['features']):
                    # Tentukan warna (handle jika index out of bound)
                    col = colors[i] if i < len(colors) else 'gray'
                    lbl = labels[i] if i < len(labels) else ''
                    
                    folium.GeoJson(
                        feature,
                        style_function=lambda x, col=col: {
                            'fillColor': col, 
                            'color': 'black', 
                            'weight': 1, 
                            'fillOpacity': 0.4
                        },
                        tooltip=f"Area Jangkauan: {lbl}"
                    ).add_to(m)
            
            # Bintang Pusat
            folium.Marker(
                [selected_center['latitude'], selected_center['longitude']],
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
                tooltip=f"PUSAT: {selected_center.get('title', 'Lokasi')}"
            ).add_to(m)
            
        except Exception as e:
            st.error(f"Gagal memuat Isochrone. Cek API Key atau Kuota. Error: {e}")
    else:
        st.warning("⚠️ Masukkan API Key di sidebar untuk melihat Isochrone.")

# --- LAYER 2: TITIK + LABEL KODE (Selalu Muncul) ---
for _, row in df_final.iterrows():
    # Jika ini titik pusat (sedang mode isochrone), skip biar gak numpuk bintang
    if selected_center is not None and row.get('title') == selected_center.get('title'):
        continue
    
    # Ambil Kode Venue
    kode_text = str(row.get('kode_venue', '?'))
    nama_venue = str(row.get('title', 'Tanpa Nama'))
    
    # 1. Gambar Titik Biru Kecil (Scatter)
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color='blue',
        fill=True,
        fill_color='cyan',
        fill_opacity=0.8,
        popup=folium.Popup(f"<b>{nama_venue}</b><br>Kode: {kode_text}", max_width=200),
        tooltip=f"{nama_venue} ({kode_text})"
    ).add_to(m)
    
    # 2. Gambar Huruf Kode di Atas Titik (Menggunakan DivIcon)
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(6, 14), # Mengatur posisi teks agar pas di tengah atas titik
            html=f'<div style="font-size: 10pt; font-weight: bold; color: black; text-shadow: 1px 1px 0 #fff;">{kode_text}</div>'
        )
    ).add_to(m)

# Tampilkan Peta
st.title("Peta Sebaran & Jangkauan Venue")
st_folium(m, width="100%", height=700)