import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import openpyxl 
import os

# --- TAMBAHAN LIBRARY UNTUK PETA ---
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import openrouteservice
from openrouteservice import convert
from folium.features import DivIcon

# Impor dari Scikit-learn (SKLEARN)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Impor dari STATSMODELS
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.multivariate.manova import MANOVA

# Impor dari FACTOR ANALYZER
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# =================================================================
# KONFIGURASI HALAMAN & CSS GRAND DESIGN PKL 65 
# =================================================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Statistik PKL 65")

# --- CUSTOM CSS: DESIGN, DARK MODE FIX, TOOLBAR & NAVBAR (SIDEBAR NAIK) ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rakkas&family=Playfair+Display:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    /* --- WARNA DARI GRAND DESIGN --- */
    :root {
        --base-cream: #FDF8E4;
        --base-terracotta: #E07A3F;
        --comp-gold: #F2C94C;
        --comp-teal: #4F8190;
        --comp-olive: #739159;
        --text-dark: #4A3B32; 
        --sidebar-bg: #FFFFFF;       
    }

    /* 1. SETUP BACKGROUND UTAMA */
    .stApp {
        background-color: var(--base-cream);
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }

    /* --- NAVBAR / HEADER ATAS --- */
    header[data-testid="stHeader"] {
        background-color: rgba(253, 248, 228, 0.95) !important;
        border-bottom: 4px solid var(--base-terracotta);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 3.5rem !important;
        z-index: 999990 !important;
    }

    /* --- TOOLBAR --- */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
        z-index: 99999999 !important;
        right: 1rem;
        top: 0.2rem;
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stToolbar"] button {
        color: var(--base-terracotta) !important;
    }

    /* 2. HEADER PADDING (KONTEN UTAMA) */
    .block-container {
        padding-top: 5rem !important; 
        padding-bottom: 2rem;
        max-width: 100%;
    }

    /* 3. TYPOGRAPHY */
    h1 {
        font-family: 'Rakkas', cursive !important;
        color: var(--base-terracotta) !important;
        font-weight: 400 !important;
        font-size: 3rem !important;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h2, h3, h4, .streamlit-expanderHeader {
        font-family: 'Playfair Display', serif !important;
        color: var(--base-terracotta) !important;
        font-weight: 700 !important;
    }

    /* 4. SIDEBAR (PERBAIKAN POSISI NAIK) */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 3px solid var(--base-terracotta);
        box-shadow: 4px 0 10px rgba(0,0,0,0.05);
    }
    
    /* INI KUNCINYA: Mengatur jarak isi sidebar dari atas */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 5rem !important; /* Jarak aman dari navbar */
        padding-bottom: 2rem !important;
    }

    /* Judul Sidebar */
    section[data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        color: var(--base-terracotta) !important;
        text-align: center;
        margin-top: 0 !important; /* Hapus margin bawaan */
    }
    
    /* Fix teks sidebar */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: var(--text-dark) !important;
    }

    /* 5. TOMBOL */
    .stButton > button {
        background: linear-gradient(to right, var(--comp-gold), var(--base-terracotta)) !important;
        color: #FFF !important;
        border: none !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15) !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2) !important;
        background: linear-gradient(to right, var(--base-terracotta), var(--comp-gold)) !important;
    }

    /* 6. TAB MENU STYLE */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid var(--comp-teal);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: var(--comp-teal);
    }
    .stTabs [aria-selected="true"] {
        color: var(--base-terracotta) !important;
        border-bottom: 3px solid var(--base-terracotta) !important;
    }

    /* 7. WIDGET INPUT */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div, .stSlider > div > div {
        background-color: #FFFFFF !important;
        border-color: var(--comp-gold) !important;
        color: var(--text-dark) !important;
    }
    .stMultiSelect div[data-baseweb="select"] span {
        color: var(--text-dark) !important;
    }
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: var(--base-terracotta) !important;
        color: white !important;
    }
    
    /* 8. FILE UPLOADER */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        border: 1px dashed var(--base-terracotta);
    }
    div[data-testid="stFileUploader"] section {
        background-color: #FDF1D6 !important; 
    }
    div[data-testid="stFileUploader"] span {
        color: var(--text-dark) !important;
    }
    
    /* 9. BOX & ALERT STYLING */
    .welcome-box {
        background-color: #FFFFFF;
        padding: 25px;
        border-left: 6px solid var(--base-terracotta);
        border-top: 2px solid var(--comp-gold);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .stAlert {
        background-color: #FFFFFF;
        border: 1px solid var(--comp-gold);
        color: var(--text-dark);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stSuccess { border-left-color: var(--comp-olive) !important; }
    .stInfo { border-left-color: var(--comp-teal) !important; }
    .stWarning { border-left-color: var(--comp-gold) !important; }
    .stError { border-left-color: #C0392B !important; }

    /* 10. TEXT COLOR GLOBAL override */
    p, label, span, div {
        color: var(--text-dark);
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
# =================================================================
# SETUP GAMBAR (SESUAI GRAND DESIGN & ASET BARU)
# =================================================================
# Menggunakan header full dan footer full yang baru
LOGO_IMG   = 'gambar/image_14.png'  # Logo Bulat untuk Sidebar
MASCOT_IMG = 'gambar/image_10.png'  # Maskot Laptop untuk Sidebar


# =================================================================
# FUNGSI BANTUAN (MAP) - SOLUSI COMPOSITE KEY (BANTUL_JETIS vs YOGYA_JETIS)
# =================================================================
@st.cache_data
def load_map_excel_data():
    folder_path = 'data'
    df_list = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.xlsx') or file.endswith('.xls'):
                file_path = os.path.join(folder_path, file)
                try:
                    temp_df = pd.read_excel(file_path)
                    temp_df.columns = [c.lower().strip() for c in temp_df.columns]
                    if 'latitude' in temp_df.columns:
                        temp_df = temp_df.rename(columns={'latitude': 'lattitude'})
                    df_list.append(temp_df)
                except: pass
    
    if not df_list: return pd.DataFrame()
    
    combined_df = pd.concat(df_list, ignore_index=True)
    if 'lattitude' in combined_df.columns and 'longitude' in combined_df.columns:
        combined_df = combined_df.dropna(subset=['lattitude', 'longitude'])
    else:
        return pd.DataFrame()

    # 1. BERSIHKAN KABUPATEN & BUAT KAB_KEY
    if 'kabupaten' in combined_df.columns:
        combined_df['kabupaten'] = combined_df['kabupaten'].astype(str).str.title().str.strip()
        combined_df['kabupaten'] = combined_df['kabupaten'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
        combined_df['kabupaten'] = combined_df['kabupaten'].replace({'Yogya': 'Yogyakarta', 'Yogyakartakarta': 'Yogyakarta'})
        
        # Key Kabupaten (Tanpa Spasi)
        combined_df['kab_key'] = combined_df['kabupaten'].str.upper().str.replace(" ", "")

    # 2. BERSIHKAN KECAMATAN & BUAT KEC_KEY
    if 'kecamatan' in combined_df.columns:
        combined_df['kecamatan'] = combined_df['kecamatan'].astype(str).str.title().str.strip()
        combined_df['kecamatan'] = combined_df['kecamatan'].str.replace(r'^(Kec\.?|Kecamatan|Kapanewon|Kemantren)\s+', '', regex=True)
        
        # Key Kecamatan (Tanpa Spasi)
        combined_df['kec_key'] = combined_df['kecamatan'].str.upper().str.replace(" ", "")

    # 3. FIX MANTRIJERON (Pindah ke Kota)
    kec_kota_jogja_keys = [
        'DANUREJAN', 'GEDONGTENGEN', 'GONDOKUSUMAN', 'GONDOMANAN', 
        'JETIS', 'KOTAGEDE', 'KRATON', 'KERATON', 'MANTRIJERON', 'MERGANGSAN', 
        'NGAMPILAN', 'PAKUALAMAN', 'TEGALREJO', 'UMBULHARJO', 'WIROBRAJAN'
    ]
    
    if 'kabupaten' in combined_df.columns and 'kec_key' in combined_df.columns:
        # Logika: Jika Kecamatan ada di daftar Kota Jogja, paksa Kabupatennnya jadi Yogyakarta
        mask_kota = combined_df['kec_key'].isin(kec_kota_jogja_keys)
        
        # TAPI HATI-HATI: JETIS ada di Bantul juga.
        # Jadi kita hanya ubah jika data aslinya BUKAN Bantul tapi masuk list kota (kasus Mantrijeron)
        # Untuk Jetis, kita percayakan pada data asli inputan mahasiswa (semoga input kabupatennya benar)
        
        # Khusus Mantrijeron (yang sering salah masuk Bantul)
        mask_mantri = combined_df['kec_key'].isin(['MANTRIJERON', 'KRATON', 'NGAMPILAN'])
        combined_df.loc[mask_mantri, 'kabupaten'] = 'Yogyakarta'
        combined_df.loc[mask_mantri, 'kab_key'] = 'YOGYAKARTA'

    # 4. [PENTING] BUAT UNIQUE KEY (GABUNGAN KAB + KEC)
    # Ini solusinya: BANTUL_JETIS beda dengan YOGYAKARTA_JETIS
    if 'kab_key' in combined_df.columns and 'kec_key' in combined_df.columns:
        combined_df['unique_key'] = combined_df['kab_key'] + "_" + combined_df['kec_key']

    return combined_df

@st.cache_data
def load_iso_data():
    folder_path = 'data'
    df_list = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            # Syarat: File Excel DAN ADA '_kode' di namanya
            if (file.endswith('.xlsx') or file.endswith('.xls')) and ('_kode' in file.lower()):
                file_path = os.path.join(folder_path, file)
                try:
                    temp_df = pd.read_excel(file_path)
                    
                    # --- CLEANING SPESIFIK UNTUK ISOCHRONE ---
                    # 1. Lowercase header & ganti spasi jadi _
                    temp_df.columns = temp_df.columns.str.lower().str.strip().str.replace(' ', '_')
                    
                    # 2. Rename 'title' jadi 'nama_venue' (karena di file kode biasanya pakai 'title')
                    if 'title' in temp_df.columns:
                        temp_df = temp_df.rename(columns={'title': 'nama_venue'})
                    
                    # 3. Pastikan kode_venue string
                    if 'kode_venue' in temp_df.columns:
                        temp_df['kode_venue'] = temp_df['kode_venue'].astype(str)
                        
                    df_list.append(temp_df)
                except: pass

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_shp_data():
    try:
        gdf = gpd.read_file("data/shp/kec_jogja.shp") 
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        return None

# Konfigurasi Kolom SHP
SHP_COL_KEC = 'nmkec'
SHP_COL_KAB = 'nmkab'

# =================================================================
# FUNGSI BANTUAN (STATISTIK - ASLI)
# =================================================================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error saat memuat file: {e}")
        return None

def interpret_correlation(r):
    r_abs = abs(r)
    if r_abs >= 0.8: return "sangat kuat"
    if r_abs >= 0.6: return "kuat"
    if r_abs >= 0.4: return "cukup"
    if r_abs >= 0.2: return "lemah"
    return "sangat lemah"

# =================================================================
# SIDEBAR UTAMA DENGAN BRANDING PKL
# =================================================================
with st.sidebar:
    # --- LOGO & MASKOT SIDEBAR ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_logo, col_mascot = st.columns([1, 1.3])
    if os.path.exists(LOGO_IMG):
        with col_logo:
            st.image(LOGO_IMG, use_container_width=True)
    if os.path.exists(MASCOT_IMG):
        with col_mascot:
            st.image(MASCOT_IMG, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True) # Spasi
    st.markdown("<h1>KONTROL PANEL</h1>", unsafe_allow_html=True) # Menggunakan H1 Rakkas
    st.markdown("---", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload File Data (Format CSV/Excel)", type=['csv', 'xls', 'xlsx'])

# =================================================================
# LOGIKA UTAMA
# =================================================================
# =============================================================================
# JIKA BELUM UPLOAD FILE (DASHBOARD PETA UTAMA / DEFAULT)
# =============================================================================
if uploaded_file is None:
    # --- JUDUL & SAMBUTAN ---
    st.markdown("<h1>DASHBOARD SEBARAN LOKASI POTENSIAL PEKERJA GIG PKL 65<br>D.I. YOGYAKARTA</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-box">
        <h3 style='margin-top:0;'>Selamat Datang di Dashboard PKL 65</h3>
        <p style='font-size: 1.1rem;'>Halaman ini menyajikan peta interaktif persebaran lokasi pekerja Gig di D.I. Yogyakarta.</p>
        <p style='font-size: 1rem; margin-top: 15px;'>Untuk memulai <b>Analisis Statistik Mendalam</b>, silakan unggah dataset Anda di panel kiri.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- LOAD DATA PETA DEFAULT ---
    df_map = load_map_excel_data()
    gdf_shape = load_shp_data()

    if not df_map.empty and gdf_shape is not None:
        
        # [PENTING] STANDARISASI KOLOM OTOMATIS
        # Ini mencegah error "Kode Venue" vs "kode_venue" vs "kode venue"
        df_map.columns = df_map.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # --- TAB NAVIGASI PETA ---
        tab_sebaran, tab_iso = st.tabs(["🗺️ Peta Sebaran (Wilayah & Titik)", "⏱️ Peta Jangkauan & Label"])

        # =================================================================
        # TAB 1: PETA SEBARAN (Choropleth & Heatmap)
        # =================================================================
        with tab_sebaran:
            st.subheader("Filter Visualisasi Sebaran")
            
            c_filter1, c_filter2, c_filter3 = st.columns(3)
            with c_filter1:
                # Pastikan kolom 'kabupaten' ada (hasil standarisasi huruf kecil)
                if 'kabupaten' in df_map.columns:
                    list_kab = sorted(df_map['kabupaten'].unique().tolist())
                    selected_kab = st.multiselect("1. Pilih Kabupaten:", list_kab, default=list_kab, key='map_kab')
                else:
                    st.error("Kolom 'kabupaten' tidak ditemukan di Excel.")
                    selected_kab = []

            with c_filter2:
                if selected_kab and 'kecamatan' in df_map.columns:
                    df_filtered_kab = df_map[df_map['kabupaten'].isin(selected_kab)]
                    list_kec = sorted(df_filtered_kab['kecamatan'].unique().tolist())
                else:
                    list_kec = []
                selected_kec = st.multiselect("2. Pilih Kecamatan:", list_kec, default=list_kec, key='map_kec')
            
            with c_filter3:
                map_mode = st.radio("3. Mode Tampilan:", ["Gabungan", "Choropleth (Wilayah)", "Heatmap (Titik)"], horizontal=True, key='map_mode_radio')

            if selected_kab and selected_kec:
                # Filter Data Utama
                final_df = df_map[
                    (df_map['kabupaten'].isin(selected_kab)) & 
                    (df_map['kecamatan'].isin(selected_kec))
                ]
                
                # Filter SHP (Logic Cleaning untuk Key SHP)
                gdf_shape['kab_key'] = gdf_shape[SHP_COL_KAB].astype(str).str.upper().str.replace(" ", "").str.replace(r'^(KAB\.?|KABUPATEN|KOTA)\s+', '', regex=True)
                gdf_shape['kec_key'] = gdf_shape[SHP_COL_KEC].astype(str).str.upper().str.replace(" ", "")
                # Normalisasi nama khusus DIY
                gdf_shape['kab_key'] = gdf_shape['kab_key'].replace({'GUNUNGKIDUL': 'GUNUNGKIDUL', 'YOGYA': 'YOGYAKARTA'})
                gdf_shape['unique_key'] = gdf_shape['kab_key'] + "_" + gdf_shape['kec_key']

                selected_kab_keys = [k.upper().replace(" ", "") for k in selected_kab]
                selected_kec_keys = [k.upper().replace(" ", "") for k in selected_kec]

                final_gdf = gdf_shape[
                    (gdf_shape['kab_key'].isin(selected_kab_keys)) &
                    (gdf_shape['kec_key'].isin(selected_kec_keys))
                ]

                if not final_gdf.empty:
                    stats_kec = final_df.groupby('unique_key').size().reset_index(name='jumlah_lokasi')
                    gdf_viz = final_gdf.merge(stats_kec, on='unique_key', how='left')
                    gdf_viz['jumlah_lokasi'] = gdf_viz['jumlah_lokasi'].fillna(0)
                    gdf_viz['Nama Kecamatan'] = gdf_viz[SHP_COL_KEC].astype(str).str.title()
                    gdf_viz['Nama Kabupaten'] = gdf_viz[SHP_COL_KAB].astype(str).str.title()

                    bounds = final_gdf.total_bounds
                    center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                    m_sebaran = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

                    # Layer Choropleth
                    if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                        cp = folium.Choropleth(
                            geo_data=gdf_viz, name='Kepadatan', data=gdf_viz,
                            columns=['unique_key', 'jumlah_lokasi'], key_on='feature.properties.unique_key',
                            fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='Jumlah Lokasi', highlight=True
                        ).add_to(m_sebaran)
                        folium.GeoJsonTooltip(fields=['Nama Kabupaten', 'Nama Kecamatan', 'jumlah_lokasi'], aliases=['Kabupaten:', 'Kecamatan:', 'Jumlah:'], localize=True).add_to(cp.geojson)

                    # Layer Heatmap
                    if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                        # Pastikan kolom latitude/longitude sesuai (sudah lowercase karena standarisasi)
                        # Cek typo umum: lattitude vs latitude
                        lat_col = 'lattitude' if 'lattitude' in final_df.columns else 'latitude'
                        lon_col = 'longitude' # Biasanya konsisten
                        
                        if lat_col in final_df.columns and lon_col in final_df.columns:
                            heat_data = final_df[[lat_col, lon_col]].dropna().values.tolist()
                            HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: '#FFD700', 0.65: '#FF8C00', 1: '#8B0000'}).add_to(m_sebaran)

                    folium.LayerControl().add_to(m_sebaran)
                    
                    st.markdown('<div style="box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-radius: 12px; border: 2px solid #E07A3F; overflow: hidden;">', unsafe_allow_html=True)
                    st_folium(m_sebaran, width=1200, height=600, key="folium_sebaran")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("Lihat Data Tabel"):
                        st.dataframe(final_df, use_container_width=True)
                else:
                    st.warning("Wilayah SHP tidak ditemukan untuk filter ini.")
            else:
                st.info("Pilih filter wilayah di atas.")

# =================================================================
        # TAB 2: PETA JANGKAUAN (MODIFIKASI: FILTER RADIUS OTOMATIS)
        # =================================================================
        with tab_iso:
            st.subheader("Analisis Jangkauan & Jarak Tempuh")
            
            # --- LOAD DATA KHUSUS ---
            df_iso = load_iso_data()
            
            if df_iso is None or df_iso.empty:
                st.warning("⚠️ Data khusus kode venue (file *_kode.xlsx) tidak ditemukan.")
            else:
                st.markdown("""
                <div style='background-color: #FDF1D6; padding: 15px; border-radius: 10px; border-left: 5px solid #F2C94C; margin-bottom: 20px;'>
                    <small><b>Mode Smart Radius:</b> 
                    <br>1. Pilih Titik Pusat.
                    <br>2. Klik Hitung.
                    <br>3. Peta hanya akan menampilkan titik-titik yang <b>masuk dalam jangkauan (radius)</b> saja.</small>
                </div>
                """, unsafe_allow_html=True)

                # --- 1. INISIALISASI SESSION STATE ---
                if 'iso_geojson' not in st.session_state: st.session_state['iso_geojson'] = None
                if 'iso_center_coord' not in st.session_state: st.session_state['iso_center_coord'] = None
                if 'iso_center_name' not in st.session_state: st.session_state['iso_center_name'] = ""
                if 'iso_speed' not in st.session_state: st.session_state['iso_speed'] = 30
                if 'iso_matrix_data' not in st.session_state: st.session_state['iso_matrix_data'] = None 
                if 'iso_targets' not in st.session_state: st.session_state['iso_targets'] = [] 

                # --- 2. LAYOUT FILTER (3 Kolom) ---
                iso_col1, iso_col2, iso_col3 = st.columns([1, 1.2, 1.5])
                
                # --- KOLOM 1: FILTER WILAYAH (KABUPATEN -> KODE) ---
                with iso_col1:
                    st.markdown("##### 1. Filter Data")
                    
                    # A. Filter Kabupaten
                    if 'kabupaten' in df_iso.columns:
                        list_kab_iso = sorted(df_iso['kabupaten'].astype(str).unique().tolist())
                        list_kab_iso.insert(0, "SEMUA KABUPATEN")
                        pilih_kab_iso = st.selectbox("1. Pilih Kabupaten:", list_kab_iso, key='iso_filter_kab')
                        
                        if pilih_kab_iso == "SEMUA KABUPATEN":
                            df_iso_step0 = df_iso.copy()
                        else:
                            df_iso_step0 = df_iso[df_iso['kabupaten'] == pilih_kab_iso]
                    else:
                        df_iso_step0 = df_iso.copy()

                    # B. Filter Kode Venue (Langsung, tanpa Kecamatan)
                    col_kode = 'kode_venue' if 'kode_venue' in df_iso_step0.columns else None
                    if col_kode:
                        unique_kodes = sorted(df_iso_step0[col_kode].astype(str).unique().tolist())
                        unique_kodes.insert(0, "SEMUA KODE")
                        pilih_kode_iso = st.selectbox(f"2. Pilih {col_kode.title()}:", unique_kodes, key='iso_filter_kode')
                        
                        if pilih_kode_iso == "SEMUA KODE":
                            df_iso_final = df_iso_step0.copy()
                        else:
                            df_iso_final = df_iso_step0[df_iso_step0[col_kode].astype(str) == pilih_kode_iso]
                    else:
                        df_iso_final = df_iso_step0.copy()

                # --- KOLOM 2: PARAMETER ---
                with iso_col2:
                    st.markdown("##### 2. Parameter")
                    api_key = st.text_input("🔑 ORS API Key", type="password", help="Wajib diisi.")
                    speed_val = st.slider("Kecepatan (km/jam)", 10, 80, 30)
                    
                    st.markdown("---")
                    enable_isodistance = st.checkbox("Aktifkan Hitung Jarak (Isodistance)", value=False)
                    
                    target_name_selection = "SEMUA TITIK" 
                    if enable_isodistance and not df_iso_final.empty:
                        c_name = 'nama_venue' if 'nama_venue' in df_iso_final.columns else 'title'
                        list_dest = sorted(df_iso_final[c_name].astype(str).unique().tolist())
                        list_dest.insert(0, "SEMUA TITIK DALAM RADIUS")
                        target_name_selection = st.selectbox("Pilih Tujuan:", list_dest)

                # --- KOLOM 3: TITIK PUSAT ---
                with iso_col3:
                    st.markdown("##### 3. Titik Pusat")
                    col_nama = 'nama_venue' if 'nama_venue' in df_iso_final.columns else ('title' if 'title' in df_iso_final.columns else df_iso_final.columns[0])
                    
                    if not df_iso_final.empty:
                        list_lokasi = sorted(df_iso_final[col_nama].astype(str).unique().tolist())
                        center_point_name = st.selectbox("Pilih Lokasi Pusat (Start):", list_lokasi, key="iso_center_select")
                    else:
                        center_point_name = None
                        st.info("Data kosong.")

                    st.caption(f"Total Data Awal: **{len(df_iso_final)}** titik.")
                    run_iso = st.button("📍 Hitung Analisis", use_container_width=True, key="btn_iso_run")

                # --- 3. LOGIKA PROSES (DENGAN FILTER RADIUS) ---
                if run_iso:
                    if api_key and center_point_name:
                        try:
                            # 1. Persiapan Data Pusat
                            row_pusat = df_iso_final[df_iso_final[col_nama] == center_point_name].iloc[0]
                            lat_c = 'lattitude' if 'lattitude' in df_iso_final.columns else 'latitude'
                            c_lat, c_lon = row_pusat[lat_c], row_pusat['longitude']

                            st.session_state['iso_center_coord'] = [c_lat, c_lon]
                            st.session_state['iso_center_name'] = center_point_name
                            st.session_state['iso_speed'] = speed_val
                            
                            client = openrouteservice.Client(key=api_key)

                            # 2. HITUNG ISOCHRONE
                            with st.spinner("1/3 Menghitung area jangkauan..."):
                                ranges_m = [(t/60) * speed_val * 1000 for t in [25, 20, 15, 10, 5]]
                                iso_res = client.isochrones(
                                    locations=[[c_lon, c_lat]], profile='driving-car',
                                    range_type='distance', range=ranges_m, units='m'
                                )
                                iso_res['features'] = sorted(iso_res['features'], key=lambda x: x['properties']['value'], reverse=True)
                                st.session_state['iso_geojson'] = iso_res

                            # 3. FILTER TITIK DALAM RADIUS (LOGIKA BARU)
                            # Kita gunakan library shapely untuk cek apakah titik ada di dalam poligon terbesar
                            with st.spinner("2/3 Memfilter titik dalam radius..."):
                                from shapely.geometry import shape, Point
                                
                                # Ambil poligon terluar (index 0 karena sudah disort reverse=True)
                                outer_polygon = shape(iso_res['features'][0]['geometry'])
                                
                                # Fungsi cek
                                def is_in_radius(row):
                                    point = Point(row['longitude'], row[lat_c])
                                    return outer_polygon.contains(point)
                                
                                # Terapkan filter ke dataframe sementara
                                df_inside = df_iso_final[df_iso_final.apply(is_in_radius, axis=1)]
                                
                                if len(df_inside) == 0:
                                    st.warning("Tidak ada titik lain yang masuk dalam jangkauan isochrone ini.")

                            # 4. HITUNG ISODISTANCE (Hanya untuk titik yang SUDAH DIFILTER)
                            if enable_isodistance and not df_inside.empty:
                                with st.spinner(f"3/3 Menghitung jarak real untuk {len(df_inside)} titik..."):
                                    if target_name_selection == "SEMUA TITIK DALAM RADIUS":
                                        df_targets = df_inside[df_inside[col_nama] != center_point_name]
                                    else:
                                        df_targets = df_inside[df_inside[col_nama] == target_name_selection]
                                    
                                    st.session_state['iso_targets'] = df_targets[col_nama].tolist()

                                    if not df_targets.empty:
                                        locations = [[c_lon, c_lat]] 
                                        dest_names = []
                                        
                                        # Limitasi API (Max 50 titik terdekat)
                                        for _, r in df_targets.head(49).iterrows():
                                            locations.append([r['longitude'], r[lat_c]])
                                            dest_names.append(r[col_nama])
                                        
                                        matrix = client.distance_matrix(
                                            locations=locations, profile='driving-car', metrics=['distance'], units='km'
                                        )
                                        distances = matrix['distances'][0][1:]
                                        result_data = list(zip(dest_names, distances))
                                        st.session_state['iso_matrix_data'] = result_data
                                    else:
                                        st.session_state['iso_matrix_data'] = None
                            else:
                                st.session_state['iso_matrix_data'] = None
                                st.session_state['iso_targets'] = []

                            st.success(f"Selesai! Ditemukan {len(df_inside)} titik dalam jangkauan.")

                        except Exception as e:
                            st.error(f"Gagal: {e}")
                    elif not api_key:
                        st.warning("⚠️ API Key kosong.")

                # --- 4. RENDER PETA ---
                if st.session_state['iso_center_coord']:
                    map_center = st.session_state['iso_center_coord']; zoom = 14
                elif not df_iso_final.empty:
                    lc = 'lattitude' if 'lattitude' in df_iso_final.columns else 'latitude'
                    map_center = [df_iso_final.iloc[0][lc], df_iso_final.iloc[0]['longitude']]; zoom = 13
                else:
                    map_center = [-7.7956, 110.3695]; zoom = 11

                m_iso = folium.Map(location=map_center, zoom_start=zoom, tiles='CartoDB positron')

                # LAYER ISOCHRONE
                if st.session_state['iso_geojson']:
                    colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
                    labels = ['20-25 Mnt', '15-20 Mnt', '10-15 Mnt', '5-10 Mnt', '< 5 Mnt']
                    for i, feature in enumerate(st.session_state['iso_geojson']['features']):
                        col = colors[i] if i < len(colors) else 'gray'
                        lbl = labels[i] if i < len(labels) else ''
                        folium.GeoJson(feature, style_function=lambda x, col=col: {'fillColor': col, 'color': 'black', 'weight': 1, 'fillOpacity': 0.6}, tooltip=f"Zona: {lbl}").add_to(m_iso)
                    
                    # Legenda Kiri
                    legend_html = """
                    <div style="position: fixed; bottom: 30px; left: 30px; width: 150px; height: 130px; 
                    background-color: white; z-index:9999; font-size:11px; border:2px solid grey; border-radius: 5px; padding: 10px; opacity: 0.9;">
                        <b>Legenda Waktu</b><br>
                        <i class="fa fa-square" style="color:#1a9641"></i> < 5 Mnt (Dekat)<br>
                        <i class="fa fa-square" style="color:#a6d96a"></i> 5-10 Mnt<br>
                        <i class="fa fa-square" style="color:#ffffbf; border:1px solid #ccc"></i> 10-15 Mnt<br>
                        <i class="fa fa-square" style="color:#fdae61"></i> 15-20 Mnt<br>
                        <i class="fa fa-square" style="color:#d7191c"></i> > 20 Mnt (Jauh)
                    </div>
                    """
                    m_iso.get_root().html.add_child(folium.Element(legend_html))

                # LAYER ISODISTANCE RESULT
                if st.session_state['iso_matrix_data']:
                    list_items = ""
                    for name, dist in st.session_state['iso_matrix_data']:
                        list_items += f"<li><b>{name}</b>: {dist:.2f} km</li>"
                    
                    dist_legend_html = f"""
                    <div style="position: fixed; bottom: 30px; right: 10px; width: 200px; max-height: 300px; 
                    background-color: white; z-index:9999; font-size:11px; border:2px solid black; border-radius: 5px; padding: 10px; opacity: 0.95; overflow-y: auto;">
                        <h5 style="margin:0; text-align:center;">Jarak Tempuh (Mobil)</h5>
                        <hr style="margin:5px 0;">
                        <ul style="padding-left: 15px; margin:0;">{list_items}</ul>
                    </div>
                    """
                    m_iso.get_root().html.add_child(folium.Element(dist_legend_html))
                    
                    if len(st.session_state['iso_matrix_data']) == 1:
                        target_name = st.session_state['iso_matrix_data'][0][0]
                        # Cari di df_iso_final (karena df_inside lokal)
                        row_t = df_iso_final[df_iso_final[col_nama] == target_name].iloc[0]
                        lat_c_col = 'lattitude' if 'lattitude' in df_iso_final.columns else 'latitude'
                        folium.PolyLine(locations=[st.session_state['iso_center_coord'], [row_t[lat_c_col], row_t['longitude']]], color="red", weight=2, dash_array='5, 5', opacity=0.8).add_to(m_iso)

                # LAYER TITIK & LABEL (YANG SUDAH DI-FILTER SECARA VISUAL)
                lat_c_col = 'lattitude' if 'lattitude' in df_iso_final.columns else 'latitude'
                
                # Tentukan DataFrame mana yang akan di-plot
                df_to_plot = df_iso_final
                
                # JIKA SUDAH ADA HASIL ISOCHRONE, KITA FILTER LAGI DISINI AGAR PETA RAPI
                if st.session_state['iso_geojson']:
                    try:
                        from shapely.geometry import shape, Point
                        outer_poly = shape(st.session_state['iso_geojson']['features'][0]['geometry'])
                        df_to_plot = df_iso_final[df_iso_final.apply(lambda x: outer_poly.contains(Point(x['longitude'], x[lat_c_col])), axis=1)]
                    except:
                        pass # Fallback ke semua data jika error

                for _, row in df_to_plot.iterrows():
                    # Skip pusat
                    if st.session_state['iso_center_name'] and row[col_nama] == st.session_state['iso_center_name']: continue
                    
                    is_target = row[col_nama] in st.session_state['iso_targets']
                    f_color = 'yellow' if is_target else 'cyan'
                    border_c = 'red' if is_target else 'blue'
                    rad = 6 if is_target else 4
                    
                    kode_txt = str(row.get(col_kode, '?')).replace('nan', '?')
                    
                    folium.CircleMarker(
                        [row[lat_c_col], row['longitude']], radius=rad, color=border_c, fill=True, fill_color=f_color, fill_opacity=0.9,
                        popup=f"<b>{row[col_nama]}</b><br>Kode: {kode_txt}", tooltip=f"{row[col_nama]}"
                    ).add_to(m_iso)
                    
                    folium.Marker(
                        [row[lat_c_col], row['longitude']],
                        icon=DivIcon(icon_size=(150,36), icon_anchor=(6, 14), html=f'<div style="font-size: 8pt; font-weight: bold;">{kode_txt}</div>')
                    ).add_to(m_iso)

                if st.session_state['iso_center_coord']:
                    folium.Marker(st.session_state['iso_center_coord'], icon=folium.Icon(color='red', icon='star'), tooltip="PUSAT").add_to(m_iso)

                st_folium(m_iso, width=1200, height=600)
    else:
        st.warning("Data peta default tidak tersedia. Pastikan fungsi load_map_excel_data() berfungsi.")
        
# === JIKA FILE SUDAH DIUPLOAD (KODE ASLI STATISTIK) ===
else:
    # Inisialisasi
    df = None
    numeric_cols = []
    categorical_cols = []
    all_cols = []

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File berhasil di-upload.")
            
            # Identifikasi tipe variabel
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            st.sidebar.subheader("Variabel Teridentifikasi")
            st.sidebar.write(f"**Kolom Numerik:** ({len(numeric_cols)})")
            st.sidebar.caption(f", ".join(numeric_cols))
            st.sidebar.write(f"**Kolom Kategorikal:** ({len(categorical_cols)})")
            st.sidebar.caption(f", ".join(categorical_cols))

    if df is not None and all_cols:
        # Judul Halaman Statistik (H1 Rakkas)
        st.markdown("<h1>ANALISIS STATISTIK</h1>", unsafe_allow_html=True)
        
        tab_data, tab_basic, tab_reg, tab_anova, tab_manova, tab_dim, tab_class = st.tabs([
            "Beranda & Data",
            "Analisis Dasar",
            "Model Regresi",
            "ANOVA",
            "MANOVA",
            "Reduksi Dimensi (PCA/EFA)",
            "Klasifikasi & Clustering"
        ])

        # -------------------------------------------------------------
        # TAB 0: RINGKASAN DATA
        # -------------------------------------------------------------
        with tab_data:
            st.header("Ringkasan dan Tampilan Data")
            
            st.subheader("Ringkasan Statistik (Variabel Numerik)")
            st.info("Statistik deskriptif dasar (rata-rata, median, min, max, dll.) untuk semua kolom numerik dalam data Anda.")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik untuk diringkas.")

            st.subheader("Tampilan Data Mentah (50 Baris Pertama)")
            st.info("Tampilan 50 baris pertama dari data yang Anda upload.")
            st.dataframe(df.head(50), use_container_width=True)

        # -------------------------------------------------------------
        # TAB 1: ANALISIS DASAR (Univariat & Bivariat)
        # -------------------------------------------------------------
        with tab_basic:
            st.header("Analisis Dasar (Univariat & Bivariat)")
            st.info("Analisis ini berfokus pada 1 atau 2 variabel pada satu waktu.")
            st.markdown("---")

            if not numeric_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                # --- ANALISIS UNIVARIAT ---
                st.subheader("Analisis Univariat (Satu Variabel)")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("Melihat sebaran frekuensi dari sebuah variabel numerik.")
                    hist_col = st.selectbox("Pilih variabel untuk Histogram:", numeric_cols, key='hist_col')
                    if hist_col:
                        # Menggunakan warna Terracotta/Gold
                        fig_hist = px.histogram(df, x=hist_col, title=f'Histogram untuk {hist_col}', marginal="box", color_discrete_sequence=['#E07A3F'])
                        fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    st.info("Menguji apakah data terdistribusi normal (simetris).")
                    norm_col = st.selectbox("Pilih variabel untuk Uji Normalitas:", numeric_cols, key='norm_col')
                    
                    if st.button("Jalankan Uji Normalitas (Shapiro-Wilk)", key='norm_btn'):
                        data_to_test = df[norm_col].dropna()
                        if len(data_to_test) < 3:
                            st.error("Uji Normalitas memerlukan setidaknya 3 sampel.")
                        else:
                            stat, p_value = stats.shapiro(data_to_test)
                            st.write(f"**P-value:** `{p_value:.4f}`")
                            
                            # Interpretasi hasil
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** Uji Normalitas (Shapiro-Wilk) bertujuan untuk mengecek apakah data pada variabel '{norm_col}' terdistribusi normal.
                                **Aturan Keputusan (Rule of Thumb):**
                                * **P-value > 0.05:** Data **TERDISTRIBUSI NORMAL**.
                                * **P-value <= 0.05:** Data **TIDAK TERDISTRIBUSI NORMAL**.
                                **Hasil Anda:** P-value Anda adalah **{p_value:.4f}**.
                                """)
                                if p_value > 0.05:
                                    st.success(f"**Kesimpulan:** Hasil Anda **NORMAL** (karena P-value > 0.05).")
                                else:
                                    st.error(f"**Kesimpulan:** Hasil Anda **TIDAK NORMAL** (karena P-value <= 0.05).")
                
                st.markdown("---")

                # --- ANALISIS BIVARIAT (KORELASI & UJI T) ---
                st.subheader("Analisis Bivariat (Dua Variabel)")
                
                # Korelasi
                st.write("**Hubungan Numerik vs Numerik (Korelasi)**")
                col3, col4 = st.columns([1, 2])
                with col3:
                    st.info("Pilih dua variabel numerik untuk melihat hubungan linear dan korelasinya.")
                    bi_x = st.selectbox("Pilih Variabel X:", numeric_cols, key='bi_x')
                    bi_y = st.selectbox("Pilih Variabel Y:", numeric_cols, key='bi_y')
                
                with col4:
                    if bi_x and bi_y and bi_x != bi_y:
                        data_bi = df[[bi_x, bi_y]].dropna()
                        # Menggunakan warna Gold/Terracotta
                        fig_scatter = px.scatter(data_bi, x=bi_x, y=bi_y, title=f"Scatter Plot: {bi_y} vs {bi_x}", trendline="ols", color_discrete_sequence=['#F2C94C'])
                        fig_scatter.update_traces(marker=dict(color='#E07A3F')) # Marker Terracotta
                        fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        st.write("**Uji Korelasi (Pearson)**")
                        corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                        st.write(f"* **Koefisien Korelasi (r):** `{corr:.4f}`")
                        st.write(f"* **P-value:** `{p_value:.4f}`")
                        
                        # Interpretasi hasil
                        with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                            st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                            st.markdown(f"""
                            **Tujuan Uji:** Uji Korelasi Pearson mengukur kekuatan dan arah hubungan *linear* antara '{bi_x}' dan '{bi_y}'.
                            **Aturan Keputusan (Rule of Thumb):**
                            1.  **P-value:** Menunjukkan apakah hubungan itu "nyata" (signifikan).
                                * `P-value > 0.05`: Hubungan **Tidak Nyata**.
                                * `P-value <= 0.05`: Hubungan **Nyata**.
                            2.  **Koefisien Korelasi (r):** Jika hubungannya nyata, ini menunjukkan kekuatan dan arah.
                            **Hasil Anda:** P-value Anda adalah **{p_value:.4f}**, Koefisien (r) Anda adalah **{corr:.4f}**.
                            """)
                            if p_value < 0.05:
                                st.success(f"**Kesimpulan:** Ya, ada hubungan yang **nyata** antara {bi_x} dan {bi_y} (karena P-value <= 0.05).")
                                st.info(f" **Kekuatan:** Hubungan ini **{interpret_correlation(corr)}** dan bersifat {'positif' if corr > 0 else 'negatif'}.")
                            else:
                                st.error(f"**Kesimpulan:** Tidak, **tidak ditemukan** hubungan yang nyata antara {bi_x} dan {bi_y} (karena P-value > 0.05).")
                    elif bi_x == bi_y:
                        st.warning("Variabel X dan Y tidak boleh sama.")
                
                st.markdown("---")

                # Uji T
                st.write("**Hubungan Kategorikal vs Numerik (Uji T)**")
                col5, col6 = st.columns([1, 2])
                with col5:
                    st.info("Membandingkan rata-rata variabel numerik di antara **DUA** kelompok.")
                    if not categorical_cols:
                        st.warning("Tidak ada kolom kategorikal yang terdeteksi untuk Uji T.")
                        cat_col_t = None
                        num_col_t = None
                    else:
                        cat_col_t = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='bi_cat_t')
                        num_col_t = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='bi_num_t')
                
                with col6:
                    if categorical_cols and cat_col_t and num_col_t:
                        groups_t = df[cat_col_t].dropna().unique()
                        
                        if len(groups_t) == 2:
                            if st.button("Jalankan Uji T", key='t_test_btn'):
                                st.write("**Uji T (Independent T-Test)**")
                                # Menggunakan palet warna Grand Design (Teal & Terracotta)
                                fig_box = px.box(df, x=cat_col_t, y=num_col_t, title=f"Distribusi {num_col_t} berdasarkan {cat_col_t}", points="all", color=cat_col_t, color_discrete_sequence=['#4F8190', '#E07A3F'])
                                fig_box.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                st.plotly_chart(fig_box, use_container_width=True)

                                group1 = df[df[cat_col_t] == groups_t[0]][num_col_t].dropna()
                                group2 = df[df[cat_col_t] == groups_t[1]][num_col_t].dropna()
                                
                                stat, p_value = stats.ttest_ind(group1, group2)
                                st.write(f"* **P-value:** `{p_value:.4f}`")
                                
                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan Uji:** Uji T membandingkan rata-rata '{num_col_t}' antara '{groups_t[0]}' dan '{groups_t[1]}'.
                                    **Aturan Keputusan (Rule of Thumb):**
                                    * **P-value > 0.05:** Perbedaan **Tidak Signifikan**.
                                    * **P-value <= 0.05:** Perbedaan **Signifikan**.
                                    **Hasil Anda:** P-value Anda adalah **{p_value:.4f}**.
                                    """)
                                    if p_value < 0.05:
                                        st.success(f"**Kesimpulan:** Ya, ada perbedaan rata-rata {num_col_t} yang **nyata** antara kelompok (karena P-value <= 0.05).")
                                    else:
                                        st.error(f"**Kesimpulan:** Tidak, **tidak ditemukan** perbedaan rata-rata {num_col_t} yang nyata (karena P-value > 0.05).")
                        elif len(groups_t) > 2:
                            st.warning(f"Variabel '{cat_col_t}' memiliki {len(groups_t)} kelompok. Pindah ke tab 'ANOVA' untuk menguji 3 kelompok atau lebih.")
                        else:
                            st.warning(f"Variabel '{cat_col_t}' hanya memiliki {len(groups_t)} kelompok. Tidak dapat diuji.")

                # ==========================================================
                # PENAMBAHAN BARU: PLOT DENGAN ERROR BAR
                # ==========================================================
                st.markdown("---")
                st.subheader("Perbandingan Rata-rata (dengan Error Bar)")
                st.info("Visualisasikan rata-rata (mean) dengan galat (error bar) untuk setiap kelompok. Mirip dengan Box Plot, tapi berfokus pada Mean dan SD/SE.")
                
                col7, col8 = st.columns([1, 2])
                
                with col7:
                    if not categorical_cols:
                        st.warning("Analisis ini memerlukan setidaknya 1 kolom kategorikal.")
                        eb_cat = None
                        eb_num = None
                    else:
                        eb_cat = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='eb_cat')
                        eb_num = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='eb_num')
                        eb_type = st.radio("Pilih Tipe Error Bar:", ["Standar Error (SE)", "Standar Deviasi (SD)"], key='eb_type')
                        eb_plot_type = st.radio("Tipe Plot:", ["Bar Chart", "Line Chart"], key='eb_plot_type')
                        
                with col8:
                    if eb_cat and eb_num:
                        if st.button("Buat Grafik Error Bar", key='eb_btn'):
                            try:
                                # 1. Hitung statistik yang diperlukan
                                df_agg = df.groupby(eb_cat)[eb_num].agg(['mean', 'std', 'count']).reset_index()
                                
                                # 2. Hitung Standar Error (SE)
                                df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count'])
                                
                                # 3. Tentukan nilai error berdasarkan pilihan user
                                error_col = 'se' if eb_type == "Standar Error (SE)" else 'std'
                                df_agg['error_val'] = df_agg[error_col]
                                
                                # 4. Buat plot
                                fig = go.Figure()
                                
                                # Warna Terracotta untuk plot
                                plot_color = '#E07A3F'

                                if eb_plot_type == "Line Chart":
                                    fig.add_trace(go.Scatter(
                                        x=df_agg[eb_cat],
                                        y=df_agg['mean'],
                                        error_y=dict(
                                            type='data',
                                            array=df_agg['error_val'],
                                            visible=True,
                                            color=plot_color
                                        ),
                                        mode='lines+markers',
                                        line=dict(color=plot_color),
                                        marker=dict(color=plot_color)
                                    ))
                                else: # Bar Chart
                                    fig.add_trace(go.Bar(
                                        x=df_agg[eb_cat],
                                        y=df_agg['mean'],
                                        error_y=dict(
                                            type='data',
                                            array=df_agg['error_val'],
                                            visible=True,
                                            color=plot_color
                                        ),
                                        marker_color=plot_color
                                    ))
                                
                                fig.update_layout(
                                    title=f"Rata-rata {eb_num} per {eb_cat} (Error Bar: {eb_type})",
                                    xaxis_title=eb_cat,
                                    yaxis_title=f"Rata-rata {eb_num}",
                                    plot_bgcolor='rgba(0,0,0,0)', 
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_family="Poppins"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                with st.expander("Lihat Data Agregat"):
                                    st.dataframe(df_agg)

                            except Exception as e:
                                st.error(f"Gagal membuat grafik: {e}")
                    
                    elif categorical_cols:
                        st.warning("Silakan pilih variabel kategorikal dan numerik.")

        # -------------------------------------------------------------
        # TAB 3: MODEL REGRESI (MODIFIKASI)
        # -------------------------------------------------------------
        with tab_reg:
            st.header("Model Regresi")
            st.info("Memodelkan hubungan antara variabel dependen (Y) dan variabel independen (X) untuk membuat prediksi.")
            st.markdown("---")

            if not numeric_cols:
                st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # --- PERUBAHAN DI SINI ---
                    reg_y_list = st.multiselect("Pilih Variabel Dependen (Y) (min. 1):", numeric_cols, key='reg_y_list')
                    
                    available_x = [col for col in numeric_cols if col not in reg_y_list]
                    reg_x = st.multiselect("Pilih Variabel Independen (X):", available_x, key='reg_x')
                
                # --- KONDISI 1: REGRESI SEDERHANA / POLINOMIAL (1Y, 1X) ---
                if len(reg_y_list) == 1 and len(reg_x) == 1:
                    reg_y = reg_y_list[0] # Ambil satu-satunya Y
                    
                    st.subheader(f"Regresi Sederhana: {reg_y} vs {reg_x[0]}")
                    
                    with col1:
                        poly_degree = st.radio(
                            "Pilih Tipe Model:",
                            [1, 2, 3],
                            format_func=lambda x: f"Linear (Derajat 1)" if x == 1 else f"Kuadratik (Derajat {x})" if x == 2 else f"Kubik (Derajat {x})",
                            key='poly_degree'
                        )
                    
                    with col2:
                        data_reg = df[[reg_y] + reg_x].dropna()
                        X_simple = data_reg[[reg_x[0]]]
                        y_simple = data_reg[reg_y]
                        
                        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        X_poly = poly_features.fit_transform(X_simple)
                        
                        model = LinearRegression()
                        model.fit(X_poly, y_simple)
                        y_pred = model.predict(X_poly)
                        r2 = r2_score(y_simple, y_pred)
                        
                        plot_df = pd.DataFrame({'X': X_simple.iloc[:, 0], 'y_true': y_simple, 'y_pred': y_pred}).sort_values(by='X')
                        
                        # Warna Gold dan Terracotta
                        fig_poly = px.scatter(plot_df, x='X', y='y_true', title=f"Model Regresi (Derajat {poly_degree})", color_discrete_sequence=['#F2C94C'])
                        fig_poly.add_trace(go.Scatter(x=plot_df['X'], y=plot_df['y_pred'], name='Garis Prediksi', line=dict(color='#E07A3F', width=3)))
                        fig_poly.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig_poly, use_container_width=True)
                        
                        st.write(f"**R-squared:** `{r2:.4f}`")
                        
                        with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                            st.success(f"**Kesimpulan (Bahasa Awam):** Model Anda **{r2*100:.2f}%** akurat. Ini berarti {r2*100:.2f}% dari perubahan pada '{reg_y}' dapat dijelaskan oleh perubahan pada '{reg_x[0]}' menggunakan model ini.")

                # --- KONDISI 2: REGRESI BERGANDA (1Y, >1X) ---
                elif len(reg_y_list) == 1 and len(reg_x) >= 2:
                    reg_y = reg_y_list[0] # Ambil satu-satunya Y
                    
                    st.subheader("Regresi Linear Berganda (OLS)")
                    st.info(f"Model: **{reg_y} = b0 + b1*{reg_x[0]} + b2*{reg_x[1]} + ...**")

                    with col2:
                        data_reg = df[[reg_y] + reg_x].dropna()
                        X_multi = data_reg[reg_x]
                        y_multi = data_reg[reg_y]
                        
                        X_with_const = sm.add_constant(X_multi)
                        model_ols = sm.OLS(y_multi, X_with_const).fit()
                        
                        st.write("**Ringkasan Model (OLS)**")
                        st.text_area("Ringkasan", model_ols.summary().as_text(), height=400)
                        
                        r_squared = model_ols.rsquared_adj
                        
                        with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                            st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                            st.markdown(f"**1. Kebaikan Model (R-squared)**\n* **Angka Anda:** *Adjusted R-squared* = **{r_squared:.4f}**.\n* **Kesimpulan:** Model Anda **{r_squared*100:.2f}%** akurat.")
                            st.markdown(f"**2. Signifikansi Variabel (Tabel `P>|t|`)**\n* **Aturan:** Jika `P>|t|` <= 0.05, variabel itu **signifikan**. Jika > 0.05, variabel itu **tidak signifikan**.")
                        
                        st.markdown("---")
                        st.subheader("Uji Asumsi")
                        st.write("**1. Multikolinearitas (VIF)**")
                        vif_data = pd.DataFrame()
                        vif_data["Variabel"] = X_with_const.columns
                        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                        st.dataframe(vif_data[vif_data["Variabel"] != 'const'])
                        
                        if (vif_data[vif_data["Variabel"] != 'const']["VIF"] > 10).any():
                            st.error(f"**Kesimpulan VIF:** Ditemukan VIF > 10. Ini mengindikasikan **multikolinearitas**.")
                        else:
                            st.success(f"**Kesimpulan VIF:** Semua VIF < 10. Asumsi **terpenuhi**.")
                        
                        st.write("**2. Heteroskedastisitas (Breusch-Pagan)**")
                        bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                        bp_p_value = bp_test[1]
                        st.write(f"* **P-value Uji Breusch-Pagan:** `{bp_p_value:.4f}`")
                        if bp_p_value < 0.05:
                            st.error(f"**Kesimpulan Hetero:** P-value < 0.05. Terindikasi **heteroskedastisitas**.")
                        else:
                            st.success(f"**Kesimpulan Hetero:** P-value > 0.05. Asumsi **terpenuhi** (homoskedastisitas).")

                # --- KONDISI 3: REGRESI MULTIVARIAT (>1Y, >0X) ---
                elif len(reg_y_list) >= 2 and len(reg_x) >= 1:
                    st.subheader("Regresi Linear Multivariat (Multi-Output)")
                    st.info(f"Model ini menjalankan regresi OLS terpisah untuk setiap variabel Y.")
                    st.write(f"Model: **({', '.join(reg_y_list)}) ~ {', '.join(reg_x)}**")

                    with col2:
                        try:
                            data_reg = df[reg_y_list + reg_x].dropna()
                            X_multi = data_reg[reg_x]
                            y_multi = data_reg[reg_y_list]
                            
                            model = LinearRegression()
                            model.fit(X_multi, y_multi)
                            y_pred = model.predict(X_multi)
                            
                            r2_scores = r2_score(y_multi, y_pred, multioutput='raw_values')
                            
                            st.write("**Koefisien Model**")
                            st.info("Setiap baris adalah model untuk 1 variabel Y. Kolom menunjukkan koefisien untuk setiap variabel X.")
                            coefs_df = pd.DataFrame(model.coef_, columns=reg_x, index=reg_y_list)
                            st.dataframe(coefs_df, use_container_width=True)
                            
                            st.write("**Intercept Model**")
                            intercept_df = pd.DataFrame(model.intercept_, index=reg_y_list, columns=['Intercept'])
                            st.dataframe(intercept_df, use_container_width=True)
                            
                            st.write("**R-squared (per Variabel Y)**")
                            r2_df = pd.DataFrame(r2_scores, index=reg_y_list, columns=['R-squared'])
                            st.dataframe(r2_df, use_container_width=True)

                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** Anda memprediksi {len(reg_y_list)} variabel Y secara bersamaan menggunakan {len(reg_x)} variabel X. Model ini pada dasarnya menjalankan {len(reg_y_list)} regresi berganda terpisah.
                                
                                **1. Koefisien Model:**
                                * Tabel ini menunjukkan bobot (pengaruh) dari setiap X terhadap setiap Y.
                                * **Contoh:** Nilai di baris `{reg_y_list[0]}` dan kolom `{reg_x[0]}` adalah koefisien `{reg_x[0]}` dalam model untuk memprediksi `{reg_y_list[0]}`.
                                
                                **2. R-squared:**
                                * Tabel ini menunjukkan seberapa baik *setiap* model Y bekerja.
                                * **Contoh:** R-squared untuk `{reg_y_list[0]}` (`{r2_scores[0]:.4f}`) berarti `{r2_scores[0]*100:.2f}%` variasi di `{reg_y_list[0]}` dapat dijelaskan oleh variabel X yang Anda pilih.
                                """)

                        except Exception as e:
                            st.error(f"Gagal menjalankan regresi multivariat: {e}")

                elif not reg_y_list or not reg_x:
                    st.info("Silakan pilih minimal 1 Variabel Dependen (Y) dan 1 Variabel Independen (X) untuk memulai.")

        # -------------------------------------------------------------
        # TAB 4: ANOVA (BAB 6)
        # -------------------------------------------------------------
        with tab_anova:
            st.header("Analisis Varians (ANOVA)")
            st.info("Membandingkan rata-rata variabel numerik di antara beberapa kelompok.")
            st.markdown("---")
            
            if not numeric_cols or not categorical_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik DAN satu kolom kategorikal.")
            else:
                # --- ANOVA ONE-WAY ---
                st.subheader("ANOVA One-Way")
                st.info("Membandingkan rata-rata dari **TIGA ATAU LEBIH** kelompok (berdasarkan 1 variabel kategorikal).")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    cat_col_anova1 = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='a1_cat')
                    num_col_anova1 = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='a1_num')
                
                with col2:
                    if cat_col_anova1 and num_col_anova1:
                        groups_a1 = df[cat_col_anova1].dropna().unique()
                        
                        if len(groups_a1) > 2:
                            if st.button("Jalankan ANOVA One-Way", key='a1_btn'):
                                # Warna Grand Design
                                fig_box_a1 = px.box(df, x=cat_col_anova1, y=num_col_anova1, title=f"Distribusi {num_col_anova1} berdasarkan {cat_col_anova1}", points="all", color=cat_col_anova1, color_discrete_sequence=['#4F8190', '#E07A3F', '#F2C94C', '#739159'])
                                fig_box_a1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                st.plotly_chart(fig_box_a1, use_container_width=True)

                                df_clean = df[[cat_col_anova1, num_col_anova1]].dropna()
                                # Sanitasi nama kolom untuk formula statsmodels
                                clean_cat = cat_col_anova1.replace(' ', '_').replace('[', '').replace(']', '')
                                clean_num = num_col_anova1.replace(' ', '_').replace('[', '').replace(']', '')
                                df_clean.columns = [clean_cat, clean_num]
                                
                                formula = f'{clean_num} ~ C({clean_cat})'
                                
                                try:
                                    model = smf.ols(formula, data=df_clean).fit()
                                    anova_table = sm.stats.anova_lm(model, typ=2)
                                    st.dataframe(anova_table)
                                    
                                    p_value_anova = anova_table['PR(>F)'][0]
                                    
                                    # Interpretasi hasil
                                    with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                        st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                        st.markdown(f"""
                                        **Tujuan Uji:** ANOVA One-Way membandingkan rata-rata '{num_col_anova1}' di antara {len(groups_a1)} kelompok.
                                        **Aturan Keputusan (Rule of Thumb):**
                                        * Lihat P-value (kolom `PR(>F)`).
                                        * **P-value > 0.05:** Perbedaan **Tidak Signifikan**.
                                        * **P-value <= 0.05:** Perbedaan **Signifikan**.
                                        **Hasil Anda:** P-value Anda adalah **{p_value_anova:.4f}**.
                                        """)
                                        if p_value_anova < 0.05:
                                            st.success(f"**Kesimpulan:** Ya, ada perbedaan rata-rata {num_col_anova1} yang **nyata** di antara kelompok-kelompok tersebut (karena P-value <= 0.05).")
                                        else:
                                            st.error(f"**Kesimpulan:** Tidak, **tidak ditemukan** perbedaan rata-rata {num_col_anova1} yang nyata (karena P-value > 0.05).")
                                except Exception as e:
                                    st.error(f"Error saat menjalankan ANOVA: {e}")
                        elif len(groups_a1) == 2:
                            st.warning(f"Variabel '{cat_col_anova1}' hanya memiliki 2 kelompok. Gunakan Uji T di tab 'Analisis Dasar'.")
                        else:
                            st.warning(f"Variabel '{cat_col_anova1}' hanya memiliki {len(groups_a1)} kelompok.")

                st.markdown("---")

                # --- ANOVA TWO-WAY ---
                st.subheader("ANOVA Two-Way")
                st.info("Membandingkan rata-rata variabel numerik berdasarkan **DUA** variabel kategorikal (dan interaksinya).")
                
                col3, col4 = st.columns([1, 2])
                with col3:
                    anova2_cat1 = st.selectbox("Pilih Variabel Kategorikal 1 (X1):", categorical_cols, key='a2_c1')
                    anova2_cat2 = st.selectbox("Pilih Variabel Kategorikal 2 (X2):", categorical_cols, key='a2_c2')
                    anova2_num = st.selectbox("Pilih Variabel Dependen (Y):", numeric_cols, key='a2_n')
                
                with col4:
                    if anova2_cat1 and anova2_cat2 and anova2_num and anova2_cat1 != anova2_cat2:
                        if st.button("Jalankan ANOVA Two-Way", key='a2_btn'):
                            df_clean = df[[anova2_cat1, anova2_cat2, anova2_num]].dropna()
                            c1, c2, n = [col.replace(' ', '_').replace('[', '').replace(']', '') for col in df_clean.columns]
                            df_clean.columns = [c1, c2, n]
                            
                            formula = f'{n} ~ C({c1}) + C({c2}) + C({c1}):C({c2})'
                            
                            try:
                                model = smf.ols(formula, data=df_clean).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                st.write("**Hasil ANOVA Two-Way:**")
                                st.dataframe(anova_table)
                                
                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan Uji:** ANOVA Two-Way menguji tiga hal:
                                    1.  Efek **{anova2_cat1}** (X1) pada {anova2_num} (Y)?
                                    2.  Efek **{anova2_cat2}** (X2) pada {anova2_num} (Y)?
                                    3.  Efek **interaksi** (X1*X2) pada Y?
                                    
                                    **Aturan Keputusan (Rule of Thumb):** Lihat `PR(>F)`. P-value <= 0.05 berarti "Ya, ada efek yang signifikan".
                                    """)
                                    
                                    p_c1 = anova_table.loc[f'C({c1})', 'PR(>F)']
                                    p_c2 = anova_table.loc[f'C({c2})', 'PR(>F)']
                                    p_int = anova_table.loc[f'C({c1}):C({c2})', 'PR(>F)']
                                    
                                    st.write(f"**1. Efek {anova2_cat1}**: {'SIGNIFIKAN' if p_c1 < 0.05 else 'TIDAK SIGNIFIKAN'} (P-value = {p_c1:.4f})")
                                    st.write(f"**2. Efek {anova2_cat2}**: {'SIGNIFIKAN' if p_c2 < 0.05 else 'TIDAK SIGNIFIKAN'} (P-value = {p_c2:.4f})")
                                    st.write(f"**3. Efek INTERAKSI**: {'SIGNIFIKAN' if p_int < 0.05 else 'TIDAK SIGNIFIKAN'} (P-value = {p_int:.4f})")

                            except Exception as e:
                                st.error(f"Error saat menjalankan ANOVA: {e}")
                    elif anova2_cat1 == anova2_cat2:
                        st.warning("Variabel Kategorikal 1 dan 2 tidak boleh sama.")

        # -------------------------------------------------------------
        # TAB 5: MANOVA (BAB 6)
        # -------------------------------------------------------------
        with tab_manova:
            st.header("Analisis MANOVA")
            st.info("Seperti ANOVA, tetapi untuk **DUA ATAU LEBIH** variabel dependen (Y) secara bersamaan.")
            st.markdown("---")

            if not numeric_cols or not categorical_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik DAN satu kolom kategorikal.")
            else:
                # --- MANOVA ONE-WAY ---
                st.subheader("MANOVA One-Way")
                st.info("Membandingkan beberapa rata-rata variabel Y berdasarkan **SATU** variabel kategorikal X.")

                col1, col2 = st.columns([1, 2])
                with col1:
                    manova1_cat = st.selectbox("Pilih Variabel Kelompok (X):", categorical_cols, key='m1_c')
                    manova1_num = st.multiselect("Pilih Variabel Dependen (Y) (minimal 2):", numeric_cols, key='m1_n')
                
                with col2:
                    if manova1_cat and len(manova1_num) >= 2:
                        if st.button("Jalankan MANOVA One-Way", key='m1_btn'):
                            df_clean = df[[manova1_cat] + manova1_num].dropna()
                            clean_cols = [col.replace(' ', '_').replace('[', '').replace(']', '') for col in df_clean.columns]
                            df_clean.columns = clean_cols
                            
                            c1 = clean_cols[0]
                            n_vars = clean_cols[1:]
                            
                            formula = f'{" + ".join(n_vars)} ~ C({c1})'
                            
                            try:
                                model = MANOVA.from_formula(formula, data=df_clean)
                                mv_test = model.mv_test()
                                
                                st.write("**Hasil MANOVA (Ringkasan Tes Multivariat):**")
                                st.dataframe(mv_test.summary_frame)
                                
                                p_value_manova = mv_test.summary_frame.loc[(f'C({c1})', "Wilks' lambda"), "Pr > F"]
                                
                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan Uji:** MANOVA menguji apakah '{manova1_cat}' memiliki efek pada *kombinasi* variabel Y.
                                    **Aturan Keputusan (Rule of Thumb):**
                                    * Lihat P-value (`Pr > F`) untuk **Wilks' lambda**.
                                    * **P-value > 0.05:** Perbedaan **Tidak Signifikan**.
                                    * **P-value <= 0.05:** Perbedaan **Signifikan**.
                                    **Hasil Anda:** P-value (Wilks' lambda) Anda adalah **{p_value_manova:.4f}**.
                                    """)
                                    if p_value_manova < 0.05:
                                        st.success(f"**Kesimpulan:** Ya, '{manova1_cat}' memiliki pengaruh yang **nyata** (karena P-value <= 0.05).")
                                    else:
                                        st.error(f"**Kesimpulan:** Tidak, '{manova1_cat}' **tidak memiliki pengaruh** yang nyata (karena P-value > 0.05).")
                            except Exception as e:
                                st.error(f"Error menjalankan MANOVA: {e}")
                    elif not manova1_cat or len(manova1_num) < 2:
                        st.warning("Silakan pilih 1 variabel kelompok dan minimal 2 variabel dependen.")

                st.markdown("---")
                
                # --- MANOVA TWO-WAY ---
                st.subheader("MANOVA Two-Way")
                st.info("Membandingkan beberapa rata-rata variabel Y berdasarkan **DUA** variabel kategorikal X.")
                
                col3, col4 = st.columns([1, 2])
                with col3:
                    manova2_cat1 = st.selectbox("Pilih Kelompok 1 (X1):", categorical_cols, key='m2_c1')
                    manova2_cat2 = st.selectbox("Pilih Kelompok 2 (X2):", categorical_cols, key='m2_c2')
                    manova2_num = st.multiselect("Pilih Variabel Dependen (Y) (minimal 2):", numeric_cols, key='m2_n')
                
                with col4:
                    if manova2_cat1 and manova2_cat2 and len(manova2_num) >= 2 and manova2_cat1 != manova2_cat2:
                        if st.button("Jalankan MANOVA Two-Way", key='m2_btn'):
                            df_clean = df[[manova2_cat1, manova2_cat2] + manova2_num].dropna()
                            clean_cols = [col.replace(' ', '_').replace('[', '').replace(']', '') for col in df_clean.columns]
                            df_clean.columns = clean_cols
                            
                            c1, c2 = clean_cols[0], clean_cols[1]
                            n_vars = clean_cols[2:]
                            
                            formula = f'{" + ".join(n_vars)} ~ C({c1}) + C({c2}) + C({c1}):C({c2})'
                            
                            try:
                                model = MANOVA.from_formula(formula, data=df_clean)
                                mv_test = model.mv_test()
                                
                                st.write("**Hasil MANOVA (Ringkasan Tes Multivariat):**")
                                st.dataframe(mv_test.summary_frame)
                                
                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan Uji:** MANOVA Two-Way menguji tiga hal (pada kombinasi Y).
                                    **Aturan Keputusan (Rule of Thumb):** Lihat `Pr > F` (Wilks' lambda). P-value <= 0.05 berarti "Ya, ada efek yang signifikan".
                                    """)
                                    
                                    p_c1 = mv_test.summary_frame.loc[(f'C({c1})', "Wilks' lambda"), "Pr > F"]
                                    p_c2 = mv_test.summary_frame.loc[(f'C({c2})', "Wilks' lambda"), "Pr > F"]
                                    p_int = mv_test.summary_frame.loc[(f'C({c1}):C({c2})', "Wilks' lambda"), "Pr > F"]

                                    st.write(f"**1. Efek {manova2_cat1}**: {'SIGNIFIKAN' if p_c1 < 0.05 else 'TIDAK SIGNIFIKAN'} (P-value = {p_c1:.4f})")
                                    st.write(f"**2. Efek {manova2_cat2}**: {'SIGNIFIKAN' if p_c2 < 0.05 else 'TIDAK SIGNIFIKAN'} (P-value = {p_c2:.4f})")
                                    st.write(f"**3. Efek INTERAKSI**: {'SIGNIFIKAN' if p_int < 0.05 else 'TIDAK SIGNIFIKAN'} (P-value = {p_int:.4f})")

                            except Exception as e:
                                st.error(f"Error menjalankan MANOVA: {e}")
                    elif not manova2_cat1 or not manova2_cat2 or len(manova2_num) < 2:
                        st.warning("Silakan pilih 2 variabel kelompok yang berbeda dan minimal 2 variabel dependen.")
                    elif manova2_cat1 == manova2_cat2:
                        st.warning("Variabel Kelompok 1 dan 2 tidak boleh sama.")

        # -------------------------------------------------------------
        # TAB 6: REDUKSI DIMENSI (PCA & EFA)
        # -------------------------------------------------------------
        with tab_dim:
            st.header("Reduksi Dimensi (Penyederhanaan Data)")
            st.info("""
            **Apa fungsi halaman ini?** Jika Anda memiliki banyak variabel (misal: 10-20 variabel) dan bingung membacanya, metode ini akan meringkasnya menjadi beberapa "Komponen" atau "Faktor" utama tanpa membuang banyak informasi penting.
            """)
            
            st.warning("**Catatan Teknis:** Data akan otomatis distandarisasi (skala disamakan) agar analisis akurat.")
            st.markdown("---")

            if len(numeric_cols) < 2:
                 st.error("Analisis ini memerlukan setidaknya 2 kolom numerik (angka).")
            else:
                try:
                    # 1. Standarisasi Data (Wajib untuk PCA/EFA)
                    scaler = StandardScaler()
                    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
                except Exception as e:
                    st.error(f"Gagal melakukan standarisasi data: {e}")
                    st.stop() 

                # =========================================================
                # BAGIAN 1: PCA (Principal Component Analysis)
                # =========================================================
                st.subheader("1. Principal Component Analysis (PCA)")
                st.markdown("""
                **Tujuan:** Meringkas data menjadi indeks atau komponen baru. 
                Contoh: Menggabungkan "Pendapatan", "Aset", dan "Pengeluaran" menjadi satu komponen bernama "Kekayaan".
                """)

                # --- STEP 1: PILIH VARIABEL ---
                st.markdown("##### Langkah 1: Pilih Variabel")
                pca_vars = st.multiselect("Pilih variabel yang ingin diringkas:", numeric_cols, default=numeric_cols, key='pca_vars')
                
                if len(pca_vars) < 2:
                    st.warning("Silakan pilih minimal 2 variabel.")
                else:
                    df_pca_ready = df_scaled[pca_vars]

                    # --- STEP 2: ANALISIS PENDUKUNG (SCREE PLOT) ---
                    st.markdown("##### Langkah 2: Analisis Pendukung (Menentukan Jumlah Komponen)")
                    with st.expander("Lihat Grafik Siku (Scree Plot) untuk bantuan", expanded=True):
                        # Hitung PCA Full dulu untuk melihat varians
                        pca_full = PCA()
                        pca_full.fit(df_pca_ready)
                        var_exp = pca_full.explained_variance_ratio_
                        cum_var = np.cumsum(var_exp)
                        
                        # Plotting Scree Plot
                        scree_df = pd.DataFrame({
                            'Komponen': range(1, len(var_exp)+1),
                            'Varians (%)': var_exp * 100,
                            'Kumulatif (%)': cum_var * 100
                        })
                        
                        col_scr1, col_scr2 = st.columns([2, 1])
                        with col_scr1:
                            fig_scree = go.Figure()
                            fig_scree.add_trace(go.Bar(x=scree_df['Komponen'], y=scree_df['Varians (%)'], name='Varians per Komponen', marker_color='#F2C94C'))
                            fig_scree.add_trace(go.Scatter(x=scree_df['Komponen'], y=scree_df['Kumulatif (%)'], name='Total Informasi (Kumulatif)', line=dict(color='#E07A3F', width=3)))
                            fig_scree.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target Informasi 70%")
                            fig_scree.update_layout(title="Scree Plot (Berapa komponen yang harus diambil?)", xaxis_title="Komponen Ke-", yaxis_title="Informasi (%)", height=400)
                            st.plotly_chart(fig_scree, use_container_width=True)
                        
                        with col_scr2:
                            st.info("""
                            **Tips Memilih:**
                            1. Cari titik di mana grafik batang mulai landai (seperti siku).
                            2. Atau lihat garis oranye: Berapa komponen yang dibutuhkan untuk mencapai **>70%** informasi?
                            """)
                            st.dataframe(scree_df.set_index('Komponen').style.format("{:.2f}%"), height=300)

                    # --- STEP 3: EKSEKUSI PCA ---
                    st.markdown("##### Langkah 3: Hasil Utama PCA")
                    n_components_pca = st.slider("Berdasarkan grafik di atas, pilih jumlah Komponen:", 1, len(pca_vars), 2, key='n_pca')
                    
                    if st.button("Proses PCA & Tampilkan Biplot", key='btn_pca'):
                        pca_final = PCA(n_components=n_components_pca)
                        components = pca_final.fit_transform(df_pca_ready)
                        loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)
                        
                        # --- INTERPRETASI LOADINGS (BOBOT) ---
                        st.write("### 1. Tabel Bobot (Loadings)")
                        st.caption("Angka yang besar (positif/negatif) menunjukkan variabel tersebut sangat berpengaruh dalam membentuk komponen.")
                        loadings_df = pd.DataFrame(
                            pca_final.components_.T, 
                            index=pca_vars, 
                            columns=[f'Komponen {i+1}' for i in range(n_components_pca)]
                        )
                        # Highlight nilai tinggi
                        st.dataframe(loadings_df.style.background_gradient(cmap="Oranges"), use_container_width=True)

                        # --- BIPLOT (VISUALISASI 2D) ---
                        if n_components_pca >= 2:
                            st.write("### 2. Biplot (Peta Variabel & Observasi)")
                            st.caption("Visualisasi ini menggabungkan sebaran data (titik) dan arah variabel (panah).")
                            
                            fig_bi = go.Figure()
                            
                            # A. Plot Titik Data (Scores)
                            fig_bi.add_trace(go.Scatter(
                                x=components[:,0], y=components[:,1],
                                mode='markers', name='Data Observasi',
                                marker=dict(color='#4F8190', opacity=0.5, size=8),
                                text=df.index
                            ))
                            
                            # B. Plot Panah (Loadings) - Di-scale agar terlihat
                            scale_factor = np.max(np.abs(components)) / np.max(np.abs(loadings[:, :2])) # Skala otomatis
                            
                            for i, feature in enumerate(pca_vars):
                                fig_bi.add_shape(
                                    type='line', x0=0, y0=0,
                                    x1=loadings[i, 0] * scale_factor,
                                    y1=loadings[i, 1] * scale_factor,
                                    line=dict(color='#E07A3F', width=2)
                                )
                                fig_bi.add_annotation(
                                    x=loadings[i, 0] * scale_factor,
                                    y=loadings[i, 1] * scale_factor,
                                    text=feature, showarrow=False,
                                    font=dict(color="#E07A3F", size=12, weight="bold")
                                )
                            
                            var_ratio = pca_final.explained_variance_ratio_
                            fig_bi.update_layout(
                                title=f"Biplot (Komponen 1: {var_ratio[0]:.1%} vs Komponen 2: {var_ratio[1]:.1%})",
                                xaxis_title=f"Komponen 1 ({var_ratio[0]:.1%})",
                                yaxis_title=f"Komponen 2 ({var_ratio[1]:.1%})",
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_bi, use_container_width=True)
                            
                            # Interpretasi Biplot untuk orang awam
                            st.info("""
                            **Cara Membaca Biplot:**
                            * **Sudut antar Panah:**
                                * Panah yang berdekatan (sudut sempit) = Variabel saling berhubungan kuat (positif).
                                * Panah berlawanan arah = Berhubungan terbalik (negatif).
                                * Panah tegak lurus (90 derajat) = Tidak berhubungan.
                            * **Arah Panah:** Menunjukkan variabel mana yang mendominasi komponen tersebut.
                            """)

                st.markdown("---")

                # =========================================================
                # BAGIAN 2: EFA (Exploratory Factor Analysis)
                # =========================================================
                st.subheader("2. Exploratory Factor Analysis (EFA)")
                st.markdown("""
                **Tujuan:** Menemukan "Konsep Tersembunyi" (Faktor Laten) yang menyebabkan variabel-variabel saling berkorelasi.
                Contoh: Nilai Matematika, Fisika, Kimia tinggi mungkin disebabkan oleh faktor tersembunyi yaitu "Kecerdasan Eksakta".
                """)

                # --- STEP 1: PILIH VARIABEL ---
                st.markdown("##### Langkah 1: Pilih Variabel")
                efa_vars = st.multiselect("Pilih variabel untuk EFA:", numeric_cols, default=numeric_cols, key='efa_vars')
                
                if len(efa_vars) < 3:
                    st.warning("EFA membutuhkan minimal 3 variabel agar hasilnya valid.")
                else:
                    df_efa_ready = df_scaled[efa_vars]

                    # --- STEP 2: UJI KELAYAKAN (ANALISIS PENDUKUNG) ---
                    st.markdown("##### Langkah 2: Uji Kelayakan Data (Analisis Pendukung)")
                    
                    if st.button("Cek Apakah Data Layak di-EFA?", key='btn_kmo'):
                        # 1. Bartlett's Test
                        chi_square_value, p_value_bartlett = calculate_bartlett_sphericity(df_efa_ready)
                        # 2. KMO Test
                        kmo_all, kmo_model = calculate_kmo(df_efa_ready)
                        
                        c_kmo1, c_kmo2 = st.columns(2)
                        with c_kmo1:
                            st.metric("Nilai KMO (Kaiser-Meyer-Olkin)", f"{kmo_model:.3f}")
                            if kmo_model > 0.6:
                                st.success("**Data Cukup Baik** (KMO > 0.6). Sampel mencukupi.")
                            else:
                                st.error("**Data Kurang** (KMO < 0.6). Tambah data atau kurangi variabel.")
                        
                        with c_kmo2:
                            st.metric("P-value Bartlett", f"{p_value_bartlett:.4f}")
                            if p_value_bartlett < 0.05:
                                st.success("**Variabel Saling Berhubungan** (P < 0.05). EFA bisa dilanjutkan.")
                            else:
                                st.error("**Variabel Tidak Berhubungan** (P > 0.05). EFA tidak berguna.")

                        # 3. Scree Plot Eigenvalues (Kaiser Criterion)
                        st.write("**Berapa Faktor yang harus dibentuk? (Kaiser Criterion: Eigenvalue > 1)**")
                        fa_temp = FactorAnalyzer(rotation=None)
                        fa_temp.fit(df_efa_ready)
                        ev, v = fa_temp.get_eigenvalues()
                        
                        fig_ev = px.line(x=range(1, len(ev)+1), y=ev, markers=True, title="Scree Plot Eigenvalue")
                        fig_ev.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Batas Eigenvalue = 1")
                        fig_ev.update_layout(xaxis_title="Faktor Ke-", yaxis_title="Eigenvalue")
                        st.plotly_chart(fig_ev, use_container_width=True)
                        
                        suggested_factors = sum(ev > 1)
                        st.info(f"**Rekomendasi:** Berdasarkan grafik di atas (nilai > 1), disarankan membuat **{suggested_factors} Faktor**.")

                    # --- STEP 3: EKSEKUSI EFA ---
                    st.markdown("##### Langkah 3: Hasil Utama EFA")
                    
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        n_factors_efa = st.slider("Jumlah Faktor:", 1, len(efa_vars)-1, 2, key='n_efa')
                    with col_e2:
                        rot_method = st.selectbox("Metode Rotasi:", ["varimax", "promax", "quartimax", "oblimin"], index=0, key='rot_efa', help="Varimax menghasilkan faktor yang saling tegak lurus (beda tegas). Promax membolehkan korelasi antar faktor.")

                    if st.button("Bentuk Faktor", key='btn_efa_run'):
                        try:
                            fa = FactorAnalyzer(n_factors=n_factors_efa, rotation=rot_method)
                            fa.fit(df_efa_ready)
                            
                            # Loadings
                            loadings_efa = pd.DataFrame(
                                fa.loadings_, 
                                index=efa_vars, 
                                columns=[f'Faktor {i+1}' for i in range(n_factors_efa)]
                            )
                            
                            st.write("### Matriks Faktor (Pattern Matrix)")
                            st.write("Angka ini menunjukkan seberapa kuat hubungan variabel dengan faktor yang terbentuk.")
                            
                            # Tampilkan dengan heatmap style
                            st.dataframe(loadings_efa.style.background_gradient(cmap="Greens"), use_container_width=True)
                            
                            # Varians dijelaskan
                            st.write("### Total Varians Dijelaskan")
                            var_efa = pd.DataFrame(fa.get_factor_variance(), index=["SS Loadings", "Proportion Var", "Cumulative Var"], columns=[f'Faktor {i+1}' for i in range(n_factors_efa)])
                            st.dataframe(var_efa)
                            
                            # Interpretasi Otomatis Sederhana
                            st.write("### Interpretasi Faktor")
                            for i in range(n_factors_efa):
                                col_name = f'Faktor {i+1}'
                                # Ambil variabel yang loadingnya > 0.4 (cutoff umum)
                                dominant_vars = loadings_efa[abs(loadings_efa[col_name]) > 0.4].index.tolist()
                                st.success(f"**{col_name}** paling mewakili variabel: **{', '.join(dominant_vars)}**")

                        except Exception as e:
                            st.error(f"Terjadi error: {e}. Coba kurangi jumlah faktor atau ganti metode rotasi.")

# -------------------------------------------------------------
        # TAB 7: KLASIFIKASI & CLUSTERING (BAB 11 & 12)
        # -------------------------------------------------------------
        with tab_class:
            st.header("Klasifikasi & Clustering")
            st.info("Modul ini digunakan untuk mengelompokkan data (Clustering) atau memprediksi kategori (Klasifikasi).")
            
            method_cls = st.radio("Pilih Metode Analisis:", 
                ["Clustering (K-Means) - Pengelompokan Otomatis", "Diskriminan (LDA) - Prediksi Kategori"], 
                horizontal=True)
            
            st.markdown("---")

            # =========================================================
            # SKENARIO 1: CLUSTERING (K-MEANS)
            # =========================================================
            if method_cls == "Clustering (K-Means) - Pengelompokan Otomatis":
                st.subheader("K-Means Clustering")
                st.markdown("""
                **Tujuan:** Mengelompokkan data yang memiliki karakteristik mirip ke dalam satu grup (Cluster).
                Contoh: Mengelompokkan daerah berdasarkan tingkat kemiskinan dan pengangguran.
                """)

                # 1. Pilih Variabel
                cl_vars = st.multiselect("Pilih variabel pembentuk cluster (Numerik):", numeric_cols, key='km_vars')
                
                if len(cl_vars) >= 2:
                    # Pre-processing: Standarisasi Data (Wajib untuk K-Means)
                    X_cl = df[cl_vars].dropna()
                    scaler = StandardScaler()
                    X_std = scaler.fit_transform(X_cl)
                    
                    # --- ANALISIS PENDUKUNG: ELBOW METHOD ---
                    st.markdown("#### 1. Analisis Pendukung: Metode Elbow")
                    with st.expander("Bantuan: Cara menentukan jumlah cluster yang tepat", expanded=True):
                        st.write("Grafik ini membantu Anda memilih jumlah cluster. Cari titik di mana garis mulai melandai (seperti siku tangan).")
                        
                        inertias = []
                        K_range = range(1, 11) # Coba 1 sampai 10 cluster
                        for k in K_range:
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)
                            km.fit(X_std)
                            inertias.append(km.inertia_)
                        
                        fig_elb = px.line(x=list(K_range), y=inertias, markers=True, title="Metode Elbow (Cari Titik Siku)")
                        fig_elb.update_layout(xaxis_title="Jumlah Cluster (k)", yaxis_title="Inersia (Error)", plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_elb, use_container_width=True)

                    # --- PROSES UTAMA ---
                    st.markdown("#### 2. Proses Clustering")
                    k_num = st.slider("Berdasarkan grafik di atas, pilih jumlah cluster:", 2, 10, 3, key='n_clust_slider')
                    
                    if st.button("Bentuk Cluster", key='btn_km_run'):
                        # Eksekusi K-Means
                        kmeans = KMeans(n_clusters=k_num, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(X_std)
                        
                        # Gabungkan hasil
                        df_res = X_cl.copy()
                        df_res['Cluster'] = labels
                        df_res['Cluster'] = df_res['Cluster'].astype(str) # Ubah ke string agar jadi kategori warna
                        
                        # Visualisasi Scatter Plot
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.write("**Visualisasi Sebaran Cluster**")
                            # Plot menggunakan 2 variabel pertama yang dipilih pengguna
                            fig_cl = px.scatter(
                                df_res, x=cl_vars[0], y=cl_vars[1], 
                                color='Cluster',
                                title=f"Peta Cluster ({cl_vars[0]} vs {cl_vars[1]})",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            fig_cl.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_cl, use_container_width=True)
                        
                        with c2:
                            st.write("**Profil Rata-rata per Cluster**")
                            st.caption("Tabel ini menunjukkan karakteristik setiap kelompok.")
                            # Hitung rata-rata tiap cluster
                            summary_cl = df_res.groupby('Cluster')[cl_vars].mean()
                            # Tampilkan dataframe (tanpa background_gradient untuk menghindari error matplotlib)
                            st.dataframe(summary_cl)
                            
                        # Interpretasi Otomatis
                        st.write("### Interpretasi Profil")
                        for cluster_id in sorted(summary_cl.index):
                            # Cari variabel yang nilainya paling tinggi di cluster ini
                            max_col = summary_cl.loc[cluster_id].idxmax()
                            max_val = summary_cl.loc[cluster_id].max()
                            st.info(f"**Cluster {cluster_id}:** Cenderung memiliki nilai **{max_col}** yang tinggi (Rata-rata: {max_val:.2f}).")

                else:
                    st.warning("Silakan pilih minimal 2 variabel numerik.")

            # =========================================================
            # SKENARIO 2: DISKRIMINAN (LDA)
            # =========================================================
            elif method_cls == "Diskriminan (LDA) - Prediksi Kategori":
                st.subheader("Linear Discriminant Analysis (LDA)")
                st.markdown("""
                **Tujuan:** Mempelajari pola dari data yang sudah ada kategorinya (Supervised), untuk melihat seberapa akurat variabel pembeda memisahkan kelompok.
                """)

                # 1. Pilih Variabel
                c_lda1, c_lda2 = st.columns(2)
                with c_lda1:
                    lda_target = st.selectbox("Variabel Target (Kategori/Grouping):", categorical_cols, key='lda_t')
                with c_lda2:
                    lda_preds = st.multiselect("Variabel Pembeda (Numerik):", numeric_cols, key='lda_p')
                
                if st.button("Jalankan Analisis Diskriminan", key='btn_lda_run'):
                    if lda_target and len(lda_preds) > 0:
                        try:
                            # Persiapan Data
                            df_lda = df[[lda_target] + lda_preds].dropna()
                            X_lda = df_lda[lda_preds]
                            y_lda = df_lda[lda_target]
                            
                            # Cek jumlah kategori
                            n_classes = len(y_lda.unique())
                            if n_classes < 2:
                                st.error("Variabel target harus memiliki minimal 2 kategori.")
                                st.stop()

                            # Eksekusi LDA
                            lda = LinearDiscriminantAnalysis()
                            lda.fit(X_lda, y_lda)
                            y_pred = lda.predict(X_lda)
                            
                            # Hitung Akurasi (Resubstitution)
                            acc = np.mean(y_pred == y_lda)
                            
                            # Tampilkan Hasil Utama
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.metric("Akurasi Model", f"{acc*100:.2f}%")
                            
                            with col_res2:
                                if acc > 0.8:
                                    st.success("**Sangat Baik.** Variabel pembeda mampu memisahkan kelompok dengan tegas.")
                                elif acc > 0.6:
                                    st.warning("**Cukup.** Ada beberapa data yang tumpang tindih antar kelompok.")
                                else:
                                    st.error("**Kurang Baik.** Variabel yang dipilih tidak cukup kuat untuk membedakan kelompok.")

                            # Koefisien Diskriminan
                            st.write("### Koefisien Diskriminan (Bobot Pembeda)")
                            st.caption("Semakin besar angka (positif/negatif), semakin penting variabel tersebut dalam membedakan kelompok.")
                            
                            # Buat DataFrame Koefisien
                            n_comps = min(n_classes - 1, len(lda_preds))
                            coef_df = pd.DataFrame(
                                lda.scalings_[:, :n_comps], 
                                index=lda_preds, 
                                columns=[f'Fungsi LD{i+1}' for i in range(n_comps)]
                            )
                            st.dataframe(coef_df)
                            
                            # Plot Proyeksi LDA
                            if n_comps >= 1:
                                st.write("### Visualisasi Sebaran Kategori")
                                X_lda_proj = lda.transform(X_lda)
                                df_proj = pd.DataFrame(X_lda_proj, columns=[f'LD{i+1}' for i in range(n_comps)])
                                df_proj['Kategori'] = y_lda.values
                                
                                if n_comps == 1:
                                    # Histogram 1D
                                    fig_lda = px.histogram(df_proj, x='LD1', color='Kategori', barmode='overlay', title="Distribusi Pemisah (1 Dimensi)")
                                else:
                                    # Scatter 2D
                                    fig_lda = px.scatter(df_proj, x='LD1', y='LD2', color='Kategori', title="Peta Sebaran LDA (2 Dimensi)")
                                
                                st.plotly_chart(fig_lda, use_container_width=True)

                        except Exception as e:
                            st.error(f"Terjadi error: {e}. Pastikan variabel target tidak memiliki nilai tunggal.")
                    else:
                        st.warning("Mohon lengkapi pilihan variabel Target dan Prediktor.")