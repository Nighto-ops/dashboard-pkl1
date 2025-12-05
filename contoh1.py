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

# === JIKA BELUM UPLOAD FILE (FITUR PETA) ===
if uploaded_file is None:
    # Judul Halaman Utama (Menggunakan Font RAKKAS dari CSS H1)
    st.markdown("<h1>DASHBOARD SEBARAN LOKASI POTENSIAL PEKERJA GIG PKL 65<br>D.I. YOGYAKARTA</h1>", unsafe_allow_html=True)
    
    # Pesan Selamat Datang dengan Styling Baru
    st.markdown("""
    <div class="welcome-box">
        <h3 style='margin-top:0;'>Selamat Datang di Dashboard PKL 65</h3>
        <p style='font-size: 1.1rem;'>Halaman ini menyajikan peta interaktif persebaran lokasi pekerja Gig di D.I. Yogyakarta. Visualisasi ini dirancang untuk memberikan gambaran spasial terkait data lapangan.</p>
        <p style='font-size: 1rem; margin-top: 15px;'>Untuk memulai <b>Analisis Statistik Mendalam</b> (seperti Uji Regresi, ANOVA, MANOVA, dan Clustering), silakan unggah dataset Anda melalui panel di sisi kiri.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df_map = load_map_excel_data()
    gdf_shape = load_shp_data()

    if not df_map.empty and gdf_shape is not None:
        
        # FITUR CEK ISI SHP
        with st.expander("FITUR BANTUAN: CEK EJAAN NAMA WILAYAH DI FILE SHP"):
            st.info("Gunakan fitur ini untuk memastikan kesesuaian nama wilayah antara data Excel dan file SHP.")
            if SHP_COL_KAB in gdf_shape.columns:
                unique_kab = gdf_shape[SHP_COL_KAB].unique()
                pilih_kab_cek = st.selectbox("Pilih Kabupaten (dari SHP):", unique_kab)
                isi_kec = gdf_shape[gdf_shape[SHP_COL_KAB] == pilih_kab_cek][SHP_COL_KEC].unique()
                st.write(f"Daftar Kecamatan yang tersedia di SHP untuk '{pilih_kab_cek}':")
                st.write(isi_kec)

        st.markdown("---")

        # FILTER PETA
        st.subheader("Filter Visualisasi Peta")
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        with c_filter1:
            list_kab = sorted(df_map['kabupaten'].unique().tolist())
            selected_kab = st.multiselect("1. Pilih Kabupaten:", list_kab, default=list_kab)
        with c_filter2:
            if selected_kab:
                df_filtered_kab = df_map[df_map['kabupaten'].isin(selected_kab)]
                list_kec = sorted(df_filtered_kab['kecamatan'].unique().tolist())
            else:
                list_kec = []
            selected_kec = st.multiselect("2. Pilih Kecamatan:", list_kec, default=list_kec)
        with c_filter3:
            map_mode = st.radio("3. Mode Tampilan Peta:", ["Gabungan", "Choropleth (Wilayah)", "Heatmap (Titik)"], horizontal=True)

# RENDER PETA
        if selected_kab and selected_kec:
            final_df = df_map[
                (df_map['kabupaten'].isin(selected_kab)) & 
                (df_map['kecamatan'].isin(selected_kec))
            ]
            
            # 1. SIAPKAN SHP & BUAT KEY
            gdf_shape['kab_key'] = gdf_shape[SHP_COL_KAB].astype(str).str.upper().str.replace(" ", "").str.replace(r'^(KAB\.?|KABUPATEN|KOTA)\s+', '', regex=True)
            gdf_shape['kec_key'] = gdf_shape[SHP_COL_KEC].astype(str).str.upper().str.replace(" ", "")
            
            # Fix typo SHP
            gdf_shape['kab_key'] = gdf_shape['kab_key'].replace({'GUNUNGKIDUL': 'GUNUNGKIDUL', 'YOGYA': 'YOGYAKARTA'})

            # --- [SOLUSI JETIS] BUAT UNIQUE KEY DI SHP JUGA ---
            gdf_shape['unique_key'] = gdf_shape['kab_key'] + "_" + gdf_shape['kec_key']
            # --------------------------------------------------

            # Konversi filter user ke key
            selected_kab_keys = [k.upper().replace(" ", "") for k in selected_kab]
            selected_kec_keys = [k.upper().replace(" ", "") for k in selected_kec]

            # Filter SHP
            final_gdf = gdf_shape[
                (gdf_shape['kab_key'].isin(selected_kab_keys)) &
                (gdf_shape['kec_key'].isin(selected_kec_keys))
            ]

            if not final_gdf.empty:
                # Hitung statistik berdasarkan UNIQUE KEY (Bukan nama kecamatan)
                stats_kec = final_df.groupby('unique_key').size().reset_index(name='jumlah_lokasi')
                
                # Merge menggunakan UNIQUE KEY
                gdf_viz = final_gdf.merge(stats_kec, on='unique_key', how='left')
                gdf_viz['jumlah_lokasi'] = gdf_viz['jumlah_lokasi'].fillna(0)
                
                # Siapkan label tooltip
                gdf_viz['Nama Kecamatan'] = gdf_viz[SHP_COL_KEC].astype(str).str.title()
                gdf_viz['Nama Kabupaten'] = gdf_viz[SHP_COL_KAB].astype(str).str.title()

                bounds = final_gdf.total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

                if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                    cp = folium.Choropleth(
                        geo_data=gdf_viz,
                        name='Kepadatan Wilayah',
                        data=gdf_viz,
                        columns=['unique_key', 'jumlah_lokasi'], # JOIN PAKAI UNIQUE KEY
                        key_on='feature.properties.unique_key',  # JOIN PAKAI UNIQUE KEY
                        fill_color='YlOrRd', 
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='Jumlah Lokasi',
                        highlight=True
                    ).add_to(m)
                    
                    # Tooltip yang lebih informatif (Ada nama Kab dan Kec)
                    folium.GeoJsonTooltip(
                        fields=['Nama Kabupaten', 'Nama Kecamatan', 'jumlah_lokasi'], 
                        aliases=['Kabupaten:', 'Kecamatan:', 'Jumlah Data:'], 
                        localize=True
                    ).add_to(cp.geojson)

                if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                    heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: '#FFD700', 0.65: '#FF8C00', 1: '#8B0000'}).add_to(m)

                folium.LayerControl().add_to(m)
                
                st.markdown('<div style="box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-radius: 12px; overflow: hidden; border: 3px solid #E07A3F;">', unsafe_allow_html=True)
                st_folium(m, width=1200, height=650)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Lihat Tabel Data Detail"): 
                    st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("Wilayah tidak ditemukan di Peta.")
        else:
            st.info("Silakan pilih Kabupaten dan Kecamatan pada panel filter di atas.")
    else:
        st.warning("Data peta (file Excel atau SHP) belum tersedia di folder 'data/'. Pastikan file .xlsx dan .shp telah diunggah dengan benar.")

# === JIKA FILE SUDAH DIUPLOAD (KODE ASLI STATISTIK) ===
else:
    # --- INISIALISASI ---
    df = None
    numeric_cols = []
    categorical_cols = []
    all_cols = []

    # --- FUNGSI BANTUAN INTERPRETASI ---
    def interpret_pvalue(p_val):
        if p_val < 0.05:
            return "Signifikan (Nyata)", "success"
        else:
            return "Tidak Signifikan (Tidak Nyata)", "warning"

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File berhasil dimuat!")
            
            # Pre-processing Nama Kolom (agar aman untuk formula statistik)
            df.columns = [c.strip().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]

            # Identifikasi tipe variabel
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📋 Info Variabel")
            st.sidebar.info(f"**{len(numeric_cols)}** Variabel Angka (Numerik)")
            st.sidebar.caption(", ".join(numeric_cols) if numeric_cols else "-")
            st.sidebar.info(f"**{len(categorical_cols)}** Variabel Kategori (Teks)")
            st.sidebar.caption(", ".join(categorical_cols) if categorical_cols else "-")

    if df is not None and all_cols:
        # Header Halaman Statistik
        st.markdown("<h1>ANALISIS STATISTIK LENGKAP</h1>", unsafe_allow_html=True)
        st.markdown("Dashboard ini mencakup analisis dasar hingga analisis multivariat kompleks sesuai modul praktikum.")
        
        # --- MENU TABS GABUNGAN (LAMA + BARU) ---
        tabs = st.tabs([
            "Data", 
            "Dasar & Korelasi",      # Fitur Lama (Basic)
            "Regresi & ANOVA",       # Fitur Lama (Regresi + ANOVA)
            "Asumsi Multivariat",    # Fitur Baru (Modul 2)
            "Uji Beda Vektor",       # Fitur Baru (Modul 3-4 Hotelling)
            "MANOVA",                # Fitur Baru (Modul 5-7)
            "PCA & Biplot",          # Fitur Baru (Modul 8-10)
            "Kanonik",               # Fitur Baru (Modul 12)
            "Klasifikasi"            # Fitur Baru (Modul 11 & 13)
        ])

        # -------------------------------------------------------------
        # 1. TAB DATA
        # -------------------------------------------------------------
        with tabs[0]:
            st.header("Eksplorasi Data Awal")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Statistik Deskriptif")
                st.write(df.describe())
            with c2:
                st.subheader("Cuplikan Data")
                st.dataframe(df.head(50), use_container_width=True)

        # -------------------------------------------------------------
        # 2. TAB DASAR & KORELASI (FITUR LAMA ANDA)
        # -------------------------------------------------------------
        with tabs[1]:
            st.header("Analisis Deskriptif & Bivariat")
            
            # A. Histogram
            st.subheader("1. Distribusi Data (Histogram)")
            hist_col = st.selectbox("Pilih Variabel:", numeric_cols, key='hist_basic')
            if hist_col:
                fig = px.histogram(df, x=hist_col, title=f"Distribusi {hist_col}", color_discrete_sequence=['#E07A3F'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # B. Korelasi & Scatter
            st.subheader("2. Hubungan Antar Variabel (Scatter Plot)")
            c1, c2 = st.columns([1, 2])
            with c1:
                sc_x = st.selectbox("Sumbu X:", numeric_cols, key='sc_x')
                sc_y = st.selectbox("Sumbu Y:", numeric_cols, key='sc_y')
            with c2:
                if sc_x and sc_y:
                    fig_sc = px.scatter(df, x=sc_x, y=sc_y, trendline='ols', title=f"{sc_y} vs {sc_x}", color_discrete_sequence=['#F2C94C'])
                    st.plotly_chart(fig_sc, use_container_width=True)
                    # Hitung korelasi
                    corr_val, p_val = stats.pearsonr(df[sc_x].dropna(), df[sc_y].dropna())
                    st.info(f"**Korelasi Pearson (r):** {corr_val:.4f} | **P-Value:** {p_val:.4f}")

        # -------------------------------------------------------------
        # 3. TAB REGRESI & ANOVA (FITUR LAMA ANDA)
        # -------------------------------------------------------------
        with tabs[2]:
            st.header("Model Regresi & ANOVA")
            
            method = st.radio("Pilih Metode:", ["Regresi Linear Berganda", "ANOVA One-Way"], horizontal=True)
            
            if method == "Regresi Linear Berganda":
                st.subheader("Regresi Linear (OLS)")
                ry = st.selectbox("Variabel Dependen (Y):", numeric_cols, key='reg_y')
                rx = st.multiselect("Variabel Independen (X):", [c for c in numeric_cols if c != ry], key='reg_x')
                
                if st.button("Proses Regresi", key='btn_reg'):
                    if ry and rx:
                        X = sm.add_constant(df[rx])
                        y = df[ry]
                        model = sm.OLS(y, X, missing='drop').fit()
                        
                        st.write(model.summary())
                        
                        # Interpretasi Singkat
                        st.success(f"**R-Squared:** {model.rsquared:.4f} (Model menjelaskan {model.rsquared*100:.1f}% variasi data)")
                    else:
                        st.warning("Pilih minimal 1 variabel X.")
            
            elif method == "ANOVA One-Way":
                st.subheader("Analysis of Variance (ANOVA)")
                av_cat = st.selectbox("Kelompok (Faktor):", categorical_cols, key='av_c')
                av_num = st.selectbox("Variabel Nilai (Y):", numeric_cols, key='av_n')
                
                if st.button("Proses ANOVA", key='btn_anova'):
                    if av_cat and av_num:
                        formula = f"{av_num} ~ C({av_cat})"
                        model = smf.ols(formula, data=df).fit()
                        aov_table = sm.stats.anova_lm(model, typ=2)
                        st.write(aov_table)
                        
                        pval = aov_table['PR(>F)'][0]
                        st.info(f"**Hasil:** Perbedaan rata-rata antar kelompok adalah **{interpret_pvalue(pval)[0]}**.")

        # -------------------------------------------------------------
        # 4. TAB ASUMSI MULTIVARIAT (MODUL 2 - BARU)
        # -------------------------------------------------------------
        with tabs[3]:
            st.header("Modul 2: Uji Normalitas Multivariat")
            st.info("Pengecekan apakah data menyebar normal secara multivariat menggunakan Chi-Square Plot.")
            
            mvn_vars = st.multiselect("Pilih Variabel Multivariat (Min 2):", numeric_cols, default=numeric_cols[:3] if len(numeric_cols)>=3 else numeric_cols, key='mvn_vars')
            
            if len(mvn_vars) >= 2:
                if st.button("Cek Normalitas (Chi-Square Plot)", key='btn_mvn'):
                    try:
                        data_m = df[mvn_vars].dropna()
                        # Mahalanobis Distance
                        x_minus_mu = data_m - np.mean(data_m, axis=0)
                        cov = np.cov(data_m.values.T)
                        inv_covmat = np.linalg.inv(cov)
                        left_term = np.dot(x_minus_mu, inv_covmat)
                        mahal = np.dot(left_term, x_minus_mu.T).diagonal()
                        
                        # QQ Plot
                        n = len(data_m)
                        p = len(mvn_vars)
                        prob = (np.arange(1, n+1) - 0.5) / n
                        chi2_q = stats.chi2.ppf(prob, df=p)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=chi2_q, y=np.sort(mahal), mode='markers', name='Data'))
                        max_val = max(max(chi2_q), max(mahal))
                        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Referensi Normal', line=dict(color='red', dash='dash')))
                        
                        fig.update_layout(title="Chi-Square Q-Q Plot", xaxis_title="Kuantil Teoretis", yaxis_title="Jarak Mahalanobis", plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.caption("**Interpretasi:** Jika titik-titik data mengikuti garis merah putus-putus, data diasumsikan berdistribusi Normal Multivariat.")
                    except Exception as e:
                        st.error(f"Gagal menghitung (Cek Multikolinearitas): {e}")

        # -------------------------------------------------------------
        # 5. TAB UJI BEDA VEKTOR (MODUL 3-4 - BARU)
        # -------------------------------------------------------------
        with tabs[4]:
            st.header("Modul 3 & 4: Uji Hotelling's T²")
            st.info("Menguji perbedaan rata-rata multivariat antara DUA kelompok independen.")
            
            hot_cat = st.selectbox("Variabel Kelompok (Harus 2 Kategori):", categorical_cols, key='hot_cat')
            hot_vars = st.multiselect("Variabel Dependen (Numerik):", numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>=2 else numeric_cols, key='hot_vars')
            
            if st.button("Jalankan Uji Hotelling", key='btn_hot'):
                if hot_cat and len(hot_vars) >= 1:
                    if df[hot_cat].nunique() == 2:
                        try:
                            # Menggunakan Statsmodels MANOVA (Equivalent Hotelling T2 untuk 2 grup)
                            formula = f'{" + ".join(hot_vars)} ~ C({hot_cat})'
                            model = MANOVA.from_formula(formula, data=df)
                            res = model.mv_test()
                            summ = res.summary_frame
                            
                            st.write(summ)
                            
                            pval = summ.loc[(f'C({hot_cat})', "Wilks' lambda"), "Pr > F"]
                            res_text, color = interpret_pvalue(pval)
                            
                            st.markdown(f"### Kesimpulan: Perbedaan antar kelompok adalah :{color}[**{res_text}**] (P={pval:.4f})")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.error("Variabel kelompok harus memiliki tepat 2 kategori untuk Hotelling T².")

        # -------------------------------------------------------------
        # 6. TAB MANOVA (MODUL 5-7 - BARU)
        # -------------------------------------------------------------
        with tabs[5]:
            st.header("Modul 5, 6, 7: MANOVA & MANCOVA")
            
            type_m = st.radio("Jenis:", ["MANOVA One-Way", "MANCOVA (dengan Kovariat)"], horizontal=True)
            
            if type_m == "MANOVA One-Way":
                mc = st.selectbox("Faktor (Kategori):", categorical_cols, key='m1_c')
                mv = st.multiselect("Variabel Respon (Y):", numeric_cols, key='m1_v')
                
                if st.button("Proses MANOVA", key='btn_m1'):
                    if mc and len(mv)>=2:
                        model = MANOVA.from_formula(f'{" + ".join(mv)} ~ C({mc})', data=df)
                        res = model.mv_test()
                        st.write(res.summary())
                        
                        pval = res.summary_frame.loc[(f'C({mc})', "Wilks' lambda"), "Pr > F"]
                        st.success(f"**Kesimpulan:** Pengaruh faktor '{mc}' adalah **{interpret_pvalue(pval)[0]}**.")
            
            else: # MANCOVA
                mc_f = st.selectbox("Faktor (Kategori):", categorical_cols, key='mc_f')
                mc_cov = st.selectbox("Kovariat (Pengontrol):", numeric_cols, key='mc_cov')
                mc_y = st.multiselect("Variabel Respon (Y):", [c for c in numeric_cols if c!=mc_cov], key='mc_y')
                
                if st.button("Proses MANCOVA", key='btn_mc'):
                    if mc_f and mc_cov and len(mc_y)>=2:
                        model = MANOVA.from_formula(f'{" + ".join(mc_y)} ~ C({mc_f}) + {mc_cov}', data=df)
                        res = model.mv_test()
                        st.write(res.summary())
                        
                        pval = res.summary_frame.loc[(f'C({mc_f})', "Wilks' lambda"), "Pr > F"]
                        st.success(f"**Kesimpulan:** Setelah dikontrol oleh {mc_cov}, pengaruh faktor '{mc_f}' adalah **{interpret_pvalue(pval)[0]}**.")

        # -------------------------------------------------------------
        # 7. TAB PCA & BIPLOT (MODUL 8 & 10 - BARU)
        # -------------------------------------------------------------
        with tabs[6]:
            st.header("Modul 8 & 10: PCA & Biplot")
            
            pc_vars = st.multiselect("Variabel untuk PCA/Biplot:", numeric_cols, default=numeric_cols, key='pc_v')
            
            if st.button("Generate Biplot", key='btn_bio'):
                if len(pc_vars)>=2:
                    # PCA Calculation
                    X = df[pc_vars].dropna()
                    X_std = StandardScaler().fit_transform(X)
                    pca = PCA(n_components=2)
                    comps = pca.fit_transform(X_std)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    
                    # Plot
                    fig = go.Figure()
                    # Points
                    fig.add_trace(go.Scatter(x=comps[:,0], y=comps[:,1], mode='markers', name='Observasi', marker=dict(color='#E07A3F', opacity=0.6)))
                    
                    # Arrows
                    scale = 1.0
                    if np.max(comps) > 0: scale = np.max(np.abs(comps)) / np.max(np.abs(loadings))
                    
                    for i, feature in enumerate(pc_vars):
                        fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i,0]*scale, y1=loadings[i,1]*scale, line=dict(color='#4F8190', width=2))
                        fig.add_annotation(x=loadings[i,0]*scale, y=loadings[i,1]*scale, text=feature, showarrow=False, font=dict(color='#4F8190'))
                    
                    var_exp = pca.explained_variance_ratio_
                    fig.update_layout(title=f"Biplot (Total Varians: {sum(var_exp)*100:.1f}%)", xaxis_title="PC1", yaxis_title="PC2", plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("**Cara Baca:** Panah yang berdekatan berarti variabel berkorelasi positif kuat. Panah berlawanan arah berkorelasi negatif.")

        # -------------------------------------------------------------
        # 8. TAB KORELASI KANONIK (MODUL 12 - BARU)
        # -------------------------------------------------------------
        with tabs[7]:
            st.header("Modul 12: Korelasi Kanonik")
            st.info("Mengukur hubungan antara dua himpunan variabel (Set X vs Set Y).")
            
            c1, c2 = st.columns(2)
            with c1:
                set_x = st.multiselect("Set Variabel X:", numeric_cols, key='cc_x')
            with c2:
                avail_y = [c for c in numeric_cols if c not in set_x]
                set_y = st.multiselect("Set Variabel Y:", avail_y, key='cc_y')
                
            if st.button("Hitung Kanonik", key='btn_cc'):
                if len(set_x)>=2 and len(set_y)>=2:
                    from sklearn.cross_decomposition import CCA
                    X_c = df[set_x].dropna(); Y_c = df[set_y].dropna()
                    common = X_c.index.intersection(Y_c.index)
                    X_c, Y_c = X_c.loc[common], Y_c.loc[common]
                    
                    n_comp = min(len(set_x), len(set_y))
                    cca = CCA(n_components=n_comp)
                    cca.fit(X_c, Y_c)
                    X_trans, Y_trans = cca.transform(X_c, Y_c)
                    
                    corrs = [np.corrcoef(X_trans[:, i], Y_trans[:, i])[0, 1] for i in range(n_comp)]
                    
                    st.metric("Korelasi Kanonik Tertinggi", f"{max(corrs):.4f}")
                    st.write("Korelasi per Variat:", corrs)

        # -------------------------------------------------------------
        # 9. TAB KLASIFIKASI & CLUSTER (MODUL 11 & 13 - BARU)
        # -------------------------------------------------------------
        with tabs[8]:
            st.header("Modul 11 & 13: Cluster & Diskriminan")
            
            method_c = st.radio("Pilih:", ["Clustering (K-Means)", "Diskriminan (LDA)"], horizontal=True)
            
            if method_c == "Clustering (K-Means)":
                st.subheader("K-Means Clustering")
                cl_vars = st.multiselect("Variabel Cluster:", numeric_cols, key='cl_v')
                
                if len(cl_vars) >= 2:
                    # Analisis Pendukung: Elbow Method
                    st.markdown("#### 1️⃣ Analisis Pendukung: Metode Elbow (Menentukan Jumlah Cluster)")
                    st.write("Grafik ini membantu menentukan jumlah cluster optimal (pilih titik siku).")
                    
                    X_cl = df[cl_vars].dropna()
                    X_std = StandardScaler().fit_transform(X_cl)
                    
                    inertias = []
                    for k in range(1, 11):
                        km = KMeans(n_clusters=k, random_state=42).fit(X_std)
                        inertias.append(km.inertia_)
                    
                    fig_elb = px.line(x=list(range(1, 11)), y=inertias, markers=True, title="Elbow Method")
                    st.plotly_chart(fig_elb, use_container_width=True)
                    
                    st.markdown("#### 2️⃣ Proses Clustering")
                    k_final = st.slider("Pilih Jumlah Cluster (k):", 2, 10, 3)
                    
                    if st.button("Bentuk Cluster", key='btn_cl'):
                        km_final = KMeans(n_clusters=k_final, random_state=42).fit(X_std)
                        df_res = X_cl.copy()
                        df_res['Cluster'] = km_final.labels_.astype(str)
                        
                        fig_res = px.scatter(df_res, x=cl_vars[0], y=cl_vars[1], color='Cluster', title=f"Hasil Cluster ({cl_vars[0]} vs {cl_vars[1]})", color_discrete_sequence=px.colors.qualitative.Bold)
                        st.plotly_chart(fig_res, use_container_width=True)
                        
                        st.write("**Profil Rata-rata per Cluster:**")
                        st.dataframe(df_res.groupby('Cluster').mean().style.highlight_max(axis=0, color='lightgreen'))

            else: # LDA
                st.subheader("Linear Discriminant Analysis (LDA)")
                ld_t = st.selectbox("Target (Kategori):", categorical_cols, key='ld_t')
                ld_p = st.multiselect("Prediktor (Numerik):", numeric_cols, key='ld_p')
                
                if st.button("Proses LDA", key='btn_ld'):
                    if ld_t and ld_p:
                        df_l = df[[ld_t]+ld_p].dropna()
                        lda = LinearDiscriminantAnalysis()
                        lda.fit(df_l[ld_p], df_l[ld_t])
                        pred = lda.predict(df_l[ld_p])
                        acc = np.mean(pred == df_l[ld_t])
                        
                        st.metric("Akurasi Klasifikasi", f"{acc*100:.2f}%")
                        st.write("**Koefisien Diskriminan:**")
                        st.dataframe(pd.DataFrame(lda.scalings_, index=ld_p, columns=[f'LD{i+1}' for i in range(lda.scalings_.shape[1])]))