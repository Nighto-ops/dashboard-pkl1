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
# KONFIGURASI HALAMAN & CSS GRAND DESIGN PKL 65 (FIXED)
# =================================================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Statistik PKL 65")

# --- CUSTOM CSS: DESIGN, DARK MODE FIX, & TOOLBAR FIX ---
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
    }

    /* 1. SETUP BACKGROUND UTAMA */
    .stApp {
        background-color: var(--base-cream);
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }

    /* 2. HEADER PADDING (Agar judul tidak tertutup) */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 2rem;
        max-width: 100%;
    }

    /* --- FIX: MEMUNCULKAN TOMBOL TOOLBAR (TITIK TIGA) --- */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
        z-index: 9999999 !important; /* Layer paling atas */
        right: 2rem;
        top: 1rem;
        background-color: rgba(253, 248, 228, 0.8); /* Background transparan agar terlihat di atas batik */
        border-radius: 8px;
        padding: 2px;
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

    /* 4. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #FDF1D6;
        border-right: 2px solid var(--base-terracotta);
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        color: var(--base-terracotta) !important;
        text-align: center;
    }
    /* Memaksa teks sidebar tetap gelap meski user dark mode */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: var(--text-dark) !important;
    }

    /* 5. TOMBOL */
    .stButton > button {
        background: linear-gradient(to right, var(--comp-gold), var(--base-terracotta)) !important;
        color: var(--base-cream) !important;
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

    /* 7. WIDGET INPUT (MULTISELECT, SELECTBOX) - FIX DARK MODE */
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
    
    /* 8. FILE UPLOADER - FIX DARK MODE */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        border: 1px dashed var(--base-terracotta);
    }
    div[data-testid="stFileUploader"] section {
        background-color: #FDF1D6 !important; /* Warna krem muda */
    }
    div[data-testid="stFileUploader"] span {
        color: var(--text-dark) !important;
    }
    
    /* 9. BOX & ALERT STYLING */
    .welcome-box {
        background-color: #FFFBF0;
        padding: 25px;
        border-left: 6px solid var(--base-terracotta);
        border-top: 2px solid var(--comp-gold);
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .stAlert {
        background-color: #FFFDF8;
        border: 1px solid var(--comp-gold);
        color: var(--text-dark);
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
HEADER_IMG = 'gambar/image_12.png'  # Header Batik Full Logo
LOGO_IMG   = 'gambar/image_14.png'  # Logo Bulat untuk Sidebar
MASCOT_IMG = 'gambar/image_10.png'  # Maskot Laptop untuk Sidebar
HANDS_IMG  = 'gambar/image_13.png'  # Gambar Tangan (di atas footer)
FOOTER_IMG = 'gambar/image_15.png'  # Footer Bar Full Sosmed

# Tampilkan Header di paling atas
if os.path.exists(HEADER_IMG):
    st.image(HEADER_IMG, use_container_width=True)

# =================================================================
# FUNGSI BANTUAN (MAP) - FIXED SESUAI GAMBAR SHP ASLI
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

    # 1. STANDARISASI KABUPATEN
    if 'kabupaten' in combined_df.columns:
        combined_df['kabupaten'] = combined_df['kabupaten'].astype(str)
        combined_df['kabupaten'] = combined_df['kabupaten'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
        combined_df['kabupaten'] = combined_df['kabupaten'].str.title().str.strip()
        combined_df['kabupaten'] = combined_df['kabupaten'].replace({
            'Gunungkidul': 'Gunung Kidul', # Di SHP biasanya Gunung Kidul (spasi)
            'Yogya': 'Yogyakarta',
            'Yogyakartakarta': 'Yogyakarta'
        })

    # 2. STANDARISASI KECAMATAN (KUNCI UTAMA)
    if 'kecamatan' in combined_df.columns:
        combined_df['kecamatan'] = combined_df['kecamatan'].astype(str).str.title().str.strip()
        
        # MAPPING: Nama Excel (Baku) -> Nama SHP (Sesuai Gambar Anda)
        combined_df['kecamatan'] = combined_df['kecamatan'].replace({
            # GUNUNG KIDUL (SHP: PAKAI SPASI)
            'Gedangsari': 'Gedang Sari',
            'Girisubo': 'Giri Subo',
            'Karangmojo': 'Karang Mojo',
            'Nglipar': 'Ngli Par',
            'Paliyan': 'Pali Yan',
            'Panggang': 'Panggang', # Cek, biasanya aman
            'Patuk': 'Patuk',
            'Playen': 'Playen',
            'Ponjong': 'Ponjong',
            'Purwosari': 'Purwo Sari',
            'Rongkop': 'Rong Kop',
            'Saptosari': 'Sapto Sari',
            'Semanu': 'Semanu',
            'Semin': 'Semin',
            'Tanjungsari': 'Tanjung Sari',
            'Tepus': 'Tepus',
            'Wonosari': 'Wono Sari',
            
            # KULON PROGO (SHP: PAKAI SPASI)
            'Galur': 'Galur',
            'Girimulyo': 'Giri Mulyo',
            'Kalibawang': 'Kali Bawang',
            'Kokap': 'Kokap',
            'Lendah': 'Lendah',
            'Nanggulan': 'Nanggulan',
            'Panjatan': 'Panjatan',
            'Pengasih': 'Pengasih',
            'Samigaluh': 'Sami Galuh',
            'Sentolo': 'Sentolo',
            'Temon': 'Temon',
            'Wates': 'Wates',

            # BANTUL (SHP: HANYA BAMBANGLIPURO YANG SPASI)
            'Bambanglipuro': 'Bambang Lipuro',
            
            # KOTA YOGYAKARTA (SHP: TERNYATA SAMBUNG / TIDAK SPASI)
            # Pastikan Excel tidak typo pakai spasi
            'Danu Rejan': 'Danurejan',
            'Gedong Tengen': 'Gedongtengen',
            'Gondo Kusuman': 'Gondokusuman',
            'Gondo Manan': 'Gondomanan',
            'Kota Gede': 'Kotagede',
            'Mantri Jeron': 'Mantrijeron',
            'Mer Gangsan': 'Mergangsan',
            'Paku Alaman': 'Pakualaman',
            'Tegal Rejo': 'Tegalrejo',
            'Umbul Harjo': 'Umbulharjo',
            'Wiro Brajan': 'Wirobrajan'
        })

    # 3. AUTO-CORRECT KABUPATEN (FIX MANTRIJERON MASUK BANTUL)
    # Jika kecamatannya termasuk di Kota Jogja, paksa kabupaten jadi 'Yogyakarta'
    kec_kota_jogja = [
        'Danurejan', 'Gedongtengen', 'Gondokusuman', 'Gondomanan', 
        'Jetis', 'Kotagede', 'Kraton', 'Mantrijeron', 'Mergangsan', 
        'Ngampilan', 'Pakualaman', 'Tegalrejo', 'Umbulharjo', 'Wirobrajan'
    ]
    
    if 'kabupaten' in combined_df.columns and 'kecamatan' in combined_df.columns:
        combined_df.loc[combined_df['kecamatan'].isin(kec_kota_jogja), 'kabupaten'] = 'Yogyakarta'

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
    st.markdown("<h1>DASHBOARD SEBARAN LOKASI PKL 65<br>D.I. YOGYAKARTA</h1>", unsafe_allow_html=True)
    
    # Pesan Selamat Datang dengan Styling Baru
    st.markdown("""
    <div class="welcome-box">
        <h3 style='margin-top:0;'>Selamat Datang di Dashboard PKL 65</h3>
        <p style='font-size: 1.1rem;'>Halaman ini menyajikan peta interaktif persebaran lokasi di D.I. Yogyakarta. Visualisasi ini dirancang untuk memberikan gambaran spasial terkait data lapangan.</p>
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
            
            # 1. Bersihkan Nama Kabupaten di SHP
            gdf_shape['kab_upper'] = gdf_shape[SHP_COL_KAB].astype(str).str.title().str.strip()
            gdf_shape['kab_upper'] = gdf_shape['kab_upper'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
            gdf_shape['kab_upper'] = gdf_shape['kab_upper'].replace({
                'Gunungkidul': 'Gunung Kidul', # Samakan dengan Excel
                'Yogya': 'Yogyakarta',
                'Yogyakartakarta': 'Yogyakarta'
            })
            
            # 2. Bersihkan Nama Kecamatan di SHP
            # PENTING: JANGAN DI-REPLACE LAGI. 
            # Biarkan apa adanya (Title Case) agar sesuai dengan gambar SHP yang Anda kirim.
            gdf_shape['kec_upper'] = gdf_shape[SHP_COL_KEC].astype(str).str.title().str.strip()

            # Filter SHP
            final_gdf = gdf_shape[
                (gdf_shape['kab_upper'].isin(selected_kab)) &
                (gdf_shape['kec_upper'].isin(selected_kec))
            ]

            if not final_gdf.empty:
                stats_kec = final_df.groupby('kecamatan').size().reset_index(name='jumlah_lokasi')
                
                # Merge Data Excel ke SHP
                gdf_viz = final_gdf.merge(stats_kec, left_on='kec_upper', right_on='kecamatan', how='left')
                gdf_viz['jumlah_lokasi'] = gdf_viz['jumlah_lokasi'].fillna(0)

                # Setup Peta
                bounds = final_gdf.total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

                # Layer Choropleth
                if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                    cp = folium.Choropleth(
                        geo_data=gdf_viz,
                        name='Kepadatan Wilayah',
                        data=gdf_viz,
                        columns=['kec_upper', 'jumlah_lokasi'],
                        key_on='feature.properties.kec_upper', # Pastikan ini sesuai kolom dummy kita
                        fill_color='YlOrRd', 
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='Jumlah Lokasi (Frekuensi)',
                        highlight=True
                    ).add_to(m)
                    
                    # Tooltip Peta
                    folium.GeoJsonTooltip(
                        fields=['kec_upper', 'jumlah_lokasi'], 
                        aliases=['Kecamatan:', 'Jumlah Data:'], 
                        localize=True
                    ).add_to(cp.geojson)

                # Layer Heatmap
                if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                    heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: '#FFD700', 0.65: '#FF8C00', 1: '#8B0000'}).add_to(m)

                folium.LayerControl().add_to(m)
                
                # Frame Peta Elegan
                st.markdown('<div style="box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-radius: 12px; overflow: hidden; border: 3px solid #E07A3F;">', unsafe_allow_html=True)
                st_folium(m, width=1200, height=650)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Tabel Detail
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Lihat Tabel Data Detail untuk Wilayah Terpilih"): 
                    st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("Wilayah SHP tidak ditemukan. Kemungkinan filter Kabupaten/Kecamatan tidak cocok dengan data SHP.")
                st.write("Nama Kabupaten di SHP (Filtered):", gdf_shape[gdf_shape['kab_upper'].isin(selected_kab)]['kab_upper'].unique())
        else:
            st.info("Silakan pilih Kabupaten dan Kecamatan pada panel filter di atas untuk menampilkan visualisasi peta.")
    else:
        st.warning("Data peta (file Excel atau SHP) belum tersedia di folder 'data/'. Pastikan file .xlsx dan .shp telah diunggah dengan benar.")

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
        st.markdown("<h1>ANALISIS STATISTIK DATA PKL</h1>", unsafe_allow_html=True)
        
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
        # TAB 6: REDUKSI DIMENSI (PCA & EFA) (BAB 8 & 9)
        # -------------------------------------------------------------
        with tab_dim:
            st.header("Reduksi Dimensi")
            st.info("Metode ini membantu menyederhanakan data Anda dengan mengurangi jumlah variabel.")
            st.warning("Penting: Kami akan **otomatis menstandardisasi data Anda** (mean=0, std=1) sebelum analisis.")
            st.markdown("---")

            if len(numeric_cols) < 2:
                 st.error("Analisis ini memerlukan setidaknya 2 kolom numerik.")
            else:
                try:
                    # Selalu standarisasi data untuk PCA/EFA
                    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
                except Exception as e:
                    st.error(f"Gagal melakukan standarisasi data: {e}")
                    st.stop() 

                # --- PCA (Bab 8) ---
                st.subheader("Principal Component Analysis (PCA)")
                st.info("Tujuan: Meringkas (mereduksi) beberapa variabel numerik menjadi lebih sedikit 'komponen' baru.")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    pca_vars = st.multiselect("Pilih variabel untuk PCA:", numeric_cols, default=numeric_cols, key='pca_vars')
                    if len(pca_vars) < 2:
                        st.warning("Pilih minimal 2 variabel untuk PCA.")
                    else:
                        n_components_pca = st.slider(
                            "Pilih jumlah Komponen PCA:",
                            min_value=1,
                            max_value=len(pca_vars),
                            value=max(1, len(pca_vars)//2),
                            key='n_pca'
                        )
                
                with col2:
                    if len(pca_vars) >= 2:
                        if st.button("Jalankan PCA", key='pca_btn'):
                            try:
                                df_scaled_pca = df_scaled[pca_vars]
                                pca = PCA(n_components=n_components_pca)
                                pca.fit(df_scaled_pca)
                                
                                st.write("**Scree Plot (Proporsi Varians yang Dijelaskan)**")
                                scree_data = pd.DataFrame({
                                    'Komponen': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                    'Explained Variance': pca.explained_variance_ratio_
                                })
                                # Warna Grand Design
                                fig_scree = px.bar(scree_data, x='Komponen', y='Explained Variance', title='Scree Plot', color_discrete_sequence=['#F2C94C'])
                                fig_scree.add_trace(go.Scatter(x=scree_data['Komponen'], y=scree_data['Explained Variance'].cumsum(), name='Kumulatif', line=dict(color='#E07A3F')))
                                fig_scree.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                st.plotly_chart(fig_scree, use_container_width=True)
                                
                                st.write("**Component Loadings (Bobot Variabel)**")
                                st.dataframe(pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f'PC{i+1}' for i in range(n_components_pca)]))

                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **1. Scree Plot:** Garis 'Kumulatif' menunjukkan seberapa banyak info (varians) yang dipertahankan.
                                    * **Hasil Anda:** {n_components_pca} komponen utama Anda menjelaskan **{pca.explained_variance_ratio_.sum()*100:.2f}%** dari total variasi.
                                    **2. Component Loadings:** Menunjukkan "resep" dari setiap Komponen Utama (PC).
                                    * Cari angka *loading* yang besar (jauh dari 0) untuk menamai komponen (misal: PC1 adalah "Faktor Senioritas").
                                    """)
                            except Exception as e:
                                st.error(f"Error menjalankan PCA: {e}")

                st.markdown("---")

                # --- EFA (Bab 9) ---
                st.subheader("Exploratory Factor Analysis (EFA)")
                st.info("Tujuan: Menemukan 'faktor' (konsep tersembunyi/laten) yang mendasari sekumpulan variabel.")
                
                col3, col4 = st.columns([1, 2])
                with col3:
                    efa_vars = st.multiselect("Pilih variabel untuk EFA:", numeric_cols, default=numeric_cols, key='efa_vars')
                    if len(efa_vars) < 3:
                          st.warning("Pilih minimal 3 variabel untuk EFA.")
                    else:
                        n_components_efa = st.slider(
                            "Pilih jumlah Faktor:",
                            min_value=1,
                            max_value=len(efa_vars)-1,
                            value=max(1, len(efa_vars)//2),
                            key='n_efa'
                        )
                        rotation = st.radio("Pilih metode Rotasi:", ["varimax", "promax"], key='efa_rot')

                with col4:
                    if len(efa_vars) >= 3:
                        if st.button("Jalankan EFA", key='efa_btn'):
                            try:
                                df_scaled_efa = df_scaled[efa_vars]
                                # Uji Kelayakan
                                chi_square_value, p_value_bartlett = calculate_bartlett_sphericity(df_scaled_efa)
                                kmo_all, kmo_model = calculate_kmo(df_scaled_efa)
                                
                                st.write("**Uji Kelayakan Data (KMO & Bartlett)**")
                                st.write(f"* **Kaiser-Meyer-Olkin (KMO) Measure:** `{kmo_model:.4f}`")
                                st.write(f"* **Bartlett's Test p-value:** `{p_value_bartlett:.4f}`")

                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan Uji Kelayakan"):
                                    st.markdown(f"""
                                    **Tujuan Uji:** Ini adalah tes "Boleh Jalan" untuk EFA.
                                    1.  **KMO:** Harus > 0.6.
                                    2.  **Bartlett:** P-value harus <= 0.05.
                                    """)
                                    if kmo_model < 0.6:
                                        st.error("**Kesimpulan KMO:** Nilai KMO < 0.6. Data **kurang ideal**.")
                                    else:
                                        st.success("**Kesimpulan KMO:** Nilai KMO > 0.6. Data **cukup baik**.")
                                    
                                    if p_value_bartlett > 0.05:
                                        st.error("**Kesimpulan Bartlett:** P-value > 0.05. Variabel tidak berkorelasi. EFA **tidak disarankan**.")
                                    else:
                                        st.success("**Kesimpulan Bartlett:** P-value < 0.05. Variabel **berkorelasi**, bagus untuk EFA.")
                                
                                if kmo_model >= 0.6 and p_value_bartlett <= 0.05:
                                    fa = FactorAnalyzer(n_factors=n_components_efa, rotation=rotation)
                                    fa.fit(df_scaled_efa)
                                    
                                    st.write(f"**Factor Loadings (Rotasi {rotation})**")
                                    st.dataframe(pd.DataFrame(fa.loadings_, index=efa_vars, columns=[f'Faktor {i+1}' for i in range(n_components_efa)]))
                                    
                                    with st.expander("Lihat Penjelasan Factor Loadings"):
                                        st.markdown(f"""
                                        **Tujuan:** Memberi "nama" pada {n_components_efa} faktor tersembunyi.
                                        **Lihat:** Cari angka *loading* yang besar (misal > 0.6) di setiap kolom Faktor.
                                        **Aturan:** Idealnya, satu variabel hanya punya *loading* tinggi di **satu** faktor saja.
                                        """)
                            except Exception as e:
                                st.error(f"Error menjalankan EFA: {e}")

        # -------------------------------------------------------------
        # TAB 7: KLASIFIKASI & CLUSTERING (BAB 11 & 12)
        # -------------------------------------------------------------
        with tab_class:
            st.header("Klasifikasi & Clustering")
            st.info("Metode ini membantu Anda mengelompokkan data Anda.")
            st.markdown("---")

            if not numeric_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                # --- Clustering (Bab 12 - Unsupervised) ---
                st.subheader("Clustering (K-Means)")
                st.info("Tujuan: Menemukan kelompok-kelompok (cluster) yang 'alami' dalam data Anda (unsupervised).")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    cluster_vars = st.multiselect(
                        "Pilih variabel numerik untuk Clustering:",
                        numeric_cols,
                        default=numeric_cols if len(numeric_cols) >= 2 else [],
                        key='clust_vars'
                    )
                    n_clusters = st.slider(
                        "Pilih jumlah Cluster (K):",
                        min_value=2,
                        max_value=10,
                        value=3,
                        key='n_clust'
                    )
                
                with col2:
                    if len(cluster_vars) >= 2:
                        if st.button("Jalankan K-Means Clustering", key='clust_btn'):
                            try:
                                df_scaled_clust = pd.DataFrame(StandardScaler().fit_transform(df[cluster_vars]), columns=cluster_vars)
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(df_scaled_clust)
                                
                                df_clean_clust = df[cluster_vars].dropna()
                                df_clean_clust['Cluster'] = cluster_labels
                                
                                st.write(f"**Visualisasi Cluster ({cluster_vars[0]} vs {cluster_vars[1]})**")
                                # Menggunakan palet warna Grand Design
                                fig_clust = px.scatter(
                                    df_clean_clust,
                                    x=cluster_vars[0],
                                    y=cluster_vars[1],
                                    color='Cluster',
                                    title=f"Hasil Clustering K-Means (K={n_clusters})",
                                    color_continuous_scale=['#F2C94C', '#4F8190', '#739159', '#E07A3F']
                                )
                                fig_clust.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                st.plotly_chart(fig_clust, use_container_width=True)
                                
                                st.write("**Pusat Cluster (Cluster Centers)**")
                                st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_vars, index=[f'Cluster {i}' for i in range(n_clusters)]))

                                # Interpretasi hasil
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"**1. Visualisasi Cluster:** Menunjukkan sebaran {n_clusters} kelompok. Apakah terlihat terpisah dengan baik?")
                                    st.markdown(f"**2. Pusat Cluster:** Menunjukkan nilai *rata-rata* (standardisasi) dari setiap variabel untuk setiap cluster. Gunakan ini untuk memberi 'nama' atau 'persona' pada setiap cluster.")
                                    
                            except Exception as e:
                                st.error(f"Error menjalankan Clustering: {e}")
                    else:
                        st.warning("Silakan pilih minimal 2 variabel untuk clustering.")

                st.markdown("---")

                # --- Discriminant Analysis (Bab 11 - Supervised) ---
                st.subheader("Analisis Diskriminan Linear (LDA)")
                st.info("Tujuan: Menemukan 'fungsi' (kombinasi linear) yang paling baik **memisahkan** kelompok yang sudah diketahui (supervised).")
                
                col3, col4 = st.columns([1, 2])
                with col3:
                    if not categorical_cols:
                        st.warning("Analisis Diskriminan memerlukan 1 variabel kategorikal (Target) untuk diprediksi.")
                        lda_target = None
                        lda_predictors = []
                    else:
                        lda_target = st.selectbox("Pilih Variabel Target (Kategorikal):", categorical_cols, key='lda_target')
                        
                        available_lda_x = [col for col in numeric_cols]
                        lda_predictors = st.multiselect(
                            "Pilih Variabel Prediktor (Numerik):",
                            available_lda_x,
                            default=available_lda_x if len(available_lda_x) >= 2 else [],
                            key='lda_preds'
                        )
                
                with col4:
                    if categorical_cols and lda_target and len(lda_predictors) >= 1:
                        if st.button("Jalankan Analisis Diskriminan", key='lda_btn'):
                            try:
                                data_lda = df[[lda_target] + lda_predictors].dropna()
                                X_lda = data_lda[lda_predictors]
                                y_lda = data_lda[lda_target]
                                
                                X_scaled_lda = StandardScaler().fit_transform(X_lda)
                                
                                n_classes = len(y_lda.unique())
                                n_components_lda = min(n_classes - 1, len(lda_predictors))

                                if n_components_lda < 1:
                                    st.error(f"Error: Jumlah diskriminan harus > 0. (Kelas: {n_classes}, Prediktor: {len(lda_predictors)})")
                                else:
                                    lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
                                    X_lda_transformed = lda.fit_transform(X_scaled_lda, y_lda)
                                    
                                    lda_accuracy = lda.score(X_scaled_lda, y_lda)
                                    
                                    st.write(f"**Akurasi Model:** `{lda_accuracy*100:.2f}%`")
                                    
                                    st.write("**Koefisien Diskriminan (Bobot Variabel)**")
                                    st.dataframe(pd.DataFrame(lda.scalings_, index=lda_predictors, columns=[f'LD{i+1}' for i in range(n_components_lda)]))

                                    # Menggunakan palet warna Grand Design
                                    if n_components_lda >= 2:
                                        lda_plot_df = pd.DataFrame(X_lda_transformed, columns=['LD1', 'LD2'])
                                        lda_plot_df['Target'] = y_lda.values
                                        fig_lda = px.scatter(lda_plot_df, x='LD1', y='LD2', color='Target', title='Plot Fungsi Diskriminan', color_discrete_sequence=['#F2C94C', '#4F8190', '#739159', '#E07A3F'])
                                        fig_lda.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                        st.plotly_chart(fig_lda, use_container_width=True)
                                    elif n_components_lda == 1:
                                        lda_plot_df = pd.DataFrame(X_lda_transformed, columns=['LD1'])
                                        lda_plot_df['Target'] = y_lda.values
                                        fig_lda = px.histogram(lda_plot_df, x='LD1', color='Target', title='Plot Fungsi Diskriminan 1D', marginal='box', color_discrete_sequence=['#F2C94C', '#4F8190', '#739159', '#E07A3F'])
                                        fig_lda.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                        st.plotly_chart(fig_lda, use_container_width=True)
                                    
                                    # Interpretasi hasil
                                    with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                        st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                        st.markdown(f"**1. Akurasi Model:** Model ini **{lda_accuracy*100:.2f}%** akurat dalam menebak '{lda_target}' berdasarkan prediktor.")
                                        st.markdown(f"**2. Koefisien Diskriminan:** Angka yang besar (jauh dari 0) menunjukkan variabel tersebut adalah pembeda yang *kuat* antar kelompok.")
                                        st.markdown(f"**3. Plot Fungsi Diskriminan:** Menunjukkan seberapa baik model memisahkan kelompok. Semakin jauh jarak antar warna, semakin baik.")
                            except Exception as e:
                                st.error(f"Error menjalankan LDA: {e}")
                    elif categorical_cols:
                        st.warning("Silakan pilih 1 variabel target (kategorikal) dan minimal 1 variabel prediktor (numerik).")

# =================================================================
# VISUAL: FOOTER IMAGE
# =================================================================
# Menampilkan tangan di atas footer bar
st.markdown("<br><br>", unsafe_allow_html=True)
col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
if os.path.exists(HANDS_IMG):
    with col_h2:
        st.image(HANDS_IMG, use_container_width=True)

# Menampilkan footer bar penuh di paling bawah
if os.path.exists(FOOTER_IMG):
    st.image(FOOTER_IMG, use_container_width=True)