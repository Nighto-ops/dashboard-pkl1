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

# =================================================================
# BAGIAN ANALISIS STATISTIK (LENGKAP & DIPERBAIKI)
# =================================================================
else:
    # --- INISIALISASI VARIABEL ---
    df = None
    numeric_cols = []
    categorical_cols = []
    all_cols = []

    # --- FUNGSI BANTUAN INTERPRETASI (BAHASA AWAM) ---
    def interpret_pvalue(p_val, alpha=0.05):
        if p_val < alpha:
            return "SIGNIFIKAN (Nyata)", "success", f"Karena P-value ({p_val:.4f}) < {alpha}, hipotesis nol ditolak. Ada pengaruh/perbedaan yang nyata."
        else:
            return "TIDAK SIGNIFIKAN (Tidak Nyata)", "warning", f"Karena P-value ({p_val:.4f}) > {alpha}, hipotesis nol diterima. Tidak cukup bukti untuk menyatakan adanya perbedaan/pengaruh."

    def check_normality(data):
        stat, p = stats.shapiro(data)
        return p

    # --- LOAD DATA ---
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success("✅ File berhasil dimuat!")
            
            # --- [CRITICAL FIX] PEMBERSIHAN & DEDUPLIKASI NAMA KOLOM ---
            # Ini mencegah error 'DuplicateError' pada library plotting
            clean_cols = [c.strip().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]
            seen = {}
            final_cols = []
            for col in clean_cols:
                if col in seen:
                    seen[col] += 1
                    final_cols.append(f"{col}_{seen[col]}") # Tambah suffix _1, _2 jika duplikat
                else:
                    seen[col] = 0
                    final_cols.append(col)
            df.columns = final_cols
            # -----------------------------------------------------------

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
        st.markdown("<h1>ANALISIS STATISTIK TERPADU</h1>", unsafe_allow_html=True)
        st.markdown("Dashboard ini menggabungkan analisis statistik dasar hingga multivariat kompleks dengan interpretasi otomatis.")
        
        # --- MENU TABS LENGKAP (GABUNGAN FITUR LAMA & MODUL BARU) ---
        list_tabs = [
            "📂 Data", 
            "📈 Analisis Dasar",        # Fitur Lama (Hist, Scatter, T-Test)
            "📐 Regresi",               # Fitur Lama + Uji Asumsi
            "📊 ANOVA",                 # Fitur Lama
            "✅ Asumsi Multivariat",    # Fitur Baru (Modul 2)
            "⚖️ Uji Hotelling",         # Fitur Baru (Modul 3-4)
            "🧮 MANOVA",                # Fitur Baru (Modul 5-7)
            "📉 PCA & Biplot",          # Fitur Baru (Modul 8-10)
            "🔗 Kanonik",               # Fitur Baru (Modul 12)
            "🧩 Cluster & Klasifikasi"  # Fitur Baru (Modul 11 & 13)
        ]
        tabs = st.tabs(list_tabs)

        # =================================================================
        # TAB 1: DATA OVERVIEW
        # =================================================================
        with tabs[0]:
            st.header("Eksplorasi Data Awal")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Statistik Deskriptif")
                st.write(df.describe())
            with c2:
                st.subheader("Tabel Data")
                st.dataframe(df, use_container_width=True)

        # =================================================================
        # TAB 2: ANALISIS DASAR (HISTOGRAM, SCATTER, T-TEST) - FITUR LAMA
        # =================================================================
        with tabs[1]:
            st.header("Analisis Univariat & Bivariat")
            
            # 1. UNIVARIAT
            st.subheader("1. Distribusi Data (Histogram)")
            col_u1, col_u2 = st.columns([1, 3])
            with col_u1:
                hist_col = st.selectbox("Pilih Variabel:", numeric_cols, key='hist_basic')
            with col_u2:
                if hist_col:
                    fig = px.histogram(df, x=hist_col, title=f"Distribusi {hist_col}", marginal="box", color_discrete_sequence=['#E07A3F'])
                    st.plotly_chart(fig, use_container_width=True)
                    # Uji Normalitas Sederhana
                    stat, p_norm = stats.shapiro(df[hist_col].dropna())
                    res_norm, col_norm, desc_norm = interpret_pvalue(p_norm)
                    if p_norm > 0.05:
                        st.success(f"Data berdistribusi NORMAL (P-value: {p_norm:.4f} > 0.05)")
                    else:
                        st.warning(f"Data TIDAK berdistribusi Normal (P-value: {p_norm:.4f} < 0.05)")

            st.markdown("---")
            
            # 2. BIVARIAT (SCATTER & KORELASI)
            st.subheader("2. Hubungan Antar Variabel (Korelasi)")
            c1, c2 = st.columns([1, 2])
            with c1:
                sc_x = st.selectbox("Variabel X:", numeric_cols, key='sc_x')
                sc_y = st.selectbox("Variabel Y:", numeric_cols, key='sc_y')
            with c2:
                if sc_x and sc_y:
                    fig_sc = px.scatter(df, x=sc_x, y=sc_y, trendline='ols', title=f"Scatter Plot: {sc_y} vs {sc_x}", color_discrete_sequence=['#F2C94C'])
                    st.plotly_chart(fig_sc, use_container_width=True)
                    
                    corr_val, p_val = stats.pearsonr(df[sc_x].dropna(), df[sc_y].dropna())
                    status, color, desc = interpret_pvalue(p_val)
                    
                    st.markdown(f"**Korelasi Pearson (r):** {corr_val:.4f}")
                    st.markdown(f"**Signifikansi:** :{color}[{status}]")
                    with st.expander("Interpretasi"):
                        st.write(desc)
                        st.write(f"Nilai r = {corr_val:.4f} menunjukkan hubungan yang **{'Sangat Kuat' if abs(corr_val)>0.8 else 'Kuat' if abs(corr_val)>0.6 else 'Sedang' if abs(corr_val)>0.4 else 'Lemah'}**.")

            st.markdown("---")

            # 3. UJI BEDA 2 KELOMPOK (T-TEST)
            st.subheader("3. Uji Beda Dua Rata-rata (Independent T-Test)")
            c_t1, c_t2 = st.columns([1, 2])
            with c_t1:
                ttest_cat = st.selectbox("Variabel Kelompok (Kategori):", categorical_cols, key='ttest_cat')
                ttest_num = st.selectbox("Variabel Nilai (Numerik):", numeric_cols, key='ttest_num')
            with c_t2:
                if ttest_cat and ttest_num:
                    grps = df[ttest_cat].dropna().unique()
                    if len(grps) == 2:
                        g1 = df[df[ttest_cat] == grps[0]][ttest_num]
                        g2 = df[df[ttest_cat] == grps[1]][ttest_num]
                        stat, p_ttest = stats.ttest_ind(g1, g2)
                        
                        status_t, color_t, desc_t = interpret_pvalue(p_ttest)
                        st.metric("P-Value T-Test", f"{p_ttest:.4f}")
                        st.markdown(f"Hasil: :{color_t}[**{status_t}**]")
                        st.caption(f"Perbedaan rata-rata {ttest_num} antara {grps[0]} dan {grps[1]}.")
                    else:
                        st.warning(f"Variabel '{ttest_cat}' memiliki {len(grps)} kategori. T-Test hanya untuk 2 kategori. Gunakan ANOVA untuk >2 kategori.")

        # =================================================================
        # TAB 3: REGRESI (FITUR LAMA + UJI ASUMSI)
        # =================================================================
        with tabs[2]:
            st.header("Analisis Regresi")
            reg_type = st.radio("Tipe Regresi:", ["Regresi Linear Sederhana/Polinomial", "Regresi Linear Berganda"], horizontal=True)
            
            if reg_type == "Regresi Linear Sederhana/Polinomial":
                c1, c2 = st.columns([1,2])
                with c1:
                    ry = st.selectbox("Variabel Y (Dependen):", numeric_cols, key='reg1_y')
                    rx = st.selectbox("Variabel X (Independen):", [c for c in numeric_cols if c!=ry], key='reg1_x')
                    degree = st.slider("Derajat Polinomial (1=Linear):", 1, 4, 1)
                with c2:
                    if ry and rx:
                        X = df[[rx]].dropna(); y = df[ry].dropna()
                        poly = PolynomialFeatures(degree)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        y_pred = model.predict(X_poly)
                        r2 = r2_score(y, y_pred)
                        
                        fig_reg = px.scatter(x=X[rx], y=y, title=f"Regresi Derajat {degree}", labels={'x':rx, 'y':ry})
                        # Sort for line plot
                        sort_idx = np.argsort(X[rx])
                        fig_reg.add_trace(go.Scatter(x=X[rx].iloc[sort_idx], y=y_pred[sort_idx], mode='lines', name='Fit Line', line=dict(color='red')))
                        st.plotly_chart(fig_reg, use_container_width=True)
                        st.success(f"**R-Squared:** {r2:.4f} (Model menjelaskan {r2*100:.1f}% varians data)")

            else: # Berganda
                c1, c2 = st.columns([1,2])
                with c1:
                    ry_m = st.selectbox("Variabel Y:", numeric_cols, key='regm_y')
                    rx_m = st.multiselect("Variabel X (Banyak):", [c for c in numeric_cols if c!=ry_m], key='regm_x')
                with c2:
                    if ry_m and len(rx_m) > 0:
                        if st.button("Jalankan Regresi Berganda", key='btn_reg_m'):
                            data_reg = df[[ry_m] + rx_m].dropna()
                            X = sm.add_constant(data_reg[rx_m])
                            y = data_reg[ry_m]
                            model = sm.OLS(y, X).fit()
                            
                            st.write(model.summary())
                            
                            # --- ANALISIS PENDUKUNG (ASUMSI KLASIK) ---
                            st.markdown("### 🔍 Uji Asumsi Klasik (Analisis Pendukung)")
                            
                            # 1. Multikolinearitas (VIF)
                            st.write("**1. Uji Multikolinearitas (VIF)**")
                            vif_data = pd.DataFrame()
                            vif_data["Variabel"] = X.columns
                            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                            st.dataframe(vif_data[vif_data['Variabel'] != 'const'])
                            if any(vif_data[vif_data['Variabel'] != 'const']['VIF'] > 10):
                                st.error("⚠️ Terdeteksi Multikolinearitas (VIF > 10). Pertimbangkan membuang variabel yang berkorelasi tinggi.")
                            else:
                                st.success("✅ Tidak ada masalah Multikolinearitas (VIF < 10).")

                            # 2. Normalitas Residual
                            st.write("**2. Uji Normalitas Residual (Shapiro-Wilk)**")
                            shapiro_stat, shapiro_p = stats.shapiro(model.resid)
                            if shapiro_p > 0.05:
                                st.success(f"✅ Residual berdistribusi Normal (P={shapiro_p:.4f} > 0.05).")
                            else:
                                st.warning(f"⚠️ Residual TIDAK Normal (P={shapiro_p:.4f} < 0.05).")

        # =================================================================
        # TAB 4: ANOVA (FITUR LAMA)
        # =================================================================
        with tabs[3]:
            st.header("Analysis of Variance (ANOVA)")
            
            anova_type = st.radio("Tipe ANOVA:", ["One-Way ANOVA", "Two-Way ANOVA"], horizontal=True)
            
            if anova_type == "One-Way ANOVA":
                c1, c2 = st.columns([1,2])
                with c1:
                    av1_cat = st.selectbox("Faktor (Kategori):", categorical_cols, key='av1_c')
                    av1_num = st.selectbox("Variabel Respon (Numerik):", numeric_cols, key='av1_n')
                with c2:
                    if st.button("Proses ANOVA One-Way"):
                        if av1_cat and av1_num:
                            # Visualisasi
                            fig = px.box(df, x=av1_cat, y=av1_num, color=av1_cat, title="Boxplot Perbandingan")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Model
                            model = smf.ols(f'{av1_num} ~ C({av1_cat})', data=df).fit()
                            aov_table = sm.stats.anova_lm(model, typ=2)
                            st.write(aov_table)
                            
                            pval = aov_table['PR(>F)'][0]
                            res_txt, col_txt, desc_txt = interpret_pvalue(pval)
                            st.markdown(f"### Kesimpulan: :{col_txt}[{res_txt}]")
                            st.write(desc_txt)

            else: # Two-Way
                c1, c2 = st.columns([1,2])
                with c1:
                    av2_cat1 = st.selectbox("Faktor 1:", categorical_cols, key='av2_c1')
                    av2_cat2 = st.selectbox("Faktor 2:", categorical_cols, key='av2_c2')
                    av2_num = st.selectbox("Variabel Respon:", numeric_cols, key='av2_n')
                with c2:
                    if st.button("Proses ANOVA Two-Way"):
                        if av2_cat1 and av2_cat2 and av2_num:
                            model = smf.ols(f'{av2_num} ~ C({av2_cat1}) + C({av2_cat2}) + C({av2_cat1}):C({av2_cat2})', data=df).fit()
                            aov_table = sm.stats.anova_lm(model, typ=2)
                            st.write(aov_table)
                            
                            with st.expander("Interpretasi Efek"):
                                st.write(f"1. Efek {av2_cat1}: {interpret_pvalue(aov_table.loc[f'C({av2_cat1})', 'PR(>F)'])[0]}")
                                st.write(f"2. Efek {av2_cat2}: {interpret_pvalue(aov_table.loc[f'C({av2_cat2})', 'PR(>F)'])[0]}")
                                st.write(f"3. Efek Interaksi: {interpret_pvalue(aov_table.loc[f'C({av2_cat1}):C({av2_cat2})', 'PR(>F)'])[0]}")

        # =================================================================
        # TAB 5: ASUMSI MULTIVARIAT (MODUL 2)
        # =================================================================
        with tabs[4]:
            st.header("Modul 2: Uji Normalitas Multivariat")
            st.info("Metode Chi-Square Plot untuk memeriksa apakah sekumpulan variabel berdistribusi normal secara bersamaan.")
            
            mvn_vars = st.multiselect("Pilih Variabel Multivariat (Min. 2):", numeric_cols, key='mvn_v')
            
            if len(mvn_vars) >= 2:
                if st.button("Cek Normalitas Multivariat", key='btn_mvn'):
                    try:
                        d = df[mvn_vars].dropna()
                        # Mahalanobis
                        m_diff = d - np.mean(d, axis=0)
                        cov = np.cov(d.values.T)
                        inv_cov = np.linalg.inv(cov)
                        mahal = np.diag(m_diff @ inv_cov @ m_diff.T)
                        
                        # Chi-Square Quantiles
                        n = len(d)
                        p = len(mvn_vars)
                        prob = (np.arange(1, n+1) - 0.5) / n
                        chi2_q = stats.chi2.ppf(prob, df=p)
                        
                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=chi2_q, y=np.sort(mahal), mode='markers', name='Data'))
                        mx = max(max(chi2_q), max(mahal))
                        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode='lines', name='Garis Normal', line=dict(color='red', dash='dash')))
                        fig.update_layout(title="Chi-Square Q-Q Plot", xaxis_title="Kuantil Teoretis", yaxis_title="Jarak Mahalanobis")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Interpretasi:**
                        * Jika titik-titik menyebar mengikuti garis merah putus-putus, asumsi **Normalitas Multivariat TERPENUHI**.
                        * Jika melengkung jauh, asumsi dilanggar (mungkin perlu transformasi data atau buang outlier).
                        """)
                    except Exception as e:
                        st.error(f"Error perhitungan: {e}. Cek apakah ada variabel yang berkorelasi sempurna.")

        # =================================================================
        # TAB 6: HOTELLING T2 (MODUL 3-4)
        # =================================================================
        with tabs[5]:
            st.header("Modul 3 & 4: Uji Hotelling's T²")
            st.write("Menguji perbedaan rata-rata vektor antara DUA kelompok independen.")
            
            c1, c2 = st.columns([1,2])
            with c1:
                hot_cat = st.selectbox("Variabel Kelompok (Harus 2 Kategori):", categorical_cols, key='ht_c')
                hot_vars = st.multiselect("Variabel Dependen (Numerik):", numeric_cols, key='ht_v')
            with c2:
                if hot_cat and len(hot_vars)>=1:
                    cats = df[hot_cat].dropna().unique()
                    if len(cats) == 2:
                        if st.button("Jalankan Uji Hotelling"):
                            # Menggunakan Statsmodels MANOVA (Equivalent Hotelling)
                            formula = f'{" + ".join(hot_vars)} ~ C({hot_cat})'
                            model = MANOVA.from_formula(formula, data=df)
                            res = model.mv_test()
                            summ = res.summary_frame
                            
                            pval = summ.loc[(f'C({hot_cat})', "Wilks' lambda"), "Pr > F"]
                            stat_txt, col_txt, desc_txt = interpret_pvalue(pval)
                            
                            st.markdown(f"### Hasil Uji: :{col_txt}[{stat_txt}]")
                            st.write(f"P-Value: {pval:.4f}")
                            
                            with st.expander("Penjelasan Detail"):
                                st.write(f"Kami menguji apakah profil {', '.join(hot_vars)} berbeda secara signifikan antara kelompok **{cats[0]}** dan **{cats[1]}**.")
                                st.write(desc_txt)
                                st.write("Tabel Statistik:")
                                st.dataframe(summ)
                    else:
                        st.warning(f"Variabel '{hot_cat}' memiliki {len(cats)} kategori. Hotelling T² hanya untuk 2 kategori.")

        # =================================================================
        # TAB 7: MANOVA & MANCOVA (MODUL 5-7)
        # =================================================================
        with tabs[6]:
            st.header("Modul 5, 6, 7: MANOVA & MANCOVA")
            
            modul_type = st.radio("Pilih Analisis:", ["MANOVA (Tanpa Kontrol)", "MANCOVA (Dengan Variabel Kontrol/Kovariat)"], horizontal=True)
            
            if modul_type == "MANOVA (Tanpa Kontrol)":
                c1, c2 = st.columns([1,2])
                with c1:
                    m_fac = st.selectbox("Faktor (Kelompok):", categorical_cols, key='m_f')
                    m_vars = st.multiselect("Variabel Respon (Y):", numeric_cols, key='m_v')
                with c2:
                    if st.button("Analisis MANOVA"):
                        if m_fac and len(m_vars)>=2:
                            model = MANOVA.from_formula(f'{" + ".join(m_vars)} ~ C({m_fac})', data=df)
                            res = model.mv_test()
                            st.write(res.summary())
                            
                            pval = res.summary_frame.loc[(f'C({m_fac})', "Wilks' lambda"), "Pr > F"]
                            st.info(f"Kesimpulan: Perbedaan antar kelompok '{m_fac}' adalah **{interpret_pvalue(pval)[0]}**.")
            
            else: # MANCOVA
                c1, c2 = st.columns([1,2])
                with c1:
                    mc_fac = st.selectbox("Faktor Utama:", categorical_cols, key='mc_f')
                    mc_cov = st.selectbox("Kovariat (Pengontrol):", numeric_cols, key='mc_c')
                    mc_vars = st.multiselect("Variabel Respon (Y):", [c for c in numeric_cols if c!=mc_cov], key='mc_v')
                with c2:
                    if st.button("Analisis MANCOVA"):
                        if mc_fac and mc_cov and len(mc_vars)>=2:
                            model = MANOVA.from_formula(f'{" + ".join(mc_vars)} ~ C({mc_fac}) + {mc_cov}', data=df)
                            res = model.mv_test()
                            
                            summ = res.summary_frame
                            pval_fac = summ.loc[(f'C({mc_fac})', "Wilks' lambda"), "Pr > F"]
                            pval_cov = summ.loc[(mc_cov, "Wilks' lambda"), "Pr > F"]
                            
                            st.write("### Interpretasi Hasil:")
                            st.write(f"1. **Efek Kovariat ({mc_cov}):** {interpret_pvalue(pval_cov)[0]}")
                            st.write(f"2. **Efek Faktor ({mc_fac}) setelah dikontrol:** {interpret_pvalue(pval_fac)[0]}")
                            st.write("Detail Statistik:")
                            st.dataframe(summ)

        # =================================================================
        # TAB 8: PCA & BIPLOT (MODUL 8 & 10)
        # =================================================================
        with tabs[7]:
            st.header("Modul 8 & 10: Analisis Komponen Utama (PCA) & Biplot")
            
            pca_cols = st.multiselect("Pilih Variabel (Numerik) untuk PCA:", numeric_cols, default=numeric_cols, key='pca_c')
            
            if len(pca_cols) >= 2:
                if st.button("Generate PCA & Biplot"):
                    # 1. Standardisasi
                    X = df[pca_cols].dropna()
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # 2. PCA Fitting
                    pca = PCA()
                    pca.fit(X_scaled)
                    var_exp = pca.explained_variance_ratio_
                    cum_var = np.cumsum(var_exp)
                    
                    # 3. Analisis Pendukung: SCREE PLOT
                    st.subheader("1. Analisis Pendukung: Scree Plot")
                    st.write("Grafik ini membantu menentukan berapa komponen yang harus diambil.")
                    scree_df = pd.DataFrame({'Komponen': range(1, len(var_exp)+1), 'Varians Dijelaskan': var_exp, 'Kumulatif': cum_var})
                    fig_scree = px.bar(scree_df, x='Komponen', y='Varians Dijelaskan', title='Scree Plot')
                    fig_scree.add_trace(go.Scatter(x=scree_df['Komponen'], y=scree_df['Kumulatif'], name='Kumulatif'))
                    st.plotly_chart(fig_scree, use_container_width=True)
                    
                    # 4. BIPLOT (2 Dimensi)
                    st.subheader("2. Biplot (Visualisasi 2 Dimensi)")
                    pca2 = PCA(n_components=2)
                    comps = pca2.fit_transform(X_scaled)
                    loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)
                    
                    fig_bi = go.Figure()
                    
                    # Titik Observasi
                    fig_bi.add_trace(go.Scatter(x=comps[:,0], y=comps[:,1], mode='markers', name='Observasi', marker=dict(color='#E07A3F', opacity=0.5)))
                    
                    # Panah Variabel
                    scale = 1.0
                    if np.max(comps) > 0: scale = np.max(np.abs(comps)) / np.max(np.abs(loadings))
                    
                    for i, feature in enumerate(pca_cols):
                        fig_bi.add_shape(type='line', x0=0, y0=0, x1=loadings[i,0]*scale, y1=loadings[i,1]*scale, line=dict(color='#4F8190', width=2))
                        fig_bi.add_annotation(x=loadings[i,0]*scale, y=loadings[i,1]*scale, text=feature, showarrow=False, font=dict(color='#4F8190', size=12, weight='bold'))
                    
                    fig_bi.update_layout(
                        title=f"Biplot (PC1: {var_exp[0]*100:.1f}% & PC2: {var_exp[1]*100:.1f}%)",
                        xaxis_title="PC1", yaxis_title="PC2", plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_bi, use_container_width=True)
                    
                    with st.expander("Cara Membaca Biplot"):
                        st.markdown("""
                        * **Sudut antar panah:** Jika lancip (<90°), variabel berkorelasi positif kuat. Jika tumpul, berkorelasi negatif. Jika siku-siku (90°), tidak berkorelasi.
                        * **Panjang panah:** Semakin panjang, semakin besar kontribusi variabel tersebut pada komponen utama.
                        * **Posisi titik:** Titik yang searah dengan panah variabel memiliki nilai tinggi pada variabel tersebut.
                        """)

        # =================================================================
        # TAB 9: KORELASI KANONIK (MODUL 12)
        # =================================================================
        with tabs[8]:
            st.header("Modul 12: Analisis Korelasi Kanonik")
            st.info("Mencari hubungan antara Dua Himpunan Variabel (Set X vs Set Y).")
            
            col_cc1, col_cc2 = st.columns(2)
            with col_cc1:
                st.subheader("Set Variabel 1 (X)")
                set_x = st.multiselect("Pilih Anggota Set X:", numeric_cols, key='cca_x')
            with col_cc2:
                st.subheader("Set Variabel 2 (Y)")
                avail_y = [c for c in numeric_cols if c not in set_x]
                set_y = st.multiselect("Pilih Anggota Set Y:", avail_y, key='cca_y')
            
            if st.button("Hitung Korelasi Kanonik"):
                if len(set_x)>=2 and len(set_y)>=2:
                    X_c = df[set_x].dropna(); Y_c = df[set_y].dropna()
                    common = X_c.index.intersection(Y_c.index)
                    X_c = X_c.loc[common]; Y_c = Y_c.loc[common]
                    
                    n_comp = min(len(set_x), len(set_y))
                    cca = CCA(n_components=n_comp)
                    cca.fit(X_c, Y_c)
                    X_trans, Y_trans = cca.transform(X_c, Y_c)
                    
                    corrs = [np.corrcoef(X_trans[:, i], Y_trans[:, i])[0, 1] for i in range(n_comp)]
                    
                    st.markdown("### Hasil Korelasi Kanonik")
                    st.metric("Korelasi Kanonik Tertinggi", f"{max(corrs):.4f}")
                    
                    canon_df = pd.DataFrame({'Variat Ke-': range(1, n_comp+1), 'Korelasi': corrs, 'Squared Corr': np.square(corrs)})
                    st.dataframe(canon_df)
                    
                    interp = "Sangat Kuat" if max(corrs)>0.7 else "Kuat" if max(corrs)>0.5 else "Lemah"
                    st.success(f"**Interpretasi:** Hubungan antara himpunan X dan himpunan Y tergolong **{interp}**.")
                else:
                    st.error("Pilih minimal 2 variabel untuk setiap set.")

        # =================================================================
        # TAB 10: CLUSTER & KLASIFIKASI (MODUL 11 & 13)
        # =================================================================
        with tabs[9]:
            st.header("Modul 11 & 13: Cluster & Klasifikasi")
            
            opsi_cls = st.radio("Pilih Metode:", ["K-Means Clustering (Pengelompokan)", "Linear Discriminant Analysis (Klasifikasi)"], horizontal=True)
            
            if opsi_cls == "K-Means Clustering (Pengelompokan)":
                st.subheader("K-Means Clustering")
                cl_vars = st.multiselect("Variabel Cluster:", numeric_cols, key='km_v')
                
                if len(cl_vars) >= 2:
                    X_cl = df[cl_vars].dropna()
                    scaler = StandardScaler()
                    X_std = scaler.fit_transform(X_cl)
                    
                    # 1. Analisis Pendukung: ELBOW METHOD
                    st.markdown("#### 1️⃣ Analisis Pendukung: Metode Elbow")
                    st.write("Grafik ini membantu Anda memilih jumlah cluster terbaik (pilih titik siku).")
                    
                    inertias = []
                    K_range = range(1, 11)
                    for k in K_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(X_std)
                        inertias.append(km.inertia_)
                    
                    fig_elb = px.line(x=list(K_range), y=inertias, markers=True, title="Elbow Method", labels={'x':'Jumlah Cluster (k)', 'y':'Inersia'})
                    st.plotly_chart(fig_elb, use_container_width=True)
                    
                    # 2. Clustering Process
                    st.markdown("#### 2️⃣ Proses Clustering")
                    n_k = st.slider("Pilih Jumlah Cluster:", 2, 10, 3)
                    
                    if st.button("Bentuk Cluster"):
                        km_final = KMeans(n_clusters=n_k, random_state=42, n_init=10)
                        labels = km_final.fit_predict(X_std)
                        
                        df_res = X_cl.copy()
                        df_res['Cluster'] = labels
                        df_res['Cluster'] = df_res['Cluster'].astype(str)
                        
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            fig_res = px.scatter(df_res, x=cl_vars[0], y=cl_vars[1], color='Cluster', title="Hasil Cluster (2 Variabel Pertama)", color_discrete_sequence=px.colors.qualitative.Bold)
                            st.plotly_chart(fig_res, use_container_width=True)
                        with c2:
                            st.write("Rata-rata Profil Cluster:")
                            st.dataframe(df_res.groupby('Cluster').mean().style.highlight_max(axis=0, color='lightgreen'))

            else: # LDA
                st.subheader("Linear Discriminant Analysis (LDA)")
                st.info("Memprediksi kategori berdasarkan variabel numerik.")
                
                ld_target = st.selectbox("Target Kategori:", categorical_cols, key='ld_t')
                ld_preds = st.multiselect("Prediktor Numerik:", numeric_cols, key='ld_p')
                
                if st.button("Jalankan LDA"):
                    if ld_target and len(ld_preds)>0:
                        try:
                            dfl = df[[ld_target]+ld_preds].dropna()
                            X = dfl[ld_preds]
                            y = dfl[ld_target]
                            
                            lda = LinearDiscriminantAnalysis()
                            lda.fit(X, y)
                            preds = lda.predict(X)
                            acc = np.mean(preds == y)
                            
                            st.metric("Akurasi Model", f"{acc*100:.2f}%")
                            
                            st.write("**Koefisien Fungsi Diskriminan (Bobot Pembeda):**")
                            coef_df = pd.DataFrame(lda.scalings_, index=ld_preds, columns=[f'LD{i+1}' for i in range(lda.scalings_.shape[1])])
                            st.dataframe(coef_df.style.background_gradient(cmap='RdYlGn'))
                            
                            st.success(f"Variabel-variabel ini mampu membedakan kategori '{ld_target}' dengan akurasi **{acc*100:.2f}%**.")
                        except Exception as e:
                            st.error(f"Error (Pastikan target minimal 2 kategori): {e}")