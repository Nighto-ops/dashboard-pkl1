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
# 1. KONFIGURASI HALAMAN & CSS GRAND DESIGN PKL 65
# =================================================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Statistik PKL 65")

st.markdown("""
<style>
    /* IMPORT FONTS: RAKKAS (Headline), PLAYFAIR (Sub), POPPINS (Body) */
    @import url('https://fonts.googleapis.com/css2?family=Rakkas&family=Playfair+Display:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    /* --- PALET WARNA GRAND DESIGN --- */
    :root {
        --base-cream: #FDF8E4;
        --base-terracotta: #E07A3F;
        --comp-gold: #F2C94C;
        --comp-teal: #4F8190;
        --comp-olive: #739159;
        --text-dark: #4A3B32; 
    }

    /* SETUP BACKGROUND UTAMA */
    .stApp {
        background-color: var(--base-cream);
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }

    /* HEADER PADDING (Agar Header Gambar Pas) */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem;
        max-width: 100%;
    }

    /* --- FIX: TOMBOL TOOLBAR/THEME (POJOK KANAN ATAS) --- */
    /* Z-Index tinggi agar tidak tertutup gambar header */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
        z-index: 99999999 !important; 
        right: 2rem;
        top: 1rem;
        background-color: rgba(253, 248, 228, 0.85); /* Background transparan */
        border-radius: 8px;
        padding: 4px;
        border: 1px solid var(--comp-gold);
    }
    
    /* TYPOGRAPHY */
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

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #FDF1D6;
        border-right: 2px solid var(--base-terracotta);
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        color: var(--base-terracotta) !important;
        text-align: center;
    }
    /* Fix teks sidebar agar terbaca di mode apapun */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: var(--text-dark) !important;
    }

    /* TOMBOL (GRADASI GOLD -> TERRACOTTA) */
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

    /* TABS MENU */
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

    /* WIDGET INPUT (FIX DARK MODE) */
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
    
    /* FILE UPLOADER */
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
    
    /* WELCOME BOX & ALERTS */
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

    /* TEXT COLOR OVERRIDE GLOBAL */
    p, label, span, div {
        color: var(--text-dark);
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. SETUP GAMBAR & ASET
# =================================================================
HEADER_IMG = 'gambar/header.png'      
FULL_IMG   = 'gambar/Full.jpg'        
LOGO_IMG   = 'gambar/image_14.png'    
MASCOT_IMG = 'gambar/image_4cfe77.png' # Maskot di dalam page
HANDS_IMG  = 'gambar/image_11.png'    
FOOTER_IMG = 'gambar/image_15.png'    

# --- HEADER (CENTERED) ---
if os.path.exists(HEADER_IMG):
    col_h1, col_h2, col_h3 = st.columns([1, 10, 1])
    with col_h2:
        st.image(HEADER_IMG, use_container_width=True)

# Fungsi Helper: Menampilkan Maskot Kecil di Tiap Tab
def show_page_mascot():
    if os.path.exists(MASCOT_IMG):
        c1, c2 = st.columns([1, 8])
        with c1:
            st.image(MASCOT_IMG, width=120) 
        st.write("") 

# =================================================================
# 3. FUNGSI LOAD DATA MAP (SOLUSI NO-SPACE / MERATAKAN NAMA)
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

    # 1. BERSIHKAN KABUPATEN
    if 'kabupaten' in combined_df.columns:
        # Standarisasi teks biasa
        combined_df['kabupaten'] = combined_df['kabupaten'].astype(str).str.title().str.strip()
        combined_df['kabupaten'] = combined_df['kabupaten'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
        # Normalisasi typo umum
        combined_df['kabupaten'] = combined_df['kabupaten'].replace({
            'Yogya': 'Yogyakarta',
            'Yogyakartakarta': 'Yogyakarta'
        })
        # BUAT KOLOM KUNCI (HURUF BESAR & TANPA SPASI)
        combined_df['kab_key'] = combined_df['kabupaten'].str.upper().str.replace(" ", "")

    # 2. BERSIHKAN KECAMATAN
    if 'kecamatan' in combined_df.columns:
        combined_df['kecamatan'] = combined_df['kecamatan'].astype(str).str.title().str.strip()
        combined_df['kecamatan'] = combined_df['kecamatan'].str.replace(r'^(Kec\.?|Kecamatan|Kapanewon|Kemantren)\s+', '', regex=True)
        
        # BUAT KOLOM KUNCI (HURUF BESAR & TANPA SPASI)
        combined_df['kec_key'] = combined_df['kecamatan'].str.upper().str.replace(" ", "")

    # 3. AUTO-CORRECT KABUPATEN (MANTRIJERON FIX)
    # Kita cek berdasarkan 'kec_key' biar aman dari typo spasi
    kec_kota_jogja_keys = [
        'DANUREJAN', 'GEDONGTENGEN', 'GONDOKUSUMAN', 'GONDOMANAN', 
        'JETIS', 'KOTAGEDE', 'KRATON', 'KERATON', 'MANTRIJERON', 'MERGANGSAN', 
        'NGAMPILAN', 'PAKUALAMAN', 'TEGALREJO', 'UMBULHARJO', 'WIROBRAJAN'
    ]
    
    if 'kabupaten' in combined_df.columns and 'kec_key' in combined_df.columns:
        # Jika kecamatannya ada di list kota jogja, ubah kabupaten jadi Yogyakarta
        # Dan update kab_key nya juga jadi YOGYAKARTA
        mask = combined_df['kec_key'].isin(kec_kota_jogja_keys)
        combined_df.loc[mask, 'kabupaten'] = 'Yogyakarta'
        combined_df.loc[mask, 'kab_key'] = 'YOGYAKARTA'

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
# 4. FUNGSI BANTUAN STATISTIK
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
# 5. SIDEBAR (LOGO SAJA)
# =================================================================
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # HANYA LOGO di Sidebar
    if os.path.exists(LOGO_IMG):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.image(LOGO_IMG, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("<h1>KONTROL PANEL</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload File Data (Format CSV/Excel)", type=['csv', 'xls', 'xlsx'])

# =================================================================
# 6. LOGIKA UTAMA (MAIN PAGE)
# =================================================================

# === JIKA BELUM UPLOAD FILE (TAMPILKAN PETA) ===
if uploaded_file is None:
    st.markdown("<h1>DASHBOARD SEBARAN LOKASI PKL 65<br>D.I. YOGYAKARTA</h1>", unsafe_allow_html=True)
    
    # --- GAMBAR 'FULL' DI ATAS BOX WELCOME ---
    if os.path.exists(FULL_IMG):
        # Center Full Image
        c1, c2, c3 = st.columns([1, 2, 1]) 
        with c2:
            st.image(FULL_IMG, use_container_width=True)
    
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
            # Filter Data Excel
            final_df = df_map[
                (df_map['kabupaten'].isin(selected_kab)) & 
                (df_map['kecamatan'].isin(selected_kec))
            ]
            
            # 1. SIAPKAN SHP - BUAT KOLOM KUNCI YANG SAMA (NO SPACE)
            # Kita tidak mengubah tampilan asli (nmkab/nmkec), tapi buat kolom baru untuk joining
            gdf_shape['kab_key'] = gdf_shape[SHP_COL_KAB].astype(str).str.upper().str.replace(" ", "").str.replace(r'^(KAB\.?|KABUPATEN|KOTA)\s+', '', regex=True)
            gdf_shape['kec_key'] = gdf_shape[SHP_COL_KEC].astype(str).str.upper().str.replace(" ", "")
            
            # Bersihkan typo khusus di SHP key jika perlu
            gdf_shape['kab_key'] = gdf_shape['kab_key'].replace({'GUNUNGKIDUL': 'GUNUNGKIDUL', 'YOGYA': 'YOGYAKARTA'})

            # Ambil key dari pilihan user (yang berasal dari Excel)
            # Kita harus convert pilihan user ke format key juga
            selected_kab_keys = [k.upper().replace(" ", "") for k in selected_kab]
            selected_kec_keys = [k.upper().replace(" ", "") for k in selected_kec]

            # Filter SHP menggunakan KEY (bukan nama cantik)
            final_gdf = gdf_shape[
                (gdf_shape['kab_key'].isin(selected_kab_keys)) &
                (gdf_shape['kec_key'].isin(selected_kec_keys))
            ]

            if not final_gdf.empty:
                # Hitung statistik per kecamatan
                # Group by 'kec_key' agar aman
                stats_kec = final_df.groupby('kec_key').size().reset_index(name='jumlah_lokasi')
                
                # Merge Data Excel ke SHP menggunakan KEY
                gdf_viz = final_gdf.merge(stats_kec, on='kec_key', how='left')
                gdf_viz['jumlah_lokasi'] = gdf_viz['jumlah_lokasi'].fillna(0)
                
                # Agar tooltip tetap cantik, kita pastikan kolom nama asli ada
                # Kita pakai nama dari SHP asli (nmkec) untuk label tampilan
                gdf_viz['Nama Kecamatan'] = gdf_viz[SHP_COL_KEC].astype(str).str.title()

                # Setup Peta
                bounds = final_gdf.total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

                if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                    cp = folium.Choropleth(
                        geo_data=gdf_viz,
                        name='Kepadatan Wilayah',
                        data=gdf_viz,
                        columns=['kec_key', 'jumlah_lokasi'], # Join pakai Key
                        key_on='feature.properties.kec_key',  # Join pakai Key
                        fill_color='YlOrRd', 
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='Jumlah Lokasi',
                        highlight=True
                    ).add_to(m)
                    
                    # Tooltip pakai nama asli yang cantik
                    folium.GeoJsonTooltip(
                        fields=['Nama Kecamatan', 'jumlah_lokasi'], 
                        aliases=['Kecamatan:', 'Jumlah Data:'], 
                        localize=True
                    ).add_to(cp.geojson)

                if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                    heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: '#FFD700', 0.65: '#FF8C00', 1: '#8B0000'}).add_to(m)

                folium.LayerControl().add_to(m)
                
                # Frame Peta
                st.markdown('<div style="box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-radius: 12px; overflow: hidden; border: 3px solid #E07A3F;">', unsafe_allow_html=True)
                st_folium(m, width=1200, height=650)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Lihat Tabel Data Detail untuk Wilayah Terpilih"): 
                    st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("Wilayah tidak ditemukan di Peta. Kemungkinan filter tidak cocok dengan SHP.")
        else:
            st.info("Silakan pilih Kabupaten dan Kecamatan pada panel filter di atas.")
    else:
        st.warning("Data peta (file Excel atau SHP) belum tersedia di folder 'data/'.")

# === JIKA FILE SUDAH DIUPLOAD (MASUK MODE STATISTIK) ===
else:
    df = None
    numeric_cols = []
    categorical_cols = []
    all_cols = []

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File berhasil di-upload.")
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            st.sidebar.subheader("Variabel Teridentifikasi")
            st.sidebar.write(f"**Kolom Numerik:** ({len(numeric_cols)})")
            st.sidebar.caption(f", ".join(numeric_cols))
            st.sidebar.write(f"**Kolom Kategorikal:** ({len(categorical_cols)})")
            st.sidebar.caption(f", ".join(categorical_cols))

    if df is not None and all_cols:
        st.markdown("<h1>ANALISIS STATISTIK DATA PKL</h1>", unsafe_allow_html=True)
        
        tab_data, tab_basic, tab_reg, tab_anova, tab_manova, tab_dim, tab_class = st.tabs([
            "Beranda & Data",
            "Analisis Dasar",
            "Model Regresi",
            "ANOVA",
            "MANOVA",
            "Reduksi Dimensi",
            "Klasifikasi & Clustering"
        ])

        # -------------------------------------------------------------
        # TAB 0: RINGKASAN DATA
        # -------------------------------------------------------------
        with tab_data:
            show_page_mascot() 
            st.header("Ringkasan dan Tampilan Data")
            st.subheader("Ringkasan Statistik")
            st.info("Statistik deskriptif dasar (rata-rata, median, min, max, dll.) untuk semua kolom numerik dalam data Anda.")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik.")
            st.subheader("Data Mentah")
            st.info("Tampilan 50 baris pertama dari data yang Anda upload.")
            st.dataframe(df.head(50), use_container_width=True)

        # -------------------------------------------------------------
        # TAB 1: ANALISIS DASAR
        # -------------------------------------------------------------
        with tab_basic:
            show_page_mascot() 
            st.header("Analisis Dasar (Univariat & Bivariat)")
            st.info("Analisis ini berfokus pada 1 atau 2 variabel pada satu waktu.")
            st.markdown("---")

            if not numeric_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                st.subheader("Analisis Univariat")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("Melihat sebaran frekuensi dari sebuah variabel numerik.")
                    hist_col = st.selectbox("Pilih variabel untuk Histogram:", numeric_cols, key='hist_col')
                    if hist_col:
                        fig_hist = px.histogram(df, x=hist_col, title=f'Histogram untuk {hist_col}', marginal="box", color_discrete_sequence=['#E07A3F'])
                        fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    st.info("Menguji apakah data terdistribusi normal (simetris).")
                    norm_col = st.selectbox("Pilih variabel untuk Uji Normalitas:", numeric_cols, key='norm_col')
                    if st.button("Jalankan Uji Normalitas", key='norm_btn'):
                        data_to_test = df[norm_col].dropna()
                        if len(data_to_test) < 3:
                            st.error("Uji Normalitas memerlukan setidaknya 3 sampel.")
                        else:
                            stat, p_value = stats.shapiro(data_to_test)
                            st.write(f"**P-value:** `{p_value:.4f}`")
                            
                            with st.expander("Lihat Penjelasan"):
                                st.write("**Rule:** P-value > 0.05 artinya Normal.")
                                if p_value > 0.05:
                                    st.success(f"**Kesimpulan:** Hasil Anda **NORMAL**.")
                                else:
                                    st.error(f"**Kesimpulan:** Hasil Anda **TIDAK NORMAL**.")
                
                st.markdown("---")
                st.subheader("Analisis Bivariat")
                col3, col4 = st.columns([1, 2])
                with col3:
                    st.info("Pilih dua variabel numerik untuk melihat hubungan linear.")
                    bi_x = st.selectbox("Pilih Variabel X:", numeric_cols, key='bi_x')
                    bi_y = st.selectbox("Pilih Variabel Y:", numeric_cols, key='bi_y')
                with col4:
                    if bi_x and bi_y and bi_x != bi_y:
                        data_bi = df[[bi_x, bi_y]].dropna()
                        fig_scatter = px.scatter(data_bi, x=bi_x, y=bi_y, title=f"{bi_y} vs {bi_x}", trendline="ols", color_discrete_sequence=['#F2C94C'])
                        fig_scatter.update_traces(marker=dict(color='#E07A3F'))
                        fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                        st.write(f"**Koefisien Korelasi:** `{corr:.4f}` ({interpret_correlation(corr)}) | **P-value:** `{p_value:.4f}`")

                st.markdown("---")
                st.write("**Uji T (Kategorikal vs Numerik)**")
                col5, col6 = st.columns([1, 2])
                with col5:
                    st.info("Membandingkan rata-rata numerik di antara DUA kelompok.")
                    if categorical_cols:
                        cat_col_t = st.selectbox("Kelompok (Kat):", categorical_cols, key='bi_cat_t')
                        num_col_t = st.selectbox("Nilai (Num):", numeric_cols, key='bi_num_t')
                    else: st.warning("Tidak ada data kategorikal.")
                with col6:
                    if categorical_cols and cat_col_t and num_col_t:
                        groups_t = df[cat_col_t].dropna().unique()
                        if len(groups_t) == 2:
                            if st.button("Jalankan Uji T", key='t_test_btn'):
                                fig_box = px.box(df, x=cat_col_t, y=num_col_t, title=f"Distribusi {num_col_t}", color=cat_col_t, color_discrete_sequence=['#4F8190', '#E07A3F'])
                                fig_box.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                st.plotly_chart(fig_box, use_container_width=True)
                                grp1 = df[df[cat_col_t] == groups_t[0]][num_col_t].dropna()
                                grp2 = df[df[cat_col_t] == groups_t[1]][num_col_t].dropna()
                                s, p = stats.ttest_ind(grp1, grp2)
                                st.write(f"**P-value:** `{p:.4f}`")
                                if p < 0.05: st.success("Perbedaan Signifikan.")
                                else: st.warning("Perbedaan Tidak Signifikan.")
                        else: st.warning("Uji T hanya untuk 2 kelompok.")

                st.markdown("---")
                st.subheader("Perbandingan Rata-rata (Error Bar)")
                st.info("Visualisasi Mean dengan Error Bar (SE/SD).")
                col7, col8 = st.columns([1, 2])
                with col7:
                    if categorical_cols:
                        eb_cat = st.selectbox("Pilih Kelompok:", categorical_cols, key='eb_cat')
                        eb_num = st.selectbox("Pilih Nilai:", numeric_cols, key='eb_num')
                        eb_type = st.radio("Tipe Error:", ["Standar Error (SE)", "Standar Deviasi (SD)"], key='eb_type')
                        eb_plot_type = st.radio("Tipe Plot:", ["Bar Chart", "Line Chart"], key='eb_plot_type')
                    else: st.warning("Butuh kolom kategorikal.")
                with col8:
                    if categorical_cols and eb_cat and eb_num:
                        if st.button("Buat Grafik Error Bar", key='eb_btn'):
                            try:
                                df_agg = df.groupby(eb_cat)[eb_num].agg(['mean', 'std', 'count']).reset_index()
                                df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count'])
                                error_val = df_agg['se'] if eb_type == "Standar Error (SE)" else df_agg['std']
                                
                                fig = go.Figure()
                                plot_color = '#E07A3F'
                                if eb_plot_type == "Line Chart":
                                    fig.add_trace(go.Scatter(x=df_agg[eb_cat], y=df_agg['mean'], error_y=dict(type='data', array=error_val, visible=True), mode='lines+markers', line=dict(color=plot_color)))
                                else:
                                    fig.add_trace(go.Bar(x=df_agg[eb_cat], y=df_agg['mean'], error_y=dict(type='data', array=error_val, visible=True), marker_color=plot_color))
                                
                                fig.update_layout(title=f"Rata-rata {eb_num} per {eb_cat}", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                                st.plotly_chart(fig, use_container_width=True)
                                with st.expander("Lihat Data Agregat"): st.dataframe(df_agg)
                            except: st.error("Gagal membuat grafik.")

        # -------------------------------------------------------------
        # TAB 3: MODEL REGRESI
        # -------------------------------------------------------------
        with tab_reg:
            show_page_mascot() 
            st.header("Model Regresi")
            st.info("Memodelkan hubungan antara variabel dependen (Y) dan independen (X).")
            st.markdown("---")
            if not numeric_cols: st.error("Butuh data numerik.")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    ry = st.multiselect("Y:", numeric_cols, key='reg_y_list')
                    rx = st.multiselect("X:", [c for c in numeric_cols if c not in ry], key='reg_x')
                
                # Regresi Sederhana
                if len(ry) == 1 and len(rx) == 1:
                    with c2:
                        poly_degree = st.radio("Tipe:", [1, 2, 3], format_func=lambda x: f"Order {x}")
                        d = df[[ry[0]] + rx].dropna()
                        
                        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        X_poly = poly.fit_transform(d[rx])
                        model = LinearRegression().fit(X_poly, d[ry[0]])
                        y_pred = model.predict(X_poly)
                        
                        d['pred'] = y_pred
                        d = d.sort_values(by=rx[0])
                        
                        fig = px.scatter(d, x=rx[0], y=ry[0], title=f"Regresi (Order {poly_degree})", color_discrete_sequence=['#F2C94C'])
                        fig.add_trace(go.Scatter(x=d[rx[0]], y=d['pred'], mode='lines', line=dict(color='#E07A3F', width=3)))
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(f"**R2 Score:** {r2_score(d[ry[0]], y_pred):.4f}")
                        
                        with st.expander("Penjelasan"):
                            st.success(f"Model ini menjelaskan {r2_score(d[ry[0]], y_pred)*100:.2f}% variansi data.")

                # Regresi Berganda (OLS Full)
                elif len(ry) == 1 and len(rx) >= 1:
                    with c2:
                        d = df[[ry[0]] + rx].dropna()
                        X = sm.add_constant(d[rx])
                        model_ols = sm.OLS(d[ry[0]], X).fit()
                        
                        st.write("**Ringkasan Model (OLS)**")
                        st.text_area("Summary", model_ols.summary().as_text(), height=400)
                        
                        # Uji Asumsi
                        st.subheader("Uji Asumsi")
                        st.write("**1. Multikolinearitas (VIF)**")
                        vif = pd.DataFrame()
                        vif["Var"] = X.columns
                        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                        st.dataframe(vif[vif["Var"]!='const'])
                        if (vif[vif["Var"]!='const']["VIF"] > 10).any():
                             st.error("VIF > 10 ditemukan (Multikolinearitas).")
                        else: st.success("VIF Aman.")
                        
                        st.write("**2. Heteroskedastisitas (Breusch-Pagan)**")
                        bp = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                        st.write(f"P-value: `{bp[1]:.4f}`")
                        if bp[1] < 0.05: st.error("Terjadi Heteroskedastisitas.")
                        else: st.success("Aman (Homoskedastisitas).")

                # Regresi Multivariat
                elif len(ry) >= 2 and len(rx) >= 1:
                    with c2:
                        try:
                            d = df[ry + rx].dropna()
                            model = LinearRegression().fit(d[rx], d[ry])
                            st.write("**Koefisien Model (Matrix)**")
                            st.dataframe(pd.DataFrame(model.coef_, columns=rx, index=ry))
                            st.write("**R-Squared per Variabel Y**")
                            r2s = r2_score(d[ry], model.predict(d[rx]), multioutput='raw_values')
                            st.dataframe(pd.DataFrame(r2s, index=ry, columns=['R2']))
                        except Exception as e: st.error(f"Error: {e}")

        # -------------------------------------------------------------
        # TAB 4: ANOVA
        # -------------------------------------------------------------
        with tab_anova:
            show_page_mascot() 
            st.header("ANOVA")
            st.info("Membandingkan rata-rata >2 kelompok.")
            if numeric_cols and categorical_cols:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("One-Way / Two-Way")
                    ac1 = st.selectbox("Faktor 1:", categorical_cols, key='a1_cat')
                    ac2 = st.selectbox("Faktor 2 (Opsional):", ["None"] + categorical_cols, key='a2_cat_opt')
                    an = st.selectbox("Nilai Y:", numeric_cols, key='a1_num')
                with c2:
                    if st.button("Proses ANOVA", key='a1_btn'):
                        if ac2 == "None":
                            # One Way
                            d = df[[ac1, an]].dropna()
                            d.columns = ['C', 'N']
                            # Boxplot
                            fig = px.box(d, x='C', y='N', color='C', color_discrete_sequence=['#4F8190', '#E07A3F', '#F2C94C'])
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                            st.plotly_chart(fig)
                            
                            res = sm.stats.anova_lm(smf.ols('N ~ C(C)', data=d).fit(), typ=2)
                            st.dataframe(res)
                            if res['PR(>F)'][0] < 0.05: st.success("Signifikan.")
                            else: st.warning("Tidak Signifikan.")
                        elif ac1 != ac2:
                            # Two Way
                            d = df[[ac1, ac2, an]].dropna()
                            cols = [c.replace(' ','_') for c in d.columns]
                            d.columns = cols
                            f = f"{cols[2]} ~ C({cols[0]}) + C({cols[1]}) + C({cols[0]}):C({cols[1]})"
                            res = sm.stats.anova_lm(smf.ols(f, data=d).fit(), typ=2)
                            st.dataframe(res)

        # -------------------------------------------------------------
        # TAB 5: MANOVA
        # -------------------------------------------------------------
        with tab_manova:
            show_page_mascot() 
            st.header("MANOVA")
            st.info("Membandingkan rata-rata banyak Y berdasarkan kelompok X.")
            c1, c2 = st.columns([1, 2])
            with c1:
                mc = st.selectbox("Grup (X):", categorical_cols, key='m1_c')
                mn = st.multiselect("Y (min 2):", numeric_cols, key='m1_n')
            with c2:
                if mc and len(mn) >= 2 and st.button("Proses MANOVA", key='m1_btn'):
                    d = df[[mc] + mn].dropna()
                    cols = [c.replace(' ','_').replace('.','') for c in d.columns]
                    d.columns = cols
                    try:
                        f = f"{' + '.join(cols[1:])} ~ C({cols[0]})"
                        res = MANOVA.from_formula(f, data=d).mv_test()
                        st.write(res.summary_frame)
                    except Exception as e: st.error(f"Gagal: {e}")

        # -------------------------------------------------------------
        # TAB 6: REDUKSI DIMENSI
        # -------------------------------------------------------------
        with tab_dim:
            show_page_mascot() 
            st.header("PCA & EFA")
            st.info("Mereduksi variabel.")
            if len(numeric_cols) >= 2:
                ds = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
                c1, c2 = st.columns([1, 2])
                with c1:
                    pv = st.multiselect("Var PCA/EFA:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key='pca_vars')
                    n_comp = st.slider("Jumlah Komponen:", 1, len(pv), 2, key='n_pca')
                with c2:
                    if len(pv) >= 2:
                        if st.button("Hitung PCA", key='pca_btn'):
                            pca = PCA(n_components=n_comp).fit(ds[pv])
                            st.write("**Explained Variance Ratio:**")
                            st.bar_chart(pca.explained_variance_ratio_)
                            st.write("**Loadings:**")
                            st.dataframe(pd.DataFrame(pca.components_.T, index=pv))
                        
                        if len(pv) >= 3 and st.button("Hitung EFA", key='efa_btn'):
                            try:
                                kmo_val, _ = calculate_kmo(ds[pv])
                                st.write(f"KMO: `{kmo_val:.3f}`")
                                if kmo_val > 0.5:
                                    fa = FactorAnalyzer(n_factors=n_comp, rotation='varimax').fit(ds[pv])
                                    st.write("**Factor Loadings:**")
                                    st.dataframe(pd.DataFrame(fa.loadings_, index=pv))
                                else: st.warning("KMO < 0.5, data kurang cocok.")
                            except: st.error("EFA Error.")

        # -------------------------------------------------------------
        # TAB 7: KLASIFIKASI & CLUSTERING
        # -------------------------------------------------------------
        with tab_class:
            show_page_mascot() 
            st.header("Clustering & LDA")
            c1, c2 = st.columns([1, 2])
            with c1:
                cv = st.multiselect("Var Cluster:", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))], key='clust_vars')
                k = st.slider("K:", 2, 8, 3, key='k_means')
            with c2:
                if len(cv) >= 2 and st.button("Start K-Means", key='clust_btn'):
                    d = df[cv].dropna()
                    X = StandardScaler().fit_transform(d)
                    d['Cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X).astype(str)
                    fig = px.scatter(d, x=cv[0], y=cv[1], color='Cluster', title=f"K-Means (K={k})", color_discrete_sequence=['#F2C94C', '#4F8190', '#739159'])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("**Cluster Centers:**")
                    st.dataframe(d.groupby('Cluster').mean())
            
            st.divider()
            st.subheader("Linear Discriminant Analysis (LDA)")
            c3, c4 = st.columns([1, 2])
            with c3:
                if categorical_cols:
                    lda_y = st.selectbox("Target (Kat):", categorical_cols, key='lda_y')
                    lda_x = st.multiselect("Prediktor (Num):", numeric_cols, key='lda_x')
            with c4:
                if categorical_cols and lda_y and lda_x:
                    if st.button("Start LDA", key='lda_btn'):
                        try:
                            d = df[[lda_y] + lda_x].dropna()
                            X_lda = StandardScaler().fit_transform(d[lda_x])
                            lda = LinearDiscriminantAnalysis()
                            lda.fit(X_lda, d[lda_y])
                            st.success(f"Akurasi: {lda.score(X_lda, d[lda_y]):.2%}")
                        except: st.error("LDA Gagal.")

# =================================================================
# 7. FOOTER IMAGE
# =================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
if os.path.exists(HANDS_IMG):
    with col_h2:
        st.image(HANDS_IMG, use_container_width=True)

if os.path.exists(FOOTER_IMG):
    st.image(FOOTER_IMG, use_container_width=True)