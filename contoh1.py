import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import openpyxl 
import os
import base64 # PENTING: Untuk mengubah gambar jadi kode agar bisa masuk Navbar

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
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Statistik PKL 65")

# --- FUNGSI BANTUAN: GAMBAR KE BASE64 (UTK NAVBAR) ---
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# =================================================================
# 2. SETUP GAMBAR & ASET
# =================================================================
# Ganti nama file BPS dan STIS sesuai file Anda
IMG_BPS  = 'gambar/logo_bps.png'   
IMG_STIS = 'gambar/logo_stis.png'  
IMG_PKL  = 'gambar/image_14.png'   

FULL_IMG   = 'gambar/Full.jpg'        
MASCOT_IMG = 'gambar/image_10.png' # Maskot Sidebar
HANDS_IMG  = 'gambar/image_13.png'    
FOOTER_IMG = 'gambar/image_15.png'
PAGE_MASCOT = 'gambar/image_4cfe77.png' # Maskot Kecil di Page

# Convert Gambar ke Base64 untuk Navbar
b64_bps  = get_img_as_base64(IMG_BPS)
b64_stis = get_img_as_base64(IMG_STIS)
b64_pkl  = get_img_as_base64(IMG_PKL)

# =================================================================
# 3. CUSTOM CSS (NAVBAR CUSTOM 3 LOGO)
# =================================================================
st.markdown(f"""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rakkas&family=Playfair+Display:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    /* --- WARNA DARI GRAND DESIGN --- */
    :root {{
        --base-cream: #FDF8E4;
        --base-terracotta: #E07A3F;
        --comp-gold: #F2C94C;
        --comp-teal: #4F8190;
        --comp-olive: #739159;
        --text-dark: #4A3B32; 
        --sidebar-bg: #FFFFFF;       
    }}

    /* 1. SETUP BACKGROUND UTAMA */
    .stApp {{
        background-color: var(--base-cream);
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }}

    /* --- NAVBAR CUSTOM (3 LOGO DI TENGAH) --- */
    /* Kita menyembunyikan header asli Streamlit sebagian, dan menimpa dengan ini */
    .custom-navbar {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 5rem;
        background-color: rgba(253, 248, 228, 0.98); /* Cream Solid/Transparan */
        border-bottom: 4px solid var(--base-terracotta);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        z-index: 999990; /* Di bawah toolbar dots */
        display: flex;
        justify-content: center; /* Tengah Horizontal */
        align-items: center;     /* Tengah Vertikal */
        gap: 30px;               /* Jarak antar logo */
    }}
    
    .custom-navbar img {{
        height: 3.5rem; /* Tinggi Logo */
        width: auto;
        transition: transform 0.3s ease;
    }}
    
    .custom-navbar img:hover {{
        transform: scale(1.1);
    }}

    /* Sembunyikan dekorasi header bawaan agar tidak tumpang tindih */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    /* --- TOOLBAR (TOMBOL SETTING) --- */
    [data-testid="stToolbar"] {{
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
        z-index: 999999; /* Paling Atas */
        right: 1rem;
        top: 1.5rem;
        background-color: rgba(255,255,255,0.5) !important;
        border-radius: 5px;
    }}
    [data-testid="stToolbar"] button {{
        color: var(--base-terracotta) !important;
    }}

    /* 2. HEADER PADDING (KONTEN UTAMA) */
    .block-container {{
        padding-top: 7rem !important; /* Jarak aman agar tidak ketutup Navbar */
        padding-bottom: 2rem;
        max-width: 100%;
    }}

    /* 3. TYPOGRAPHY */
    h1 {{
        font-family: 'Rakkas', cursive !important;
        color: var(--base-terracotta) !important;
        font-weight: 400 !important;
        font-size: 3rem !important;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    h2, h3, h4, .streamlit-expanderHeader {{
        font-family: 'Playfair Display', serif !important;
        color: var(--base-terracotta) !important;
        font-weight: 700 !important;
    }}

    /* 4. SIDEBAR */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg);
        border-right: 3px solid var(--base-terracotta);
        box-shadow: 4px 0 10px rgba(0,0,0,0.05);
    }}
    
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 6rem !important;
        padding-bottom: 2rem !important;
    }}

    section[data-testid="stSidebar"] h1 {{
        font-size: 1.8rem !important;
        color: var(--base-terracotta) !important;
        text-align: center;
        margin-top: 0 !important; 
    }}
    
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {{
        color: var(--text-dark) !important;
    }}

    /* 5. TOMBOL */
    .stButton > button {{
        background: linear-gradient(to right, var(--comp-gold), var(--base-terracotta)) !important;
        color: #FFF !important;
        border: none !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15) !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2) !important;
        background: linear-gradient(to right, var(--base-terracotta), var(--comp-gold)) !important;
    }}

    /* 6. TAB MENU STYLE */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 2px solid var(--comp-teal);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: var(--comp-teal);
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--base-terracotta) !important;
        border-bottom: 3px solid var(--base-terracotta) !important;
    }}

    /* 7. WIDGET INPUT */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div, .stSlider > div > div {{
        background-color: #FFFFFF !important;
        border-color: var(--comp-gold) !important;
        color: var(--text-dark) !important;
    }}
    .stMultiSelect div[data-baseweb="select"] span {{
        color: var(--text-dark) !important;
    }}
    .stMultiSelect div[data-baseweb="tag"] {{
        background-color: var(--base-terracotta) !important;
        color: white !important;
    }}
    
    /* 8. FILE UPLOADER */
    div[data-testid="stFileUploader"] {{
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        border: 1px dashed var(--base-terracotta);
    }}
    div[data-testid="stFileUploader"] section {{
        background-color: #FDF1D6 !important; 
    }}
    div[data-testid="stFileUploader"] span {{
        color: var(--text-dark) !important;
    }}
    
    /* 9. BOX & ALERT STYLING */
    .welcome-box {{
        background-color: #FFFFFF;
        padding: 25px;
        border-left: 6px solid var(--base-terracotta);
        border-top: 2px solid var(--comp-gold);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-radius: 12px;
        margin-bottom: 25px;
    }}
    .stAlert {{
        background-color: #FFFFFF;
        border: 1px solid var(--comp-gold);
        color: var(--text-dark);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    .stSuccess {{ border-left-color: var(--comp-olive) !important; }}
    .stInfo {{ border-left-color: var(--comp-teal) !important; }}
    .stWarning {{ border-left-color: var(--comp-gold) !important; }}
    .stError {{ border-left-color: #C0392B !important; }}

    /* 10. TEXT OVERRIDE */
    p, label, span, div {{
        color: var(--text-dark);
    }}
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
</style>

<div class="custom-navbar">
    <img src="data:image/png;base64,{b64_bps}" title="BPS">
    <img src="data:image/png;base64,{b64_stis}" title="Politeknik Statistika STIS">
    <img src="data:image/png;base64,{b64_pkl}" title="PKL 65">
</div>
""", unsafe_allow_html=True)

# Helper Maskot Kecil
def show_page_mascot():
    if os.path.exists(PAGE_MASCOT):
        c1, c2 = st.columns([1, 8])
        with c1:
            st.image(PAGE_MASCOT, width=120) 
        st.write("") 

# =================================================================
# 4. FUNGSI LOAD DATA MAP (PERBAIKAN NAMA WILAYAH LENGKAP)
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

    # FIX TYPO KABUPATEN
    if 'kabupaten' in combined_df.columns:
        combined_df['kabupaten'] = combined_df['kabupaten'].astype(str)
        combined_df['kabupaten'] = combined_df['kabupaten'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
        combined_df['kabupaten'] = combined_df['kabupaten'].str.title().str.strip()
        combined_df['kabupaten'] = combined_df['kabupaten'].replace({
            'Gunungkidul': 'Gunung Kidul',
            'Yogya': 'Yogyakarta',
            'Yogyakartakarta': 'Yogyakarta'
        })

    # FIX TYPO KECAMATAN (MAPPING AGAR COCOK DENGAN SHP)
    if 'kecamatan' in combined_df.columns:
        combined_df['kecamatan'] = combined_df['kecamatan'].astype(str).str.title().str.strip()
        
        combined_df['kecamatan'] = combined_df['kecamatan'].replace({
            # GUNUNG KIDUL (SHP: PAKAI SPASI)
            'Gedangsari': 'Gedang Sari',
            'Girisubo': 'Giri Subo',
            'Karangmojo': 'Karang Mojo',
            'Nglipar': 'Ngli Par',
            'Paliyan': 'Pali Yan',
            'Purwosari': 'Purwo Sari',
            'Rongkop': 'Rong Kop',
            'Saptosari': 'Sapto Sari',
            'Tanjungsari': 'Tanjung Sari',
            'Wonosari': 'Wono Sari',
            
            # KULON PROGO (SHP: PAKAI SPASI)
            'Girimulyo': 'Giri Mulyo',
            'Kalibawang': 'Kali Bawang',
            'Samigaluh': 'Sami Galuh',
            
            # BANTUL (SHP: Bambang Lipuro PAKAI SPASI)
            'Bambanglipuro': 'Bambang Lipuro',
            
            # KOTA YOGYAKARTA (SHP: SAMBUNG)
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

    # AUTO-CORRECT KABUPATEN
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
# 5. FUNGSI BANTUAN STATISTIK
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
# 6. SIDEBAR (HANYA MASKOT)
# =================================================================
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # HANYA MASKOT
    if os.path.exists(MASCOT_IMG):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.image(MASCOT_IMG, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("<h1>KONTROL PANEL</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload File Data (Format CSV/Excel)", type=['csv', 'xls', 'xlsx'])

# =================================================================
# 7. LOGIKA UTAMA (MAIN PAGE)
# =================================================================

# === JIKA BELUM UPLOAD FILE (TAMPILKAN PETA) ===
if uploaded_file is None:
    st.markdown("<h1>DASHBOARD SEBARAN LOKASI PKL 65<br>D.I. YOGYAKARTA</h1>", unsafe_allow_html=True)
    
    # --- GAMBAR 'FULL' DI ATAS BOX WELCOME ---
    if os.path.exists(FULL_IMG):
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
            final_df = df_map[
                (df_map['kabupaten'].isin(selected_kab)) & 
                (df_map['kecamatan'].isin(selected_kec))
            ]
            
            # 1. Bersihkan Nama Kabupaten di SHP
            gdf_shape['kab_upper'] = gdf_shape[SHP_COL_KAB].astype(str).str.title().str.strip()
            gdf_shape['kab_upper'] = gdf_shape['kab_upper'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
            gdf_shape['kab_upper'] = gdf_shape['kab_upper'].replace({
                'Gunungkidul': 'Gunung Kidul',
                'Yogya': 'Yogyakarta',
                'Yogyakartakarta': 'Yogyakarta'
            })
            
            # 2. Bersihkan Nama Kecamatan di SHP
            gdf_shape['kec_upper'] = gdf_shape[SHP_COL_KEC].astype(str).str.title().str.strip()
            gdf_shape['kec_upper'] = gdf_shape['kec_upper'].str.replace(r'^(Kec\.?|Kecamatan|Kapanewon|Kemantren)\s+', '', regex=True)

            final_gdf = gdf_shape[
                (gdf_shape['kab_upper'].isin(selected_kab)) &
                (gdf_shape['kec_upper'].isin(selected_kec))
            ]

            if not final_gdf.empty:
                stats_kec = final_df.groupby('kecamatan').size().reset_index(name='jumlah_lokasi')
                gdf_viz = final_gdf.merge(stats_kec, left_on='kec_upper', right_on='kecamatan', how='left')
                gdf_viz['jumlah_lokasi'] = gdf_viz['jumlah_lokasi'].fillna(0)

                bounds = final_gdf.total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

                if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                    cp = folium.Choropleth(
                        geo_data=gdf_viz,
                        name='Kepadatan Wilayah',
                        data=gdf_viz,
                        columns=['kec_upper', 'jumlah_lokasi'],
                        key_on='feature.properties.kec_upper',
                        fill_color='YlOrRd', 
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='Jumlah Lokasi (Frekuensi)',
                        highlight=True
                    ).add_to(m)
                    folium.GeoJsonTooltip(fields=['kec_upper', 'jumlah_lokasi'], aliases=['Kecamatan:', 'Jumlah Data:'], localize=True).add_to(cp.geojson)

                if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                    heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: '#FFD700', 0.65: '#FF8C00', 1: '#8B0000'}).add_to(m)

                folium.LayerControl().add_to(m)
                
                st.markdown('<div style="box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-radius: 12px; overflow: hidden; border: 3px solid #E07A3F;">', unsafe_allow_html=True)
                st_folium(m, width=1200, height=650)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Lihat Tabel Data Detail untuk Wilayah Terpilih"): 
                    st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("Wilayah pada file SHP tidak ditemukan.")
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
            st.markdown("---")

            if not numeric_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                st.subheader("Analisis Univariat")
                col1, col2 = st.columns(2)
                with col1:
                    hist_col = st.selectbox("Pilih variabel untuk Histogram:", numeric_cols, key='hist_col')
                    if hist_col:
                        fig_hist = px.histogram(df, x=hist_col, title=f'Histogram untuk {hist_col}', marginal="box", color_discrete_sequence=['#E07A3F'])
                        fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    norm_col = st.selectbox("Pilih variabel untuk Uji Normalitas:", numeric_cols, key='norm_col')
                    if st.button("Jalankan Uji Normalitas", key='norm_btn'):
                        data_to_test = df[norm_col].dropna()
                        if len(data_to_test) < 3:
                            st.error("Uji Normalitas memerlukan setidaknya 3 sampel.")
                        else:
                            stat, p_value = stats.shapiro(data_to_test)
                            st.write(f"**P-value:** `{p_value:.4f}`")
                            if p_value > 0.05:
                                st.success(f"**Kesimpulan:** Hasil Anda **NORMAL**.")
                            else:
                                st.error(f"**Kesimpulan:** Hasil Anda **TIDAK NORMAL**.")
                
                st.markdown("---")
                st.subheader("Analisis Bivariat")
                col3, col4 = st.columns([1, 2])
                with col3:
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
                        st.write(f"Koefisien: `{corr:.4f}` | P-value: `{p_value:.4f}`")

                st.markdown("---")
                st.write("**Uji T (Kategorikal vs Numerik)**")
                col5, col6 = st.columns([1, 2])
                with col5:
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
                                st.write(f"P-value: `{p:.4f}`")
                        else: st.warning("Uji T hanya untuk 2 kelompok.")

                st.markdown("---")
                st.subheader("Perbandingan Rata-rata (Error Bar)")
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
            st.markdown("---")
            if not numeric_cols: st.error("Butuh data numerik.")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    ry = st.multiselect("Y:", numeric_cols, key='reg_y_list')
                    rx = st.multiselect("X:", [c for c in numeric_cols if c not in ry], key='reg_x')
                
                if len(ry) == 1 and len(rx) == 1:
                    with c2:
                        d = df[[ry[0]] + rx].dropna()
                        model = LinearRegression().fit(d[rx], d[ry[0]])
                        y_pred = model.predict(d[rx])
                        fig = px.scatter(d, x=rx[0], y=ry[0], trendline='ols', color_discrete_sequence=['#F2C94C'])
                        fig.update_traces(marker=dict(color='#E07A3F'))
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(f"R2 Score: {r2_score(d[ry[0]], y_pred):.4f}")

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
                        
                        st.write("**2. Heteroskedastisitas (Breusch-Pagan)**")
                        bp = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                        st.write(f"P-value: `{bp[1]:.4f}`")
                        if bp[1] < 0.05: st.error("Terjadi Heteroskedastisitas.")
                        else: st.success("Aman (Homoskedastisitas).")

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
            if numeric_cols and categorical_cols:
                c1, c2 = st.columns([1, 2])
                with c1:
                    ac = st.selectbox("Grup:", categorical_cols, key='a1_cat')
                    an = st.selectbox("Nilai:", numeric_cols, key='a1_num')
                with c2:
                    if st.button("Proses ANOVA", key='a1_btn'):
                        d = df[[ac, an]].dropna()
                        d.columns = ['C', 'N']
                        res = sm.stats.anova_lm(smf.ols('N ~ C(C)', data=d).fit(), typ=2)
                        st.dataframe(res)

        # -------------------------------------------------------------
        # TAB 5: MANOVA
        # -------------------------------------------------------------
        with tab_manova:
            show_page_mascot() 
            st.header("MANOVA")
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
            if len(numeric_cols) >= 2:
                ds = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
                c1, c2 = st.columns([1, 2])
                with c1:
                    pv = st.multiselect("Var PCA:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key='pca_vars')
                with c2:
                    if len(pv) >= 2 and st.button("Hitung PCA", key='pca_btn'):
                        pca = PCA().fit(ds[pv])
                        st.bar_chart(pca.explained_variance_ratio_)

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

# =================================================================
# 8. FOOTER IMAGE
# =================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
if os.path.exists(HANDS_IMG):
    with col_h2:
        st.image(HANDS_IMG, use_container_width=True)

if os.path.exists(FOOTER_IMG):
    st.image(FOOTER_IMG, use_container_width=True)