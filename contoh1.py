import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import openpyxl 
import os
from PIL import Image

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
# KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(
    layout="wide", 
    page_title="Tools Analisis Statistik - PKL STIS",
    page_icon="📊"
)

# =================================================================
# CUSTOM CSS & DESIGN (ELEGAN & BATIK THEME)
# =================================================================
def local_css():
    st.markdown("""
    <style>
        /* IMPOR FONT AGAR LEBIH ELEGAN */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@400;700&display=swap');

        /* 1. BACKGROUND UTAMA (CREAM LEMBUT AGAR MENYATU DENGAN BATIK) */
        .stApp {
            background-color: #FFFCF5;
            font-family: 'Lato', sans-serif;
        }

        /* 2. MENGHILANGKAN PADDING ATAS AGAR HEADER BATIK NEMPEL */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 5rem;
        }

        /* 3. JUDUL (H1, H2, H3) - MENGGUNAKAN FONT SERIF MEWAH */
        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
            color: #8E44AD; /* Di-override di bawah agar sesuai tema */
            color: #A04000 !important; /* Warna Cokelat/Bata Batik */
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        /* 4. SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 2px solid #F5CBA7; /* Garis batas warna peach/emas */
        }

        /* 5. TOMBOL (GRADASI EMAS - ORANYE) */
        .stButton>button {
            background: linear-gradient(90deg, #F39C12 0%, #D35400 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 24px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 8px rgba(0,0,0,0.3);
            color: #fff;
        }

        /* 6. TABS (TAB MENU) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background-color: white;
            padding: 10px;
            border-radius: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 10px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FEF5E7 !important;
            color: #D35400 !important;
            border-bottom: 3px solid #D35400;
        }

        /* 7. DATAFRAME / TABEL */
        div[data-testid="stDataFrame"] {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #F5CBA7;
        }

        /* 8. PESAN (INFO, SUCCESS, WARNING) - STYLE LEBIH SOFT */
        .stAlert {
            border-radius: 10px;
            border-left: 5px solid #D35400;
        }
        
        /* 9. EXPANDER */
        .streamlit-expanderHeader {
            background-color: white;
            border-radius: 10px;
            color: #A04000;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# =================================================================
# HEADER IMAGE DISPLAY (VISUAL)
# =================================================================
# Menampilkan Header Batik di bagian paling atas konten utama
header_path = 'gambar/3.jpg' # Menggunakan gambar header batik yang kamu kirim
if os.path.exists(header_path):
    st.image(header_path, use_container_width=True)
else:
    # Fallback jika file tidak ada di folder gambar
    st.warning("Gambar header (gambar/3.jpg) tidak ditemukan.")

# =================================================================
# FUNGSI BANTUAN (MAP)
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

    # FIX TYPO DI EXCEL
    if 'kabupaten' in combined_df.columns:
        combined_df['kabupaten'] = combined_df['kabupaten'].astype(str)
        combined_df['kabupaten'] = combined_df['kabupaten'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
        combined_df['kabupaten'] = combined_df['kabupaten'].str.title().str.strip()
        combined_df['kabupaten'] = combined_df['kabupaten'].replace({
            'Gunungkidul': 'Gunung Kidul',
            'Yogya': 'Yogyakarta',
            'Yogyakartakarta': 'Yogyakarta'
        })

    if 'kecamatan' in combined_df.columns:
        combined_df['kecamatan'] = combined_df['kecamatan'].astype(str)
        combined_df['kecamatan'] = combined_df['kecamatan'].str.replace(r'^(Kec\.?|Kecamatan|Kapanewon|Kemantren)\s+', '', regex=True)
        combined_df['kecamatan'] = combined_df['kecamatan'].str.title().str.strip()
        
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
# SIDEBAR UTAMA
# =================================================================
with st.sidebar:
    # MENAMPILKAN MASKOT DI SIDEBAR
    mascot_path = 'gambar/Gundatala_Riset 5.jpg'
    logo_path = 'gambar/LOGO-PKL_REV8.jpg'
    
    # Pilih mau nampilin logo atau maskot, atau keduanya
    col_sb1, col_sb2 = st.columns(2)
    if os.path.exists(logo_path):
        with col_sb1:
            st.image(logo_path, use_container_width=True)
    if os.path.exists(mascot_path):
        with col_sb2:
            st.image(mascot_path, use_container_width=True)
    
    st.markdown("---")
    st.title("Kontrol Panel")
    uploaded_file = st.file_uploader("Upload File Anda", type=['csv', 'xls', 'xlsx'])

# =================================================================
# LOGIKA UTAMA
# =================================================================

# === JIKA BELUM UPLOAD FILE (FITUR PETA) ===
if uploaded_file is None:
    # Spacer agar tidak terlalu mepet header gambar
    st.write("") 
    st.title("Dashboard Sebaran Lokasi DIY")
    st.markdown("""
    <div style='background-color: white; padding: 15px; border-radius: 10px; border-left: 6px solid #F39C12; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        Selamat datang di <b>Dashboard Analisis PKL</b>. <br>
        Di sini Anda dapat melihat peta interaktif persebaran lokasi. Untuk melakukan analisis statistik mendalam (Regresi, ANOVA, dll), silakan <b>Upload Data</b> melalui panel di sebelah kiri.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    df_map = load_map_excel_data()
    gdf_shape = load_shp_data()

    if not df_map.empty and gdf_shape is not None:
        
        # FITUR CEK ISI SHP
        with st.expander("🔍 CEK NAMA ASLI DI FILE SHP (KLIK UNTUK MEMBUKA)"):
            st.info("Gunakan ini untuk melihat ejaan asli.")
            if SHP_COL_KAB in gdf_shape.columns:
                unique_kab = gdf_shape[SHP_COL_KAB].unique()
                pilih_kab_cek = st.selectbox("Isi Kolom Kabupaten (SHP):", unique_kab)
                isi_kec = gdf_shape[gdf_shape[SHP_COL_KAB] == pilih_kab_cek][SHP_COL_KEC].unique()
                st.write(f"Isi Kecamatan untuk '{pilih_kab_cek}':")
                st.write(isi_kec)

        # FILTER PETA (Dibuat dalam Container agar rapi)
        with st.container():
            st.subheader("🛠️ Filter Wilayah")
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
                map_mode = st.radio("3. Tampilan Peta:", ["Gabungan", "Choropleth (Wilayah)", "Heatmap (Titik)"], horizontal=True)

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
            
            # --- FIX TYPO / SPASI (SHP -> EXCEL) ---
            gdf_shape['kec_upper'] = gdf_shape['kec_upper'].replace({
                'Gedang Sari': 'Gedangsari', # <--- FIX GEDANG SARI
                'Sapto Sari': 'Saptosari',    
                'Karang Mojo': 'Karangmojo', 
                'Giri Subo': 'Girisubo',
                'Purwo Sari': 'Purwosari',
                'Gondo Kusuman': 'Gondokusuman',
                'Gondo Manan': 'Gondomanan',
                'Danu Rejan': 'Danurejan',
                'Mer Gangsan': 'Mergangsan',
                'Gedong Tengen': 'Gedongtengen',
                'Umbul Harjo': 'Umbulharjo',
                'Paku Alaman': 'Pakualaman',
                'Kota Gede': 'Kotagede',
                'Mantri Jeron': 'Mantrijeron',
                'Wiro Brajan': 'Wirobrajan',
                'Tegal Rejo': 'Tegalrejo'
            })

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
                m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron") # Peta dasar bersih

                if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                    cp = folium.Choropleth(
                        geo_data=gdf_viz,
                        name='Kepadatan Wilayah',
                        data=gdf_viz,
                        columns=['kec_upper', 'jumlah_lokasi'],
                        key_on='feature.properties.kec_upper',
                        fill_color='YlOrRd', # Warna Oranye Merah sesuai tema
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='Jumlah Lokasi',
                        highlight=True
                    ).add_to(m)
                    folium.GeoJsonTooltip(fields=['kec_upper', 'jumlah_lokasi'], aliases=['Kecamatan:', 'Jumlah Data:'], localize=True).add_to(cp.geojson)

                if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                    heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

                folium.LayerControl().add_to(m)
                
                # Container untuk Peta agar ada shadow
                st.markdown("<div style='box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden;'>", unsafe_allow_html=True)
                st_folium(m, width=1200, height=600)
                st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("Lihat Data Tabel Detail"): st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("Wilayah SHP tidak ditemukan. Cek ejaan di fitur 'Cek Nama SHP' di atas.")
        else:
            st.info("Silakan pilih Kabupaten dan Kecamatan di atas untuk menampilkan peta.")
    else:
        st.warning("Data peta (Excel/SHP) belum siap di folder data/. Pastikan file .xlsx dan .shp ada.")

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
            st.sidebar.success("File berhasil di-upload!")
            
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
        st.title("Analisis Statistik")
        
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
            st.header("Ringkasan dan Tampilan Data")
            
            col_summ1, col_summ2 = st.columns([1, 2])
            with col_summ1:
                 st.info("Berikut adalah ringkasan statistik deskriptif dari data yang Anda unggah.")
            
            st.subheader("Ringkasan Statistik (Variabel Numerik)")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik untuk diringkas.")

            st.subheader("Tampilan Data Mentah (50 Baris Pertama)")
            st.dataframe(df.head(50), use_container_width=True)

        # -------------------------------------------------------------
        # TAB 1: ANALISIS DASAR (Univariat & Bivariat)
        # -------------------------------------------------------------
        with tab_basic:
            st.header("Analisis Dasar (Univariat & Bivariat)")
            st.markdown("---")

            if not numeric_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                # --- ANALISIS UNIVARIAT ---
                st.subheader("1. Analisis Univariat (Satu Variabel)")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Histogram")
                    hist_col = st.selectbox("Pilih variabel untuk Histogram:", numeric_cols, key='hist_col')
                    if hist_col:
                        fig_hist = px.histogram(df, x=hist_col, title=f'Distribusi: {hist_col}', marginal="box", color_discrete_sequence=['#D35400'])
                        st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    st.markdown("##### Uji Normalitas")
                    norm_col = st.selectbox("Pilih variabel untuk Uji Normalitas:", numeric_cols, key='norm_col')
                    
                    if st.button("Jalankan Uji Shapiro-Wilk", key='norm_btn'):
                        data_to_test = df[norm_col].dropna()
                        if len(data_to_test) < 3:
                            st.error("Data terlalu sedikit (<3).")
                        else:
                            stat, p_value = stats.shapiro(data_to_test)
                            st.metric("P-Value", f"{p_value:.4f}")
                            
                            if p_value > 0.05:
                                st.success(f"**NORMAL** (P-value > 0.05).")
                            else:
                                st.error(f"**TIDAK NORMAL** (P-value <= 0.05).")
                
                st.markdown("---")

                # --- ANALISIS BIVARIAT ---
                st.subheader("2. Analisis Bivariat (Dua Variabel)")
                
                # Korelasi
                st.markdown("#### A. Hubungan Numerik vs Numerik")
                col3, col4 = st.columns([1, 2])
                with col3:
                    bi_x = st.selectbox("Pilih Variabel X:", numeric_cols, key='bi_x')
                    bi_y = st.selectbox("Pilih Variabel Y:", numeric_cols, key='bi_y')
                
                with col4:
                    if bi_x and bi_y and bi_x != bi_y:
                        data_bi = df[[bi_x, bi_y]].dropna()
                        fig_scatter = px.scatter(data_bi, x=bi_x, y=bi_y, title=f"{bi_y} vs {bi_x}", trendline="ols", color_discrete_sequence=['#A04000'])
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                        st.info(f"Korelasi Pearson: **{corr:.4f}** ({interpret_correlation(corr)}) | P-value: **{p_value:.4f}**")
                    elif bi_x == bi_y:
                        st.warning("Variabel X dan Y sama.")
                
                st.markdown("---")

                # Uji T
                st.markdown("#### B. Hubungan Kategorikal vs Numerik (Uji T)")
                col5, col6 = st.columns([1, 2])
                with col5:
                    if not categorical_cols:
                        st.warning("Butuh data kategorikal.")
                        cat_col_t, num_col_t = None, None
                    else:
                        cat_col_t = st.selectbox("Kelompok (Kat):", categorical_cols, key='bi_cat_t')
                        num_col_t = st.selectbox("Nilai (Num):", numeric_cols, key='bi_num_t')
                
                with col6:
                    if categorical_cols and cat_col_t and num_col_t:
                        groups_t = df[cat_col_t].dropna().unique()
                        if len(groups_t) == 2:
                            if st.button("Jalankan Uji T", key='t_test_btn'):
                                fig_box = px.box(df, x=cat_col_t, y=num_col_t, color=cat_col_t, color_discrete_sequence=px.colors.qualitative.Bold)
                                st.plotly_chart(fig_box, use_container_width=True)

                                group1 = df[df[cat_col_t] == groups_t[0]][num_col_t].dropna()
                                group2 = df[df[cat_col_t] == groups_t[1]][num_col_t].dropna()
                                stat, p_value = stats.ttest_ind(group1, group2)
                                
                                st.write(f"**Hasil Uji T:** P-value = `{p_value:.4f}`")
                                if p_value < 0.05:
                                    st.success("Perbedaan Signifikan (Nyata).")
                                else:
                                    st.warning("Perbedaan Tidak Signifikan.")
                        else:
                            st.warning(f"Variabel '{cat_col_t}' memiliki {len(groups_t)} kelompok. Uji T hanya untuk 2 kelompok.")

                # Error Bar Plot
                st.markdown("---")
                st.markdown("#### C. Plot Rata-rata & Error")
                col7, col8 = st.columns([1, 2])
                with col7:
                    if categorical_cols:
                        eb_cat = st.selectbox("Kelompok:", categorical_cols, key='eb_cat')
                        eb_num = st.selectbox("Nilai:", numeric_cols, key='eb_num')
                        eb_type = st.radio("Error:", ["Standar Error (SE)", "Standar Deviasi (SD)"], key='eb_type')
                    else:
                        st.warning("Butuh data kategorikal.")
                        eb_cat, eb_num = None, None

                with col8:
                    if eb_cat and eb_num:
                        if st.button("Buat Grafik Error Bar", key='eb_btn'):
                            df_agg = df.groupby(eb_cat)[eb_num].agg(['mean', 'std', 'count']).reset_index()
                            df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count'])
                            error_val = df_agg['se'] if eb_type == "Standar Error (SE)" else df_agg['std']
                            
                            fig = go.Figure(data=go.Bar(
                                x=df_agg[eb_cat], y=df_agg['mean'],
                                error_y=dict(type='data', array=error_val, visible=True),
                                marker_color='#F39C12' 
                            ))
                            fig.update_layout(title=f"Rata-rata {eb_num} per {eb_cat}", yaxis_title=f"Rata-rata {eb_num}")
                            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------
        # TAB 3: MODEL REGRESI
        # -------------------------------------------------------------
        with tab_reg:
            st.header("Model Regresi Linear")
            st.markdown("---")
            if not numeric_cols:
                st.error("Perlu data numerik.")
            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    reg_y_list = st.multiselect("Variabel Dependen (Y):", numeric_cols, key='reg_y_list')
                    available_x = [col for col in numeric_cols if col not in reg_y_list]
                    reg_x = st.multiselect("Variabel Independen (X):", available_x, key='reg_x')

                # REGRESI SEDERHANA / POLINOMIAL
                if len(reg_y_list) == 1 and len(reg_x) == 1:
                    reg_y = reg_y_list[0]
                    with col1:
                        poly_degree = st.radio("Tipe:", [1, 2, 3], format_func=lambda x: f"Orde {x}", key='poly_degree')
                    with col2:
                        data_reg = df[[reg_y] + reg_x].dropna()
                        X = data_reg[[reg_x[0]]]
                        y = data_reg[reg_y]
                        
                        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        y_pred = model.predict(X_poly)
                        r2 = r2_score(y, y_pred)

                        plot_df = pd.DataFrame({'X': X.iloc[:, 0], 'y': y, 'pred': y_pred}).sort_values('X')
                        fig = px.scatter(plot_df, x='X', y='y', title=f"Regresi Orde {poly_degree}", color_discrete_sequence=['#D35400'])
                        fig.add_trace(go.Scatter(x=plot_df['X'], y=plot_df['pred'], mode='lines', name='Fit', line=dict(color='#2ECC71', width=3)))
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"**R-Squared:** {r2:.4f}")

                # REGRESI BERGANDA
                elif len(reg_y_list) == 1 and len(reg_x) >= 2:
                    reg_y = reg_y_list[0]
                    with col2:
                        data_reg = df[[reg_y] + reg_x].dropna()
                        X = sm.add_constant(data_reg[reg_x])
                        model = sm.OLS(data_reg[reg_y], X).fit()
                        st.write(model.summary())
                        
                        st.subheader("Uji Asumsi Klasik")
                        # VIF
                        vif = pd.DataFrame()
                        vif["Var"] = X.columns
                        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                        col_vif, col_hetero = st.columns(2)
                        with col_vif:
                            st.write("**1. Multikolinearitas (VIF)**")
                            st.dataframe(vif[vif["Var"] != 'const'])
                        with col_hetero:
                            st.write("**2. Heteroskedastisitas**")
                            bp_test = het_breuschpagan(model.resid, model.model.exog)
                            st.write(f"P-value BP: `{bp_test[1]:.4f}`")
                            if bp_test[1] < 0.05: st.error("Terjadi Heteroskedastisitas")
                            else: st.success("Aman (Homoskedastisitas)")

                # REGRESI MULTIVARIAT
                elif len(reg_y_list) >= 2 and len(reg_x) >= 1:
                    with col2:
                        data_reg = df[reg_y_list + reg_x].dropna()
                        model = LinearRegression().fit(data_reg[reg_x], data_reg[reg_y_list])
                        st.write("**Koefisien (Matrix):**")
                        st.dataframe(pd.DataFrame(model.coef_, columns=reg_x, index=reg_y_list))

                elif not reg_y_list or not reg_x:
                    st.info("Pilih variabel Y dan X di samping.")

        # -------------------------------------------------------------
        # TAB 4: ANOVA
        # -------------------------------------------------------------
        with tab_anova:
            st.header("Analysis of Variance (ANOVA)")
            if not numeric_cols or not categorical_cols:
                 st.error("Butuh data numerik dan kategorikal.")
            else:
                st.subheader("One-Way ANOVA")
                c1, c2 = st.columns([1, 2])
                with c1:
                    a1_cat = st.selectbox("Kelompok:", categorical_cols, key='a1_cat')
                    a1_num = st.selectbox("Nilai:", numeric_cols, key='a1_num')
                with c2:
                    if a1_cat and a1_num:
                        if st.button("Proses ANOVA", key='a1_btn'):
                            clean_cols = {a1_cat: 'C', a1_num: 'N'}
                            d = df[[a1_cat, a1_num]].dropna().rename(columns=clean_cols)
                            model = smf.ols('N ~ C(C)', data=d).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            st.dataframe(anova_table)
                            p = anova_table['PR(>F)'][0]
                            if p < 0.05: st.success(f"Signifikan (P={p:.4f})")
                            else: st.warning(f"Tidak Signifikan (P={p:.4f})")

                st.divider()
                st.subheader("Two-Way ANOVA")
                c3, c4 = st.columns([1, 2])
                with c3:
                    a2_c1 = st.selectbox("Faktor 1:", categorical_cols, key='a2_c1')
                    a2_c2 = st.selectbox("Faktor 2:", categorical_cols, key='a2_c2')
                    a2_n = st.selectbox("Nilai Y:", numeric_cols, key='a2_n')
                with c4:
                    if a2_c1 and a2_c2 and a2_n and a2_c1 != a2_c2:
                        if st.button("Proses Two-Way", key='a2_btn'):
                            d = df[[a2_c1, a2_c2, a2_n]].dropna()
                            cols = [c.replace(' ','_') for c in d.columns]
                            d.columns = cols
                            f = f"{cols[2]} ~ C({cols[0]}) + C({cols[1]}) + C({cols[0]}):C({cols[1]})"
                            st.write(sm.stats.anova_lm(smf.ols(f, data=d).fit(), typ=2))

        # -------------------------------------------------------------
        # TAB 5: MANOVA
        # -------------------------------------------------------------
        with tab_manova:
            st.header("Multivariate ANOVA (MANOVA)")
            c1, c2 = st.columns([1, 2])
            with c1:
                m1_cat = st.selectbox("Kelompok (X):", categorical_cols, key='m1_c')
                m1_nums = st.multiselect("Variabel Dependen (Ys):", numeric_cols, key='m1_n')
            with c2:
                if m1_cat and len(m1_nums) >= 2:
                    if st.button("Jalankan MANOVA", key='m1_btn'):
                        d = df[[m1_cat] + m1_nums].dropna()
                        cols = [c.replace(' ','_').replace('.','') for c in d.columns]
                        d.columns = cols
                        f = f"{' + '.join(cols[1:])} ~ C({cols[0]})"
                        try:
                            res = MANOVA.from_formula(f, data=d).mv_test()
                            st.write(res.summary_frame)
                        except Exception as e: st.error(f"Error: {e}")

        # -------------------------------------------------------------
        # TAB 6: REDUKSI DIMENSI
        # -------------------------------------------------------------
        with tab_dim:
            st.header("PCA & EFA")
            if len(numeric_cols) < 2: st.error("Butuh minimal 2 variabel numerik.")
            else:
                df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
                
                st.subheader("Principal Component Analysis (PCA)")
                c1, c2 = st.columns([1, 2])
                with c1:
                    pca_vars = st.multiselect("Variabel PCA:", numeric_cols, default=numeric_cols[:4], key='pca_vars')
                    n_pca = st.slider("Jumlah Komponen:", 1, len(pca_vars), 2, key='n_pca')
                with c2:
                    if len(pca_vars) >= 2 and st.button("Hitung PCA", key='pca_btn'):
                        pca = PCA(n_components=n_pca).fit(df_scaled[pca_vars])
                        # Scree Plot
                        ev = pca.explained_variance_ratio_
                        fig = px.bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, title="Scree Plot")
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("**Loadings:**")
                        st.dataframe(pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f"PC{i+1}" for i in range(n_pca)]))

                st.divider()
                st.subheader("Exploratory Factor Analysis (EFA)")
                c3, c4 = st.columns([1, 2])
                with c3:
                    efa_vars = st.multiselect("Variabel EFA:", numeric_cols, default=numeric_cols[:4], key='efa_vars')
                    n_efa = st.slider("Faktor:", 1, len(efa_vars)-1, 2, key='n_efa')
                with c4:
                    if len(efa_vars) >= 3 and st.button("Hitung EFA", key='efa_btn'):
                        d_efa = df_scaled[efa_vars]
                        kmo_val, _ = calculate_kmo(d_efa)
                        _, bartlett_p = calculate_bartlett_sphericity(d_efa)
                        
                        col_kmo, col_bart = st.columns(2)
                        col_kmo.metric("KMO", f"{kmo_val:.3f}")
                        col_bart.metric("Bartlett P-val", f"{bartlett_p:.4f}")

                        if kmo_val > 0.5 and bartlett_p < 0.05:
                            fa = FactorAnalyzer(n_factors=n_efa, rotation='varimax').fit(d_efa)
                            st.write("**Factor Loadings:**")
                            st.dataframe(pd.DataFrame(fa.loadings_, index=efa_vars, columns=[f"F{i+1}" for i in range(n_efa)]))
                        else:
                            st.warning("Data belum memenuhi syarat EFA (KMO < 0.5 atau Bartlett tidak signifikan).")

        # -------------------------------------------------------------
        # TAB 7: KLASIFIKASI & CLUSTERING
        # -------------------------------------------------------------
        with tab_class:
            st.header("Clustering & Klasifikasi")
            
            st.subheader("K-Means Clustering")
            c1, c2 = st.columns([1, 2])
            with c1:
                clust_vars = st.multiselect("Variabel:", numeric_cols, default=numeric_cols[:2], key='clust_vars')
                k = st.slider("Jumlah Cluster (K):", 2, 8, 3, key='k_means')
            with c2:
                if len(clust_vars) >= 2 and st.button("Start Clustering", key='clust_btn'):
                    X = StandardScaler().fit_transform(df[clust_vars].dropna())
                    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
                    d_clust = df[clust_vars].dropna()
                    d_clust['Cluster'] = labels.astype(str)
                    
                    fig = px.scatter(d_clust, x=clust_vars[0], y=clust_vars[1], color='Cluster', 
                                     title=f"K-Means (K={k})", color_discrete_sequence=px.colors.qualitative.Prism)
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Linear Discriminant Analysis (LDA)")
            c3, c4 = st.columns([1, 2])
            with c3:
                lda_y = st.selectbox("Target (Kat):", categorical_cols, key='lda_y')
                lda_x = st.multiselect("Prediktor (Num):", numeric_cols, key='lda_x')
            with c4:
                if lda_y and lda_x:
                    if st.button("Start LDA", key='lda_btn'):
                        d = df[[lda_y]+lda_x].dropna()
                        lda = LinearDiscriminantAnalysis()
                        X_lda = lda.fit_transform(d[lda_x], d[lda_y])
                        st.success(f"Akurasi: {lda.score(d[lda_x], d[lda_y]):.2%}")
                        
                        if X_lda.shape[1] >= 2:
                            d_plot = pd.DataFrame(X_lda[:, :2], columns=['LD1', 'LD2'])
                            d_plot['Target'] = d[lda_y].values
                            fig = px.scatter(d_plot, x='LD1', y='LD2', color='Target', title="LDA Plot")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            d_plot = pd.DataFrame(X_lda[:, 0], columns=['LD1'])
                            d_plot['Target'] = d[lda_y].values
                            fig = px.histogram(d_plot, x='LD1', color='Target', barmode='overlay')
                            st.plotly_chart(fig, use_container_width=True)