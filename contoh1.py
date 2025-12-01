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
# CUSTOM CSS (TEMA DOMINAN ORANYE - TANPA PUTIH)
# =================================================================
def local_css():
    st.markdown("""
    <style>
        /* IMPOR FONT */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@400;700&display=swap');

        /* 1. BACKGROUND UTAMA (GRADASI ORANYE HANGAT) */
        .stApp {
            /* Background gradasi dari Oranye Muda ke Peach Gelap */
            background: linear-gradient(180deg, #FFF3E0 0%, #FFE0B2 100%);
            font-family: 'Lato', sans-serif;
            color: #3E2723; /* Teks Coklat Tua agar kontras dan enak dibaca */
        }

        /* 2. MENGHILANGKAN PADDING ATAS */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 5rem;
        }

        /* 3. JUDUL (H1, H2, H3) */
        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
            color: #BF360C !important; /* Merah Bata / Burnt Orange */
            text-shadow: 1px 1px 0px rgba(255,255,255,0.2);
        }

        /* 4. SIDEBAR (ORANYE LEBIH PEKAT) */
        [data-testid="stSidebar"] {
            background-color: #FFCC80; /* Orange Medium */
            border-right: 2px solid #EF6C00;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
             color: #3E2723 !important;
        }

        /* 5. TOMBOL (GRADASI ORANYE KUAT) */
        .stButton>button {
            background: linear-gradient(90deg, #FF6F00 0%, #E65100 100%);
            color: #FFF3E0;
            border: 1px solid #BF360C;
            border-radius: 25px;
            padding: 10px 24px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #E65100 0%, #BF360C 100%);
            color: white;
        }

        /* 6. TABS (MENU TAB) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background-color: #FFE0B2; /* Dasarnya Oranye Muda */
            padding: 10px;
            border-radius: 15px;
            box-shadow: inset 0 0 5px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 10px;
            font-weight: 600;
            color: #5D4037;
        }
        .stTabs [aria-selected="true"] {
            background-color: #EF6C00 !important; /* Oranye Menyala saat aktif */
            color: #FFFFFF !important;
            border-bottom: none;
        }

        /* 7. WIDGET & INPUT (Selectbox, Multiselect, Uploader) */
        .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
            background-color: #FFF8E1; /* Kuning/Oranye Pucat */
            color: #3E2723;
            border-color: #FFB74D;
        }
        
        /* 8. DATAFRAME / TABEL */
        div[data-testid="stDataFrame"] {
            background-color: #FFF8E1; /* Background Tabel Kuning Gading */
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #FFAB91;
        }

        /* 9. INFO BOX / ALERT */
        .stAlert {
            background-color: #FFECB3; /* Background Info Oranye Pucat */
            color: #3E2723;
            border: 1px solid #FF9800;
        }
        
        /* 10. EXPANDER */
        .streamlit-expanderHeader {
            background-color: #FFE0B2; /* Header Expander Oranye */
            border-radius: 5px;
            color: #E65100;
            font-weight: bold;
        }
        div[data-testid="stExpanderDetails"] {
            background-color: #FFF3E0;
            border: 1px solid #FFCC80;
            border-top: none;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# =================================================================
# HEADER IMAGE DISPLAY (VISUAL)
# =================================================================
header_path = 'gambar/3.jpg' 
if os.path.exists(header_path):
    st.image(header_path, use_container_width=True)
else:
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
        
    return combined_df

@st.cache_data
def load_shp_data():
    try:
        gdf = gpd.read_file("data/shp/kec_jogja.shp") 
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        return None

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
    mascot_path = 'gambar/Gundatala_Riset 5.jpg'
    logo_path = 'gambar/LOGO-PKL_REV8.jpg'
    
    col_sb1, col_sb2 = st.columns(2)
    if os.path.exists(logo_path):
        with col_sb1:
            st.image(logo_path, use_container_width=True)
    if os.path.exists(mascot_path):
        with col_sb2:
            st.image(mascot_path, use_container_width=True)
    
    st.markdown("---")
    st.title("🎛️ Kontrol Panel")
    uploaded_file = st.file_uploader("📂 1. Upload File Anda", type=['csv', 'xls', 'xlsx'])

# =================================================================
# LOGIKA UTAMA
# =================================================================

# === JIKA BELUM UPLOAD FILE ===
if uploaded_file is None:
    st.write("") 
    st.title("🌏 Dashboard Sebaran Lokasi DIY")
    st.markdown("""
    <div style='background-color: #FFCC80; padding: 15px; border-radius: 10px; border-left: 6px solid #BF360C; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #3E2723;'>
        Selamat datang di <b>Dashboard Analisis PKL</b>. <br>
        Silakan gunakan panel di sebelah kiri untuk <b>Upload Data</b> dan memulai analisis statistik.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    df_map = load_map_excel_data()
    gdf_shape = load_shp_data()

    if not df_map.empty and gdf_shape is not None:
        
        with st.expander("🔍 CEK NAMA ASLI DI FILE SHP"):
            st.info("Gunakan ini untuk melihat ejaan asli.")
            if SHP_COL_KAB in gdf_shape.columns:
                unique_kab = gdf_shape[SHP_COL_KAB].unique()
                pilih_kab_cek = st.selectbox("Isi Kolom Kabupaten (SHP):", unique_kab)
                isi_kec = gdf_shape[gdf_shape[SHP_COL_KAB] == pilih_kab_cek][SHP_COL_KEC].unique()
                st.write(f"Isi Kecamatan untuk '{pilih_kab_cek}':")
                st.write(isi_kec)

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

        if selected_kab and selected_kec:
            final_df = df_map[
                (df_map['kabupaten'].isin(selected_kab)) & 
                (df_map['kecamatan'].isin(selected_kec))
            ]
            
            gdf_shape['kab_upper'] = gdf_shape[SHP_COL_KAB].astype(str).str.title().str.strip()
            gdf_shape['kab_upper'] = gdf_shape['kab_upper'].str.replace(r'^(Kab\.?|Kabupaten|Kota)\s+', '', regex=True)
            gdf_shape['kab_upper'] = gdf_shape['kab_upper'].replace({
                'Gunungkidul': 'Gunung Kidul', 'Yogya': 'Yogyakarta', 'Yogyakartakarta': 'Yogyakarta'
            })
            
            gdf_shape['kec_upper'] = gdf_shape[SHP_COL_KEC].astype(str).str.title().str.strip()
            gdf_shape['kec_upper'] = gdf_shape['kec_upper'].str.replace(r'^(Kec\.?|Kecamatan|Kapanewon|Kemantren)\s+', '', regex=True)
            
            gdf_shape['kec_upper'] = gdf_shape['kec_upper'].replace({
                'Gedang Sari': 'Gedangsari', 'Sapto Sari': 'Saptosari', 'Karang Mojo': 'Karangmojo', 
                'Giri Subo': 'Girisubo', 'Purwo Sari': 'Purwosari', 'Gondo Kusuman': 'Gondokusuman',
                'Gondo Manan': 'Gondomanan', 'Danu Rejan': 'Danurejan', 'Mer Gangsan': 'Mergangsan',
                'Gedong Tengen': 'Gedongtengen', 'Umbul Harjo': 'Umbulharjo', 'Paku Alaman': 'Pakualaman',
                'Kota Gede': 'Kotagede', 'Mantri Jeron': 'Mantrijeron', 'Wiro Brajan': 'Wirobrajan',
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
                m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron") 

                if map_mode in ["Gabungan", "Choropleth (Wilayah)"]:
                    cp = folium.Choropleth(
                        geo_data=gdf_viz,
                        name='Kepadatan Wilayah',
                        data=gdf_viz,
                        columns=['kec_upper', 'jumlah_lokasi'],
                        key_on='feature.properties.kec_upper',
                        fill_color='Oranges', # Peta menggunakan tema Oranges
                        fill_opacity=0.8,
                        line_opacity=0.2,
                        legend_name='Jumlah Lokasi',
                        highlight=True
                    ).add_to(m)
                    folium.GeoJsonTooltip(fields=['kec_upper', 'jumlah_lokasi'], aliases=['Kecamatan:', 'Jumlah Data:'], localize=True).add_to(cp.geojson)

                if map_mode in ["Gabungan", "Heatmap (Titik)"]:
                    heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, name='Heatmap', radius=15, gradient={0.4: 'orange', 0.65: 'red', 1: 'maroon'}).add_to(m)

                folium.LayerControl().add_to(m)
                
                st.markdown("<div style='border: 4px solid #FFB74D; border-radius: 10px; overflow: hidden;'>", unsafe_allow_html=True)
                st_folium(m, width=1200, height=600)
                st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("📄 Lihat Data Tabel Detail"): st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("Wilayah SHP tidak ditemukan.")
        else:
            st.info("👋 Silakan pilih Kabupaten dan Kecamatan di atas.")
    else:
        st.warning("⚠️ Data peta belum siap.")

# === JIKA FILE SUDAH DIUPLOAD ===
else:
    df = None
    numeric_cols = []
    categorical_cols = []
    all_cols = []

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success("✅ File berhasil di-upload!")
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            st.sidebar.subheader("📋 Variabel Teridentifikasi")
            st.sidebar.write(f"**Kolom Numerik:** ({len(numeric_cols)})")
            st.sidebar.write(f"**Kolom Kategorikal:** ({len(categorical_cols)})")

    if df is not None and all_cols:
        st.title("📈 Analisis Statistik")
        
        tab_data, tab_basic, tab_reg, tab_anova, tab_manova, tab_dim, tab_class = st.tabs([
            "🏠 Beranda & Data", "📊 Analisis Dasar", "📉 Model Regresi",
            "🧪 ANOVA", "🧬 MANOVA", "📐 Reduksi Dimensi", "🎯 Klasifikasi & Clustering"
        ])

        with tab_data:
            st.header("Ringkasan dan Tampilan Data")
            col_summ1, col_summ2 = st.columns([1, 2])
            with col_summ1:
                 st.info("Ringkasan statistik dari data Anda.")
            st.subheader("Ringkasan Statistik")
            if numeric_cols: st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            st.subheader("Data Mentah")
            st.dataframe(df.head(50), use_container_width=True)

        with tab_basic:
            st.header("Analisis Dasar")
            st.markdown("---")
            if not numeric_cols: st.error("Perlu data numerik.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    hist_col = st.selectbox("Histogram:", numeric_cols, key='hist_col')
                    if hist_col:
                        fig_hist = px.histogram(df, x=hist_col, title=f'Distribusi: {hist_col}', marginal="box", color_discrete_sequence=['#E65100'])
                        st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    norm_col = st.selectbox("Uji Normalitas:", numeric_cols, key='norm_col')
                    if st.button("Uji Shapiro-Wilk", key='norm_btn'):
                        data_to_test = df[norm_col].dropna()
                        if len(data_to_test) < 3: st.error("Data kurang.")
                        else:
                            stat, p_value = stats.shapiro(data_to_test)
                            st.metric("P-Value", f"{p_value:.4f}")
                            if p_value > 0.05: st.success("NORMAL")
                            else: st.error("TIDAK NORMAL")
                
                st.markdown("---")
                col3, col4 = st.columns([1, 2])
                with col3:
                    bi_x = st.selectbox("X:", numeric_cols, key='bi_x')
                    bi_y = st.selectbox("Y:", numeric_cols, key='bi_y')
                with col4:
                    if bi_x and bi_y and bi_x != bi_y:
                        data_bi = df[[bi_x, bi_y]].dropna()
                        fig_scatter = px.scatter(data_bi, x=bi_x, y=bi_y, trendline="ols", color_discrete_sequence=['#BF360C'])
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                        st.info(f"Korelasi: {corr:.4f} ({interpret_correlation(corr)})")

        with tab_reg:
            st.header("Regresi Linear")
            if not numeric_cols: st.error("Perlu data numerik.")
            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    reg_y_list = st.multiselect("Y:", numeric_cols, key='reg_y_list')
                    available_x = [col for col in numeric_cols if col not in reg_y_list]
                    reg_x = st.multiselect("X:", available_x, key='reg_x')

                if len(reg_y_list) == 1 and len(reg_x) == 1:
                    reg_y = reg_y_list[0]
                    with col2:
                        data_reg = df[[reg_y] + reg_x].dropna()
                        model = LinearRegression().fit(data_reg[[reg_x[0]]], data_reg[reg_y])
                        y_pred = model.predict(data_reg[[reg_x[0]]])
                        fig = px.scatter(data_reg, x=reg_x[0], y=reg_y, trendline='ols', color_discrete_sequence=['#E65100'])
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"R2 Score: {r2_score(data_reg[reg_y], y_pred):.4f}")
                elif len(reg_y_list) == 1 and len(reg_x) >= 2:
                    with col2:
                        data_reg = df[[reg_y_list[0]] + reg_x].dropna()
                        X = sm.add_constant(data_reg[reg_x])
                        model = sm.OLS(data_reg[reg_y_list[0]], X).fit()
                        st.write(model.summary())

        with tab_anova:
            st.header("ANOVA")
            if numeric_cols and categorical_cols:
                c1, c2 = st.columns([1, 2])
                with c1:
                    a1_cat = st.selectbox("Grup:", categorical_cols, key='a1_cat')
                    a1_num = st.selectbox("Nilai:", numeric_cols, key='a1_num')
                with c2:
                    if st.button("Proses ANOVA", key='a1_btn'):
                        d = df[[a1_cat, a1_num]].dropna()
                        d.columns = ['C', 'N']
                        res = sm.stats.anova_lm(smf.ols('N ~ C(C)', data=d).fit(), typ=2)
                        st.dataframe(res)
        
        with tab_manova:
            st.header("MANOVA")
            c1, c2 = st.columns([1, 2])
            with c1:
                m1_cat = st.selectbox("Grup (X):", categorical_cols, key='m1_c')
                m1_nums = st.multiselect("Y (min 2):", numeric_cols, key='m1_n')
            with c2:
                if m1_cat and len(m1_nums) >= 2 and st.button("Jalankan MANOVA", key='m1_btn'):
                    d = df[[m1_cat] + m1_nums].dropna()
                    cols = [c.replace(' ','_').replace('.','') for c in d.columns]
                    d.columns = cols
                    try:
                        st.write(MANOVA.from_formula(f"{' + '.join(cols[1:])} ~ C({cols[0]})", data=d).mv_test().summary_frame)
                    except: st.error("Gagal menjalankan MANOVA.")

        with tab_dim:
            st.header("PCA & EFA")
            if len(numeric_cols) >= 2:
                df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
                c1, c2 = st.columns([1, 2])
                with c1:
                    pca_vars = st.multiselect("Var PCA:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key='pca_vars')
                with c2:
                    if len(pca_vars) >= 2 and st.button("Hitung PCA", key='pca_btn'):
                        pca = PCA().fit(df_scaled[pca_vars])
                        st.bar_chart(pca.explained_variance_ratio_)

        with tab_class:
            st.header("Clustering & LDA")
            c1, c2 = st.columns([1, 2])
            with c1:
                clust_vars = st.multiselect("Var Cluster:", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))], key='clust_vars')
                k = st.slider("K:", 2, 8, 3, key='k_means')
            with c2:
                if len(clust_vars) >= 2 and st.button("Start K-Means", key='clust_btn'):
                    d = df[clust_vars].dropna()
                    X = StandardScaler().fit_transform(d)
                    d['Cluster'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X).astype(str)
                    fig = px.scatter(d, x=clust_vars[0], y=clust_vars[1], color='Cluster', color_discrete_sequence=px.colors.qualitative.Dark24)
                    st.plotly_chart(fig, use_container_width=True)