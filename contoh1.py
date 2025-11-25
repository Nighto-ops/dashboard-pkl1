import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import openpyxl 

# Impor SKLEARN (Machine Learning)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Impor STATSMODELS (Statistik Lanjut)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.multivariate.manova import MANOVA

# Impor FACTOR ANALYZER
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# =================================================================
# KONFIGURASI HALAMAN & VARIABEL GLOBAL
# =================================================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Statistik & Geospasial")

# Konfigurasi Nama Kolom SHP (Sesuai file kec_jogja.shp Anda)
SHP_COL_KEC = 'nmkec'  # Kolom Kecamatan di SHP
SHP_COL_KAB = 'nmkab'  # Kolom Kabupaten di SHP

# =================================================================
# FUNGSI LOAD DATA
# =================================================================

@st.cache_data
def load_user_data(file):
    """Fungsi untuk memuat data statistik user"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error memuat file: {e}")
        return None

@st.cache_data
def load_map_excel_data():
    """Fungsi memuat 5 file Excel untuk Peta"""
    files = [
        "data/Kota Yogyakarta.xlsx", "data/Bantul.xlsx", 
        "data/Sleman.xlsx", "data/Kulon Progo.xlsx", "data/Gunung Kidul.xlsx"
    ]
    df_list = []
    for f in files:
        try:
            temp_df = pd.read_excel(f)
            df_list.append(temp_df)
        except: pass # Skip jika file tidak ditemukan
            
    if not df_list: return pd.DataFrame()
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = [c.lower() for c in combined_df.columns]
    # Pastikan data memiliki koordinat
    return combined_df.dropna(subset=['lattitude', 'longitude'])

@st.cache_data
def load_shp_data():
    """Fungsi memuat Shapefile batas wilayah"""
    try:
        # Path ke file SHP Anda
        gdf = gpd.read_file("data/shp/kec_jogja.shp") 
        return gdf.to_crs(epsg=4326) # Wajib convert ke Lat/Lon
    except Exception as e:
        return None

def interpret_correlation(r):
    r_abs = abs(r)
    if r_abs >= 0.8: return "sangat kuat"
    if r_abs >= 0.6: return "kuat"
    if r_abs >= 0.4: return "cukup"
    if r_abs >= 0.2: return "lemah"
    return "sangat lemah"

# =================================================================
# SIDEBAR & LOGIKA UTAMA
# =================================================================

st.sidebar.title("Kontrol Panel")

# Upload File untuk memicu pergantian tampilan
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload File Data Statistik",
    type=['csv', 'xls', 'xlsx'],
    help="Upload file Anda untuk masuk ke mode Analisis Statistik. Hapus file (x) untuk kembali ke Peta."
)

# =================================================================
# MODE 1: PETA GEOSPASIAL (DEFAULT / BELUM UPLOAD)
# =================================================================
if uploaded_file is None:
    st.title("📍 Dashboard Sebaran Lokasi D.I. Yogyakarta")
    st.markdown("Peta interaktif persebaran lokasi. Gunakan filter di sidebar untuk spesifikasi wilayah.")
    
    # Load Data
    df_map = load_map_excel_data()
    gdf_shape = load_shp_data()

    if not df_map.empty and gdf_shape is not None:
        
        # --- SIDEBAR FILTER PETA ---
        st.sidebar.markdown("---")
        st.sidebar.header("Filter Peta")
        
        # 1. Filter Kabupaten
        list_kab = sorted(df_map['kabupaten'].unique().tolist())
        selected_kab = st.sidebar.multiselect("Pilih Kabupaten:", list_kab, default=list_kab)
        
        # 2. Filter Kecamatan (Cascading: Hanya muncul sesuai Kabupaten yg dipilih)
        if selected_kab:
            df_filtered_kab = df_map[df_map['kabupaten'].isin(selected_kab)]
            list_kec = sorted(df_filtered_kab['kecamatan'].unique().tolist())
        else:
            list_kec = []
            
        selected_kec = st.sidebar.multiselect("Pilih Kecamatan:", list_kec, default=list_kec)

        # --- LOGIKA FILTER DATA ---
        if selected_kab and selected_kec:
            # A. Filter Data Titik (Excel)
            final_df = df_map[
                (df_map['kabupaten'].isin(selected_kab)) & 
                (df_map['kecamatan'].isin(selected_kec))
            ]
            
            # B. Filter Data Wilayah (SHP)
            # Normalisasi nama agar match (Uppercase)
            gdf_shape['kab_upper'] = gdf_shape[SHP_COL_KAB].str.upper()
            gdf_shape['kec_upper'] = gdf_shape[SHP_COL_KEC].str.upper()
            
            sel_kab_upper = [x.upper() for x in selected_kab]
            sel_kec_upper = [x.upper() for x in selected_kec]
            
            # Filter SHP berdasarkan input user
            final_gdf = gdf_shape[
                (gdf_shape['kab_upper'].isin(sel_kab_upper)) &
                (gdf_shape['kec_upper'].isin(sel_kec_upper))
            ]

            # --- VISUALISASI PETA ---
            if not final_gdf.empty:
                # 1. Hitung Statistik per Kecamatan untuk Warna Peta
                final_df['kec_upper'] = final_df['kecamatan'].str.upper()
                stats_kec = final_df.groupby('kec_upper').size().reset_index(name='jumlah_lokasi')
                
                # Merge SHP dengan Statistik
                gdf_viz = final_gdf.merge(stats_kec, on='kec_upper', how='left')
                gdf_viz['jumlah_lokasi'] = gdf_viz['jumlah_lokasi'].fillna(0)

                # 2. Tentukan Pusat Peta Otomatis
                bounds = final_gdf.total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                
                m = folium.Map(location=center, zoom_start=11)

                # 3. Layer Choropleth (Wilayah)
                cp = folium.Choropleth(
                    geo_data=gdf_viz,
                    name='Kepadatan Wilayah',
                    data=gdf_viz,
                    columns=['kec_upper', 'jumlah_lokasi'],
                    key_on='feature.properties.kec_upper',
                    fill_color='YlOrRd',
                    fill_opacity=0.6,
                    line_opacity=0.5,
                    legend_name='Jumlah Lokasi',
                    highlight=True
                ).add_to(m)

                # 4. Tooltip Interaktif (Hover)
                folium.GeoJsonTooltip(
                    fields=['kec_upper', 'jumlah_lokasi'],
                    aliases=['Kecamatan:', 'Jumlah Data:'],
                    localize=True,
                    style="background-color: white; border: 1px solid black; padding: 5px;"
                ).add_to(cp.geojson)

                # 5. Layer Heatmap (Titik)
                heat_data = final_df[['lattitude', 'longitude']].values.tolist()
                HeatMap(heat_data, name='Heatmap', radius=15).add_to(m)

                folium.LayerControl().add_to(m)
                st_folium(m, width=1400, height=600)
                
                # Tabel Data
                with st.expander(f"Lihat Data Detail ({len(final_df)} Lokasi)"):
                    st.dataframe(final_df)
            else:
                st.warning("Wilayah tidak ditemukan di file SHP. Cek kesesuaian nama kecamatan.")
        else:
            st.info("👈 Silakan pilih Kabupaten dan Kecamatan di sidebar kiri untuk menampilkan peta.")

    else:
        st.error("Gagal memuat Data Excel atau SHP. Pastikan file ada di folder 'data/' dan 'data/shp_files/'.")


# =================================================================
# MODE 2: DASHBOARD STATISTIK (JIKA USER UPLOAD FILE)
# =================================================================
else:
    df = load_user_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success("✅ File Berhasil Dimuat!")
        
        # Identifikasi Kolom
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        st.sidebar.write("---")
        st.sidebar.subheader("Variabel Data:")
        st.sidebar.write(f"🔢 Numerik: {len(numeric_cols)}")
        st.sidebar.write(f"🔤 Kategorikal: {len(categorical_cols)}")

        st.title("📊 Dashboard Analisis Statistik")
        st.markdown(f"Sedang menganalisis: **{uploaded_file.name}**")

        # TABS ANALISIS
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Data View", "Analisis Dasar", "Regresi", "ANOVA", 
            "MANOVA", "Reduksi Dimensi", "Clustering"
        ])

        # --- TAB 1: DATA VIEW ---
        with tab1:
            st.subheader("Ringkasan Data")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            st.subheader("Sampel Data (50 Baris)")
            st.dataframe(df.head(50), use_container_width=True)

        # --- TAB 2: ANALISIS DASAR ---
        with tab2:
            st.subheader("Analisis Univariat & Bivariat")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**1. Histogram & Normalitas**")
                v_hist = st.selectbox("Pilih Variabel:", numeric_cols, key='hist')
                if v_hist:
                    st.plotly_chart(px.histogram(df, x=v_hist, marginal="box"), use_container_width=True)
                    if st.button("Cek Normalitas (Shapiro)", key='shp'):
                        stat, p = stats.shapiro(df[v_hist].dropna())
                        st.info(f"P-value: {p:.4f} ({'Normal' if p > 0.05 else 'Tidak Normal'})")

            with c2:
                st.markdown("**2. Scatter & Korelasi**")
                vx = st.selectbox("X:", numeric_cols, key='sc_x')
                vy = st.selectbox("Y:", numeric_cols, key='sc_y')
                if vx and vy:
                    st.plotly_chart(px.scatter(df, x=vx, y=vy, trendline='ols'), use_container_width=True)
                    corr, _ = stats.pearsonr(df[vx].fillna(0), df[vy].fillna(0))
                    st.info(f"Korelasi Pearson: {corr:.4f} ({interpret_correlation(corr)})")
            
            st.markdown("---")
            st.markdown("**3. Perbandingan Kelompok (Error Bar)**")
            ce1, ce2 = st.columns([1, 2])
            with ce1:
                eb_cat = st.selectbox("Kelompok (Kat):", categorical_cols, key='eb_c')
                eb_num = st.selectbox("Nilai (Num):", numeric_cols, key='eb_n')
            with ce2:
                if eb_cat and eb_num:
                    if st.button("Buat Error Bar", key='btn_eb'):
                        agg = df.groupby(eb_cat)[eb_num].agg(['mean', 'std', 'count']).reset_index()
                        agg['se'] = agg['std'] / np.sqrt(agg['count'])
                        fig = go.Figure(go.Bar(
                            x=agg[eb_cat], y=agg['mean'],
                            error_y=dict(type='data', array=agg['se'], visible=True)
                        ))
                        fig.update_layout(title=f"Rata-rata {eb_num} per {eb_cat} (Standard Error)", yaxis_title=eb_num)
                        st.plotly_chart(fig, use_container_width=True)

        # --- TAB 3: REGRESI ---
        with tab3:
            st.subheader("Model Regresi Linear")
            col_reg1, col_reg2 = st.columns([1, 2])
            with col_reg1:
                y_var = st.multiselect("Variabel Dependen (Y):", numeric_cols, key='reg_y')
                x_vars = st.multiselect("Variabel Independen (X):", [c for c in numeric_cols if c not in y_var], key='reg_x')
            
            with col_reg2:
                if y_var and x_vars:
                    # Regresi Sederhana/Berganda
                    if len(y_var) == 1:
                        target = y_var[0]
                        X = sm.add_constant(df[x_vars].dropna())
                        y = df.loc[X.index, target]
                        model = sm.OLS(y, X).fit()
                        st.write(model.summary())
                    # Regresi Multivariate
                    else:
                        st.info("Regresi Multivariate (Banyak Y) menggunakan LinearRegression SKLearn")
                        df_reg = df[x_vars + y_var].dropna()
                        X = df_reg[x_vars]
                        Y = df_reg[y_var]
                        model = LinearRegression().fit(X, Y)
                        st.write("R-Squared per target:", model.score(X, Y))
                        st.write("Koefisien:", pd.DataFrame(model.coef_, columns=x_vars, index=y_var))

        # --- TAB 4: ANOVA ---
        with tab4:
            st.subheader("One-Way ANOVA")
            if categorical_cols and numeric_cols:
                ca = st.selectbox("Faktor (Kategorikal):", categorical_cols, key='anova_c')
                na = st.selectbox("Respon (Numerik):", numeric_cols, key='anova_n')
                if st.button("Jalankan ANOVA", key='btn_anova'):
                    df_a = df[[ca, na]].dropna()
                    df_a.columns = ['Faktor', 'Respon'] # Rename biar aman formula
                    model = smf.ols('Respon ~ C(Faktor)', data=df_a).fit()
                    anova_t = sm.stats.anova_lm(model, typ=2)
                    st.dataframe(anova_t)
                    if anova_t['PR(>F)'][0] < 0.05:
                        st.success("Terdapat perbedaan signifikan antar kelompok (P < 0.05)")
                    else:
                        st.warning("Tidak ada perbedaan signifikan (P > 0.05)")

        # --- TAB 5: MANOVA ---
        with tab5:
            st.subheader("MANOVA (Multivariate ANOVA)")
            if categorical_cols and len(numeric_cols) >= 2:
                cm = st.selectbox("Kelompok (X):", categorical_cols, key='man_c')
                nms = st.multiselect("Variabel Dependen (Y):", numeric_cols, key='man_n')
                
                if cm and len(nms) >= 2:
                    if st.button("Jalankan MANOVA", key='btn_man'):
                        try:
                            df_m = df[[cm] + nms].dropna()
                            cols_clean = [c.replace(' ','_').replace('.','') for c in df_m.columns]
                            df_m.columns = cols_clean
                            formula = f"{' + '.join(cols_clean[1:])} ~ C({cols_clean[0]})"
                            
                            manova = MANOVA.from_formula(formula, data=df_m)
                            st.write(manova.mv_test().summary())
                        except Exception as e:
                            st.error(f"Gagal: {e}")

        # --- TAB 6: REDUKSI DIMENSI ---
        with tab6:
            st.subheader("PCA & EFA")
            vars_dim = st.multiselect("Pilih Variabel:", numeric_cols, default=numeric_cols, key='dim_v')
            
            if len(vars_dim) >= 2:
                st.markdown("---")
                # PCA
                n_comp = st.slider("Jumlah Komponen PCA:", 1, len(vars_dim), 2)
                if st.button("Hitung PCA", key='btn_pca'):
                    x_sc = StandardScaler().fit_transform(df[vars_dim].dropna())
                    pca = PCA(n_components=n_comp).fit(x_sc)
                    st.write("Explained Variance:", pca.explained_variance_ratio_)
                    loadings = pd.DataFrame(pca.components_.T, index=vars_dim, columns=[f'PC{i+1}' for i in range(n_comp)])
                    st.dataframe(loadings)

                st.markdown("---")
                # EFA
                if st.button("Hitung EFA (Factor Analysis)", key='btn_efa'):
                    x_sc = StandardScaler().fit_transform(df[vars_dim].dropna())
                    kmo_all, kmo_model = calculate_kmo(pd.DataFrame(x_sc))
                    st.write(f"KMO Score: {kmo_model:.4f} (>0.6 Baik)")
                    
                    fa = FactorAnalyzer(n_factors=n_comp, rotation='varimax')
                    fa.fit(x_sc)
                    st.write("Factor Loadings (Varimax):")
                    st.dataframe(pd.DataFrame(fa.loadings_, index=vars_dim))

        # --- TAB 7: CLUSTERING ---
        with tab7:
            st.subheader("K-Means Clustering")
            vars_cl = st.multiselect("Variabel Clustering:", numeric_cols, key='cl_v')
            k = st.slider("Jumlah Cluster (K):", 2, 8, 3)
            
            if len(vars_cl) >= 2 and st.button("Jalankan K-Means", key='btn_cl'):
                df_cl = df[vars_cl].dropna()
                x_sc = StandardScaler().fit_transform(df_cl)
                km = KMeans(n_clusters=k, random_state=42).fit(x_sc)
                df_cl['Cluster'] = km.labels_.astype(str)
                
                fig = px.scatter(df_cl, x=vars_cl[0], y=vars_cl[1], color='Cluster', title="Visualisasi Cluster")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Rata-rata per Cluster:")
                st.dataframe(df_cl.groupby('Cluster').mean())