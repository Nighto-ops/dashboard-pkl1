import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import openpyxl 

# --- LIBRARY TAMBAHAN UNTUK PETA (GEOSPASIAL) ---
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
# ------------------------------------------------

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
# KONFIGURASI HALAMAN UTAMA
# =================================================================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Statistik & Geospasial")

# =================================================================
# FUNGSI BANTUAN
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

# --- FUNGSI LOAD DATA PETA (GEOSPASIAL) ---
@st.cache_data
def load_map_data():
    # Pastikan file-file ini ada di folder 'data'
    files = [
        "data/Kota Yogyakarta.xlsx",
        "data/Bantul.xlsx",
        "data/Sleman.xlsx",
        "data/Kulon Progo.xlsx",
        "data/Gunung Kidul.xlsx"
    ]
    df_list = []
    for f in files:
        try:
            temp_df = pd.read_excel(f)
            df_list.append(temp_df)
        except Exception:
            pass # Lanjut jika satu file error/hilang
            
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    # Standarisasi nama kolom ke lowercase agar aman
    combined_df.columns = [c.lower() for c in combined_df.columns]
    # Bersihkan data tanpa koordinat
    combined_df = combined_df.dropna(subset=['lattitude', 'longitude'])
    return combined_df

@st.cache_data
def load_shp():
    # Ganti path ini sesuai lokasi file SHP Anda
    shp_path = "data/shp/kec_jogja.shp" 
    try:
        gdf = gpd.read_file(shp_path)
        # Transformasi ke CRS Lat/Lon (WGS84)
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        # Jangan tampilkan error di UI utama agar tidak mengganggu jika SHP belum ada
        print(f"Error loading SHP: {e}") 
        return None

# =================================================================
# SIDEBAR UTAMA
# =================================================================

st.sidebar.title("Kontrol Panel")

uploaded_file = st.sidebar.file_uploader(
    "1. Upload File Statistik",
    type=['csv', 'xls', 'xlsx'],
    help="Upload data Anda untuk analisis statistik. Jika kosong, Peta akan ditampilkan."
)

# =================================================================
# LOGIKA TAMPILAN (SWITCHING VIEWS)
# =================================================================

# === KONDISI 1: PETA (JIKA BELUM UPLOAD FILE) ===
if uploaded_file is None:
    st.title("Dashboard Sebaran Lokasi D.I. Yogyakarta")
    st.markdown("Berikut adalah peta persebaran data lokasi (Bantul, Sleman, Kulon Progo, Gunung Kidul, Kota Yogya).")
    st.info(" **Info:** Silakan upload file Excel/CSV di sidebar kiri untuk masuk ke menu **Analisis Statistik**.")

    # 1. Load Data
    df_map = load_map_data()
    gdf_kecamatan = load_shp()

    if not df_map.empty and gdf_kecamatan is not None:
        
        # 2. Filter Wilayah (Interaktif)
        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            all_kab = df_map['kabupaten'].unique().tolist() if 'kabupaten' in df_map.columns else []
            selected_kab = st.multiselect("Filter Kabupaten/Kota:", all_kab, default=all_kab)
        
        # Filter Dataframe
        if selected_kab:
            filtered_map_df = df_map[df_map['kabupaten'].isin(selected_kab)]
        else:
            filtered_map_df = df_map

        # 3. Proses Agregasi Data untuk Choropleth
        if 'kecamatan' in filtered_map_df.columns:
            # Standarisasi nama kecamatan (Upper Case)
            filtered_map_df['kecamatan_upper'] = filtered_map_df['kecamatan'].str.upper()
            
            # --- PENTING: SESUAIKAN NAMA KOLOM SHP DI SINI ---
            # Cek nama kolom di file SHP Anda (biasanya 'WADMKC', 'NAMOBJ', atau 'KECAMATAN')
            shp_col_name = 'WADMKC' # <--- GANTI INI JIKA WARNA PETA TIDAK MUNCUL
            # -------------------------------------------------

            if shp_col_name in gdf_kecamatan.columns:
                gdf_kecamatan['kecamatan_upper'] = gdf_kecamatan[shp_col_name].str.upper()
                
                # Hitung jumlah lokasi per kecamatan
                kecamatan_stats = filtered_map_df.groupby('kecamatan_upper').size().reset_index(name='jumlah_lokasi')
                
                # Gabungkan Data Statistik ke SHP
                gdf_final = gdf_kecamatan.merge(kecamatan_stats, on='kecamatan_upper', how='left')
                gdf_final['jumlah_lokasi'] = gdf_final['jumlah_lokasi'].fillna(0)

                # 4. Render Peta Folium
                # Titik tengah Jogja
                m = folium.Map(location=[-7.88, 110.45], zoom_start=10)

                # Layer 1: Choropleth (Wilayah)
                cp = folium.Choropleth(
                    geo_data=gdf_final,
                    name='Choropleth (Kepadatan)',
                    data=gdf_final,
                    columns=['kecamatan_upper', 'jumlah_lokasi'],
                    key_on='feature.properties.kecamatan_upper',
                    fill_color='YlOrRd',
                    fill_opacity=0.6,
                    line_opacity=0.2,
                    legend_name='Jumlah Lokasi',
                    highlight=True
                ).add_to(m)

                # Tooltip untuk Choropleth
                folium.GeoJsonTooltip(
                    fields=['kecamatan_upper', 'jumlah_lokasi'],
                    aliases=['Kecamatan:', 'Jumlah Lokasi:'],
                    localize=True
                ).add_to(cp.geojson)

                # Layer 2: Heatmap (Titik)
                # Ambil lat, lon, dan bobot (jika ada kolom jumlah ulasan, gunakan. jika tidak, bobot=1)
                if 'jumlah ulasan' in filtered_map_df.columns:
                    # Mengisi NaN dengan 0 agar tidak error
                    heat_data = filtered_map_df[['lattitude', 'longitude', 'jumlah ulasan']].fillna(0).values.tolist()
                else:
                    heat_data = filtered_map_df[['lattitude', 'longitude']].values.tolist()
                
                HeatMap(heat_data, name='Heatmap Persebaran', radius=12, blur=15).add_to(m)

                # Layer Control
                folium.LayerControl().add_to(m)

                # Tampilkan
                st_folium(m, width=1400, height=600)
                
                # Tampilkan Data Tabel di bawah peta
                with st.expander("Lihat Data Lokasi Terpilih"):
                    st.dataframe(filtered_map_df)

            else:
                st.error(f"Kolom kecamatan '{shp_col_name}' tidak ditemukan di file SHP. Harap cek nama kolom SHP Anda.")
        else:
            st.warning("Kolom 'kecamatan' tidak ditemukan di data Excel.")
    else:
        st.warning("Data peta belum siap. Pastikan file Excel dan SHP ada di folder 'data/'.")


# === KONDISI 2: DASHBOARD STATISTIK (JIKA USER UPLOAD FILE) ===
else:
    # Inisialisasi Variable
    df = None
    numeric_cols = []
    categorical_cols = []
    all_cols = []

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

        # JUDUL DASHBOARD STATISTIK
        st.title("Tools Analisis Statistik")
        st.markdown(f"*Analisis File: **{uploaded_file.name}***")

        # TABS ANALISIS
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
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik untuk diringkas.")
            st.subheader("Tampilan Data Mentah (50 Baris Pertama)")
            st.dataframe(df.head(50), use_container_width=True)

        # -------------------------------------------------------------
        # TAB 1: ANALISIS DASAR
        # -------------------------------------------------------------
        with tab_basic:
            st.header("Analisis Dasar (Univariat & Bivariat)")
            st.markdown("---")
            if not numeric_cols:
                 st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
            else:
                # UNIVARIAT
                st.subheader("Analisis Univariat")
                col1, col2 = st.columns(2)
                with col1:
                    hist_col = st.selectbox("Pilih variabel Histogram:", numeric_cols, key='hist_col')
                    if hist_col:
                        fig_hist = px.histogram(df, x=hist_col, title=f'Histogram {hist_col}', marginal="box")
                        st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    norm_col = st.selectbox("Pilih variabel Uji Normalitas:", numeric_cols, key='norm_col')
                    if st.button("Uji Normalitas (Shapiro-Wilk)", key='norm_btn'):
                        data_test = df[norm_col].dropna()
                        if len(data_test) >= 3:
                            stat, p = stats.shapiro(data_test)
                            st.write(f"**P-value:** `{p:.4f}`")
                            if p > 0.05: st.success("Data Terdistribusi Normal (P > 0.05)")
                            else: st.error("Data TIDAK Normal (P <= 0.05)")
                        else: st.error("Data kurang dari 3 sampel.")
                
                st.markdown("---")
                # BIVARIAT
                st.subheader("Analisis Bivariat")
                col3, col4 = st.columns([1, 2])
                with col3:
                    bi_x = st.selectbox("Variabel X:", numeric_cols, key='bi_x')
                    bi_y = st.selectbox("Variabel Y:", numeric_cols, key='bi_y')
                with col4:
                    if bi_x and bi_y:
                        fig_scat = px.scatter(df, x=bi_x, y=bi_y, trendline="ols", title=f"{bi_y} vs {bi_x}")
                        st.plotly_chart(fig_scat, use_container_width=True)
                        corr, p_corr = stats.pearsonr(df[bi_x].fillna(0), df[bi_y].fillna(0)) # Simple fillna for demo
                        st.write(f"**Korelasi Pearson:** `{corr:.4f}` ({interpret_correlation(corr)})")

                # UJI T & ERROR BAR
                st.markdown("---")
                st.subheader("Perbandingan Kelompok (Uji T & Error Bar)")
                col7, col8 = st.columns([1, 2])
                with col7:
                    if categorical_cols:
                        eb_cat = st.selectbox("Variabel Kelompok:", categorical_cols, key='eb_cat')
                        eb_num = st.selectbox("Variabel Numerik:", numeric_cols, key='eb_num')
                        eb_type = st.radio("Error Bar:", ["Standar Error (SE)", "Standar Deviasi (SD)"], key='eb_type')
                    else:
                        st.warning("Perlu kolom kategorikal.")
                        eb_cat, eb_num = None, None
                
                with col8:
                    if eb_cat and eb_num:
                        if st.button("Buat Grafik Error Bar", key='eb_btn'):
                            try:
                                df_agg = df.groupby(eb_cat)[eb_num].agg(['mean', 'std', 'count']).reset_index()
                                df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count'])
                                error_val = df_agg['se'] if "SE" in eb_type else df_agg['std']
                                
                                fig = go.Figure(go.Bar(
                                    x=df_agg[eb_cat], y=df_agg['mean'],
                                    error_y=dict(type='data', array=error_val, visible=True)
                                ))
                                fig.update_layout(title=f"Rata-rata {eb_num} per {eb_cat}", yaxis_title=eb_num)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e: st.error(e)

        # -------------------------------------------------------------
        # TAB 2: MODEL REGRESI
        # -------------------------------------------------------------
        with tab_reg:
            st.header("Model Regresi")
            if not numeric_cols:
                st.error("Perlu data numerik.")
            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    reg_y_list = st.multiselect("Variabel Dependen (Y):", numeric_cols, key='reg_y_list')
                    avail_x = [c for c in numeric_cols if c not in reg_y_list]
                    reg_x = st.multiselect("Variabel Independen (X):", avail_x, key='reg_x')
                
                with col2:
                    # Regresi Sederhana / Polinomial
                    if len(reg_y_list) == 1 and len(reg_x) == 1:
                        reg_y = reg_y_list[0]
                        poly_degree = st.radio("Derajat Polinomial:", [1, 2, 3], key='poly')
                        
                        data_reg = df[[reg_y] + reg_x].dropna()
                        X = data_reg[[reg_x[0]]]
                        y = data_reg[reg_y]
                        
                        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        y_pred = model.predict(X_poly)
                        r2 = r2_score(y, y_pred)
                        
                        st.write(f"**R-Squared:** `{r2:.4f}`")
                        plot_df = pd.DataFrame({'X': X.iloc[:,0], 'y': y, 'pred': y_pred}).sort_values('X')
                        fig = px.scatter(plot_df, x='X', y='y', title=f"Regresi {reg_y} vs {reg_x[0]}")
                        fig.add_trace(go.Scatter(x=plot_df['X'], y=plot_df['pred'], mode='lines', name='Fit', line=dict(color='red')))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Regresi Berganda (OLS Statsmodels)
                    elif len(reg_y_list) == 1 and len(reg_x) >= 2:
                        reg_y = reg_y_list[0]
                        data_reg = df[[reg_y] + reg_x].dropna()
                        X = sm.add_constant(data_reg[reg_x])
                        y = data_reg[reg_y]
                        model = sm.OLS(y, X).fit()
                        st.text_area("OLS Summary", model.summary().as_text(), height=300)
                        
                        # Uji Asumsi Klasik Simple
                        if st.checkbox("Cek Multikolinearitas (VIF)"):
                            vif = pd.DataFrame()
                            vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                            vif["Fitur"] = X.columns
                            st.dataframe(vif)

        # -------------------------------------------------------------
        # TAB 3: ANOVA
        # -------------------------------------------------------------
        with tab_anova:
            st.header("ANOVA")
            if numeric_cols and categorical_cols:
                col1, col2 = st.columns([1, 2])
                with col1:
                    a_cat = st.selectbox("Kelompok (Kategorikal):", categorical_cols, key='a_cat')
                    a_num = st.selectbox("Nilai (Numerik):", numeric_cols, key='a_num')
                with col2:
                    if st.button("Jalankan ANOVA One-Way", key='a_btn'):
                        try:
                            df_clean = df[[a_cat, a_num]].dropna()
                            # Clean column names for formula
                            df_clean.columns = ['Cat', 'Num']
                            model = smf.ols('Num ~ C(Cat)', data=df_clean).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            st.dataframe(anova_table)
                            p_val = anova_table['PR(>F)'][0]
                            if p_val < 0.05: st.success("Perbedaan Signifikan (P < 0.05)")
                            else: st.warning("Tidak Signifikan (P > 0.05)")
                        except Exception as e: st.error(e)
            else:
                st.warning("Butuh Data Numerik & Kategorikal.")

        # -------------------------------------------------------------
        # TAB 4: MANOVA
        # -------------------------------------------------------------
        with tab_manova:
            st.header("MANOVA")
            if numeric_cols and categorical_cols:
                m_cat = st.selectbox("Kelompok (X):", categorical_cols, key='m_cat')
                m_num = st.multiselect("Variabel Dependen (Y):", numeric_cols, key='m_num')
                
                if m_cat and len(m_num) >= 2:
                    if st.button("Jalankan MANOVA", key='m_btn'):
                        try:
                            df_clean = df[[m_cat] + m_num].dropna()
                            # Rename cols to remove spaces
                            clean_cols = [c.replace(' ','_') for c in df_clean.columns]
                            df_clean.columns = clean_cols
                            formula = f"{' + '.join(clean_cols[1:])} ~ C({clean_cols[0]})"
                            
                            ma = MANOVA.from_formula(formula, data=df_clean)
                            res = ma.mv_test()
                            st.dataframe(res.summary_frame)
                        except Exception as e: st.error(f"Error: {e}")

        # -------------------------------------------------------------
        # TAB 5: REDUKSI DIMENSI
        # -------------------------------------------------------------
        with tab_dim:
            st.header("PCA / EFA")
            pca_vars = st.multiselect("Pilih Variabel:", numeric_cols, default=numeric_cols, key='pca_vars')
            if len(pca_vars) >= 2:
                n_comp = st.slider("Jumlah Komponen:", 1, len(pca_vars), 2, key='n_pca')
                if st.button("Jalankan PCA", key='pca_btn'):
                    x_scaled = StandardScaler().fit_transform(df[pca_vars].dropna())
                    pca = PCA(n_components=n_comp)
                    pca.fit(x_scaled)
                    
                    st.write("**Explained Variance Ratio:**", pca.explained_variance_ratio_)
                    st.write("**Loadings:**")
                    loadings = pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f'PC{i+1}' for i in range(n_comp)])
                    st.dataframe(loadings)

        # -------------------------------------------------------------
        # TAB 6: KLASIFIKASI & CLUSTERING
        # -------------------------------------------------------------
        with tab_class:
            st.header("Clustering (K-Means)")
            clust_vars = st.multiselect("Variabel Clustering:", numeric_cols, key='c_vars')
            k = st.slider("Jumlah Cluster (K):", 2, 8, 3, key='k_clust')
            
            if len(clust_vars) >= 2 and st.button("Jalankan K-Means", key='k_btn'):
                df_c = df[clust_vars].dropna()
                x_scaled = StandardScaler().fit_transform(df_c)
                kmeans = KMeans(n_clusters=k, random_state=42).fit(x_scaled)
                df_c['Cluster'] = kmeans.labels_
                
                fig = px.scatter(df_c, x=clust_vars[0], y=clust_vars[1], color=df_c['Cluster'].astype(str), title="Hasil Clustering")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Format file tidak didukung.")