import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import openpyxl 

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
# FUNGSI BANTUAN
# =================================================================

@st.cache_data
def load_data(file):
    """Memuat data dari file yang diupload (CSV atau Excel)."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error saat memuat file: {e}")
        return None

# Fungsi bantuan baru untuk interpretasi korelasi
def interpret_correlation(r):
    """Memberikan interpretasi kualitatif untuk nilai koefisien korelasi."""
    r_abs = abs(r)
    if r_abs >= 0.8: return "sangat kuat"
    if r_abs >= 0.6: return "kuat"
    if r_abs >= 0.4: return "cukup"
    if r_abs >= 0.2: return "lemah"
    return "sangat lemah"

# =================================================================
# KONFIGURASI HALAMAN UTAMA
# =================================================================
st.set_page_config(layout="wide")
st.title("Tools Analisis Statistik")
st.markdown("*Masukkan datamu, dan lakukan sesukamu*")

# =================================================================
# SIDEBAR: UPLOAD FILE & IDENTIFIKASI VARIABEL
# =================================================================

st.sidebar.title("Kontrol Panel")

uploaded_file = st.sidebar.file_uploader(
    "1. Upload File Anda",
    type=['csv', 'xls', 'xlsx'],
    help="Hanya file CSV, XLS, atau XLSX yang didukung."
)

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

else:
    st.info("Silakan upload file data (CSV atau Excel) di sidebar kiri untuk memulai.")

# =================================================================
# HALAMAN UTAMA: TAMPILAN ANALISIS
# =================================================================

if df is not None and all_cols:
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
                    fig_hist = px.histogram(df, x=hist_col, title=f'Histogram untuk {hist_col}', marginal="box")
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
                        
                        # PERUBAHAN v5: Interpretasi lebih rinci
                        with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                            st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                            st.markdown(f"""
                            **Tujuan Uji:** Uji Normalitas (Shapiro-Wilk) bertujuan untuk mengecek apakah data pada variabel '{norm_col}' terdistribusi normal (simetris, berbentuk seperti lonceng). Ini adalah asumsi penting untuk banyak uji statistik lainnya.

                            **Aturan Keputusan (Rule of Thumb):**
                            * **P-value > 0.05:** Data **TERDISTRIBUSI NORMAL**.
                            * **P-value <= 0.05:** Data **TIDAK TERDISTRIBUSI NORMAL**.

                            **Hasil Anda:**
                            * P-value Anda adalah **{p_value:.4f}**.
                            """)
                            if p_value > 0.05:
                                st.success(f"**Kesimpulan:** Hasil Anda **NORMAL** (karena P-value > 0.05).")
                                st.info(" **Bahasa Awam:** Ini adalah hasil yang baik. Data Anda terlihat simetris, mirip kurva lonceng. Anda bisa melanjutkan dengan uji statistik yang mengasumsikan normalitas (seperti Uji T atau ANOVA) dengan percaya diri.")
                            else:
                                st.error(f"**Kesimpulan:** Hasil Anda **TIDAK NORMAL** (karena P-value <= 0.05).")
                                st.warning(" **Bahasa Awam:** Data Anda miring/tidak rata. Hasil ini menunjukkan bahwa Anda mungkin perlu berhati-hati saat menggunakan uji statistik yang mengasumsikan normalitas (seperti Uji T atau ANOVA). Pertimbangkan untuk menggunakan uji non-parametrik.")
            
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
                    fig_scatter = px.scatter(data_bi, x=bi_x, y=bi_y, title=f"Scatter Plot: {bi_y} vs {bi_x}", trendline="ols")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    st.write("**Uji Korelasi (Pearson)**")
                    corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                    st.write(f"* **Koefisien Korelasi (r):** `{corr:.4f}`")
                    st.write(f"* **P-value:** `{p_value:.4f}`")
                    
                    # PERUBAHAN v5: Interpretasi lebih rinci
                    with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                        st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                        st.markdown(f"""
                        **Tujuan Uji:** Uji Korelasi Pearson mengukur kekuatan dan arah hubungan *linear* (garis lurus) antara '{bi_x}' dan '{bi_y}'.

                        **Aturan Keputusan (Rule of Thumb):**
                        1.  **P-value:** Menunjukkan apakah hubungan itu "nyata" (signifikan).
                            * `P-value > 0.05`: Hubungan **Tidak Nyata** (kemungkinan hanya kebetulan).
                            * `P-value <= 0.05`: Hubungan **Nyata** (signifikan).
                        2.  **Koefisien Korelasi (r):** Jika hubungannya nyata, ini menunjukkan kekuatan dan arah.
                            * `Tanda Positif (+) `: Jika {bi_x} naik, {bi_y} ikut naik.
                            * `Tanda Negatif (-) `: Jika {bi_x} naik, {bi_y} malah turun.
                            * `Kekuatan`: Diukur dari {abs(corr):.2f} (skala 0 s/d 1).

                        **Hasil Anda:**
                        * P-value Anda adalah **{p_value:.4f}**.
                        * Koefisien (r) Anda adalah **{corr:.4f}**.
                        """)
                        if p_value < 0.05:
                            st.success(f"**Kesimpulan:** Ya, ada hubungan yang **nyata** antara {bi_x} dan {bi_y} (karena P-value <= 0.05).")
                            st.info(f" **Bahasa Awam:** Kekuatan hubungan ini **{interpret_correlation(corr)}**. Tanda **{'+' if corr > 0 else '-'}** menunjukkan bahwa ketika {bi_x} naik, {bi_y} cenderung {'naik' if corr > 0 else 'turun'}.")
                        else:
                            st.error(f"**Kesimpulan:** Tidak, **tidak ditemukan** hubungan yang nyata antara {bi_x} dan {bi_y} (karena P-value > 0.05).")
                            st.info(f" **Bahasa Awam:** Meskipun Anda mungkin melihat pola di grafik, uji statistik menunjukkan pola itu kemungkinan besar hanya kebetulan saja.")
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
                else:
                    cat_col_t = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='bi_cat_t')
                    num_col_t = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='bi_num_t')
            
            with col6:
                if categorical_cols and cat_col_t and num_col_t:
                    groups_t = df[cat_col_t].dropna().unique()
                    
                    if len(groups_t) == 2:
                        if st.button("Jalankan Uji T", key='t_test_btn'):
                            st.write("**Uji T (Independent T-Test)**")
                            fig_box = px.box(df, x=cat_col_t, y=num_col_t, title=f"Distribusi {num_col_t} berdasarkan {cat_col_t}", points="all")
                            st.plotly_chart(fig_box, use_container_width=True)

                            group1 = df[df[cat_col_t] == groups_t[0]][num_col_t].dropna()
                            group2 = df[df[cat_col_t] == groups_t[1]][num_col_t].dropna()
                            
                            stat, p_value = stats.ttest_ind(group1, group2)
                            st.write(f"* **P-value:** `{p_value:.4f}`")
                            
                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** Uji T adalah untuk membandingkan nilai rata-rata dari variabel '{num_col_t}' antara dua kelompok: '{groups_t[0]}' dan '{groups_t[1]}'.
                                
                                **Pertanyaan:** Apakah perbedaan rata-rata yang terlihat di grafik itu "nyata" (signifikan) atau hanya kebetulan?

                                **Aturan Keputusan (Rule of Thumb):**
                                * **P-value > 0.05:** Perbedaan **Tidak Signifikan** (rata-rata kedua kelompok dianggap sama).
                                * **P-value <= 0.05:** Perbedaan **Signifikan** (rata-rata kedua kelompok berbeda secara nyata).

                                **Hasil Anda:**
                                * P-value Anda adalah **{p_value:.4f}**.
                                """)
                                if p_value < 0.05:
                                    st.success(f"**Kesimpulan:** Ya, ada perbedaan rata-rata {num_col_t} yang **nyata** antara kelompok {groups_t[0]} dan {groups_t[1]} (karena P-value <= 0.05).")
                                else:
                                    st.error(f"**Kesimpulan:** Tidak, **tidak ditemukan** perbedaan rata-rata {num_col_t} yang nyata antara kelompok {groups_t[0]} dan {groups_t[1]} (karena P-value > 0.05).")
                                    st.info(f" **Bahasa Awam:** Meskipun rata-rata mereka terlihat berbeda di grafik, perbedaan itu tidak cukup besar untuk dianggap nyata secara statistik. Kemungkinan itu hanya kebetulan.")
                    elif len(groups_t) > 2:
                        st.warning(f"Variabel '{cat_col_t}' memiliki {len(groups_t)} kelompok. Pindah ke tab 'ANOVA' untuk menguji 3 kelompok atau lebih.")
                    else:
                        st.warning(f"Variabel '{cat_col_t}' hanya memiliki {len(groups_t)} kelompok. Tidak dapat diuji.")

    # -------------------------------------------------------------
    # TAB 3: MODEL REGRESI (BAB 7)
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
                reg_y = st.selectbox("Pilih Variabel Dependen (Y):", numeric_cols, key='reg_y')
                available_x = [col for col in numeric_cols if col != reg_y]
                reg_x = st.multiselect("Pilih Variabel Independen (X):", available_x, key='reg_x')
            
            if reg_y and len(reg_x) == 1:
                # --- JIKA 1 VAR X: REGRESI SEDERHANA (Linear, Kuadratik, Kubik) ---
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
                    fig_poly = px.scatter(plot_df, x='X', y='y_true', title=f"Model Regresi (Derajat {poly_degree})")
                    fig_poly.add_trace(go.Scatter(x=plot_df['X'], y=plot_df['y_pred'], name='Garis Prediksi', line=dict(color='red')))
                    st.plotly_chart(fig_poly, use_container_width=True)
                    
                    st.write(f"**R-squared:** `{r2:.4f}`")
                    
                    # PERUBAHAN v5: Interpretasi lebih rinci
                    with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                        st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                        st.markdown(f"""
                        **Tujuan Uji:** Regresi adalah model prediksi. Kita mencoba memprediksi nilai **{reg_y}** (Variabel Y) menggunakan nilai **{reg_x[0]}** (Variabel X).

                        **Aturan Keputusan (Rule of Thumb):**
                        * **R-squared:** Menunjukkan seberapa bagus model Anda (dalam %). Semakin tinggi (mendekati 1.0), semakin baik.
                        * `Model Linear (Derajat 1)`: Gunakan jika Anda percaya hubungannya lurus (seperti di grafik korelasi).
                        * `Model Kuadratik/Kubik`: Gunakan jika Anda percaya hubungannya melengkung (misal: meningkat lalu melambat).

                        **Hasil Anda:**
                        * Model Regresi (Derajat {poly_degree}) Anda memiliki R-squared **{r2:.4f}**.
                        """)
                        st.success(f"**Kesimpulan (Bahasa Awam):** Model Anda **{r2*100:.2f}%** akurat. Ini berarti {r2*100:.2f}% dari perubahan pada variabel '{reg_y}' dapat dijelaskan oleh perubahan pada '{reg_x[0]}' menggunakan model ini.")

            elif reg_y and len(reg_x) >= 2:
                 # --- JIKA 2+ VAR X: REGRESI LINEAR BERGANDA ---
                st.subheader("Regresi Linear Berganda")
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
                    
                    # PERUBAHAN v5: Interpretasi lebih rinci
                    with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                        st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                        st.markdown(f"""
                        **Tujuan Uji:** Regresi Linear Berganda mencoba memprediksi satu variabel **{reg_y}** (Dependen) menggunakan *kombinasi* dari beberapa variabel lain (Independen).

                        ---
                        **1. Kebaikan Model (R-squared)**
                        * **Angka Anda:** *Adjusted R-squared* = **{r_squared:.4f}**.
                        * **Kesimpulan:** Model Anda **{r_squared*100:.2f}%** akurat. Ini berarti {r_squared*100:.2f}% dari perubahan pada '{reg_y}' dapat dijelaskan oleh *kombinasi* variabel X yang Anda pilih.

                        ---
                        **2. Signifikansi Variabel (Tabel `P>|t|`)**
                        * **Tujuan:** Melihat variabel X mana yang *benar-benar* berpengaruh terhadap Y.
                        * **Lihat:** Di "Ringkasan Model (OLS)" di atas, lihat kolom `P>|t|` untuk setiap variabel X Anda.
                        * **Aturan:** Jika `P>|t|` <= 0.05, variabel itu **signifikan** (berpengaruh nyata). Jika > 0.05, variabel itu **tidak signifikan** (mungkin bisa dibuang dari model).

                        ---
                        **3. Uji Asumsi: Multikolinearitas (VIF)**
                        * **Tujuan:** Mengecek apakah ada variabel X yang "tumpang tindih" (mengukur hal yang sama). Ini bisa membuat model tidak stabil.
                        * **Lihat:** Tabel VIF di bawah.
                        * **Aturan:** Jika ada VIF **> 10**, itu masalah.
                        """)
                    
                    vif_data = pd.DataFrame()
                    vif_data["Variabel"] = X_with_const.columns
                    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                    st.dataframe(vif_data[vif_data["Variabel"] != 'const'])
                    
                    if (vif_data[vif_data["Variabel"] != 'const']["VIF"] > 10).any():
                        st.error(f"**Kesimpulan VIF:** Ditemukan VIF > 10. Ini mengindikasikan **multikolinearitas**. Variabel X Anda tumpang tindih, yang bisa mengacaukan hasil model.")
                    else:
                        st.success(f"**Kesimpulan VIF:** Semua VIF < 10. Asumsi **terpenuhi**. Variabel X Anda tidak tumpang tindih.")
                    
                    st.markdown("---")
                    
                    st.subheader("Uji Asumsi: Heteroskedastisitas (Breusch-Pagan)")
                    st.info("Tujuan: Mengecek apakah tingkat *error* (kesalahan prediksi) model Anda konsisten.")
                    
                    bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                    bp_p_value = bp_test[1]
                    st.write(f"* **P-value Uji Breusch-Pagan:** `{bp_p_value:.4f}`")
                    st.markdown(f"""
                    **Aturan:** Jika P-value <= 0.05, itu masalah (terjadi Heteroskedastisitas).
                    """)
                    if bp_p_value < 0.05:
                        st.error(f"**Kesimpulan Hetero:** P-value < 0.05. Terindikasi **heteroskedastisitas**. Tingkat error model Anda tidak konsisten, yang mengurangi keandalan model.")
                    else:
                        st.success(f"**Kesimpulan Hetero:** P-value > 0.05. Asumsi **terpenuhi**. Tingkat error model Anda konsisten (homoskedastisitas). Ini bagus!")
            
            elif not reg_y or not reg_x:
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
                            fig_box_a1 = px.box(df, x=cat_col_anova1, y=num_col_anova1, title=f"Distribusi {num_col_anova1} berdasarkan {cat_col_anova1}", points="all")
                            st.plotly_chart(fig_box_a1, use_container_width=True)

                            df_clean = df[[cat_col_anova1, num_col_anova1]].dropna()
                            clean_cat = cat_col_anova1.replace(' ', '_').replace('[', '').replace(']', '')
                            clean_num = num_col_anova1.replace(' ', '_').replace('[', '').replace(']', '')
                            df_clean.columns = [clean_cat, clean_num]
                            
                            formula = f'{clean_num} ~ C({clean_cat})'
                            
                            try:
                                model = smf.ols(formula, data=df_clean).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                st.dataframe(anova_table)
                                
                                p_value_anova = anova_table['PR(>F)'][0]
                                
                                # PERUBAHAN v5: Interpretasi lebih rinci
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan Uji:** ANOVA One-Way membandingkan rata-rata '{num_col_anova1}' di antara {len(groups_a1)} kelompok dari '{cat_col_anova1}'.
                                    
                                    **Pertanyaan:** Apakah setidaknya ada *satu* kelompok yang rata-ratanya berbeda secara nyata dari yang lain?

                                    **Aturan Keputusan (Rule of Thumb):**
                                    * Lihat P-value (kolom `PR(>F)`).
                                    * **P-value > 0.05:** Perbedaan **Tidak Signifikan** (rata-rata semua kelompok dianggap sama).
                                    * **P-value <= 0.05:** Perbedaan **Signifikan** (setidaknya satu kelompok berbeda).

                                    **Hasil Anda:**
                                    * P-value Anda adalah **{p_value_anova:.4f}**.
                                    """)
                                    if p_value_anova < 0.05:
                                        st.success(f"**Kesimpulan:** Ya, ada perbedaan rata-rata {num_col_anova1} yang **nyata** di antara kelompok-kelompok tersebut (karena P-value <= 0.05).")
                                    else:
                                        st.error(f"**Kesimpulan:** Tidak, **tidak ditemukan** perbedaan rata-rata {num_col_anova1} yang nyata di antara kelompok-kelompok tersebut (karena P-value > 0.05).")
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
                            
                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** ANOVA Two-Way menguji tiga hal secara bersamaan:
                                1.  Apakah **{anova2_cat1}** (X1) memiliki efek pada **{anova2_num}** (Y)?
                                2.  Apakah **{anova2_cat2}** (X2) memiliki efek pada **{anova2_num}** (Y)?
                                3.  Apakah **interaksi** antara X1 dan X2 memiliki efek unik pada Y?
                                
                                **Aturan Keputusan (Rule of Thumb):**
                                * Lihat kolom `PR(>F)` untuk setiap baris. P-value <= 0.05 berarti "Ya, ada efek yang signifikan".
                                
                                **Hasil Anda:**
                                """)
                                
                                p_c1 = anova_table.loc[f'C({c1})', 'PR(>F)']
                                p_c2 = anova_table.loc[f'C({c2})', 'PR(>F)']
                                p_int = anova_table.loc[f'C({c1}):C({c2})', 'PR(>F)']
                                
                                if p_c1 < 0.05:
                                    st.success(f"**1. Efek {anova2_cat1}**: SIGNIFIKAN (P-value = {p_c1:.4f}).")
                                else:
                                    st.error(f"**1. Efek {anova2_cat1}**: TIDAK SIGNIFIKAN (P-value = {p_c1:.4f}).")
                                
                                if p_c2 < 0.05:
                                    st.success(f"**2. Efek {anova2_cat2}**: SIGNIFIKAN (P-value = {p_c2:.4f}).")
                                else:
                                    st.error(f"**2. Efek {anova2_cat2}**: TIDAK SIGNIFIKAN (P-value = {p_c2:.4f}).")
                                    
                                if p_int < 0.05:
                                    st.warning(f"**3. Efek INTERAKSI**: SIGNIFIKAN (P-value = {p_int:.4f}).")
                                    st.info(f" **Bahasa Awam (Interaksi):** Ini adalah temuan penting! Ini berarti efek dari {anova2_cat1} pada {anova2_num} **bergantung** pada apa kelompok {anova2_cat2} nya. Keduanya tidak bisa dilihat secara terpisah.")
                                else:
                                    st.info(f"**3. Efek INTERAKSI**: TIDAK SIGNIFIKAN (P-value = {p_int:.4f}).")
                                    st.info(f" **Bahasa Awam (Interaksi):** Efek {anova2_cat1} dan {anova2_cat2} bersifat independen (terpisah).")
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
                            
                            # PERBAIKAN v3.2
                            p_value_manova = mv_test.summary_frame.loc[(f'C({c1})', "Wilks' lambda"), "Pr > F"]
                            
                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** MANOVA One-Way menguji apakah variabel kelompok '{manova1_cat}' memiliki efek pada *kombinasi* variabel-variabel dependen ({', '.join(manova1_num)}) secara bersamaan.

                                **Aturan Keputusan (Rule of Thumb):**
                                * Lihat P-value (kolom `Pr > F`) untuk statistik **Wilks' lambda**.
                                * **P-value > 0.05:** Perbedaan **Tidak Signifikan**.
                                * **P-value <= 0.05:** Perbedaan **Signifikan**.

                                **Hasil Anda:**
                                * P-value (Wilks' lambda) Anda adalah **{p_value_manova:.4f}**.
                                """)
                                if p_value_manova < 0.05:
                                    st.success(f"**Kesimpulan:** Ya, '{manova1_cat}' memiliki pengaruh yang **nyata** terhadap setidaknya satu dari variabel dependen yang Anda uji (karena P-value <= 0.05).")
                                else:
                                    st.error(f"**Kesimpulan:** Tidak, '{manova1_cat}' **tidak memiliki pengaruh** yang nyata terhadap kombinasi variabel dependen Anda (karena P-value > 0.05).")
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
                            
                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** MANOVA Two-Way menguji tiga hal (mirip ANOVA Two-Way), tetapi pada *kombinasi* variabel Y ({', '.join(manova2_num)}) secara bersamaan.
                                
                                **Aturan Keputusan (Rule of Thumb):**
                                * Lihat P-value (`Pr > F` untuk **Wilks' lambda**) untuk setiap baris. P-value <= 0.05 berarti "Ya, ada efek yang signifikan".
                                
                                **Hasil Anda:**
                                """)
                                
                                # PERBAIKAN v3.2
                                p_c1 = mv_test.summary_frame.loc[(f'C({c1})', "Wilks' lambda"), "Pr > F"]
                                p_c2 = mv_test.summary_frame.loc[(f'C({c2})', "Wilks' lambda"), "Pr > F"]
                                p_int = mv_test.summary_frame.loc[(f'C({c1}):C({c2})', "Wilks' lambda"), "Pr > F"]

                                if p_c1 < 0.05:
                                    st.success(f"**1. Efek {manova2_cat1}**: SIGNIFIKAN (P-value = {p_c1:.4f}).")
                                else:
                                    st.error(f"**1. Efek {manova2_cat1}**: TIDAK SIGNIFIKAN (P-value = {p_c1:.4f}).")
                                
                                if p_c2 < 0.05:
                                    st.success(f"**2. Efek {manova2_cat2}**: SIGNIFIKAN (P-value = {p_c2:.4f}).")
                                else:
                                    st.error(f"**2. Efek {manova2_cat2}**: TIDAK SIGNIFIKAN (P-value = {p_c2:.4f}).")
                                    
                                if p_int < 0.05:
                                    st.warning(f"**3. Efek INTERAKSI**: SIGNIFIKAN (P-value = {p_int:.4f}).")
                                    st.info(f" **Bahasa Awam (Interaksi):** Ini adalah temuan penting! Efek dari {manova2_cat1} **bergantung** pada apa kelompok {manova2_cat2} nya.")
                                else:
                                    st.info(f"**3. Efek INTERAKSI**: TIDAK SIGNIFIKAN (P-value = {p_int:.4f}).")
                                    st.info(f" **Bahasa Awam (Interaksi):** Efek {manova2_cat1} dan {manova2_cat2} bersifat independen (terpisah).")

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
        st.info("Metode ini membantu menyederhanakan data Anda dengan mengurangi jumlah variabel, "
                "baik dengan meringkas (PCA) atau menemukan faktor tersembunyi (EFA).")
        st.warning("Penting: Analisis di tab ini sangat sensitif terhadap skala data. "
                   "Kami akan **otomatis menstandardisasi data Anda** (mean=0, std=1) sebelum analisis.")
        st.markdown("---")

        if len(numeric_cols) < 2:
             st.error("Analisis ini memerlukan setidaknya 2 kolom numerik.")
        else:
            try:
                df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
            except Exception as e:
                st.error(f"Gagal melakukan standarisasi data: {e}")
                st.stop() 

            # --- PCA (Bab 8) ---
            st.subheader("Principal Component Analysis (PCA)")
            st.info("Tujuan: Meringkas (mereduksi) beberapa variabel numerik menjadi lebih sedikit 'komponen' "
                    "baru sambil mempertahankan sebanyak mungkin informasi (varians).")
            
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
                            fig_scree = px.bar(scree_data, x='Komponen', y='Explained Variance', title='Scree Plot')
                            fig_scree.add_trace(go.Scatter(x=scree_data['Komponen'], y=scree_data['Explained Variance'].cumsum(), name='Kumulatif'))
                            st.plotly_chart(fig_scree, use_container_width=True)
                            
                            st.write("**Component Loadings (Bobot Variabel)**")
                            st.dataframe(pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f'PC{i+1}' for i in range(n_components_pca)]))

                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** PCA adalah teknik untuk **meringkas** {len(pca_vars)} variabel Anda menjadi {n_components_pca} "Komponen Utama" (PC) baru, yang merupakan inti dari data Anda.

                                ---
                                **1. Seberapa Bagus Ringkasan Ini? (Scree Plot)**
                                * **Lihat:** Garis 'Kumulatif' pada Scree Plot di atas.
                                * **Hasil Anda:** {n_components_pca} komponen utama Anda menjelaskan **{pca.explained_variance_ratio_.sum()*100:.2f}%** dari total variasi di data Anda.
                                * **Kesimpulan:** Angka ini menunjukkan seberapa banyak "informasi" asli yang berhasil dipertahankan. (Misal, 70-80% itu sangat bagus).

                                ---
                                **2. Apa Arti dari Setiap Komponen? (Component Loadings)**
                                * **Tujuan:** Ini adalah bagian terpenting. Kita memberi "nama" pada komponen baru.
                                * **Lihat:** Tabel "Component Loadings" di atas. Cari angka *loading* yang besar (jauh dari 0, misal > 0.6 atau < -0.6) di setiap kolom PC.
                                * **Contoh:** Jika `gaji` dan `lama_bekerja` punya *loading* tinggi di `PC1`, maka `PC1` bisa kita sebut sebagai "Faktor Senioritas & Kompensasi".
                                """)
                        except Exception as e:
                            st.error(f"Error menjalankan PCA: {e}")

            st.markdown("---")

            # --- EFA (Bab 9) ---
            st.subheader("Exploratory Factor Analysis (EFA)")
            st.info("Tujuan: Menemukan 'faktor' (konsep tersembunyi/laten) yang mendasari sekumpulan variabel numerik.")
            
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

                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan Uji Kelayakan"):
                                st.markdown(f"""
                                **Tujuan Uji:** Ini adalah tes "Boleh Jalan" untuk EFA.
                                1.  **KMO:** Mengukur kecukupan sampling. **Aturan:** Harus > 0.6 (semakin dekat ke 1, semakin baik).
                                2.  **Bartlett:** Mengecek apakah variabel Anda saling berkorelasi (ini yang kita inginkan). **Aturan:** P-value harus <= 0.05.
                                """)
                                if kmo_model < 0.6:
                                    st.error("**Kesimpulan KMO:** Nilai KMO di bawah 0.6. Data Anda **kurang ideal** untuk analisis faktor.")
                                else:
                                    st.success("**Kesimpulan KMO:** Nilai KMO di atas 0.6. Data Anda **cukup baik** untuk analisis faktor.")
                                
                                if p_value_bartlett > 0.05:
                                    st.error("**Kesimpulan Bartlett:** P-value > 0.05. Variabel Anda tidak saling berkorelasi. Analisis faktor **tidak disarankan**.")
                                else:
                                    st.success("**Kesimpulan Bartlett:** P-value < 0.05. Variabel Anda **berkorelasi**, yang bagus untuk analisis faktor.")
                            
                            if kmo_model >= 0.6 and p_value_bartlett <= 0.05:
                                fa = FactorAnalyzer(n_factors=n_components_efa, rotation=rotation)
                                fa.fit(df_scaled_efa)
                                
                                st.write(f"**Factor Loadings (Rotasi {rotation})**")
                                st.dataframe(pd.DataFrame(fa.loadings_, index=efa_vars, columns=[f'Faktor {i+1}' for i in range(n_components_efa)]))
                                
                                with st.expander("Lihat Penjelasan Factor Loadings"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan:** Ini adalah hasil utama. Kita memberi "nama" pada {n_components_efa} faktor tersembunyi.
                                    **Lihat:** Tabel "Factor Loadings" di atas. Cari angka *loading* yang besar (misal > 0.6) di setiap kolom Faktor.
                                    
                                    **Aturan:** Idealnya, satu variabel hanya punya *loading* tinggi di **satu** faktor saja.
                                    * **Contoh:** Jika `skor_kepuasan` dan `skor_loyalitas` *loading*-nya tinggi di `Faktor 1`, sementara `gaji` dan `skor_kinerja` tinggi di `Faktor 2`, Anda telah menemukan dua konsep berbeda: "Kepuasan Kerja" dan "Nilai Karyawan".
                                    """)
                        except Exception as e:
                            st.error(f"Error menjalankan EFA: {e}")

    # -------------------------------------------------------------
    # TAB 7: KLASIFIKASI & CLUSTERING (BAB 11 & 12)
    # -------------------------------------------------------------
    with tab_class:
        st.header("Klasifikasi & Clustering")
        st.info("Metode ini membantu Anda mengelompokkan data Anda, baik ke dalam kelompok yang sudah ada (Klasifikasi) "
                "atau menemukan kelompok baru yang tersembunyi (Clustering).")
        st.markdown("---")

        if not numeric_cols:
             st.error("Analisis ini memerlukan setidaknya satu kolom numerik.")
        else:
            # --- Clustering (Bab 12 - Unsupervised) ---
            st.subheader("Clustering (K-Means)")
            st.info("Tujuan: Menemukan kelompok-kelompok (cluster) yang 'alami' dalam data Anda, "
                    "tanpa mengetahui kelompoknya terlebih dahulu (unsupervised).")
            
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
                            fig_clust = px.scatter(
                                df_clean_clust,
                                x=cluster_vars[0],
                                y=cluster_vars[1],
                                color='Cluster',
                                title=f"Hasil Clustering K-Means (K={n_clusters})",
                                color_continuous_scale=px.colors.qualitative.Plotly
                            )
                            st.plotly_chart(fig_clust, use_container_width=True)
                            
                            st.write("**Pusat Cluster (Cluster Centers)**")
                            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_vars, index=[f'Cluster {i}' for i in range(n_clusters)]))

                            # PERUBAHAN v5: Interpretasi lebih rinci
                            with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                st.markdown(f"""
                                **Tujuan Uji:** K-Means Clustering adalah metode untuk **menemukan kelompok-kelompok (cluster)** yang tersembunyi di dalam data Anda.

                                **Hasil Anda:** Data Anda telah dibagi menjadi **{n_clusters}** kelompok.

                                **1. Bagaimana Tampilan Kelompoknya? (Visualisasi Cluster)**
                                * **Lihat:** Grafik scatter plot di atas. Setiap warna mewakili satu cluster.
                                * **Interpretasi:** Plot ini membantu Anda memvalidasi apakah {n_clusters} adalah jumlah cluster yang tepat. Apakah kelompoknya terlihat jelas terpisah?

                                **2. Apa Ciri-ciri Setiap Kelompok? (Pusat Cluster)**
                                * **Lihat:** Tabel "Pusat Cluster" di atas. Ini adalah bagian terpenting.
                                * **Interpretasi:** Tabel ini menunjukkan nilai *rata-rata* dari setiap variabel untuk setiap cluster (dalam skala standar).
                                * **Contoh:**
                                    * `Cluster 0` mungkin memiliki `{cluster_vars[0]}` tinggi (misal: 1.5) dan `{cluster_vars[1]}` tinggi (misal: 1.2).
                                    * `Cluster 1` mungkin memiliki `{cluster_vars[0]}` rendah (misal: -0.8) dan `{cluster_vars[1]}` rendah (misal: -1.0).
                                * Dengan melihat ini, Anda bisa memberi "nama" atau "persona" pada setiap cluster (misal: "Kinerja Tinggi/Gaji Tinggi" vs "Kinerja Rendah/Gaji Rendah").
                                """)
                            
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

                                if n_components_lda >= 2:
                                    lda_plot_df = pd.DataFrame(X_lda_transformed, columns=['LD1', 'LD2'])
                                    lda_plot_df['Target'] = y_lda.values
                                    fig_lda = px.scatter(lda_plot_df, x='LD1', y='LD2', color='Target', title='Plot Fungsi Diskriminan')
                                    st.plotly_chart(fig_lda, use_container_width=True)
                                elif n_components_lda == 1:
                                    lda_plot_df = pd.DataFrame(X_lda_transformed, columns=['LD1'])
                                    lda_plot_df['Target'] = y_lda.values
                                    fig_lda = px.histogram(lda_plot_df, x='LD1', color='Target', title='Plot Fungsi Diskriminan 1D', marginal='box')
                                    st.plotly_chart(fig_lda, use_container_width=True)
                                
                                # PERUBAHAN v5: Interpretasi lebih rinci
                                with st.expander("Lihat Penjelasan dan Interpretasi Hasil"):
                                    st.subheader("Bagaimana Cara Membaca Hasil Ini?")
                                    st.markdown(f"""
                                    **Tujuan Uji:** Analisis Diskriminan (LDA) adalah model *supervised*. Tujuannya adalah untuk **memprediksi** keanggotaan kelompok (Target: **{lda_target}**) menggunakan variabel-variabel prediktor Anda.

                                    ---
                                    **1. Seberapa Akurat Model Ini? (Akurasi Model)**
                                    * **Angka Anda:** Akurasi Model = **{lda_accuracy*100:.2f}%**.
                                    * **Kesimpulan:** Model ini {lda_accuracy*100:.2f}% akurat dalam **menebak** '{lda_target}' seseorang, hanya dengan melihat variabel prediktor yang Anda pilih. (Akurasi yang tinggi berarti variabel prediktor Anda sangat baik dalam membedakan kelompok).

                                    ---
                                    **2. Variabel Apa yang Paling Penting? (Koefisien Diskriminan)**
                                    * **Lihat:** Tabel "Koefisien Diskriminan" di atas.
                                    * **Interpretasi:** Angka yang besar (jauh dari 0, baik positif atau negatif) menunjukkan variabel tersebut adalah pembeda yang *kuat* antar kelompok. Angka yang dekat dengan 0 berarti variabel itu tidak terlalu penting.

                                    ---
                                    **3. Bagaimana Model Ini Memisahkan Kelompok? (Plot Fungsi Diskriminan)**
                                    * **Lihat:** Grafik di atas. LD1 (sumbu X) dan LD2 (sumbu Y) adalah "fungsi" atau "resep" baru yang dibuat model untuk memaksimalkan pemisahan kelompok.
                                    * **Interpretasi:** Semakin jauh jarak titik-titik (kelompok) berwarna di plot, semakin baik model Anda dalam membedakan mereka.
                                    """)
                        except Exception as e:
                            st.error(f"Error menjalankan LDA: {e}")
                else:
                    st.warning("Silakan pilih 1 variabel target (kategorikal) dan minimal 1 variabel prediktor (numerik).")