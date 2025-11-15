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

# =C ================================================================
# KONFIGURASI HALAMAN UTAMA
# =================================================================
st.set_page_config(layout="wide")
st.title("🛠️ Platform Analisis Statistik Multivariat")
st.markdown("*Berdasarkan acuan: 'Applied Multivariate Statistical Analysis' (Johnson & Wichern)*")

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
    # Struktur Tab Baru (Permintaan Desain)
    tab_data, tab_basic, tab_reg, tab_anova, tab_manova, tab_dim, tab_class = st.tabs([
        "🏠 Beranda & Data",
        "📊 Analisis Dasar",
        "📈 Model Regresi",
        "🔬 ANOVA",
        "🧩 MANOVA",
        "🧬 Reduksi Dimensi (PCA/EFA)",
        "🎯 Klasifikasi & Clustering"
    ])

    # -------------------------------------------------------------
    # TAB 0: RINGKASAN DATA
    # -------------------------------------------------------------
    with tab_data:
        st.header("Ringkasan dan Tampilan Data")
        
        st.subheader("Ringkasan Statistik (Variabel Numerik)")
        st.info("Statistik deskriptif dasar untuk semua kolom numerik dalam data Anda.")
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
                st.info("Menguji apakah data terdistribusi normal.")
                norm_col = st.selectbox("Pilih variabel untuk Uji Normalitas:", numeric_cols, key='norm_col')
                
                if st.button("Jalankan Uji Normalitas (Shapiro-Wilk)", key='norm_btn'):
                    data_to_test = df[norm_col].dropna()
                    if len(data_to_test) < 3:
                        st.error("Uji Normalitas memerlukan setidaknya 3 sampel.")
                    else:
                        stat, p_value = stats.shapiro(data_to_test)
                        st.write(f"**P-value:** `{p_value:.4f}`")
                        if p_value > 0.05:
                            st.success(f"**Interpretasi Teknis:** P-value > 0.05. Gagal menolak H0. Data **terdistribusi normal**.")
                            st.info("💡 **Bahasa Awam:** Data Anda terlihat simetris seperti lonceng (bell curve) yang normal.")
                        else:
                            st.error(f"**Interpretasi Teknis:** P-value <= 0.05. Menolak H0. Data **TIDAK terdistribusi normal**.")
                            st.info("💡 **Bahasa Awam:** Data Anda miring/tidak rata, bukan seperti lonceng (bell curve) yang normal.")
            
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
                    
                    if p_value < 0.05:
                        st.success(f"**Interpretasi Teknis:** Terdapat korelasi yang signifikan secara statistik (p < 0.05).")
                        st.info(f"💡 **Bahasa Awam:** Ya, ada hubungan yang nyata antara {bi_x} dan {bi_y}.")
                    else:
                        st.error(f"**Interpretasi Teknis:** Tidak terdapat korelasi yang signifikan secara statistik (p > 0.05).")
                        st.info(f"💡 **Bahasa Awam:** Tidak ditemukan hubungan yang jelas antara {bi_x} dan {bi_y}.")
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
                            
                            if p_value < 0.05:
                                st.success(f"**Interpretasi Teknis:** Terdapat perbedaan rata-rata yang signifikan antara {groups_t[0]} dan {groups_t[1]}.")
                                st.info(f"💡 **Bahasa Awam:** Ya, ada perbedaan nilai yang nyata antara kelompok {groups_t[0]} dan {groups_t[1]}.")
                            else:
                                st.error(f"**Interpretasi Teknis:** Tidak ada perbedaan rata-rata yang signifikan.")
                                st.info(f"💡 **Bahasa Awam:** Tidak, perbedaan nilai antara kedua kelompok sepertinya hanya kebetulan saja (tidak signifikan).")
                    elif len(groups_t) > 2:
                        st.warning(f"Variabel '{cat_col_t}' memiliki {len(groups_t)} kelompok. Pindah ke tab 'ANOVA' untuk menguji 3 kelompok atau lebih.")
                    else:
                        st.warning(f"Variabel '{cat_col_t}' hanya memiliki {len(groups_t)} kelompok. Tidak dapat diuji.")

    # -------------------------------------------------------------
    # TAB 3: MODEL REGRESI (BAB 7)
    # -------------------------------------------------------------
    with tab_reg:
        st.header("Model Regresi (Bab 7)")
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
                    # Siapkan data
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
                    st.success(f"**Interpretasi Teknis:** {r2*100:.2f}% variasi pada **{reg_y}** dapat dijelaskan oleh model ini.")
                    st.info(f"💡 **Bahasa Awam:** Model ini {r2*100:.2f}% akurat dalam memprediksi {reg_y} menggunakan {reg_x[0]}.")

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
                    st.success(f"**Interpretasi R-squared (Adj.):** {r_squared*100:.2f}% variasi pada **{reg_y}** "
                               f"dapat dijelaskan oleh variabel-variabel X yang dipilih.")
                    st.info(f"💡 **Bahasa Awam:** Kombinasi variabel X ini {r_squared*100:.2f}% akurat dalam menjelaskan {reg_y}.")
                    
                    st.markdown("---")
                    
                    st.write("**:mag: Uji Diagnostik Asumsi Regresi**")
                    
                    st.info("**1. Uji Multikolinearitas (VIF)**\n"
                            "Tujuan: Mendeteksi korelasi yang tinggi antar variabel Independen (X).")
                    
                    vif_data = pd.DataFrame()
                    vif_data["Variabel"] = X_with_const.columns
                    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                    st.dataframe(vif_data[vif_data["Variabel"] != 'const'])
                    
                    if (vif_data[vif_data["Variabel"] != 'const']["VIF"] > 10).any():
                        st.error("**Interpretasi Teknis:** Ditemukan VIF > 10. Ini mengindikasikan adanya **multikolinearitas** kuat.")
                        st.info("💡 **Bahasa Awam:** Variabel X Anda 'saling tumpang tindih'. Mereka mengukur hal yang terlalu mirip.")
                    else:
                        st.success("**Interpretasi Teknis:** Semua VIF < 10. Tidak ada indikasi multikolinearitas yang kuat.")
                        st.info("💡 **Bahasa Awam:** Bagus! Variabel X Anda 'berdiri sendiri' dan tidak saling tumpang tindih.")

                    st.info("**2. Uji Heteroskedastisitas (Breusch-Pagan)**\n"
                            "Tujuan: Menguji apakah varians dari residual (error) konstan.")
                    
                    bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                    bp_p_value = bp_test[1]
                    
                    st.write(f"* **P-value Uji Breusch-Pagan:** `{bp_p_value:.4f}`")
                    
                    if bp_p_value < 0.05:
                        st.error(f"**Interpretasi Teknis:** P-value < 0.05. Menolak H0. Terindikasi **heteroskedastisitas**.")
                        st.info("💡 **Bahasa Awam:** Tingkat kesalahan (error) pada model Anda tidak konsisten.")
                    else:
                        st.success(f"**Interpretasi Teknis:** P-value > 0.05. Gagal menolak H0. Asumsi **homoskedastisitas** terpenuhi.")
                        st.info("💡 **Bahasa Awam:** Bagus! Tingkat kesalahan (error) pada model Anda konsisten.")
            
            elif not reg_y or not reg_x:
                st.info("Silakan pilih minimal 1 Variabel Dependen (Y) dan 1 Variabel Independen (X) untuk memulai.")

    # -------------------------------------------------------------
    # TAB 4: ANOVA (BAB 6)
    # -------------------------------------------------------------
    with tab_anova:
        st.header("Analisis Varians (ANOVA) (Bab 6)")
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
                                if p_value_anova < 0.05:
                                    st.success(f"**Interpretasi Teknis:** P-value ({p_value_anova:.4f}) < 0.05. "
                                               "Setidaknya ada satu kelompok yang memiliki rata-rata yang berbeda secara signifikan.")
                                    st.info("💡 **Bahasa Awam:** Ya, ada perbedaan nilai yang nyata di antara kelompok-kelompok tersebut.")
                                else:
                                    st.error(f"**Interpretasi Teknis:** P-value ({p_value_anova:.4f}) > 0.05. "
                                             "Tidak ada perbedaan rata-rata yang signifikan antar kelompok.")
                                    st.info("💡 **Bahasa Awam:** Tidak, perbedaan nilai antar kelompok sepertinya hanya kebetulan saja.")
                            except Exception as e:
                                st.error(f"Error saat menjalankan ANOVA: {e}. Pastikan nama kolom valid.")
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
                            
                            st.subheader("Interpretasi Sederhana (Bahasa Awam)")
                            p_c1 = anova_table.loc[f'C({c1})', 'PR(>F)']
                            p_c2 = anova_table.loc[f'C({c2})', 'PR(>F)']
                            p_int = anova_table.loc[f'C({c1}):C({c2})', 'PR(>F)']
                            
                            if p_c1 < 0.05:
                                st.success(f"**{anova2_cat1}**: BERPENGARUH SIGNIFIKAN.")
                            else:
                                st.error(f"**{anova2_cat1}**: TIDAK BERPENGARUH SIGNIFIKAN.")
                            
                            if p_c2 < 0.05:
                                st.success(f"**{anova2_cat2}**: BERPENGARUH SIGNIFIKAN.")
                            else:
                                st.error(f"**{anova2_cat2}**: TIDAK BERPENGARUH SIGNIFIKAN.")
                                
                            if p_int < 0.05:
                                st.warning(f"**INTERAKSI ({anova2_cat1}:{anova2_cat2})**: ADA EFEK INTERAKSI SIGNIFIKAN.")
                                st.info(f"💡 **Bahasa Awam:** Efek dari {anova2_cat1} pada nilai **bergantung** pada {anova2_cat2}.")
                            else:
                                st.info(f"**INTERAKSI ({anova2_cat1}:{anova2_cat2})**: TIDAK ADA EFEK INTERAKSI SIGNIFIKAN.")
                                st.info(f"💡 **Bahasa Awam:** Efek {anova2_cat1} dan {anova2_cat2} bersifat independen (terpisah).")

                        except Exception as e:
                            st.error(f"Error saat menjalankan ANOVA: {e}. Pastikan variabel memiliki > 1 level.")
                elif anova2_cat1 == anova2_cat2:
                    st.warning("Variabel Kategorikal 1 dan 2 tidak boleh sama.")

    # -------------------------------------------------------------
    # TAB 5: MANOVA (BAB 6)
    # -------------------------------------------------------------
    with tab_manova:
        st.header("Analisis MANOVA (Bab 6)")
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
                            
                            if p_value_manova < 0.05:
                                st.success(f"**Interpretasi Teknis:** P-value ({p_value_manova:.4f}) < 0.05. "
                                           f"Terdapat perbedaan yang signifikan antar kelompok.")
                                st.info(f"💡 **Bahasa Awam:** Ya, {manova1_cat} memiliki pengaruh yang nyata terhadap nilai-nilai ({', '.join(manova1_num)}) secara bersamaan.")
                            else:
                                st.error(f"**Interpretasi Teknis:** P-value ({p_value_manova:.4f}) > 0.05. "
                                         "Tidak ada perbedaan yang signifikan antar kelompok.")
                                st.info(f"💡 **Bahasa Awam:** Tidak, {manova1_cat} sepertinya tidak memiliki pengaruh apa-apa terhadap nilai-nilai ({', '.join(manova1_num)}) secara bersamaan.")
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
                            
                            st.subheader("Interpretasi Sederhana (Wilks' Lambda P-value)")
                            # PERBAIKAN v3.2
                            p_c1 = mv_test.summary_frame.loc[(f'C({c1})', "Wilks' lambda"), "Pr > F"]
                            p_c2 = mv_test.summary_frame.loc[(f'C({c2})', "Wilks' lambda"), "Pr > F"]
                            p_int = mv_test.summary_frame.loc[(f'C({c1}):C({c2})', "Wilks' lambda"), "Pr > F"]

                            if p_c1 < 0.05:
                                st.success(f"**{manova2_cat1}**: BERPENGARUH SIGNIFIKAN.")
                            else:
                                st.error(f"**{manova2_cat1}**: TIDAK BERPENGARUH SIGNIFIKAN.")
                            
                            if p_c2 < 0.05:
                                st.success(f"**{manova2_cat2}**: BERPENGARUH SIGNIFIKAN.")
                            else:
                                st.error(f"**{manova2_cat2}**: TIDAK BERPENGARUH SIGNIFIKAN.")
                                
                            if p_int < 0.05:
                                st.warning(f"**INTERAKSI ({manova2_cat1}:{manova2_cat2})**: ADA EFEK INTERAKSI SIGNIFIKAN.")
                            else:
                                st.info(f"**INTERAKSI ({manova2_cat1}:{manova2_cat2})**: TIDAK ADA EFEK INTERAKSI SIGNIFIKAN.")

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
        st.header("Reduksi Dimensi (Bab 8 & 9)")
        st.info("Metode ini membantu menyederhanakan data Anda dengan mengurangi jumlah variabel, "
                "baik dengan meringkas (PCA) atau menemukan faktor tersembunyi (EFA).")
        st.warning("Penting: Analisis di tab ini sangat sensitif terhadap skala data. "
                   "Kami akan **otomatis menstandardisasi data Anda** (mean=0, std=1) sebelum analisis.")
        st.markdown("---")

        if len(numeric_cols) < 2:
             st.error("Analisis ini memerlukan setidaknya 2 kolom numerik.")
        else:
            # Standarisasi data untuk tab ini
            try:
                df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[numeric_cols]), columns=numeric_cols)
            except Exception as e:
                st.error(f"Gagal melakukan standarisasi data: {e}")
                st.stop() # Hentikan eksekusi di tab ini jika scaling gagal

            # --- PCA (Bab 8) ---
            st.subheader("Principal Component Analysis (PCA)")
            st.info("Tujuan: Meringkas (mereduksi) beberapa variabel numerik menjadi lebih sedikit 'komponen' "
                    "baru sambil mempertahankan sebanyak mungkin informasi (varians).")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                n_components_pca = st.slider(
                    "Pilih jumlah Komponen PCA:",
                    min_value=1,
                    max_value=len(numeric_cols),
                    value=max(1, len(numeric_cols)//2),
                    key='n_pca'
                )
            
            with col2:
                if st.button("Jalankan PCA", key='pca_btn'):
                    try:
                        pca = PCA(n_components=n_components_pca)
                        pca.fit(df_scaled)
                        
                        st.write("**Scree Plot (Proporsi Varians yang Dijelaskan)**")
                        scree_data = pd.DataFrame({
                            'Komponen': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                            'Explained Variance': pca.explained_variance_ratio_
                        })
                        fig_scree = px.bar(scree_data, x='Komponen', y='Explained Variance', title='Scree Plot')
                        fig_scree.add_trace(go.Scatter(x=scree_data['Komponen'], y=scree_data['Explained Variance'].cumsum(), name='Kumulatif'))
                        st.plotly_chart(fig_scree, use_container_width=True)
                        
                        st.write("**Eigenvalues (Varians per Komponen)**")
                        st.dataframe(pd.DataFrame(pca.explained_variance_, index=[f'PC{i+1}' for i in range(len(pca.explained_variance_))], columns=['Eigenvalue']))
                        
                        st.write(f"**Total Varians Dijelaskan oleh {n_components_pca} Komponen:** `{pca.explained_variance_ratio_.sum()*100:.2f}%`")
                        
                        st.write("**Component Loadings (Bobot Variabel)**")
                        st.dataframe(pd.DataFrame(pca.components_.T, index=numeric_cols, columns=[f'PC{i+1}' for i in range(n_components_pca)]))
                        
                        st.success(f"**Interpretasi Teknis:** Model PCA {n_components_pca} komponen berhasil dibuat, menjelaskan {pca.explained_variance_ratio_.sum()*100:.2f}% dari total varians.")
                        st.info(f"💡 **Bahasa Awam:** Anda telah berhasil 'meringkas' {len(numeric_cols)} variabel Anda menjadi {n_components_pca} komponen utama. Lihat tabel 'Component Loadings' untuk melihat variabel mana yang paling berkontribusi pada setiap komponen baru.")
                        
                    except Exception as e:
                        st.error(f"Error menjalankan PCA: {e}")

            st.markdown("---")

            # --- EFA (Bab 9) ---
            st.subheader("Exploratory Factor Analysis (EFA)")
            st.info("Tujuan: Menemukan 'faktor' (konsep tersembunyi/laten) yang mendasari sekumpulan variabel numerik.")
            
            col3, col4 = st.columns([1, 2])
            with col3:
                n_components_efa = st.slider(
                    "Pilih jumlah Faktor:",
                    min_value=1,
                    max_value=len(numeric_cols),
                    value=max(1, len(numeric_cols)//2),
                    key='n_efa'
                )
                rotation = st.radio("Pilih metode Rotasi:", ["varimax", "promax"], key='efa_rot')

            with col4:
                if st.button("Jalankan EFA", key='efa_btn'):
                    try:
                        # Uji Kelayakan
                        chi_square_value, p_value_bartlett = calculate_bartlett_sphericity(df_scaled)
                        kmo_all, kmo_model = calculate_kmo(df_scaled)
                        
                        st.write("**Uji Kelayakan Data (KMO & Bartlett)**")
                        st.write(f"* **Kaiser-Meyer-Olkin (KMO) Measure:** `{kmo_model:.4f}`")
                        st.write(f"* **Bartlett's Test p-value:** `{p_value_bartlett:.4f}`")

                        if kmo_model < 0.6:
                            st.error("**Interpretasi KMO:** Nilai KMO di bawah 0.6. Data Anda **kurang ideal** untuk analisis faktor.")
                        else:
                            st.success("**Interpretasi KMO:** Nilai KMO di atas 0.6. Data Anda **cukup baik** untuk analisis faktor.")
                        
                        if p_value_bartlett > 0.05:
                            st.error("**Interpretasi Bartlett:** P-value > 0.05. Matriks korelasi Anda mungkin *identity matrix*. Analisis faktor **tidak disarankan**.")
                        else:
                            st.success("**Interpretasi Bartlett:** P-value < 0.05. Variabel Anda **berkorelasi**, yang bagus untuk analisis faktor.")
                        
                        if kmo_model >= 0.6 and p_value_bartlett <= 0.05:
                            fa = FactorAnalyzer(n_factors=n_components_efa, rotation=rotation)
                            fa.fit(df_scaled)
                            
                            st.write(f"**Factor Loadings (Rotasi {rotation})**")
                            st.info("Ini menunjukkan seberapa kuat setiap variabel 'mewakili' faktor tersembunyi.")
                            st.dataframe(pd.DataFrame(fa.loadings_, index=numeric_cols, columns=[f'Faktor {i+1}' for i in range(n_components_efa)]))
                            
                            st.success(f"**Interpretasi Teknis:** Model EFA {n_components_efa} faktor berhasil dibuat.")
                            st.info(f"💡 **Bahasa Awam:** Anda telah menemukan {n_components_efa} 'konsep tersembunyi'. Lihat tabel di atas. Variabel dengan angka besar (misal > 0.6) pada 'Faktor 1' adalah variabel-variabel yang mengukur konsep tersebut.")
                        
                    except Exception as e:
                        st.error(f"Error menjalankan EFA: {e}")

    # -------------------------------------------------------------
    # TAB 7: KLASIFIKASI & CLUSTERING (BAB 11 & 12)
    # -------------------------------------------------------------
    with tab_class:
        st.header("Klasifikasi & Clustering (Bab 11 & 12)")
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
                            
                            df_clean_clust = df.dropna(subset=cluster_vars)
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
                            
                            st.success(f"**Interpretasi Teknis:** Data berhasil dikelompokkan menjadi {n_clusters} cluster.")
                            st.info(f"💡 **Bahasa Awam:** Data Anda telah dibagi menjadi {n_clusters} kelompok. Lihat grafik untuk melihat persebarannya dan tabel 'Pusat Cluster' untuk melihat karakteristik rata-rata setiap kelompok.")
                            
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
                                st.success(f"**Interpretasi Teknis:** Model berhasil dibuat dengan akurasi {lda_accuracy*100:.2f}% dalam membedakan kelompok.")
                                st.info(f"💡 **Bahasa Awam:** Model ini {lda_accuracy*100:.2f}% akurat dalam menebak '{lda_target}' hanya dengan melihat variabel prediktor yang Anda pilih.")

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
                                    
                        except Exception as e:
                            st.error(f"Error menjalankan LDA: {e}")
                else:
                    st.warning("Silakan pilih 1 variabel target (kategorikal) dan minimal 1 variabel prediktor (numerik).")