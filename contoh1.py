import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.multivariate.manova import MANOVA
import openpyxl 

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

# =================================================================
# KONFIGURASI HALAMAN UTAMA
# =================================================================
st.set_page_config(layout="wide")
st.title("Statistical Analysis Tools v3")

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

if df is not None:
    # PERUBAHAN: Struktur Tab dirombak agar lebih logis
    tab_data, tab_uni, tab_bi, tab_reg, tab_anova, tab_manova = st.tabs([
        "Ringkasan Data",
        "Analisis Univariat", 
        "Analisis Bivariat",
        "Model Regresi", # BARU (Permintaan #1)
        "ANOVA",         # BARU (Permintaan #2)
        "MANOVA"         # BARU (Permintaan #3)
    ])

    # -------------------------------------------------------------
    # TAB 0: RINGKASAN DATA
    # -------------------------------------------------------------
    with tab_data:
        st.header("Ringkasan dan Tampilan Data")
        
        st.subheader("Ringkasan Statistik (Variabel Numerik)")
        st.info("Statistik deskriptif dasar untuk semua kolom numerik dalam data Anda.")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        st.subheader("Tampilan Data Mentah (50 Baris Pertama)")
        st.info("Tampilan 50 baris pertama dari data yang Anda upload.")
        st.dataframe(df.head(50), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 1: ANALISIS UNIVARIAT (Satu Variabel)
    # -------------------------------------------------------------
    with tab_uni:
        st.header("Analisis Univariat (Satu Variabel)")
        st.info("Analisis ini berfokus pada satu variabel pada satu waktu.")
        st.markdown("---")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi (Histogram)")
            st.info("Melihat sebaran frekuensi dari sebuah variabel numerik.")
            hist_col = st.selectbox("Pilih variabel:", numeric_cols, key='hist_col')
            if hist_col:
                fig_hist = px.histogram(df, x=hist_col, title=f'Histogram untuk {hist_col}', marginal="box")
                st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Uji Normalitas (Shapiro-Wilk)")
            st.info("Menguji apakah data terdistribusi normal. Penting untuk banyak uji inferensia.")
            norm_col = st.selectbox("Pilih variabel:", numeric_cols, key='norm_col')
            
            if st.button("Jalankan Uji Normalitas", key='norm_btn'):
                data_to_test = df[norm_col].dropna()
                if len(data_to_test) < 3:
                    st.error("Uji Normalitas memerlukan setidaknya 3 sampel.")
                else:
                    stat, p_value = stats.shapiro(data_to_test)
                    st.write(f"**P-value:** `{p_value:.4f}`")
                    
                    # PERUBAHAN: Tambah Interpretasi Awam (Permintaan #4)
                    if p_value > 0.05:
                        st.success(f"**Interpretasi Teknis:** P-value > 0.05. Gagal menolak H0. Data **terdistribusi normal**.")
                        st.info("💡 **Bahasa Awam:** Data Anda terlihat simetris seperti lonceng (bell curve) yang normal.")
                    else:
                        st.error(f"**Interpretasi Teknis:** P-value <= 0.05. Menolak H0. Data **TIDAK terdistribusi normal**.")
                        st.info("💡 **Bahasa Awam:** Data Anda miring/tidak rata, bukan seperti lonceng (bell curve) yang normal.")

    # -------------------------------------------------------------
    # TAB 2: ANALISIS BIVARIAT (Dua Variabel)
    # -------------------------------------------------------------
    with tab_bi:
        st.header("Analisis Bivariat (Dua Variabel)")
        st.info("Analisis ini mengeksplorasi hubungan antara dua variabel.")
        st.markdown("---")

        st.subheader("Hubungan Numerik vs Numerik (Korelasi)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("Pilih dua variabel numerik untuk melihat hubungan linear dan korelasinya.")
            bi_x = st.selectbox("Pilih Variabel X:", numeric_cols, key='bi_x')
            bi_y = st.selectbox("Pilih Variabel Y:", numeric_cols, key='bi_y')
        
        with col2:
            if bi_x and bi_y and bi_x != bi_y:
                data_bi = df[[bi_x, bi_y]].dropna()
                
                # Plot
                fig_scatter = px.scatter(data_bi, x=bi_x, y=bi_y, title=f"Scatter Plot: {bi_y} vs {bi_x}", trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Uji Korelasi
                st.write("**Uji Korelasi (Pearson)**")
                st.info("Mengukur kekuatan dan arah hubungan linear antara dua variabel numerik. (Nilai antara -1 dan 1)")
                corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                st.write(f"* **Koefisien Korelasi (r):** `{corr:.4f}`")
                st.write(f"* **P-value:** `{p_value:.4f}`")
                
                # PERUBAHAN: Tambah Interpretasi Awam (Permintaan #4)
                if p_value < 0.05:
                    st.success(f"**Interpretasi Teknis:** Terdapat korelasi yang signifikan secara statistik (p < 0.05).")
                    st.info(f"💡 **Bahasa Awam:** Ya, ada hubungan yang nyata antara {bi_x} dan {bi_y}. Saat satu naik, yang lain cenderung naik (jika r positif) atau turun (jika r negatif).")
                else:
                    st.error(f"**Interpretasi Teknis:** Tidak terdapat korelasi yang signifikan secara statistik (p > 0.05).")
                    st.info(f"💡 **Bahasa Awam:** Tidak ditemukan hubungan yang jelas antara {bi_x} dan {bi_y}. Perubahannya terlihat acak.")
            elif bi_x == bi_y:
                st.warning("Variabel X dan Y tidak boleh sama.")
        
        st.markdown("---")
        
        st.subheader("Hubungan Kategorikal vs Numerik (Uji T)")
        st.info("Membandingkan rata-rata variabel numerik di antara **DUA** kelompok.")
        col3, col4 = st.columns([1, 2])
        with col3:
            cat_col_t = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='bi_cat_t')
            num_col_t = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='bi_num_t')
        
        with col4:
            if cat_col_t and num_col_t:
                groups_t = df[cat_col_t].unique()
                
                if len(groups_t) == 2:
                    st.write("**Uji T (Independent T-Test)**")
                    fig_box = px.box(df, x=cat_col_t, y=num_col_t, title=f"Distribusi {num_col_t} berdasarkan {cat_col_t}", points="all")
                    st.plotly_chart(fig_box, use_container_width=True)

                    group1 = df[df[cat_col_t] == groups_t[0]][num_col_t].dropna()
                    group2 = df[df[cat_col_t] == groups_t[1]][num_col_t].dropna()
                    
                    stat, p_value = stats.ttest_ind(group1, group2)
                    st.write(f"* **P-value:** `{p_value:.4f}`")
                    
                    # PERUBAHAN: Tambah Interpretasi Awam (Permintaan #4)
                    if p_value < 0.05:
                        st.success(f"**Interpretasi Teknis:** Terdapat perbedaan rata-rata yang signifikan antara {groups_t[0]} dan {groups_t[1]}.")
                        st.info(f"💡 **Bahasa Awam:** Ya, ada perbedaan nilai yang nyata antara kelompok {groups_t[0]} dan {groups_t[1]}.")
                    else:
                        st.error(f"**Interpretasi Teknis:** Tidak ada perbedaan rata-rata yang signifikan.")
                        st.info(f"💡 **Bahasa Awam:** Tidak, perbedaan nilai antara kedua kelompok sepertinya hanya kebetulan saja (tidak signifikan).")
                else:
                    st.warning(f"Variabel '{cat_col_t}' tidak memiliki 2 kelompok. Pindah ke tab 'ANOVA' untuk menguji 3 kelompok atau lebih.")
        
        st.markdown("---")
        
        st.subheader("Matriks Korelasi (Multivariat)")
        st.info("Melihat korelasi linear antara semua variabel numerik secara bersamaan.")
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap Matriks Korelasi")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Membutuhkan setidaknya 2 variabel numerik untuk matriks korelasi.")
            
    # -------------------------------------------------------------
    # TAB 3: MODEL REGRESI (PERMINTAAN #1)
    # -------------------------------------------------------------
    with tab_reg:
        st.header("Model Regresi")
        st.info("Memodelkan hubungan antara variabel dependen (Y) dan variabel independen (X) untuk membuat prediksi.")
        st.markdown("---")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            reg_y = st.selectbox("Pilih Variabel Dependen (Y):", numeric_cols, key='reg_y')
            available_x = [col for col in numeric_cols if col != reg_y]
            reg_x = st.multiselect("Pilih Variabel IndependEN (X):", available_x, key='reg_x')
        
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
                
                # Transformasi X
                poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_poly = poly_features.fit_transform(X_simple)
                
                # Buat model
                model = LinearRegression()
                model.fit(X_poly, y_simple)
                y_pred = model.predict(X_poly)
                r2 = r2_score(y_simple, y_pred)
                
                # Plot
                plot_df = pd.DataFrame({'X': X_simple.iloc[:, 0], 'y_true': y_simple, 'y_pred': y_pred}).sort_values(by='X')
                fig_poly = px.scatter(plot_df, x='X', y='y_true', title=f"Model Regresi (Derajat {poly_degree})")
                fig_poly.add_trace(go.Scatter(x=plot_df['X'], y=plot_df['y_pred'], name='Garis Prediksi', line=dict(color='red')))
                st.plotly_chart(fig_poly, use_container_width=True)
                
                # Interpretasi
                st.write(f"**R-squared:** `{r2:.4f}`")
                st.success(f"**Interpretasi Teknis:** {r2*100:.2f}% variasi pada **{reg_y}** dapat dijelaskan oleh model ini.")
                st.info(f"💡 **Bahasa Awam:** Model ini {r2*100:.2f}% akurat dalam memprediksi {reg_y} menggunakan {reg_x[0]}.")

        elif reg_y and len(reg_x) >= 2:
             # --- JIKA 2+ VAR X: REGRESI LINEAR BERGANDA ---
            st.subheader("Regresi Linear Berganda")
            st.info(f"Model: **{reg_y} = b0 + b1*{reg_x[0]} + b2*{reg_x[1]} + ...**")

            with col2:
                # Siapkan data
                data_reg = df[[reg_y] + reg_x].dropna()
                X_multi = data_reg[reg_x]
                y_multi = data_reg[reg_y]
                
                X_with_const = sm.add_constant(X_multi)
                model_ols = sm.OLS(y_multi, X_with_const).fit()
                
                st.write("**Ringkasan Model (OLS)**")
                st.text_area("Ringkasan", model_ols.summary().as_text(), height=400)
                
                # Interpretasi R-squared
                r_squared = model_ols.rsquared_adj
                st.success(f"**Interpretasi R-squared (Adj.):** {r_squared*100:.2f}% variasi pada **{reg_y}** "
                           f"dapat dijelaskan oleh variabel-variabel X yang dipilih.")
                st.info(f"💡 **Bahasa Awam:** Kombinasi variabel X ini {r_squared*100:.2f}% akurat dalam menjelaskan {reg_y}.")
                
                st.markdown("---")
                
                # --- UJI DIAGNOSTIK ---
                st.write("**:mag: Uji Diagnostik Asumsi Regresi**")
                
                # 1. Uji Multikolinearitas (VIF)
                st.info("**1. Uji Multikolinearitas (VIF)**\n"
                        "Tujuan: Mendeteksi korelasi yang tinggi antar variabel Independen (X).")
                
                vif_data = pd.DataFrame()
                vif_data["Variabel"] = X_with_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                st.dataframe(vif_data[vif_data["Variabel"] != 'const'])
                
                if (vif_data[vif_data["Variabel"] != 'const']["VIF"] > 10).any():
                    st.error("**Interpretasi Teknis:** Ditemukan VIF > 10. Ini mengindikasikan adanya **multikolinearitas** kuat.")
                    st.info("💡 **Bahasa Awam:** Variabel X Anda 'saling tumpang tindih'. Mereka mengukur hal yang terlalu mirip, sehingga sulit untuk membedakan efeknya masing-masing.")
                else:
                    st.success("**Interpretasi Teknis:** Semua VIF < 10. Tidak ada indikasi multikolinearitas yang kuat.")
                    st.info("💡 **Bahasa Awam:** Bagus! Variabel X Anda 'berdiri sendiri' dan tidak saling tumpang tindih.")

                # 2. Uji Heteroskedastisitas (Breusch-Pagan)
                st.info("**2. Uji Heteroskedastisitas (Breusch-Pagan)**\n"
                        "Tujuan: Menguji apakah varians dari residual (error) konstan.")
                
                bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                bp_p_value = bp_test[1]
                
                st.write(f"* **P-value Uji Breusch-Pagan:** `{bp_p_value:.4f}`")
                
                if bp_p_value < 0.05:
                    st.error(f"**Interpretasi Teknis:** P-value < 0.05. Menolak H0. Terindikasi **heteroskedastisitas**.")
                    st.info("💡 **Bahasa Awam:** Tingkat kesalahan (error) pada model Anda tidak konsisten. Prediksi Anda mungkin lebih akurat untuk data rendah dan kurang akurat untuk data tinggi (atau sebaliknya).")
                else:
                    st.success(f"**Interpretasi Teknis:** P-value > 0.05. Gagal menolak H0. Asumsi **homoskedastisitas** terpenuhi.")
                    st.info("💡 **Bahasa Awam:** Bagus! Tingkat kesalahan (error) pada model Anda konsisten di semua level data.")
        
        elif not reg_y or not reg_x:
            st.info("Silakan pilih minimal 1 Variabel Dependen (Y) dan 1 Variabel Independen (X) untuk memulai.")

    # -------------------------------------------------------------
    # TAB 4: ANOVA (PERMINTAAN #2)
    # -------------------------------------------------------------
    with tab_anova:
        st.header("Analisis Varians (ANOVA)")
        st.info("Membandingkan rata-rata variabel numerik di antara beberapa kelompok.")
        st.markdown("---")
        
        # --- ANOVA ONE-WAY ---
        st.subheader("ANOVA One-Way")
        st.info("Membandingkan rata-rata dari **TIGA ATAU LEBIH** kelompok (berdasarkan 1 variabel kategorikal).")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            cat_col_anova1 = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='a1_cat')
            num_col_anova1 = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='a1_num')
        
        with col2:
            if cat_col_anova1 and num_col_anova1:
                groups_a1 = df[cat_col_anova1].unique()
                
                if len(groups_a1) > 2:
                    # Boxplot
                    fig_box_a1 = px.box(df, x=cat_col_anova1, y=num_col_anova1, title=f"Distribusi {num_col_anova1} berdasarkan {cat_col_anova1}", points="all")
                    st.plotly_chart(fig_box_a1, use_container_width=True)

                    # Bersihkan nama kolom
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
                        st.error(f"Error saat menjalankan ANOVA: {e}. Mungkin karena nama kolom yang tidak valid.")
                else:
                    st.warning(f"Variabel '{cat_col_anova1}' hanya memiliki {len(groups_a1)} kelompok. Gunakan Uji T di tab 'Analisis Bivariat' untuk 2 kelompok.")

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
                    c1, c2, n = [col.replace(' ', '_') for col in df_clean.columns]
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
                            st.success(f"**{anova2_cat1}**: BERPENGARUH SIGNIFIKAN. (Ada perbedaan nilai berdasarkan {anova2_cat1})")
                        else:
                            st.error(f"**{anova2_cat1}**: TIDAK BERPENGARUH SIGNIFIKAN.")
                        
                        if p_c2 < 0.05:
                            st.success(f"**{anova2_cat2}**: BERPENGARUH SIGNIFIKAN. (Ada perbedaan nilai berdasarkan {anova2_cat2})")
                        else:
                            st.error(f"**{anova2_cat2}**: TIDAK BERPENGARUH SIGNIFIKAN.")
                            
                        if p_int < 0.05:
                            st.warning(f"**INTERAKSI ({anova2_cat1}:{anova2_cat2})**: ADA EFEK INTERAKSI SIGNIFIKAN.")
                            st.info(f"💡 **Bahasa Awam:** Efek dari {anova2_cat1} pada nilai **bergantung** pada {anova2_cat2} (dan sebaliknya). Keduanya tidak bisa dilihat terpisah.")
                        else:
                            st.info(f"**INTERAKSI ({anova2_cat1}:{anova2_cat2})**: TIDAK ADA EFEK INTERAKSI SIGNIFIKAN.")
                            st.info(f"💡 **Bahasa Awam:** Efek {anova2_cat1} dan {anova2_cat2} bersifat independen (terpisah).")

                    except Exception as e:
                        st.error(f"Error saat menjalankan ANOVA: {e}. Pastikan variabel memiliki > 1 level.")
            elif anova2_cat1 == anova2_cat2:
                st.warning("Variabel Kategorikal 1 dan 2 tidak boleh sama.")

    # -------------------------------------------------------------
    # TAB 5: MANOVA (PERMINTAAN #3)
    # -------------------------------------------------------------
    with tab_manova:
        st.header("Analisis MANOVA")
        st.info("Seperti ANOVA, tetapi untuk **DUA ATAU LEBIH** variabel dependen (Y) secara bersamaan.")
        st.markdown("---")

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
                        # PERBAIKAN (Permintaan #3): Menggunakan summary_frame
                        st.dataframe(mv_test.summary_frame)
                        
                        # PERBAIKAN (Permintaan #3): Mengambil p-value dari summary_frame
                        p_value_manova = mv_test.summary_frame.loc[f'C({c1})', "Wilks' lambda_Pr > F"]
                        
                        if p_value_manova < 0.05:
                            st.success(f"**Interpretasi Teknis:** P-value ({p_value_manova:.4f}) < 0.05. "
                                       f"Terdapat perbedaan yang signifikan antar kelompok pada setidaknya satu dari variabel dependen.")
                            st.info(f"💡 **Bahasa Awam:** Ya, {manova1_cat} memiliki pengaruh yang nyata terhadap nilai-nilai ({', '.join(manova1_num)}) secara bersamaan.")
                        else:
                            st.error(f"**Interpretasi Teknis:** P-value ({p_value_manova:.4f}) > 0.05. "
                                     "Tidak ada perbedaan yang signifikan antar kelompok.")
                            st.info(f"💡 **Bahasa Awam:** Tidak, {manova1_cat} sepertinya tidak memiliki pengaruh apa-apa terhadap nilai-nilai ({', '.join(manova1_num)}) secara bersamaan.")
                    except Exception as e:
                        st.error(f"Error menjalankan MANOVA: {e}. Pastikan kelompok memiliki lebih dari 1 anggota.")
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
                        p_c1 = mv_test.summary_frame.loc[f'C({c1})', "Wilks' lambda_Pr > F"]
                        p_c2 = mv_test.summary_frame.loc[f'C({c2})', "Wilks' lambda_Pr > F"]
                        p_int = mv_test.summary_frame.loc[f'C({c1}):C({c2})', "Wilks' lambda_Pr > F"]

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