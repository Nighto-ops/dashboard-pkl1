import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.multivariate.manova import MANOVA

# Diperlukan untuk membaca file Excel
import openpyxl 


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
st.title("Statistical Analysis Tools")

# =================================================================
# SIDEBAR: UPLOAD FILE & IDENTIFIKASI VARIABEL
# =================================================================

st.sidebar.title("Kontrol Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Anda",
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
    # Memisahkan tab berdasarkan Permintaan #1 (Univariat, Bivariat, Multivariat)
    tab_data, tab_uni, tab_bi, tab_multi = st.tabs([
        "Ringkasan Data",
        "Analisis Univariat", 
        "Analisis Bivariat", 
        "Analisis Multivariat"
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
    # TAB 1: ANALISIS UNIVARIAT
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
                    if p_value > 0.05:
                        st.success(f"**Interpretasi:** P-value > 0.05. Data **terdistribusi normal**.")
                    else:
                        st.error(f"**Interpretasi:** P-value <= 0.05. Data **TIDAK terdistribusi normal**.")

    # -------------------------------------------------------------
    # TAB 2: ANALISIS BIVARIAT
    # -------------------------------------------------------------
    with tab_bi:
        st.header("Analisis Bivariat (Dua Variabel)")
        st.info("Analisis ini mengeksplorasi hubungan antara dua variabel.")
        st.markdown("---")

        st.subheader("Hubungan Numerik vs Numerik")
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
                st.info("Mengukur kekuatan dan arah hubungan linear antara dua variabel numerik.")
                corr, p_value = stats.pearsonr(data_bi[bi_x], data_bi[bi_y])
                st.write(f"* **Koefisien Korelasi (r):** `{corr:.4f}`")
                st.write(f"* **P-value:** `{p_value:.4f}`")
                
                if p_value < 0.05:
                    st.success(f"**Interpretasi:** Terdapat korelasi yang signifikan secara statistik (p < 0.05).")
                else:
                    st.error(f"**Interpretasi:** Tidak terdapat korelasi yang signifikan secara statistik (p > 0.05).")
            elif bi_x == bi_y:
                st.warning("Variabel X dan Y tidak boleh sama.")
        
        st.markdown("---")

        st.subheader("Hubungan Kategorikal vs Numerik")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("Membandingkan rata-rata variabel numerik di antara kelompok yang berbeda.")
            cat_col = st.selectbox("Pilih Variabel Kelompok (Kategorikal):", categorical_cols, key='bi_cat')
            num_col = st.selectbox("Pilih Variabel Nilai (Numerik):", numeric_cols, key='bi_num')
        
        with col2:
            if cat_col and num_col:
                groups = df[cat_col].unique()
                
                # Boxplot
                fig_box = px.box(df, x=cat_col, y=num_col, title=f"Distribusi {num_col} berdasarkan {cat_col}", points="all")
                st.plotly_chart(fig_box, use_container_width=True)

                # Menjalankan Uji T atau ANOVA One-Way
                if len(groups) == 2:
                    # Uji T
                    st.write("**Uji T (Independent T-Test)**")
                    st.info("Membandingkan rata-rata dari DUA kelompok.")
                    group1 = df[df[cat_col] == groups[0]][num_col].dropna()
                    group2 = df[df[cat_col] == groups[1]][num_col].dropna()
                    
                    stat, p_value = stats.ttest_ind(group1, group2)
                    st.write(f"* **P-value:** `{p_value:.4f}`")
                    if p_value < 0.05:
                        st.success(f"**Interpretasi:** Terdapat perbedaan rata-rata yang signifikan antara {groups[0]} dan {groups[1]}.")
                    else:
                        st.error(f"**Interpretasi:** Tidak ada perbedaan rata-rata yang signifikan.")
                        
                elif len(groups) > 2:
                    # ANOVA One-Way
                    st.write("**Uji ANOVA One-Way**")
                    st.info("Membandingkan rata-rata dari TIGA ATAU LEBIH kelompok.")
                    
                    # Kita harus membersihkan nama kolom agar sesuai dengan formula Statsmodels (tanpa spasi/simbol)
                    df_clean = df[[cat_col, num_col]].dropna()
                    clean_cat = cat_col.replace(' ', '_').replace('[', '').replace(']', '')
                    clean_num = num_col.replace(' ', '_').replace('[', '').replace(']', '')
                    df_clean.columns = [clean_cat, clean_num]
                    
                    # Formula: 'Nilai ~ C(Kelompok)'
                    # C() memberitahu statsmodels bahwa ini adalah variabel kategorikal
                    formula = f'{clean_num} ~ C({clean_cat})'
                    
                    try:
                        model = smf.ols(formula, data=df_clean).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        st.dataframe(anova_table)
                        
                        p_value_anova = anova_table['PR(>F)'][0]
                        if p_value_anova < 0.05:
                            st.success(f"**Interpretasi:** P-value ({p_value_anova:.4f}) < 0.05. "
                                       "Setidaknya ada satu kelompok yang memiliki rata-rata yang berbeda secara signifikan.")
                        else:
                            st.error(f"**Interpretasi:** P-value ({p_value_anova:.4f}) > 0.05. "
                                     "Tidak ada perbedaan rata-rata yang signifikan antar kelompok.")
                    except Exception as e:
                        st.error(f"Error saat menjalankan ANOVA: {e}. Mungkin karena nama kolom yang tidak valid.")
                
                else:
                    st.warning("Variabel kategorikal hanya memiliki 1 kelompok unik. Tidak bisa diuji.")


    # -------------------------------------------------------------
    # TAB 3: ANALISIS MULTIVARIAT
    # -------------------------------------------------------------
    with tab_multi:
        st.header("Analisis Multivariat (Tiga atau Lebih Variabel)")
        st.info("Analisis yang melibatkan 3 atau lebih variabel secara bersamaan.")
        st.markdown("---")

        st.subheader("Matriks Korelasi")
        st.info("Melihat korelasi linear antara semua variabel numerik secara bersamaan.")
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap
            fig_heatmap = px.imshow(corr_matrix, 
                                    text_auto=True, 
                                    aspect="auto",
                                    title="Heatmap Matriks Korelasi")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.write("Tabel Matriks Korelasi:")
            st.dataframe(corr_matrix)
        else:
            st.warning("Membutuhkan setidaknya 2 variabel numerik untuk matriks korelasi.")
        
        st.markdown("---")

        st.subheader("Regresi Linear Berganda")
        st.info(
            "Memodelkan hubungan antara satu variabel dependen (Y) dan DUA ATAU LEBIH variabel independen (X).\n"
            "Model: **Y = b0 + b1*X1 + b2*X2 + ...**"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            reg_y = st.selectbox("Pilih Variabel Dependen (Y):", numeric_cols, key='reg_y_multi')
            
            # Filter agar Y tidak muncul di pilihan X
            available_x = [col for col in numeric_cols if col != reg_y]
            reg_x = st.multiselect("Pilih Variabel Independen (X) (minimal 2):", available_x, key='reg_x_multi')

        with col2:
            if reg_y and len(reg_x) >= 2:
                # Siapkan data
                data_reg = df[[reg_y] + reg_x].dropna()
                X = data_reg[reg_x]
                y = data_reg[reg_y]
                
                # Tambahkan konstanta (b0)
                X_with_const = sm.add_constant(X)
                
                # Fit model OLS (Ordinary Least Squares)
                model_ols = sm.OLS(y, X_with_const).fit()
                
                st.write("**Ringkasan Model (OLS)**")
                st.info("Ini adalah output standar dari model regresi, perhatikan R-squared dan P-value (P>|t|).")
                # Tampilkan ringkasan dalam st.text_area agar bisa di-scroll
                st.text_area("Ringkasan", model_ols.summary().as_text(), height=400)
                
                # Interpretasi
                r_squared = model_ols.rsquared_adj
                st.success(f"**Interpretasi R-squared (Adj.):** {r_squared*100:.2f}% variasi pada **{reg_y}** "
                           f"dapat dijelaskan oleh variabel-variabel X yang dipilih.")
                
                st.markdown("---")
                
                # --- UJI DIAGNOSTIK (Permintaan #2) ---
                st.write("**:mag: Uji Diagnostik Asumsi Regresi**")
                
                # 1. Uji Multikolinearitas (VIF)
                st.info("**1. Uji Multikolinearitas (VIF)**\n"
                        "Tujuan: Mendeteksi korelasi yang tinggi antar variabel Independen (X).")
                
                vif_data = pd.DataFrame()
                vif_data["Variabel"] = X_with_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                
                # Tampilkan VIF (abaikan 'const')
                st.dataframe(vif_data[vif_data["Variabel"] != 'const'])
                
                if (vif_data[vif_data["Variabel"] != 'const']["VIF"] > 10).any():
                    st.error("**Interpretasi:** Ditemukan VIF > 10. Ini mengindikasikan adanya **multikolinearitas** "
                             "yang kuat. Pertimbangkan untuk menghapus salah satu variabel X yang berkorelasi tinggi.")
                else:
                    st.success("**Interpretasi:** Semua VIF < 10. Tidak ada indikasi multikolinearitas yang kuat.")

                # 2. Uji Heteroskedastisitas (Breusch-Pagan)
                st.info("**2. Uji Heteroskedastisitas (Breusch-Pagan)**\n"
                        "Tujuan: Menguji apakah varians dari residual (error) konstan.\n"
                        "* H0: Varians konstan (Homoskedastisitas)\n"
                        "* H1: Varians tidak konstan (Heteroskedastisitas)")
                
                bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
                bp_p_value = bp_test[1]
                
                st.write(f"* **P-value Uji Breusch-Pagan:** `{bp_p_value:.4f}`")
                
                if bp_p_value < 0.05:
                    st.error(f"**Interpretasi:** P-value < 0.05. Kita menolak H0. "
                             "Ini mengindikasikan adanya **heteroskedastisitas**. "
                             "Pertimbangkan transformasi data (misal: log) atau gunakan *robust standard errors*.")
                else:
                    st.success(f"**Interpretasi:** P-value > 0.05. Kita gagal menolak H0. "
                               "Asumsi **homoskedastisitas** terpenuhi.")
            
            elif reg_y and len(reg_x) < 2:
                st.warning("Silakan pilih setidaknya 2 variabel independen (X) untuk regresi berganda.")

        st.markdown("---")

        st.subheader("ANOVA Two-Way")
        st.info(
            "Tujuan: Membandingkan rata-rata variabel numerik berdasarkan DUA variabel kategorikal.\n"
            "Uji ini melihat efek masing-masing variabel (efek utama) dan efek interaksi keduanya."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            anova2_cat1 = st.selectbox("Pilih Variabel Kategorikal 1 (X1):", categorical_cols, key='a2_c1')
        with col2:
            anova2_cat2 = st.selectbox("Pilih Variabel Kategorikal 2 (X2):", categorical_cols, key='a2_c2')
        with col3:
            anova2_num = st.selectbox("Pilih Variabel Dependen (Y):", numeric_cols, key='a2_n')
        
        if anova2_cat1 and anova2_cat2 and anova2_num and anova2_cat1 != anova2_cat2:
            if st.button("Jalankan ANOVA Two-Way"):
                # Bersihkan nama kolom
                df_clean = df[[anova2_cat1, anova2_cat2, anova2_num]].dropna()
                c1, c2, n = [col.replace(' ', '_') for col in df_clean.columns]
                df_clean.columns = [c1, c2, n]
                
                # Formula: Y ~ C(X1) + C(X2) + C(X1):C(X2)
                # C(X1):C(X2) adalah efek interaksi
                formula = f'{n} ~ C({c1}) + C({c2}) + C({c1}):C({c2})'
                
                try:
                    model = smf.ols(formula, data=df_clean).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.write("**Hasil ANOVA Two-Way:**")
                    st.dataframe(anova_table)
                    
                    st.info("**Interpretasi (lihat kolom 'PR(>F)'):**\n"
                            f"* **Efek {c1}:** P-value `{anova_table['PR(>F)'][0]:.4f}`. Jika < 0.05, ada efek utama dari {c1}.\n"
                            f"* **Efek {c2}:** P-value `{anova_table['PR(>F)'][1]:.4f}`. Jika < 0.05, ada efek utama dari {c2}.\n"
                            f"* **Efek Interaksi ({c1}:{c2}):** P-value `{anova_table['PR(>F)'][2]:.4f}`. Jika < 0.05, "
                            "ada efek interaksi (efek satu variabel bergantung pada level variabel lainnya).")
                except Exception as e:
                    st.error(f"Error saat menjalankan ANOVA: {e}. Pastikan variabel memiliki > 1 level.")
        
        st.markdown("---")
        
        st.subheader("MANOVA (Multivariate Analysis of Variance)")
        st.info(
            "Tujuan: Seperti ANOVA, tetapi untuk DUA ATAU LEBIH variabel dependen (Y) secara bersamaan.\n"
            "Contoh: Apakah 'IPK' **dan** 'Lama Studi' berbeda berdasarkan 'Fakultas'?"
        )
        
        st.write("**MANOVA One-Way**")
        col1, col2 = st.columns([1, 2])
        with col1:
            manova1_cat = st.selectbox("Pilih Variabel Kelompok (X):", categorical_cols, key='m1_c')
            manova1_num = st.multiselect("Pilih Variabel Dependen (Y) (minimal 2):", numeric_cols, key='m1_n')
        
        if manova1_cat and len(manova1_num) >= 2:
            if st.button("Jalankan MANOVA One-Way"):
                # Bersihkan nama
                df_clean = df[[manova1_cat] + manova1_num].dropna()
                clean_cols = [col.replace(' ', '_') for col in df_clean.columns]
                df_clean.columns = clean_cols
                
                c1 = clean_cols[0]
                n_vars = clean_cols[1:]
                
                # Formula: Y1 + Y2 ~ C(X)
                formula = f'{" + ".join(n_vars)} ~ C({c1})'
                
                try:
                    model = MANOVA.from_formula(formula, data=df_clean)
                    mv_test = model.mv_test()
                    
                    st.write("**Hasil MANOVA (Wilks' Lambda):**")
                    st.dataframe(mv_test.results[c1]['stat'])
                    
                    p_value_manova = mv_test.results[c1]['stat'].loc['Wilks\' lambda', 'Pr > F']
                    if p_value_manova < 0.05:
                        st.success(f"**Interpretasi:** P-value ({p_value_manova:.4f}) < 0.05. "
                                   "Terdapat perbedaan yang signifikan secara statistik antar kelompok "
                                   f"pada setidaknya satu dari variabel dependen ({', '.join(n_vars)}).")
                    else:
                        st.error(f"**Interpretasi:** P-value ({p_value_manova:.4f}) > 0.05. "
                                 "Tidak ada perbedaan yang signifikan antar kelompok.")
                except Exception as e:
                    st.error(f"Error menjalankan MANOVA: {e}")
        
        st.write("**MANOVA Two-Way**")
        col1, col2, col3 = st.columns(3)
        with col1:
            manova2_cat1 = st.selectbox("Pilih Kelompok 1 (X1):", categorical_cols, key='m2_c1')
        with col2:
            manova2_cat2 = st.selectbox("Pilih Kelompok 2 (X2):", categorical_cols, key='m2_c2')
        with col3:
            manova2_num = st.multiselect("Pilih Variabel Dependen (Y) (minimal 2):", numeric_cols, key='m2_n')
        
        if manova2_cat1 and manova2_cat2 and len(manova2_num) >= 2 and manova2_cat1 != manova2_cat2:
            if st.button("Jalankan MANOVA Two-Way"):
                # Bersihkan nama
                df_clean = df[[manova2_cat1, manova2_cat2] + manova2_num].dropna()
                clean_cols = [col.replace(' ', '_') for col in df_clean.columns]
                df_clean.columns = clean_cols
                
                c1, c2 = clean_cols[0], clean_cols[1]
                n_vars = clean_cols[2:]
                
                # Formula: Y1 + Y2 ~ C(X1) + C(X2) + C(X1):C(X2)
                formula = f'{" + ".join(n_vars)} ~ C({c1}) + C({c2}) + C({c1}):C({c2})'
                
                try:
                    model = MANOVA.from_formula(formula, data=df_clean)
                    mv_test = model.mv_test()
                    
                    st.write("**Hasil MANOVA (Wilks' Lambda):**")
                    st.info("Perhatikan P-value (Pr > F) untuk setiap efek (X1, X2, dan Interaksi)")
                    st.dataframe(mv_test.summary_frame)
                except Exception as e:
                    st.error(f"Error menjalankan MANOVA: {e}")