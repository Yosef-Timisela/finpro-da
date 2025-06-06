import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import scikit_posthocs as sp
import statsmodels.api as sm

from scipy import stats
from scipy.stats import shapiro, skew, kurtosis, levene, kruskal
from scipy.signal import argrelextrema
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

pd.set_option('display.max_rows', None)  # Menampilkan semua baris
pd.set_option('display.max_columns', None)  # Jika ada banyak kolom



def Project():
    st.markdown("<h1 style='text-align: center;'>A/B Testing for Promotion Strategy Research</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left;'>Background</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>A fast-food chain tests three marketing campaigns for a new menu item. The item launches in randomly selected markets," \
    "each using a different campaign. Weekly sales are tracked for four weeks to determine which campaign is most effective.</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Question?</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>Do the 3 promotion strategies lead to different levels of sales performance?</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left;'>Goals</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>Analyze using A/B Testing to determine which marketing strategy is the most effective.</div>", unsafe_allow_html=True)
    
    st.subheader('')
    with st.expander("Data Exploration"):
        st.markdown("<h2 style='text-align: center;'>Data Description</h2>", unsafe_allow_html=True)
        st.write('MarketID: unique identifier for market')
        st.write('MarketSize: size of market area by sales')
        st.write('LocationID: unique identifier for store location')
        st.write('AgeOfStore: age of store in years')
        st.write('Promotion: one of three promotions that were tested')
        st.write('week: one of four weeks when the promotions were run')
        st.write('SalesInThousands: sales amount for a specific LocationID, Promotion, and week')

        df_marketing = pd.read_csv('WA_Marketing-Campaign.csv')
        st.markdown("<h2 style='text-align: justify;'>Datasets</h2>", unsafe_allow_html=True)
        st.dataframe(df_marketing)
        if 'Unnamed: 0' in df_marketing.columns:
             df_marketing.drop(columns='Unnamed: 0', inplace=True)
        info_df = pd.DataFrame({
            "Kolom": df_marketing.columns,
            "Non-Null Count": df_marketing.notnull().sum().values,
            "Tipe Data": df_marketing.dtypes.values
            })
        
        df_marketing = df_marketing.rename(columns={
            'MarketID': 'market_id',
            'MarketSize': 'market_size',
            'LocationID': 'location_id',
            'AgeOfStore': 'age_of_store',
            'Promotion': 'promotion',
            'SalesInThousands': 'sales(in_thousands)'
        })

        # Ubah tipe data kolom tertentu menjadi string (object)
        df_marketing['market_id'] = df_marketing['market_id'].astype(str)
        df_marketing['location_id'] = df_marketing['location_id'].astype(str)
        df_marketing['promotion'] = df_marketing['promotion'].astype(str)
        df_marketing['week'] = df_marketing['week'].astype(str)

        # Tampilkan data dan info dengan Streamlit
        st.subheader("Preview Data (Top 10 Rows)")
        st.dataframe(df_marketing.head(10))

        st.subheader("📋 Data Information")
        st.dataframe(info_df)

        st.markdown("<h2 style='text-align: center;'>Converting Data Types</h2>", unsafe_allow_html=True)

        st.subheader("Datasets")
        st.dataframe(df_marketing)
        buffer_after = df_marketing.dtypes.astype(str).to_string()
        st.text("Results:\n" + buffer_after)

        # 🔁 Konversi kolom menjadi string
        df_marketing['market_id'] = df_marketing['market_id'].astype(str)
        df_marketing['location_id'] = df_marketing['location_id'].astype(str)
        df_marketing['promotion'] = df_marketing['promotion'].astype(str)
        df_marketing['week'] = df_marketing['week'].astype(str)
        
        
        df = df_marketing.copy()
        df = df.groupby(
            ['market_id', 'market_size', 'location_id', 'age_of_store', 'promotion'])['sales(in_thousands)'].sum().reset_index()

        # Fungsi distribusi versi Streamlit
        def distribution_check(data, variable):
            st.subheader("Data Distribution Check")

            fig, axes = plt.subplots(
                nrows=len(variable), ncols=3,
                figsize=(16, 4 * len(variable))
            )

            if len(variable) == 1:
                var = variable[0]

                # Histogram
                sns.histplot(data[var], bins=30, kde=True, ax=axes[0])
                axes[0].set_title(f'Histogram of {var}')
                axes[0].grid(axis='y', color='grey', linestyle='--', alpha=0.5)

                # Q-Q Plot
                stats.probplot(data[var], dist='norm', plot=axes[1])
                axes[1].set_title(f'Q-Q Plot of {var}')
                axes[1].grid(axis='y', color='grey', linestyle='--', alpha=0.5)

                # Box Plot
                sns.boxplot(data[var], orient='h', ax=axes[2])
                axes[2].set_title(f'Box Plot of {var}')
                axes[2].grid(axis='y', color='grey', linestyle='--', alpha=0.5)

            else:
                for i, var in enumerate(variable):

                    # Histogram
                    sns.histplot(data[var], bins=30, kde=True, ax=axes[i, 0])
                    axes[i, 0].set_title(f'Histogram of {var}')
                    axes[i, 0].grid(axis='y', color='grey', linestyle='--', alpha=0.5)

                    # Q-Q Plot
                    stats.probplot(data[var], dist='norm', plot=axes[i, 1])
                    axes[i, 1].set_title(f'Q-Q Plot of {var}')
                    axes[i, 1].grid(axis='y', color='grey', linestyle='--', alpha=0.5)

                    # Box Plot
                    sns.boxplot(data[var], orient='h', ax=axes[i, 2])
                    axes[i, 2].set_title(f'Box Plot of {var}')
                    axes[i, 2].grid(axis='y', color='grey', linestyle='--', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)

        # Contoh pemanggilan fungsi di Streamlit app
        st.markdown("<h2 style='text-align: center;'>Distribution Data Visualization</h2>", unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        selected_vars = st.multiselect(
            "Select numerical variable for checking the distribution:",
            numeric_cols,
            default=['sales(in_thousands)']
        )

        if selected_vars:
            distribution_check(df, selected_vars)
        else:
            st.info("Pilih minimal satu variabel untuk memulai visualisasi.")

        # EDA
        st.markdown("<h2 style='text-align: center;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
        st.write("1. What is the relationship between store age and sales performance?\
                 Is store age positively correlated with sales performance?")

        # Membuat figure dan axes
        fig, ax = plt.subplots(figsize=(12, 5))

        # Scatter plot
        ax.scatter(df['age_of_store'], df['sales(in_thousands)'])
        ax.set_title('Scatter Plot Age of Store vs. Total Sales (In Thousands)')
        ax.set_xlabel('Age of Store')
        ax.set_ylabel('sales(in_thousands)')
        ax.grid(color='gray', linestyle='--', alpha=0.5)

        # Menambahkan trend line
        z = np.polyfit(df['age_of_store'], df['sales(in_thousands)'], 1)
        p = np.poly1d(z)
        ax.plot(df['age_of_store'], p(df['age_of_store']), "r-", linewidth=0.5)

        # Tampilkan plot di Streamlit
        st.pyplot(fig)

        # Judul Aplikasi
        st.write("2. Which type of promotion achieved the highest sales figures? Compare also in each market size.")

        # Grouping dan Pivot Table
        multiple_bar = df.groupby(['market_size', 'promotion'])['sales(in_thousands)'].mean().reset_index(name='avg_sales')
        multiple_bar = multiple_bar.sort_values(by='avg_sales', ascending=False)

        stacked_data = pd.pivot_table(
            multiple_bar,
            index='market_size',
            columns='promotion',
            values='avg_sales',
            margins=True,
            aggfunc='mean'
        )

        stacked_data = stacked_data.drop(index='All').sort_values(by='All').drop(columns='All')

        # Buat figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Definisikan posisi dan warna bar
        bar_positions = np.arange(len(stacked_data.index))
        bottom_values = np.zeros(len(stacked_data.index))
        colors = {'1': '#AA0000', '2': '#00AA00', '3': '#0000AA'}

        # Plot bar horizontal bertumpuk
        for label in stacked_data.columns:
            ax.barh(
                bar_positions,
                stacked_data[label],
                color=colors.get(label, '#333333'),
                edgecolor='white',
                label=f'Promotion {label}',
                left=bottom_values,
            )
            for i, value in enumerate(stacked_data[label]):
                if value > 0:
                    ax.text(
                        bottom_values[i] + value / 2,
                        i,
                        f'{value:.2f}',
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='white'
                    )
            bottom_values += stacked_data[label]

        # Atur tampilan grafik
        ax.set_yticks(bar_positions)
        ax.set_yticklabels(stacked_data.index)
        ax.set_title('Average Sales by Promotion Type for Each Market Size')
        ax.set_xlabel('Average Sales')
        ax.set_ylabel('Market Size')
        ax.legend(title='Promotion Type')

        # Tampilkan grafik di Streamlit
        st.pyplot(fig)

        st.write("3. What is the average sales over the last 4 weeks at each location?")

        # Hitung total sales per location
        sales_per_store = df.groupby('location_id')['sales(in_thousands)'].sum().reset_index()
        sales_per_store = sales_per_store.sort_values(by='sales(in_thousands)', ascending=False)

        # Buat plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=sales_per_store.head(10),
            x='location_id',
            y='sales(in_thousands)',
            color='cyan',
            ax=ax
        )

        ax.set_title('Total Sales per Location (Thousands)')
        ax.set_ylim(0, 420)
        ax.set_ylabel('Total Sales')
        ax.set_xlabel('Location ID')
        ax.grid(axis='y', color='grey', linestyle='--', alpha=0.5)

        # Tambahkan label di atas bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')

        plt.tight_layout()

        # Tampilkan plot di Streamlit
        st.pyplot(fig)

        st.write("4. Which locations have total sales above the average?\
                 Which type of promotion has the greatest impact on boosting sales at these locations?")

        # Hitung rata-rata sales per lokasi dan promosi
        multiple_bar = df.groupby(['location_id', 'promotion'])['sales(in_thousands)'].mean().reset_index()
        multiple_bar = multiple_bar.sort_values(by='sales(in_thousands)', ascending=False)

        # Hitung rata-rata keseluruhan
        avg_sales_all = multiple_bar['sales(in_thousands)'].mean()
        st.write(f"**Average Sales (All):** {avg_sales_all:.2f} thousand")

        # Filter hanya yang di atas rata-rata
        above_avg = multiple_bar[multiple_bar['sales(in_thousands)'] > avg_sales_all]
        st.write(f"**Number of location & promotion pairs above average:** {len(above_avg)}")

        # Buat plot
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(
            data=above_avg,
            x='location_id',
            y='sales(in_thousands)',
            hue='promotion',
            ax=ax
        )

        ax.set_title('Locations with Total Sales Above Average')
        ax.set_ylabel('Sales (In Thousands)')
        ax.set_xlabel('Location ID')
        ax.grid(axis='y', color='grey', linestyle='--', alpha=0.5)

        # Tambahkan label ke bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f')

        plt.tight_layout()

        # Tampilkan grafik di Streamlit
        st.pyplot(fig)

        st.write("5. How are the sales performances for each market size?\
                 Is market size positively correlated with sales performance?")
        
        # Hitung rata-rata sales per market size
        sales_per_size = df.groupby('market_size')['sales(in_thousands)'] \
            .mean() \
            .rename('avg_sales(in_thousands)') \
            .reset_index() \
            .sort_values(by='avg_sales(in_thousands)', ascending=False)

        # Tampilkan dataframe (opsional)
        st.dataframe(sales_per_size)

        # Buat plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=sales_per_size,
            x='market_size',
            y='avg_sales(in_thousands)',
            color='magenta',
            ax=ax
        )

        ax.set_title('Avg Sales per Market Size (In Thousands)')
        ax.set_ylim(0, 320)
        ax.set_ylabel('Average Sales')
        ax.set_xlabel('Market Size')
        ax.grid(axis='y', color='grey', linestyle='--', alpha=0.5)

        # Tambahkan label ke bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')

        plt.tight_layout()

        # Tampilkan grafik di Streamlit
        st.pyplot(fig)

        # Group data
        df_grouped = df.groupby('market_size')['sales(in_thousands)'] \
            .agg(['count', 'sum']) \
            .rename(columns={'count': 'Total Market', 'sum': 'Total Sales'}) \
            .reset_index()

        # Tampilkan data (opsional)
        st.dataframe(df_grouped)

        # Membuat plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart untuk Total Market
        ax1.bar(
            df_grouped['market_size'],
            df_grouped['Total Market'],
            color='orange',
            label='Total Market'
        )
        ax1.set_xlabel('Market Size')
        ax1.set_ylabel('Total Market', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(axis='y', color='grey', linestyle='--', alpha=0.5)
        ax1.set_title('Total Market and Total Sales by Market Size')

        # Line chart untuk Total Sales
        ax2 = ax1.twinx()
        ax2.plot(
            df_grouped['market_size'],
            df_grouped['Total Sales'],
            color='red',
            marker='o',
            linestyle='-',
            linewidth=2.5,
            label='Total Sales'
        )
        ax2.set_ylabel('Total Sales', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 15000)

        # Menampilkan legend gabungan
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))

        # Tampilkan plot di Streamlit
        st.pyplot(fig)

        st.write("6. What are the sales trends for each type of promotion?")

        # Group data by promotion and week
        sales_per_promo_week = df_marketing.groupby(['promotion', 'week'])['sales(in_thousands)'] \
            .mean().reset_index()

        # Buat plot
        fig, ax = plt.subplots(figsize=(9, 5))

        sns.lineplot(
            data=sales_per_promo_week,
            x='week',
            y='sales(in_thousands)',
            hue='promotion',
            linewidth=3,
            marker='o',
            palette=['#FF0000', '#2AAF70', '#0000FF'],
            ax=ax
        )

        ax.set_title('Trendline of Average Sales')
        ax.set_ylabel("Average Sales")
        ax.set_xlabel("Week")
        ax.legend(title='Promotion Type')
        ax.set_ylim(45, 60)
        ax.grid(axis='y', color='grey', linestyle='--', alpha=0.7)

        # Optional: Tambahkan label ke setiap titik (bisa jadi padat jika terlalu banyak)
        # Jika kamu hanya ingin menampilkan sebagian, filter dulu
        # Hanya menampilkan label untuk minggu-minggu tertentu agar tidak berantakan
        for _, row in sales_per_promo_week.iterrows():
            ax.annotate(
                text=f"{row['sales(in_thousands)']:.2f}",
                xy=(row['week'], row['sales(in_thousands)']),
                xytext=(0, 7),
                textcoords="offset points",
                ha='center',
                color='black',  # gunakan warna gelap agar terbaca di atas garis
                fontsize=8
            )

        plt.tight_layout()

        # Tampilkan plot di Streamlit
        st.pyplot(fig)

    with st.expander("Assumption Test"):
        st.markdown("<h2 style='text-align: center;'>Normality Test</h2>", unsafe_allow_html=True)
        st.write('H0 : Data follows a normal distribution')
        st.write('H1 : Data does not follow a normal distribution')
        
        # Significance level
        alpha = 0.05
        st.write(f"Significance level (alpha): **{alpha}**")

        # Jalankan Shapiro-Wilk test untuk setiap group promosi
        st.subheader("Results:")
        for group in df['promotion'].unique():
            sales_group = df['sales(in_thousands)'][df['promotion'] == group]
            _, p_value = shapiro(sales_group)
            
            st.write(f"**Promotion Group {group}**")
            st.write(f"- p-value: `{p_value:.4f}`")

            if p_value < alpha:
                st.error(f"⛔ Group {group}'s data **does NOT follow** a normal distribution.")
            else:
                st.success(f"✅ Group {group}'s data **follows** a normal distribution.")
        
        distribution_check(df, ['sales(in_thousands)'])

        # Bimodal Distribution Check (1)

        # Hitung nilai skewness, kurtosis, dan BC
        sales_data = df['sales(in_thousands)']
        gamma = skew(sales_data)
        kappa = kurtosis(sales_data, fisher=True)
        n = len(sales_data)

        BC = (gamma**2 + 1) / (kappa + (3 * (n - 1)**2) / ((n - 2) * (n - 3)))

        # Tampilkan hasil
        st.markdown(f"**Skewness (γ):** `{gamma:.4f}`")
        st.markdown(f"**Excess Kurtosis (κ):** `{kappa:.4f}`")
        st.markdown(f"**Bimodality Coefficient (BC):** `{BC:.4f}`")

        # Interpretasi
        if BC > 0.555:
            st.warning("📛 The distribution **can be categorized as bimodal** (BC > 0.555)")
        else:
            st.success("✅ The distribution **can be categorized as unimodal** (BC ≤ 0.555)")

        # Data
        sales_data = df['sales(in_thousands)'].values.reshape(-1, 1)

        # Grid search to find the optimal bandwidth
        st.write("🔍 Performing grid search for optimal KDE bandwidth...")
        bandwidths = np.logspace(-1, 1, 20)
        grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
        grid.fit(sales_data)

        best_bandwidth = grid.best_params_['bandwidth']
        st.write(f"✅ Optimal bandwidth found: `{best_bandwidth:.4f}`")

        # Fit KDE model with optimal bandwidth
        kde = KernelDensity(bandwidth=best_bandwidth)
        kde.fit(sales_data)

        # Estimate density over a grid
        x_grid = np.linspace(sales_data.min(), sales_data.max(), 1000).reshape(-1, 1)
        log_dens = kde.score_samples(x_grid)
        density = np.exp(log_dens)

        # Detect peaks (modes)
        peaks = argrelextrema(density, np.greater)[0]
        num_modes = len(peaks)
        modes = x_grid[peaks]

        # Show number of modes
        st.subheader("Mode Detection Results")
        st.write(f"🔢 **Number of detected modes:** `{num_modes}`")
        if num_modes > 1:
            st.warning("📛 The distribution **can be categorized as bimodal**")
        else:
            st.success("✅ The distribution **can be categorized as unimodal**")

        # Tampilkan nilai mode
        st.write("📌 **Detected Mode Values (approx.):**")
        st.write(modes.flatten().round(2))

        # Visualisasi
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sales_data, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black')
        ax.plot(x_grid, density, color='blue', label='Kernel Density Estimate')
        ax.scatter(modes, density[peaks], color='magenta', s=100, label='Detected Modes')
        ax.set_xlabel('sales(in_thousands)')
        ax.set_ylabel('Density')
        ax.set_title('Kernel Density Estimation with Mode Detection')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig)

        # Threshold yang sudah ditentukan berdasarkan histogram
        threshold = 275
        st.markdown(f"**Threshold to separate groups:** {threshold}")

        # Pisahkan data berdasarkan threshold
        df_1 = df[df['sales(in_thousands)'] < threshold]
        df_2 = df[df['sales(in_thousands)'] >= threshold]

        # Tampilkan info ukuran data
        st.write(f"📊 Data (df_1) shape: {df_1.shape}")
        st.write(f"📊 Data (df_2) shape: {df_2.shape}")

        st.markdown("---")

        # Jumlah data per tipe promosi di df_1
        st.subheader("Number of Data in df_1 per Promotion Type")
        promo_counts_df1 = df_1.groupby('promotion')['promotion'].count()
        st.table(promo_counts_df1)

        st.markdown("---")

        # Jumlah data per tipe promosi di df_2
        st.subheader("Number of Data in df_2 per Promotion Type")
        promo_counts_df2 = df_2.groupby('promotion')['promotion'].count()
        st.table(promo_counts_df2)

        alpha = 0.05
        st.markdown(f"**Significance level (alpha):** {alpha}")

        def normality_test(df_subset, subset_name):
            st.header(f"Normality check: {subset_name}")
            st.markdown("""
            **Hypothesis:**  
            - H0: The data follows a normal distribution  
            - H1: The data does not follow a normal distribution
            """)
            
            results = []
            for group in df_subset['promotion'].unique():
                _, p_value = shapiro(df_subset['sales(in_thousands)'][df_subset['promotion'] == group])
                normality = "follows a normal distribution" if p_value >= alpha else "doesn't follow a normal distribution"
                results.append((group, p_value, normality))
            
            # Tampilkan tabel hasil
            for group, p_value, normality in results:
                st.write(f"- Shapiro-Wilk test for group **{group}**: p-value = `{p_value:.4f}` → Data **{normality}**")

        # Jalankan untuk df_1 dan df_2
        normality_test(df_1, "df_1 (sales < 275)")
        st.markdown("---")
        normality_test(df_2, "df_2 (sales ≥ 275)")

        st.markdown("<h2 style='text-align: center;'>Homogeneity Test</h2>", unsafe_allow_html=True)

        alpha = 0.05
        st.markdown(f"**Significance level (alpha):** `{alpha}`")

        def homogeneity_test(df_subset, subset_name):
            st.header(f"Homogeneity Check: {subset_name}")
            st.markdown("""
            **Hypothesis:**  
            - H0: The data are homogeneous  
            - H1: The data are not homogeneous
            """)

            # Kelompokkan berdasarkan promotion
            sales_by_group = [df_subset['sales(in_thousands)'][df_subset['promotion'] == group] for group in df_subset['promotion'].unique()]
            
            # Levene Test
            statistic, p_value = levene(*sales_by_group)
            is_homogeneous = "The data are homogeneous ✅" if p_value >= alpha else "The data are not homogeneous ❌"

            # Tampilkan hasil
            st.write(f"- **Levene’s Test p-value**: `{p_value:.4f}`")
            st.success(is_homogeneous) if p_value >= alpha else st.error(is_homogeneous)

        # Tes untuk df_1 dan df_2
        homogeneity_test(df_1, "df_1 (sales < 275)")
        st.markdown("---")
        homogeneity_test(df_2, "df_2 (sales ≥ 275)")

        st.markdown("<h2 style='text-align: center;'>Difference Test</h2>", unsafe_allow_html=True)

        st.subheader("Kruskal Wallis Test for df_1")

        st.write("""
        **Hypothesis**
        - **H0**: There is no difference in sales median/distribution for each promotion type  
        - **H1**: At least one type of promotion has a different sales median/distribution
        """)

        alpha = 0.05
        st.write(f"**Significance level (alpha):** `{alpha}`")

        # Ambil subset per promotion type
        df1_1 = df_1['sales(in_thousands)'][df_1['promotion'] == '1']
        df1_2 = df_1['sales(in_thousands)'][df_1['promotion'] == '2']
        df1_3 = df_1['sales(in_thousands)'][df_1['promotion'] == '3']

        # Kruskal-Wallis test
        statistic, p_value = kruskal(df1_1, df1_2, df1_3)

        # Tampilkan hasil
        st.write(f"**Kruskal-Wallis Test Statistic:** `{statistic:.4f}`")
        st.write(f"**p-value:** `{p_value:.4f}`")

        # Interpretasi
        if p_value < alpha:
            st.error("🔴 At least one type of promotion has a different sales median/distribution.")
        else:
            st.success("🟢 There is no difference in sales median/distribution for each promotion type.")
        
        # Jalankan Dunn's test
        dunn = sp.posthoc_dunn([df1_1, df1_2, df1_3], p_adjust='bonferroni')

        # Rename indeks dan kolom untuk keterbacaan
        dunn.index = ['Promotion 1', 'Promotion 2', 'Promotion 3']
        dunn.columns = ['Promotion 1', 'Promotion 2', 'Promotion 3']

        # Tampilkan tabel hasil
        st.dataframe(dunn.style.background_gradient(cmap='Blues').format(precision=4))

        # Interpretasi cepat (opsional)
        st.markdown("""
        **Interpretation Tip:**  
        - A p-value < 0.05 (after Bonferroni correction) indicates a statistically significant difference in median sales between the two promotion groups.
        """)

        # Fungsi untuk menghitung mean rank
        def calculate_mean_ranks(data, group_col, value_col):
            combined_data = pd.concat([data[value_col], data[group_col]], axis=1)
            combined_data = combined_data.sort_values(value_col)
            combined_data['rank'] = np.arange(1, len(combined_data) + 1)
            mean_ranks = combined_data.groupby(group_col)['rank'].mean().reset_index()
            mean_ranks = mean_ranks.rename(columns={'rank': 'Mean Rank'})
            return mean_ranks

        st.subheader("Mean Ranks per Promotion Group (df_1)")

        # Hitung mean rank untuk df_1
        mean_ranks_df1 = calculate_mean_ranks(df_1, 'promotion', 'sales(in_thousands)')

        # Tampilkan hasil dalam tabel
        st.dataframe(mean_ranks_df1.style.format({'Mean Rank': '{:.2f}'}))

        # Catatan tambahan (opsional)
        st.markdown("""
        **Interpretation Tip:**  
        - A smaller mean rank indicates lower sales value in the group.  
        - Used in the interpretation of non-parametric tests such as Kruskal-Wallis.
        """)

        # Fungsi untuk menghitung median
        def calculate_median(data, group_col, value_col):
            medians = data.groupby(group_col)[value_col].median().reset_index()
            medians = medians.rename(columns={value_col: 'Median'})
            return medians

        st.subheader("Median Sales per Promotion Group (df_1)")

        # Hitung median untuk df_1
        median_df1 = calculate_median(df_1, 'promotion', 'sales(in_thousands)')

        # Tampilkan hasil median
        st.dataframe(median_df1.style.format({'Median': '{:.2f}'}))

        # Catatan interpretasi (opsional)
        st.markdown("""
        **Note:**  
        - Median is used to find out the middle value of the distribution of sales data per type of promotion.  
        - Suitable for use on data that is not normally distributed or contains outliers.
        """)

        st.subheader("ANOVA Test for df_2")

        # Hypothesis
        st.markdown("""
        **Hypothesis**  
        - **H0**: There is no difference in average sales for each promotion type  
        - **H1**: At least one type of promotion has a different average sales  
        """)

        # Significance level
        alpha = 0.05
        st.write(f"**Significance level (alpha):** {alpha}")

        # Extracting data
        df2_1 = df_2['sales(in_thousands)'][df_2['promotion'] == '1']
        df2_2 = df_2['sales(in_thousands)'][df_2['promotion'] == '2']
        df2_3 = df_2['sales(in_thousands)'][df_2['promotion'] == '3']

        # ANOVA test
        stat, p = stats.f_oneway(df2_1, df2_2, df2_3)

        # Results
        st.write(f"**ANOVA Test Results:**")
        st.write(f"- Statistic: {stat:.4f}")
        st.write(f"- P-value: {p:.4f}")

        # Interpretation
        if p < alpha:
            st.error("At least one type of promotion has a different average sales (Reject H0)")
        else:
            st.success("There is no difference in average sales for each promotion type (Fail to reject H0)")

        # Tukey HSD test
        tukey = pairwise_tukeyhsd(endog=df_2['sales(in_thousands)'], groups=df_2['promotion'], alpha=0.05)

        # Show summary as a table
        st.write("**Tukey HSD Test Summary:**")
        st.dataframe(tukey.summary().data[1:], use_container_width=True, 
                    hide_index=True, 
                    column_config={
                        "group1": "Group 1",
                        "group2": "Group 2",
                        "meandiff": "Mean Diff",
                        "p-adj": "P-Value",
                        "lower": "Lower CI",
                        "upper": "Upper CI",
                        "reject": "Reject H0?"
                    })
        
        # Hitung rata-rata penjualan per jenis promosi
        average_sales_per_promotion = df.groupby('promotion')['sales(in_thousands)'].mean().rename('Average Sales').reset_index()

        # Tampilkan dalam bentuk tabel
        st.dataframe(average_sales_per_promotion, use_container_width=True)

    with st.expander("Conclusions"):
        st.markdown("""
                    1. **From the EDA analysis**, we can see that the **Large Market has the highest average sales** compared to other market sizes.

                    2. **Promotion Performance**:
                    - **The first promotion type is the most effective**, generating the highest average sales over the 4-week period.
                    - **The third promotion type** follows closely behind.
                    - **The second promotion type performs the weakest**, with much lower sales compared to the other two.

                    3. **Insights on store age and sales**:
                    Some newer stores, especially in Large Markets, have achieved high sales. More analysis is needed to understand why. Possible reasons include:
                    - **Effective marketing** – using a promotion strategy that worked well in that location.
                    - **Strong product fit** – the new product matches customer preferences in the area.
                    - **Helpful external factors** – like good store location, local events, or holidays that boosted sales.

                    4. Also, some older stores continue to perform well and should be supported to keep up their strong sales.
                    """)
