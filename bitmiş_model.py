import pandas as pd
import random
import pulp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import numpy as np
from fpdf import FPDF  # Make sure to import FPDF for PDF generation
import seaborn as sns  # Import seaborn for correlation heatmap
import matplotlib.pyplot as plt  # Import matplotlib for the heatmap plot
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Sayfa ayarları
st.set_page_config(page_title="Üretim Yönetim Sistemi", page_icon="🛠️", layout="wide")

# Başlık ve giriş
st.title("🛠️ Üretim Yönetim Sistemi")
st.markdown("""**Bu platform, üretim planlamanızı optimize etmek ve performans göstergelerinizi analiz etmek için geliştirilmiştir.**
Veri analizine dayalı interaktif grafikler ve detaylı görselleştirmelerle üretim sürecinizi iyileştirin.""")

# Sidebar ayarları
st.sidebar.title("⚙️ Ayarlar ve Filtreler")
selected_operator = st.sidebar.selectbox("Operatör Seçin", ['O_1', 'O_2', 'O_3'])
selected_machine = st.sidebar.selectbox("Makine Seçin", ['M_1', 'M_2', 'M_3'])
selected_shift = st.sidebar.selectbox("Vardiya Seçin", ['V_1', 'V_2'])
selected_theme = st.sidebar.selectbox("Tema Seçin", ['Plotly', 'Seaborn'])

# Model parametreleri
operators = ['O_1', 'O_2', 'O_3']
machines = ['M_1', 'M_2', 'M_3']
shifts = ['V_1', 'V_2']
products = ['P_1', 'P_2', 'P_3']
tasks = ['T_1', 'T_2', 'T_3']  # Define tasks
random.seed(42)

# Kurulum süreleri ve hata oranları
setup_times = {(i, j, k, p): random.randint(1, 20) for i in operators for j in machines for k in shifts for p in products}
error_rates = {(i, j, k, p): round(random.uniform(0.01, 0.26), 2) for i in operators for j in machines for k in shifts for p in products}
skill_fit = {(i, j): random.randint(50, 100) for i in operators for j in machines}
max_error_rate = {'P_1': 0.2, 'P_2': 0.15, 'P_3': 0.25}
min_skill_score = {'P_1': 60, 'P_2': 70, 'P_3': 65}
max_daily_work_minutes = 1440  # Günlük maksimum çalışma süresi (dakika)
max_weekly_days = 6  # Haftalık maksimum çalışma günü sayısı
rest_time = 30  # Dinlenme süresi (dakika)
overtime_limit = 90  # Maksimum fazla mesai süresi (dakika)
task_durations = {
    'T_1': 30,
    'T_2': 45,
    'T_3': 25
}

# Model oluşturma
model = pulp.LpProblem("Operator_Assignment", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", (operators, machines, shifts, products), cat="Binary")

# Amaç fonksiyonu
model += pulp.lpSum((setup_times[i, j, k, p] + error_rates[i, j, k, p] - 0.1 * skill_fit[i, j]) * x[i][j][k][p]
                    for i in operators for j in machines for k in shifts for p in products)

# Kısıtlar
# 1. Toplam kurulum süresi kısıtı
model += pulp.lpSum(setup_times[i, j, k, p] * x[i][j][k][p] for i in operators for j in machines for k in shifts for p in products) <= 60

# 2. Ürün başına hata oranı kısıtı
for p in products:
    model += pulp.lpSum(error_rates[i, j, k, p] * x[i][j][k][p] for i in operators for j in machines for k in shifts) <= max_error_rate[p]

# 3. Makine ve vardiya kapsama kısıtı
for j in machines:
    for k in shifts:
        model += pulp.lpSum(x[i][j][k][p] for i in operators for p in products) >= 1

# 4. Operatör başına tek atama kısıtı
for i in operators:
    for k in shifts:
        model += pulp.lpSum(x[i][j][k][p] for j in machines for p in products) <= 1

# 5. Minimum yetenek puanı kısıtı
for i in operators:
    for j in machines:
        for k in shifts:
            for p in products:
                if skill_fit[i, j] < min_skill_score[p]:
                    model += x[i][j][k][p] == 0

# 6. Günlük çalışma süresi kısıtı (her operatör için)
for i in operators:
    model += pulp.lpSum(task_durations[task] * x[i][j][k][p]
                         for j in machines for k in shifts for p in products for task in tasks) <= max_daily_work_minutes - rest_time

# 7. Haftalık çalışma günü sınırı
model += pulp.lpSum(x[i][j][k][p] for i in operators for j in machines for k in shifts for p in products) <= max_weekly_days

# 8. Günlük tek vardiya kısıtı (her operatör gün içinde sadece bir vardiyada çalışabilir)
for i in operators:
    for k in shifts:
        model += pulp.lpSum(x[i][j][k][p] for j in machines for p in products) <= 1

# 9. Makine başına tek operatör kısıtı (aynı vardiyada bir makine yalnızca bir operatör tarafından kullanılabilir)
for j in machines:
    for k in shifts:
        model += pulp.lpSum(x[i][j][k][p] for i in operators for p in products) <= 1

# 10. Fazla mesai kısıtlaması
model += pulp.lpSum(task_durations[task] * x[i][j][k][p]
                     for i in operators for j in machines for k in shifts for p in products for task in tasks) <= max_daily_work_minutes + overtime_limit

# 11. Dinlenme süreleri kısıtlaması
model += pulp.lpSum(task_durations[task] * x[i][j][k][p]
                     for i in operators for j in machines for k in shifts for p in products for task in tasks) + rest_time <= max_daily_work_minutes

# Modeli çöz
model.solve()

# Çözüm durumu
solution_status = pulp.LpStatus[model.status]
st.markdown(f"### Çözüm Durumu: {solution_status}")

# Sonuçları DataFrame olarak dönüştür
results = []
for i in operators:
    for j in machines:
        for k in shifts:
            for l in products:
                if x[i][j][k][l].varValue == 1:
                    results.append([i, j, k, l, setup_times[i, j, k, l], error_rates[i, j, k, l], skill_fit[i, j]])

df_results = pd.DataFrame(results, columns=["Operatör", "Makine", "Vardiya", "Ürün", "Kurulum Süresi", "Hata Oranı", "Yetenek Skoru"])

# Atama Tablosu
st.subheader("📋 Atama Tablosu")
st.dataframe(df_results)

# İleri Düzey Görselleştirmeler
if not df_results.empty:
    st.subheader("📊 Atama Sonuçları ve Analizler")

    # Özet İstatistikler
    col1, col2 = st.columns(2)
    col1.metric("Toplam Kurulum Süresi", f"{df_results['Kurulum Süresi'].sum()} dk")
    col2.metric("Ortalama Hata Oranı", f"{df_results['Hata Oranı'].mean():.2%}")

    # Makineye Göre Kurulum Süresi
    fig1 = px.bar(df_results, x="Makine", y="Kurulum Süresi", color="Operatör", title="Makineye Göre Kurulum Süresi", barmode='group')
    st.plotly_chart(fig1)

    # Hata Oranı Analizi
    fig2 = go.Figure()
    for operator in df_results["Operatör"].unique():
        filtered_df = df_results[df_results["Operatör"] == operator]
        fig2.add_trace(go.Scatter(x=filtered_df["Ürün"], y=filtered_df["Hata Oranı"], mode='lines+markers', name=operator))
    fig2.update_layout(title="Operatör Bazında Hata Oranı Analizi", xaxis_title="Ürün", yaxis_title="Hata Oranı")
    st.plotly_chart(fig2)

    # Yetenek Skoru Dağılımı
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Yetenek Skoru", data=df_results)
    plt.title("Yetenek Skoru Dağılımı")
    plt.xlabel("Yetenek Skoru")
    plt.tight_layout()
    st.pyplot(plt)

# PDF raporu oluşturma
if st.button("PDF Raporu Oluştur"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Üretim Yönetim Sistemi Raporu", ln=True, align='C')

    # Raporun detayları
    for index, row in df_results.iterrows():
        pdf.cell(0, 10, f"Operatör: {row['Operatör']}, Makine: {row['Makine']}, Vardiya: {row['Vardiya']}, Ürün: {row['Ürün']}, Kurulum Süresi: {row['Kurulum Süresi']} dk, Hata Oranı: {row['Hata Oranı']:.2%}, Yetenek Skoru: {row['Yetenek Skoru']}", ln=True)

    pdf_file_path = "rapor.pdf"
    pdf.output(pdf_file_path)

    st.success(f"PDF raporu oluşturuldu! [İndir]({pdf_file_path})")
