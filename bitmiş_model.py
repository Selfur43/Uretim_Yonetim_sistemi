import pandas as pd
import random
import pulp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Sayfa ayarları
st.set_page_config(page_title="Üretim Yönetim Sistemi", page_icon="🛠️", layout="wide")

# Başlık ve giriş
st.title("🛠️ Üretim Yönetim Sistemi")
st.markdown("""
**Bu platform, üretim planlamanızı optimize etmek ve performans göstergelerinizi analiz etmek için geliştirilmiştir.**
Veri analizine dayalı interaktif grafikler ve detaylı görselleştirmelerle üretim sürecinizi iyileştirin.
""")

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
random.seed(42)

# Kurulum süreleri ve hata oranları
setup_times = {(i, j, k, p): random.randint(10, 20) for i in operators for j in machines for k in shifts for p in products}
error_rates = {(i, j, k, p): round(random.uniform(0.01, 0.1), 2) for i in operators for j in machines for k in shifts for p in products}
skill_fit = {(i, j): random.randint(50, 100) for i in operators for j in machines}
max_error_rate = {'P_1': 0.2, 'P_2': 0.15, 'P_3': 0.25}
min_skill_score = {'P_1': 60, 'P_2': 70, 'P_3': 65}
max_work_time = 16 * 60

# Model oluşturma
model = pulp.LpProblem("Operator_Assignment", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", (operators, machines, shifts, products), cat="Binary")

# Amaç fonksiyonu
model += pulp.lpSum((setup_times[i, j, k, p] + error_rates[i, j, k, p] - 0.1 * skill_fit[i, j]) * x[i][j][k][p]
                    for i in operators for j in machines for k in shifts for p in products)

# Kısıtlar
model += pulp.lpSum(setup_times[i, j, k, p] * x[i][j][k][p] for i in operators for j in machines for k in shifts for p in products) <= 300

for p in products:
    model += pulp.lpSum(error_rates[i, j, k, p] * x[i][j][k][p] for i in operators for j in machines for k in shifts) <= max_error_rate[p]

for j in machines:
    for k in shifts:
        model += pulp.lpSum(x[i][j][k][p] for i in operators for p in products) >= 1

for i in operators:
    for k in shifts:
        model += pulp.lpSum(x[i][j][k][p] for j in machines for p in products) <= 1

for i in operators:
    for j in machines:
        for k in shifts:
            for p in products:
                if skill_fit[i, j] < min_skill_score[p]:
                    model += x[i][j][k][p] == 0

for i in operators:
    model += pulp.lpSum(x[i][j][k][p] * setup_times[i, j, k, p] for j in machines for k in shifts for p in products) <= max_work_time

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
    fig1 = px.bar(df_results, x="Makine", y="Kurulum Süresi", color="Ürün",
                  title="Makineye Göre Kurulum Süresi",
                  labels={'Kurulum Süresi': 'Kurulum Süresi (dk)', 'Makine': 'Makine Adı'},
                  template=selected_theme.lower(), hover_data=['Operatör', 'Yetenek Skoru'])
    fig1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig1, use_container_width=True)

    # Vardiyalara Göre Hata Oranı
    fig2 = alt.Chart(df_results).mark_line(point=True).encode(
        x='Vardiya',
        y='Hata Oranı',
        color='Makine',
        tooltip=['Makine', 'Ürün', 'Operatör', 'Yetenek Skoru', 'Kurulum Süresi']
    ).properties(
        title="Vardiyalara Göre Hata Oranı",
        width=700
    ).interactive()
    st.altair_chart(fig2, use_container_width=True)

    # Kurulum Süresi ve Yetenek Skoru Dağılımı
    fig3 = px.scatter(df_results, x="Kurulum Süresi", y="Yetenek Skoru", color="Makine",
                      title="Kurulum Süresi ve Yetenek Skoru Dağılımı",
                      size="Hata Oranı", hover_data=['Operatör', 'Vardiya', 'Ürün'],
                      template=selected_theme.lower())
    st.plotly_chart(fig3, use_container_width=True)

    # Korelasyon Isı Haritası
    st.subheader("📈 Korelasyon Isı Haritası")
    numeric_cols = df_results.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df_results[numeric_cols].corr()

    fig4 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="Viridis",
        hoverongaps=False))
    fig4.update_layout(title="Korelasyon Isı Haritası", template=selected_theme.lower())
    st.plotly_chart(fig4, use_container_width=True)

    # Regresyon Modelleri
    st.subheader("📊 Gelişmiş Regresyon Modelleri")

    # Regresyon için veri hazırlama
    X = df_results[['Kurulum Süresi', 'Yetenek Skoru']].values
    y = df_results['Hata Oranı'].values

    # Train-Test Bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lineer Regresyon
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # Rastgele Orman Regresyonu
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Model Performans Gösterimi
    st.write("Lineer Regresyon Modeli Performansı:")
    st.write(f"Ortalama Kare Hatası: {mean_squared_error(y_test, y_pred_linear):.4f}")
    st.write(f"R² Skoru: {r2_score(y_test, y_pred_linear):.4f}")

    st.write("Rastgele Orman Regresyon Modeli Performansı:")
    st.write(f"Ortalama Kare Hatası: {mean_squared_error(y_test, y_pred_rf):.4f}")
    st.write(f"R² Skoru: {r2_score(y_test, y_pred_rf):.4f}")

    # Tahmin Sonuçları
    st.write("Tahmin Sonuçları:")
    comparison_df = pd.DataFrame({
        'Gerçek Değerler': y_test,
        'Basit Regresyon Tahminleri': y_pred_linear,
        'Rastgele Orman Tahminleri': y_pred_rf
    })
    st.write(comparison_df)

    # Modelleri Kaydet
    joblib.dump(linear_model, 'linear_regression_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    st.success("Modeller kaydedildi: 'linear_regression_model.pkl' ve 'random_forest_model.pkl'")

# Sonuçları Kaydetme
if st.button("Sonuçları PDF Olarak İndir"):
    df_results.to_csv("sonuclar.csv", index=False)
    st.success("Sonuçlar CSV dosyası olarak kaydedildi.")
