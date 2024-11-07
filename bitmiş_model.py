import pandas as pd
import random
import pulp
import streamlit as st

# Rastgelelik için sabit
random.seed(42)

# Streamlit sayfa ayarları
st.set_page_config(page_title="Gelişmiş Kaynak Bölümü Planlama", page_icon="🔧", layout="wide")

# Başlık ve Giriş
st.title("🔧 Gelişmiş Kaynak Bölümü Planlama - Operatör Atama Problemi")
st.markdown("**Operatörlerin deneyim seviyesine göre hata oranı ve ayar süresini optimize edin.**")

# Sidebar ayarları
st.sidebar.title("⚙️ Ayarlar ve Filtreler")
selected_operator = st.sidebar.selectbox("Operatör Seç", ['O_1', 'O_2', 'O_3', 'O_4', 'O_5'])
selected_machine = st.sidebar.selectbox("Makine Seç", ['Kaynak_M_1', 'Kaynak_M_2', 'Kaynak_M_3'])
selected_shift = st.sidebar.selectbox("Vardiya Seç", ['Sabah', 'Akşam'])
selected_theme = st.sidebar.selectbox("Tema Seç", ['Plotly', 'Seaborn'])

# Model Parametreleri
operators = ['O_1', 'O_2', 'O_3', 'O_4', 'O_5']
machines = ['Kaynak_M_1', 'Kaynak_M_2', 'Kaynak_M_3']
shifts = ['Sabah', 'Akşam']
products = ['Metal_P_1', 'Metal_P_2', 'Metal_P_3']
workdays = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma']
hours = list(range(1, 9))  # Günde 8 saat çalışma

# Parametreleri oluştur
setup_times = {(i, j, k, p, d, h): random.randint(5, 20) for i in operators for j in machines for k in shifts for p in products for d in workdays for h in hours}
error_rates = {(i, j, k, p, d, h): round(random.uniform(0.01, 0.12), 2) for i in operators for j in machines for k in shifts for p in products for d in workdays for h in hours}
skill_fit = {(i, j): random.randint(60, 100) for i in operators for j in machines}
learning_factor = 0.9  # Öğrenme eğrisi katsayısı
max_error_rates = {'Metal_P_1': 0.08, 'Metal_P_2': 0.05, 'Metal_P_3': 0.10}
min_skill_score = {'Metal_P_1': 60, 'Metal_P_2': 75, 'Metal_P_3': 70}
max_daily_work_minutes = 8 * 60  # Günde maksimum 8 saat (480 dakika)
weekly_work_limit = 45 * 60  # Haftalık çalışma limiti 45 saat (2700 dakika)
overtime_limit = 60  # Maksimum ek mesai 60 dakika ile sınırlı
daily_break_minutes = 30  # Günlük mola süresi 30 dakika

# Karar değişkeni (her operatör, makine, vardiya, ürün, gün ve saat için)
x = pulp.LpVariable.dicts("x", (operators, machines, shifts, products, workdays, hours), cat="Binary")

# Optimizasyon modelini başlat (minimizasyon problemi)
model = pulp.LpProblem("Deneyime_Gore_Operator_Atama", pulp.LpMinimize)

# Amaç fonksiyonu: Toplam kurulum süresi, hata oranları ve ek mesai maliyetlerini minimize et
overtime_penalty = 2  # Ek mesai maliyeti artırıcı katsayı
model += pulp.lpSum(
    ((setup_times[i, j, k, p, d, h] * learning_factor ** (1 / skill_fit[i, j]) + 
     error_rates[i, j, k, p, d, h] * max_daily_work_minutes - 0.05 * skill_fit[i, j] * max_daily_work_minutes) +
     overtime_penalty * (1 if setup_times[i, j, k, p, d, h] > max_daily_work_minutes else 0))
    * x[i][j][k][p][d][h] 
    for i in operators for j in machines for k in shifts for p in products for d in workdays for h in hours
)

# Kısıtlar

# 1. Günlük çalışma süresi ve mola dahilinde çalışma
for i in operators:
    for k in shifts:
        for d in workdays:
            model += pulp.lpSum(setup_times[i, j, k, p, d, h] * x[i][j][k][p][d][h] for j in machines for p in products for h in hours) <= max_daily_work_minutes - daily_break_minutes

# 2. Ürün bazında hata oranı sınırı
for j in machines:
    for k in shifts:
        for d in workdays:
            model += pulp.lpSum(error_rates[i, j, k, p, d, h] * x[i][j][k][p][d][h] for i in operators for p in products for h in hours) <= sum(max_error_rates[p] for p in products) / len(products)

# 3. Yetenek eşiği ve gelişim süreci
for i in operators:
    for j in machines:
        for k in shifts:
            for p in products:
                for d in workdays:
                    for h in hours:
                        if skill_fit[i, j] < min_skill_score[p]:
                            model += x[i][j][k][p][d][h] == 0

# 4. Haftalık çalışma limiti (her operatör için en fazla 45 saat)
for i in operators:
    model += pulp.lpSum(setup_times[i, j, k, p, d, h] * x[i][j][k][p][d][h] for j in machines for k in shifts for p in products for d in workdays for h in hours) <= weekly_work_limit

# 5. Vardiya çakışma kısıtlaması
for i in operators:
    for d in workdays:
        model += pulp.lpSum(x[i][j][k][p][d][h] for j in machines for k in shifts for p in products for h in hours) <= 8

# 6. Rotasyon ve iş çeşitlendirme
for i in operators:
    for d in workdays:
        for j in machines:
            for p in products:
                # Her gün aynı makine ve ürün için aynı operatör 4 saatten fazla çalışmasın
                model += pulp.lpSum(x[i][j][k][p][d][h] for k in shifts for h in hours) <= 4

# Modeli çöz
model.solve()

# Çözüm durumu
solution_status = pulp.LpStatus[model.status]
st.markdown(f"### Çözüm Durumu: {solution_status}")

# Sonuçları DataFrame'e çevir
results = []
if solution_status == "Optimal":
    for i in operators:
        for j in machines:
            for k in shifts:
                for p in products:
                    for d in workdays:
                        for h in hours:
                            if x[i][j][k][p][d][h].varValue == 1:
                                results.append([i, j, k, p, d, h, setup_times[(i, j, k, p, d, h)], error_rates[(i, j, k, p, d, h)], skill_fit[(i, j)]])
    df_results = pd.DataFrame(results, columns=["Operatör", "Makine", "Vardiya", "Ürün", "Gün", "Saat", "Kurulum Süresi (dk)", "Hata Oranı (%)", "Yetenek Skoru"])
else:
    st.warning("Optimal çözüm bulunamadı.")
    df_results = pd.DataFrame()

# Atama Tablosunu Göster
if not df_results.empty:
    st.subheader("📋 Atama Tablosu")
    st.dataframe(df_results)
else:
    st.write("Veri bulunamadı.")
