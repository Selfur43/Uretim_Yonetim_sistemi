import pandas as pd
import random
import pulp
import streamlit as st

# Seed for reproducibility
random.seed(42)

# Streamlit page settings
st.set_page_config(page_title="Production Management System", page_icon="🛠️", layout="wide")

# Title and introduction
st.title("🛠️ Production Management System")
st.markdown("**Optimize your production planning and analyze performance indicators on this platform.**")

# Sidebar settings
st.sidebar.title("⚙️ Settings and Filters")
selected_operator = st.sidebar.selectbox("Select Operator", ['O_1', 'O_2', 'O_3'])
selected_machine = st.sidebar.selectbox("Select Machine", ['M_1', 'M_2', 'M_3'])
selected_shift = st.sidebar.selectbox("Select Shift", ['V_1', 'V_2'])
selected_theme = st.sidebar.selectbox("Select Theme", ['Plotly', 'Seaborn'])

# Model parameters
operators = ['O_1', 'O_2', 'O_3']
machines = ['M_1', 'M_2', 'M_3']
shifts = ['V_1', 'V_2']
products = ['P_1', 'P_2', 'P_3']
workdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
hours = list(range(1, 11))  # 10-hour workday

# Generate parameters
setup_times = {(i, j, k, p, d, h): random.randint(1, 20) for i in operators for j in machines for k in shifts for p in products for d in workdays for h in hours}
error_rates = {(i, j, k, p, d, h): round(random.uniform(0.01, 0.26), 2) for i in operators for j in machines for k in shifts for p in products for d in workdays for h in hours}
skill_fit = {(i, j): random.randint(50, 100) for i in operators for j in machines}
max_error_rates = {'P_1': 0.2, 'P_2': 0.15, 'P_3': 0.25}
min_skill_score = {'P_1': 60, 'P_2': 70, 'P_3': 65}
max_daily_work_minutes = 10 * 60  # Maximum daily working time in minutes (10 hours * 60 minutes)
max_weekly_workdays = 5
rest_time = 30
overtime_limit = 90

# Define decision variable for each operator, machine, shift, product, day, and hour
x = pulp.LpVariable.dicts("x", (operators, machines, shifts, products, workdays, hours), cat="Binary")

# Initialize the optimization model as a minimization problem
model = pulp.LpProblem("Operator_Assignment", pulp.LpMinimize)

# Objective function: minimize setup times and error rates
model += pulp.lpSum(
    (setup_times[i, j, k, p, d, h] + error_rates[i, j, k, p, d, h] * max_daily_work_minutes - 0.05 * skill_fit[i, j] * max_daily_work_minutes) * x[i][j][k][p][d][h]
    for i in operators for j in machines for k in shifts for p in products for d in workdays for h in hours
)

# Constraints

# 1. Total setup time per shift and day
for i in operators:
    for k in shifts:
        for d in workdays:
            model += pulp.lpSum(setup_times[i, j, k, p, d, h] * x[i][j][k][p][d][h] for j in machines for p in products for h in hours) <= max_daily_work_minutes - rest_time

# 2. Average error rate per machine, shift, and day
for j in machines:
    for k in shifts:
        for d in workdays:
            model += pulp.lpSum(error_rates[i, j, k, p, d, h] * x[i][j][k][p][d][h] for i in operators for p in products for h in hours) <= sum(max_error_rates[p] for p in products) / len(products)

# 3. At least one operator per machine, shift, and day
for j in machines:
    for k in shifts:
        for d in workdays:
            model += pulp.lpSum(x[i][j][k][p][d][h] for i in operators for p in products for h in hours) >= 1

# 4. Single assignment per operator per shift and day
for i in operators:
    for k in shifts:
        for d in workdays:
            model += pulp.lpSum(x[i][j][k][p][d][h] for j in machines for p in products for h in hours) <= 1

# 5. Skill score threshold constraint
for i in operators:
    for j in machines:
        for k in shifts:
            for p in products:
                for d in workdays:
                    for h in hours:
                        if skill_fit[i, j] < min_skill_score[p]:
                            model += x[i][j][k][p][d][h] == 0

# 6. Weekly working day limit for each operator
for i in operators:
    model += pulp.lpSum(x[i][j][k][p][d][h] for j in machines for k in shifts for p in products for d in workdays for h in hours) <= max_weekly_workdays * 10

# 7. Daily working hours and single shift constraint for each operator
for i in operators:
    for d in workdays:
        model += pulp.lpSum(x[i][j][k][p][d][h] for j in machines for k in shifts for p in products for h in hours) <= 10

# 8. Ensure no overlap in shifts for any operator.
for i in operators:
    for d in workdays:
        model += pulp.lpSum(x[i][j][k][p][d][h] for j in machines for k in shifts for p in products for h in hours) <= 10

# 9. Daily overtime constraint for each operator
for i in operators:
    for d in workdays:
        model += pulp.lpSum(setup_times[i, j, k, p, d, h] * x[i][j][k][p][d][h] for j in machines for k in shifts for p in products for h in hours) <= max_daily_work_minutes + overtime_limit

# Solve the model
model.solve()

# Solution status
solution_status = pulp.LpStatus[model.status]
st.markdown(f"### Solution Status: {solution_status}")

# Convert results to DataFrame
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
    df_results = pd.DataFrame(results, columns=["Operator", "Machine", "Shift", "Product", "Day", "Hour", "Setup Time (min)", "Error Rate (%)", "Skill Score"])
else:
    st.warning("No optimal solution found.")
    df_results = pd.DataFrame()

# Display assignment table
if not df_results.empty:
    st.subheader("📋 Assignment Table")
    st.dataframe(df_results)
else:
    st.write("No data available for visualizations.")
