import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from trello import create_board, create_list, create_card

st.set_page_config(page_title="AI Task Management", layout="centered")
st.title("🧠 AI-Powered Task Management Dashboard")

# Load processed data
df = pd.read_csv(r"C:\Users\mishr\OneDrive\Documents\Task_Management_System\learning\balanced_tasks.csv")

# ----------------------- Section 1: Task Load Per User -----------------------
st.subheader("📊 Task Load Per User")
task_counts = df['assigned_to'].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(task_counts.index, task_counts.values, color='skyblue')
ax1.set_ylabel("Number of Tasks")
ax1.set_title("Total Tasks Assigned Per User")
st.pyplot(fig1)

# ----------------------- Section 2: High Priority Tasks -----------------------
st.subheader("🔥 High Priority Tasks Per User")
high_tasks = df[df['predicted_priority'] == 1]
high_counts = high_tasks['assigned_to'].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(high_counts.index, high_counts.values, color='salmon')
ax2.set_ylabel("High Priority Tasks")
ax2.set_title("High Priority Task Load")
st.pyplot(fig2)

# ----------------------- Section 3: Feature Importance -----------------------
st.subheader("🔍 Feature Importance")

# Load trained Random Forest model and feature names
rf, feature_names = joblib.load(r"C:\Users\mishr\OneDrive\Documents\Task_Management_System\learning\week3_rf_model.pkl")
importances = rf.feature_importances_

fig3, ax3 = plt.subplots()
ax3.barh(feature_names, importances, color='green')
ax3.set_xlabel("Importance Score")
ax3.set_title("Feature Importance for Priority Prediction")
st.pyplot(fig3)

# ----------------------- Section 4: Trello Automation -----------------------
st.subheader("📤 Automate Trello Task Creation")

# Input for board name
board_title = st.text_input("Enter Trello Board Name", value="AI Task Board")

if st.button("🚀 Push Tasks to Trello"):
    with st.spinner("Creating board and pushing tasks..."):
        # Step 1: Create board
        board_id = create_board(board_title)
        st.success(f"✅ Created board '{board_title}'")

        # Step 2: Create lists and push tasks
        users = df['assigned_to'].unique()
        for user in users:
            list_id = create_list(board_id, user)
            user_tasks = df[df['assigned_to'] == user]

            for _, row in user_tasks.iterrows():
                task = row['task_text']
                due = row['due_date']
                priority = "High" if row['predicted_priority'] == 1 else "Normal"
                category = row.get('category', '')
                status = row.get('status', '')
                length = row.get('text_length', '')
                days_left = row.get('due_days_left', '')

                card_title = f"[{priority}] {task}"
                card_desc = f"""
**Category:** {category}  
**Status:** {status}  
**Due:** {due}  
**Length:** {length} words  
**Days Left:** {days_left}
"""
                create_card(list_id, card_title)
                st.write(f"✅ Sent: {card_title} to {user}'s list")

        st.success("🎉 All tasks successfully sent to Trello!")
