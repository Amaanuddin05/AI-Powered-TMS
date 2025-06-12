import random
import pandas as pd
from datetime import datetime, timedelta

# Task properties
categories = ['Bug', 'Feature', 'Documentation', 'Meeting', 'Research']
priorities = ['Low', 'Medium', 'High']
statuses = ['To Do', 'In Progress', 'Done']

task_phrases = {
    'Bug': ['Fix login issue', 'Resolve payment bug', 'Crash on launch', 'UI glitch in menu', 'Null pointer error'],
    'Feature': ['Add dark mode', 'Implement search filter', 'Integrate chat module', 'Build notification system'],
    'Documentation': ['Update user manual', 'Write README', 'Improve API docs', 'Document new features'],
    'Meeting': ['Schedule sprint planning', 'Weekly sync-up', 'Client presentation', 'Demo product roadmap'],
    'Research': ['Explore new ML model', 'Compare cloud providers', 'Read paper on LLMs', 'Benchmark database']
}

# Generate 100 tasks per category
data = []
task_id = 1
for cat in categories:
    for _ in range(100):  # 100 per category
        text = random.choice(task_phrases[cat])
        priority = random.choice(priorities)
        status = random.choice(statuses)
        due = datetime.now() + timedelta(days=random.randint(1, 30))
        data.append({
            'task_id': task_id,
            'task_text': text,
            'category': cat,
            'priority': priority,
            'status': status,
            'due_date': due.strftime('%Y-%m-%d')
        })
        task_id += 1

# Save as CSV
df = pd.DataFrame(data)
df.to_csv("synthetic_tasks_500.csv", index=False)
print("Saved as 'synthetic_tasks_500.csv'")
