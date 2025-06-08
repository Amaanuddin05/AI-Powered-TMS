import random
import pandas as pd
from datetime import datetime, timedelta

# Define options
categories = ['Bug', 'Feature', 'Documentation', 'Meeting', 'Research']
priorities = ['Low', 'Medium', 'High']
statuses = ['To Do', 'In Progress', 'Done']
tasks = [
    'Fix login issue', 'Implement search functionality', 'Update README file',
    'Team sync-up', 'Test new release', 'Write report on AI trends',
    'Refactor dashboard code', 'Add analytics module', 'Design new logo',
    'Prepare presentation', 'Optimize database queries'
]

# Generate synthetic data
def generate_tasks(n=100):
    data = []
    for i in range(1, n+1):
        task_text = random.choice(tasks)
        category = random.choice(categories)
        priority = random.choice(priorities)
        status = random.choice(statuses)
        due_date = datetime.now() + timedelta(days=random.randint(1, 30))
        data.append({
            'task_id': i,
            'task_text': task_text,
            'category': category,
            'priority': priority,
            'status': status,
            'due_date': due_date.strftime("%Y-%m-%d")
        })
    return pd.DataFrame(data)

# Create and save dataset
df = generate_tasks(100)
df.to_csv("synthetic_tasks.csv", index=False)
print("Synthetic dataset saved as 'synthetic_tasks.csv'")
