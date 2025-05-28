#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output


# In[52]:


# --- 1. Load and clean data ---
df = pd.read_csv('laptops.csv', encoding='latin1')
df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]

df = df.rename(columns={
    'Model_Na': 'Model_Name',
    'Screen_Siz': 'Screen_Size',
    'Operating': 'Operating_System',
    'Operating_Weight': 'Weight',
    'Price_Euros': 'Price_Euros'
})

df['RAM'] = df['RAM'].astype(str).str.replace('GB', '', regex=False).str.extract('(\d+)').astype(float)
df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).str.extract('(\d+\.?\d*)').astype(float)
df['Price_Euros'] = df['Price_Euros'].astype(str).str.replace(',', '.', regex=False).str.extract('(\d+\.?\d*)').astype(float)

def storage_to_gb(val):
    if pd.isnull(val):
        return np.nan
    val = str(val).upper()
    total = 0
    for part in val.split('+'):
        part = part.strip()
        try:
            if 'TB' in part:
                num = ''.join([c for c in part if c.isdigit() or c == '.'])
                total += float(num) * 1024
            elif 'GB' in part:
                num = ''.join([c for c in part if c.isdigit() or c == '.'])
                total += float(num)
        except:
            continue  # skip parts like 'SSD' or malformed chunks
    return total if total > 0 else np.nan

df['Storage_GB'] = df['Storage'].apply(storage_to_gb)
df['Operating_System'] = df['Operating_System'].astype(str).str.strip()
df['Manufacturer'] = df['Manufacturer'].astype(str).str.strip()
df['Category'] = df['Category'].astype(str).str.strip()


# In[53]:


#Widget Setup

# Brand (with ALL)
brands = list(sorted(df['Manufacturer'].dropna().unique()))
brands.insert(0, "ALL")
manufacturer_widget = widgets.SelectMultiple(
    options=brands,
    value=("ALL",),
    description='Brand:',
    style={'description_width': 'initial'}
)

# Category (with ALL)
categories = list(sorted(df['Category'].dropna().unique()))
categories.insert(0, "ALL")
category_widget = widgets.Dropdown(
    options=categories,
    value="ALL",
    description='Category:',
    style={'description_width': 'initial'}
)

# OS (with ALL)
oses = list(sorted(df['Operating_System'].dropna().unique()))
oses.insert(0, "ALL")
os_widget = widgets.SelectMultiple(
    options=oses,
    value=("ALL",),
    description='OS:',
    style={'description_width': 'initial'}
)

# RAM
ram_widget = widgets.IntSlider(
    value=8, min=int(df['RAM'].min()), max=int(df['RAM'].max()), step=4,
    description='Min RAM (GB):',
    style={'description_width': 'initial'}
)

# Storage: Only standard sizes
standard_storage = [256, 512, 1024, 2048]  # Add more if dataset has larger sizes
storage_widget = widgets.SelectionSlider(
    options=standard_storage,
    value=256,
    description='Min Storage (GB):',
    style={'description_width': 'initial'}
)

# Weight
weight_widget = widgets.FloatSlider(
    value=2.0, min=float(df['Weight'].min()), max=float(df['Weight'].max()), step=0.1,
    description='Max Weight (kg):',
    style={'description_width': 'initial'}
)

# Price
price_widget = widgets.IntSlider(
    value=1000, min=int(df['Price_Euros'].min()), max=int(df['Price_Euros'].max()), step=50,
    description='Budget (€):',
    style={'description_width': 'initial'}
)

# Screen Size
screen_widget = widgets.FloatSlider(
    value=13.0, 
    min=float(df['Screen_Size'].str.replace('"','').astype(float).min()), 
    max=float(df['Screen_Size'].str.replace('"','').astype(float).max()), 
    step=0.1,
    description='Min Screen (in):',
    style={'description_width': 'initial'}
)

button = widgets.Button(description='Recommend Devices', button_style='success')
output = widgets.Output()


# In[54]:


# --- 2. Currency setup ---
EXCHANGE_RATES = {'€': 1, '$': 1.08}
CURRENCY_LABELS = {'€': 'Euro (€)', '$': 'Dollar ($)'}

df['Price_Dollars'] = df['Price_Euros'] * EXCHANGE_RATES['$']


# In[55]:


# --- 3. Widget Setup ---
brands = list(sorted(df['Manufacturer'].dropna().unique()))
brands.insert(0, "ALL")
manufacturer_widget = widgets.SelectMultiple(
    options=brands, value=("ALL",), description='Brand:', style={'description_width': 'initial'}
)

categories = list(sorted(df['Category'].dropna().unique()))
categories.insert(0, "ALL")
category_widget = widgets.Dropdown(
    options=categories, value="ALL", description='Category:', style={'description_width': 'initial'}
)

oses = list(sorted(df['Operating_System'].dropna().unique()))
oses.insert(0, "ALL")
os_widget = widgets.SelectMultiple(
    options=oses, value=("ALL",), description='OS:', style={'description_width': 'initial'}
)

ram_widget = widgets.IntSlider(
    value=8, min=int(df['RAM'].min()), max=int(df['RAM'].max()), step=4,
    description='Min RAM (GB):', style={'description_width': 'initial'}
)
standard_storage = [256, 512, 1024, 2048]
storage_widget = widgets.SelectionSlider(
    options=standard_storage, value=256, description='Min Storage (GB):',
    style={'description_width': 'initial'}
)
weight_widget = widgets.FloatSlider(
    value=2.0, min=float(df['Weight'].min()), max=float(df['Weight'].max()), step=0.1,
    description='Max Weight (kg):', style={'description_width': 'initial'}
)
screen_widget = widgets.FloatSlider(
    value=13.0,
    min=float(df['Screen_Size'].str.replace('"','').astype(float).min()),
    max=float(df['Screen_Size'].str.replace('"','').astype(float).max()),
    step=0.1,
    description='Min Screen (in):', style={'description_width': 'initial'}
)

# Currency dropdown
currency_widget = widgets.Dropdown(
    options=[('Euro (€)', '€'), ('Dollar ($)', '$')],
    value='€',
    description='Currency:', style={'description_width': 'initial'}
)

# Budget slider (created dynamically)
def get_budget_slider(selected_currency, df):
    rate = EXCHANGE_RATES[selected_currency]
    min_value = int(df['Price_Euros'].min() * rate)
    max_value = int(df['Price_Euros'].max() * rate)
    return widgets.IntSlider(
        value=min_value,
        min=min_value,
        max=max_value,
        step=50,
        description=f"Budget ({selected_currency}):",
        style={'description_width': 'initial'}
    )

price_widget = get_budget_slider(currency_widget.value, df)

def on_currency_change(change):
    with output:
        global price_widget
        rate = EXCHANGE_RATES[change['new']]
        # Save current value in euros before switching
        current_value_euro = price_widget.value / EXCHANGE_RATES[currency_widget.value]
        new_slider = get_budget_slider(change['new'], df)
        # Try to keep previous value if in range
        new_val = int(current_value_euro * rate)
        if new_slider.min <= new_val <= new_slider.max:
            new_slider.value = new_val
        price_widget.value = new_slider.value
        price_widget.min = new_slider.min
        price_widget.max = new_slider.max
        price_widget.description = new_slider.description

currency_widget.observe(on_currency_change, names='value')

button = widgets.Button(description='Recommend Devices', button_style='success')
output = widgets.Output()


# In[56]:


# --- 4. Use-case Scoring ---
def compute_usecase_scores(row):
    gaming = (row['RAM'] >= 16) and (('Nvidia' in str(row['GPU'])) or ('AMD' in str(row['GPU'])))
    editing = (row['RAM'] >= 16) and any(x in str(row['CPU']) for x in ['i7', 'Ryzen 7', 'i9'])
    art = 'IPS' in str(row['Screen']) or 'OLED' in str(row['Screen'])
    return pd.Series({
        'Gaming_Score': int(gaming),
        'Editing_Score': int(editing),
        'Art_Score': int(art)
    })

scores = df.apply(compute_usecase_scores, axis=1)
df = pd.concat([df, scores], axis=1)


# In[57]:


# --- 5. Recommendation Logic with currency-aware filtering/output ---
def recommend_devices(b):
    with output:
        clear_output()
        selected_currency = currency_widget.value
        rate = EXCHANGE_RATES[selected_currency]
        filtered = df.copy()
        if "ALL" not in manufacturer_widget.value:
            filtered = filtered[filtered['Manufacturer'].isin(manufacturer_widget.value)]
        if "ALL" not in os_widget.value:
            filtered = filtered[filtered['Operating_System'].isin(os_widget.value)]
        if category_widget.value != "ALL":
            filtered = filtered[filtered['Category'] == category_widget.value]
        filtered = filtered[
            (filtered['RAM'] >= ram_widget.value) &
            (filtered['Storage_GB'] >= storage_widget.value) &
            (filtered['Weight'] <= weight_widget.value) &
            ((filtered['Price_Euros'] * rate) <= price_widget.value) &
            (filtered['Screen_Size'].str.replace('"','').astype(float) >= screen_widget.value)
        ]
        if filtered.empty:
            print('No devices match your criteria.')
            return
        recs = filtered.sort_values(by='Price_Euros').head(10)
        recs = recs.copy()
        recs['Display_Price'] = (recs['Price_Euros'] * rate).round(2)
        display_cols = [
            'Manufacturer', 'Model_Name', 'CPU', 'GPU', 'RAM', 'Storage', 'Screen', 'Operating_System',
            'Weight', 'Display_Price', 'Gaming_Score', 'Editing_Score', 'Art_Score'
        ]
        display(recs[display_cols].rename(columns={'Display_Price': f'Price ({selected_currency})'}))
        plt.figure(figsize=(10,5))
        sns.barplot(x='Model_Name', y='Display_Price', hue='Manufacturer', data=recs, dodge=False)
        plt.title(f'Top Device Recommendations (by Price in {selected_currency})')
        plt.xticks(rotation=25)
        plt.ylabel(f'Price ({selected_currency})')
        plt.tight_layout()
        plt.show()

button.on_click(recommend_devices)

display(
    manufacturer_widget, category_widget, os_widget, ram_widget, storage_widget,
    weight_widget, currency_widget, price_widget, screen_widget, button, output
)


# In[ ]:




