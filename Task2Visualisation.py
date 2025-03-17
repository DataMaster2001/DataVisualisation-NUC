import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Diabetes Factors Dashboard",
    page_icon="🩺",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('diabetes_dataset.csv')
        
        outcome_map = {0: "No Diabetes", 1: "Diabetes"}
        df['Diabetes_Status'] = df['Outcome'].map(outcome_map)
        
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['<30', '30-39', '40-49', '50-59', '60+']
        )
        
        df['BMI_Category'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
df = load_data()

st.title("Diabetes Health Data Dashboard")
st.markdown("""
This interactive dashboard can be used to explore relationships between different health data and diabetes, the data has 9538 records of patients.
""")

st.sidebar.header("Filters")

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['Age'].min()) if df is not None else 18,
    max_value=int(df['Age'].max()) if df is not None else 90,
    value=(18, 90)
)

bmi_range = st.sidebar.slider(
    "BMI Range",
    min_value=float(df['BMI'].min()) if df is not None else 15.0,
    max_value=float(df['BMI'].max()) if df is not None else 50.0,
    value=(15.0, 50.0)
)

glucose_range = st.sidebar.slider(
    "Glucose Range",
    min_value=float(df['Glucose'].min()) if df is not None else 50.0,
    max_value=float(df['Glucose'].max()) if df is not None else 200.0,
    value=(50.0, 200.0)
)

if df is not None:
    filtered_df = df[
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1]) &
        (df['BMI'] >= bmi_range[0]) & 
        (df['BMI'] <= bmi_range[1]) &
        (df['Glucose'] >= glucose_range[0]) & 
        (df['Glucose'] <= glucose_range[1])
    ]
    
    st.sidebar.markdown(f"**Filtered Data:** {filtered_df.shape[0]} records")
    
    diabetes_counts = filtered_df['Diabetes_Status'].value_counts()
    st.sidebar.markdown("### Diabetes Distribution")
    st.sidebar.text(f"No Diabetes: {diabetes_counts.get('No Diabetes', 0)}")
    st.sidebar.text(f"Diabetes: {diabetes_counts.get('Diabetes', 0)}")
    
    st.sidebar.markdown("### Explanation of Data Fields")

    data_explanations = {
        "Age": "Age of the patient in years.",
        "BMI": "Body Mass Index (BMI), calculated as weight (kg) / height (m²).",
        "Glucose": "Plasma glucose concentration in mg/dL.",
        "BloodPressure": "Diastolic blood pressure (mm Hg).",
        "HbA1c": "Hemoglobin A1c percentage, indicating average blood sugar levels over the past 3 months.",
        "LDL": "Low-Density Lipoprotein (bad cholesterol) in mg/dL.",
        "HDL": "High-Density Lipoprotein (good cholesterol) in mg/dL.",
        "Triglycerides": "Level of triglycerides (fat) in the blood (mg/dL).",
        "Outcome": "Indicates whether the patient has diabetes (1) or not (0)."
    }

    selected_field = st.sidebar.selectbox("Select a data field to learn more:", list(data_explanations.keys()))

    st.sidebar.info(data_explanations[selected_field])

    row1_col1, row1_col2 = st.columns([1, 2])
    
    with row1_col1:
        st.markdown("### Key Metrics")
        
        total_count = filtered_df.shape[0]
        diabetes_percent = (filtered_df['Outcome'] == 1).mean() * 100
        avg_age = filtered_df['Age'].mean()
        avg_bmi = filtered_df['BMI'].mean()
        avg_glucose = filtered_df['Glucose'].mean()
        
        st.metric("Total Patients", f"{total_count}")
        st.metric("Diabetes Rate", f"{diabetes_percent:.1f}%")
        st.metric("Average Age", f"{avg_age:.1f}")
        st.metric("Average BMI", f"{avg_bmi:.1f}")
        st.metric("Average Glucose", f"{avg_glucose:.1f}")
        
    with row1_col2:
        st.markdown("### Diabetes Distribution by Age Group")
        
        age_diabetes_counts = filtered_df.groupby(['Age_Group', 'Diabetes_Status']).size().reset_index(name='Count')
        
        fig = px.bar(
            age_diabetes_counts,
            x='Age_Group',
            y='Count',
            color='Diabetes_Status',
            barmode='group',
            title='Diabetes Distribution by Age Group',
            color_discrete_sequence=['indianred', 'skyblue']
        )
        
        fig.update_layout(
            xaxis_title='Age Group',
            yaxis_title='Count',
            legend_title='Diabetes Status'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    risk_factors_tab1= st.tabs(['Visuals'])
    
    with risk_factors_tab1[0]:
        st.markdown("### Distribution of Diabetics by BMI Category")

        
        patient_group = st.radio(
            "Select Patient Group:",
            ["All Patients", "Diabetics Only", "Non-Diabetics Only"],
            horizontal=True
        )
        
    if patient_group == "Diabetics Only":
        filtered_for_chart = filtered_df[filtered_df['Outcome'] == 1]
        title_suffix = "Diabetics"
        color = '#e74c3c'  
    elif patient_group == "Non-Diabetics Only":
        filtered_for_chart = filtered_df[filtered_df['Outcome'] == 0]
        title_suffix = "Non-Diabetics"
        color = '#3498db' 
    else:
        filtered_for_chart = filtered_df
        title_suffix = "All Patients"
        color = None 
        
    bmi_counts = filtered_for_chart['BMI_Category'].value_counts().reset_index()
    bmi_counts.columns = ['BMI_Category', 'Count']
    bmi_counts = bmi_counts.sort_values(by='BMI_Category')
    bmi_color_map = {
        "Underweight": "#FF2B2B",
        "Normal": "#FFABAB",
        "Overweight": "#83C9FF",
        "Obese": "#0068C9"
    }
    total = bmi_counts['Count'].sum()
    bmi_counts['Percentage'] = (bmi_counts['Count'] / total * 100).round(1)

    fig = px.pie(
        bmi_counts,
        names='BMI_Category',
        values='Count',
        title=f'BMI Category Distribution for {title_suffix}',
        hover_data=['Percentage'],
        labels={'BMI_Category': 'BMI Category', 'Count': 'Number of Patients'},
        color='BMI_Category', 
        color_discrete_map=bmi_color_map 
    )

    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]}%<extra></extra>'
    )
    fig.update_layout(
        height=500,
        hoverlabel=dict(
            bgcolor="black",
            font_size=16,
            font_family="Rockwell"
        ),
        legend_title="BMI Category",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Total: {total} patients",
        showarrow=False,
        font=dict(size=14),
        bgcolor=None,
        bordercolor="grey",
        borderwidth=1,
        borderpad=4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Relationship Between Variables")

    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown("#### Correlation Heatmap")
        
        numeric_cols = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 'LDL', 'HDL', 'Triglycerides', 'Outcome']
        corr_matrix = filtered_df[numeric_cols].corr('kendall')
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r'
            # title='Correlation between Variables'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with row3_col2:
        st.markdown("#### BMI vs Glucose by Diabetes Status")
        
        fig = px.scatter(
            filtered_df,
            x='BMI',
            y='Glucose',
            color='Diabetes_Status',
            size='Age',
            hover_data=['Age', 'BloodPressure', 'HbA1c'],
            color_discrete_sequence=['#3498db', '#e74c3c'],
            opacity=0.7
            # title='BMI vs Glucose Level by Diabetes Status'
        )
        
        fig.update_layout(
            xaxis_title='BMI',
            yaxis_title='Glucose Level',
            legend_title='Diabetes Status',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
