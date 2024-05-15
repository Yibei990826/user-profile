import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings('ignore')

from ixchel.functions.data import *
from ixchel.functions.plots import *
from ixchel.functions.utils import *
from ixchel.functions.processing import *
from ixchel.functions.pricing import *

def create_section_header(title):
    # Use Markdown for the header and an optional HTML horizontal rule for styling
    st.markdown(f"""
    ### {title}
    <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
    """, unsafe_allow_html=True)

def color_number(value, condition):
    color = "green" if condition else "red"
    return f"<span style='color: {color};'>{value}</span>"

def create_info_box(title, values):
    create_section_header(title)
    for k, v in values.items():
        st.markdown(f"<b>{k}</b>: {v}", unsafe_allow_html=True)

def get_monthly_cost_analysis(email_drop, response_status, requests_user_monthly_df):
    # Define the SQL query to fetch daily request data by user, grouped by day.
    query_tokens_user_monthly = f"""
        SELECT
            email,
            COUNT(*) AS total_requests,
            DATE_TRUNC('month', created_at) as timestamp,
            COALESCE(SUM(input_tokens),0) AS total_input_tokens,
            COALESCE(SUM(output_tokens),0) AS total_output_tokens,
            COALESCE(SUM(finetune_tokens),0) AS total_finetune_tokens
        FROM
            requests_with_users_info
        WHERE
            email IS NOT NULL
            AND email NOT LIKE ALL({email_drop})
            AND response_status = ANY ({response_status})
        GROUP BY
            email, timestamp
        ORDER BY
            timestamp, email;
        """
    requests_tokens_monthly = fetch_data_supabase(query=query_tokens_user_monthly)
    requests_tokens_monthly = pd.DataFrame(requests_tokens_monthly)
    requests_tokens_monthly["timestamp"] = (
        pd.to_datetime(requests_tokens_monthly["timestamp"])
        .dt.normalize()
        .dt.tz_localize(None)
    )

    # Calculate costs by calling the function
    requests_tokens_monthly = apply_token_costs_to_dataframe(requests_tokens_monthly)

    # Lets add company and message
    requests_tokens_monthly = get_company_info_min(requests_tokens_monthly)
    total_costs_monthly = requests_tokens_monthly.groupby("timestamp").agg(
        {
            "cost_input": "sum",
            "cost_output": "sum",
            "cost_finetune": "sum",
            "total_cost": "sum",
        }
    )

    # Perform the merge using the converted 'timestamp' columns
    requests_tokens_monthly = pd.merge(
        requests_tokens_monthly,
        requests_user_monthly_df[["email", "timestamp_MS", "active"]],
        left_on=["email", "timestamp"],
        right_on=["email", "timestamp_MS"],
        how="left",
    )

    # Drop the now redundant 'timestamp_MS' column from 'monthly_active_users'
    requests_tokens_monthly = requests_tokens_monthly.drop(columns=["timestamp_MS"])

    # Perform the groupby and aggregation
    aggregated_costs_and_requests = (
        requests_tokens_monthly.groupby(["email"])
        .agg(
            {
                "total_cost": "sum",  # Sum of total_cost
                "active": "mean",  # Average of active (as an integer now)
                "company": "first",  # Just keep the first company name encountered
                "message": "first",  # Just keep the first message encountered
                "total_requests": "sum",
            }
        )
        .reset_index()
    )
    return total_costs_monthly, requests_tokens_monthly, aggregated_costs_and_requests

def plot_pmf_engagement(cohort_df, matrix_actions, freq):
    # Ensure cohort dates are in datetime format
    cohort_df['cohort'] = pd.to_datetime(cohort_df['cohort'])
    dates = pd.date_range(start="2023-07-01", periods=10, freq='M')
    values = range(9)
    date_strings = dates.strftime('%Y-%m-%d').tolist()
    metrics = ['new', 'retained', 'expanded', 'resurrected', 'contracted', 'churned']
    colors = ['#a2d2ba', '#327556', '#b0d6e2', '#2f5293', '#f5baa6', '#ab3c33']
    bar_data = []
    for i, metric in enumerate(metrics):
        bar_data.append(go.Bar(
                            name=metric, 
                            x=date_strings, 
                            y=matrix_actions[:,i],
                            marker=dict(color=colors[i % len(colors)]),  # Cycle through colors
                            hoverinfo='all',  # Show all hover info
                        ))

    layout = go.Layout(
            # title='PMF-Engagement',
            xaxis=dict(
                title="Time",
                showgrid=False,
                linecolor='black',
                title_font=dict(size=14)
            ),
            yaxis=dict(
                title="Number of actions",
                showgrid=True,
                gridcolor='lightgray',
                linecolor='black',
                title_font=dict(size=14),
                zeroline=False,
            ),
            barmode='relative',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True,
            width=900,  # Specify the width of the figure
            height=400  # Specify the height of the figure
        )
    fig = go.Figure(data=bar_data, layout=layout)
    return fig

def display_pmf_engagement(pmf_engagement_plot):
    st.markdown(
        """
        ### User Engagement Plot 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(pmf_engagement_plot)

def plot_pmf_retention_adj(cohort_df, retention_pmf_df):
    fig = go.Figure()
    # Customize layout according to Tufte's principles
    layout = dict(
        font=dict(size=10),
        # title="Retention per Cohort",
        xaxis=dict(title="Month", showgrid=False, zeroline=False),
        yaxis=dict(title="Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        hovermode='x',
        width=900,  # Specify the width of the figure
        height=400
    )

    # Add a trace for each cohort
    for cohort in cohort_df['cohort'].unique():
        retention_df_cohort = retention_pmf_df[retention_pmf_df.cohort==cohort].reset_index(drop=True)
        fig.add_trace(
            go.Scatter(
                x=retention_df_cohort.month,
                y=retention_df_cohort.percentage,
                mode='lines+markers',
                name=str(cohort),
                line=dict(width=1),
                marker=dict(size=4)
            )
        )

    fig.update_layout(layout)
    return fig

def display_retention_plot(retention_plot):
    st.markdown(
        """
        ### User Retention Plot 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(retention_plot)

def avg_retention(retention_pmf_df):
    retention_pmf_df['cohort_ym'] = retention_pmf_df['cohort'].dt.strftime('%Y-%m')
    melted_df = retention_pmf_df.melt(id_vars=['cohort_ym', 'month'], value_vars=['percentage_adjusted'])
    melted_df.iloc[-1, melted_df.columns.get_loc('value')] = 100
    pivot_df = melted_df.pivot(index='cohort_ym', columns='month', values='value')
    average_rates = pivot_df.mean(axis=0).reset_index()
    average_rates.columns = ['month', 'pct']
    
    fig = go.Figure()
    layout = dict(
        font=dict(size=10),
        # title="Average Retention per Cohort",
        xaxis=dict(title="Month", showgrid=False, zeroline=False),
        yaxis=dict(title="Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        # hovermode='x',
        width=900,  # Specify the width of the figure
        height=400
    )

    fig.add_trace(
        go.Scatter(
            x=average_rates['month'],
            y=average_rates['pct'],
            mode='lines+markers+text',
            line=dict(width=1),
            marker=dict(size=4),
            text=[f"{pct:.1f}%" for pct in average_rates['pct']],
            textposition='bottom center'
        )
    )

    fig.update_layout(layout)
    return fig

def display_avg_line(avg_line_plot):
    st.markdown(
        """
        ### Average Retention Plot 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(avg_line_plot)

def retention_chart(retention_pmf_df, retention_threshold = 15):
    retention_pmf_df['cohort_ym'] = retention_pmf_df['cohort'].dt.strftime('%Y-%m')
    n_users = retention_pmf_df.groupby(['cohort'])['count'].max().tolist()
    melted_df = retention_pmf_df.melt(id_vars=['cohort_ym', 'month'], value_vars=['percentage_adjusted'])
    melted_df.iloc[-1, melted_df.columns.get_loc('value')] = 100
    pivot_df = melted_df.pivot(index='cohort_ym', columns='month', values='value').fillna('')
    pivot_df.insert(0, 'n_users', n_users)

    formatted_values = pivot_df.applymap(lambda x: f"{x:.1f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "")
    formatted_values['n_users'] = pivot_df['n_users'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")

    header_values = ['Cohort'] + ['n_users'] + [f'Month {i}' for i in range(0, len(pivot_df.columns)-1)]
    cell_values = [formatted_values.index.tolist()] + [formatted_values[col].tolist() for col in formatted_values.columns]
    fill_colors = [['white'] + ['white'] * len(formatted_values.index)]
    for col in formatted_values.columns:
        if col == 'n_users':
            fill_colors.append(['white'] * len(formatted_values))
        else:
            fill_colors.append(
                ['white' if val == '' or val == 'nan' else
                 '#b0d6e2' if float(val[:-1]) > retention_threshold else '#f5baa6' for val in formatted_values[col]]
            )
            
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            fill_color='white',
            align='center',
            font=dict(size=14),
            # line=dict(color='black', width=1)
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            align='center',
            font=dict(size=12),
            height=30
        )
    )])
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0.92,
        y1=0.92,
        xref='paper',
        yref='paper',
        line=dict(color='black', width=1)
    )
    
    # Customize layout
    fig.update_layout(
        # title="Retention Table",
        width=900,
        height=500,
        font=dict(size=14)
    )
    return fig

def display_retention_chart(retention_chart_plot):
    st.markdown(
        """
        ### Retention Chart 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(retention_chart_plot)

def retention_revenue_chart(retention_revenue, revenue_threshold = 50):
    retention_revenue['cohort_ym'] = retention_revenue['cohort'].dt.strftime('%Y-%m')
    base_cost = retention_revenue.groupby(['cohort'])['base_cost'].max().tolist()
    melted_df = retention_revenue.melt(id_vars=['cohort_ym', 'month'], value_vars=['pct'])
    pivot_df = melted_df.pivot(index='cohort_ym', columns='month', values='value').fillna('')
    pivot_df.insert(0, 'base_cost', base_cost)

    formatted_values = pivot_df.applymap(lambda x: f"{x:.1f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "")
    formatted_values['base_cost'] = pivot_df['base_cost'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

    header_values = ['Cohort'] + ['base_cost'] + [f'Month {i}' for i in range(0, len(pivot_df.columns)-1)]
    cell_values = [formatted_values.index.tolist()] + [formatted_values[col].tolist() for col in formatted_values.columns]
    fill_colors = [['white'] + ['white'] * len(formatted_values.index)]
    for col in formatted_values.columns:
        if col == 'base_cost':
            fill_colors.append(['white'] * len(formatted_values))
        else:
            fill_colors.append(
                ['white' if val == '' or val == 'nan' else
                 '#b0d6e2' if float(val[:-1]) > revenue_threshold else '#f5baa6' for val in formatted_values[col]]
            )
            
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            fill_color='white',
            align='center',
            font=dict(size=14),
            # line=dict(color='black', width=1)
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            align='center',
            font=dict(size=12),
            height=30
        )
    )])
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0.92,
        y1=0.92,
        xref='paper',
        yref='paper',
        line=dict(color='black', width=1)
    )
    
    # Customize layout
    fig.update_layout(
        # title="Revenue Retention Table",
        width=900,
        height=500,
        font=dict(size=14)
    )
    return fig

def display_retention_revenue_chart(retention_revenue_chart_plot):
    st.markdown(
        """
        ### Retention Revenue Chart 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(retention_revenue_chart_plot)

def main():
    st.title(f'PMF Dashboard')
    email_drop = "ARRAY['ynsyilmaz34@gmail.com', 'yunus.yilmaz@pentesters.oneleet.com','testynsy+1@gmail.com', '%nixtla.io%', 'freeze1111@gmail.com', 'loama18@gmail.com', 'example_user@gmail.com','cristiani.challu@gmail.com', 'max@nixtla.io', 'gabi.weizman@gmail.com', 'fede.garza.ramirez@gmail.com', 'eduardo@panl.app','fxuanming@gmail.com','azulramirez420@gmail.com','garzaazul420@gmail.com']" 
    response_status = "ARRAY[200, 301, 302, 400, 401, 403, 404, 429, 500]"
    path = "ARRAY['/timegpt', '/forecast', '/timegpt_multi_series', '/timegpt_multi_series_anomalies', '/timegpt_multi_series_cross_validation', '/timegpt_multi_series_historic', '/timegpt_historic', '/historic_forecast','/forecast_multi_series','/historic_forecast_multi_series','/anomaly_detection_multi_series','/cross_validation_multi_series']"
    used_treshhold = 4
    activity_threshold = 4
    activity_days = 30
    invited_threshold = 4
    period_pmf = 'M'
    freq_pmf = 'MS'

    current_month = datetime.datetime.now().replace(day=1)
    last_month = current_month - relativedelta(months=1)
    last_2_month = last_month - relativedelta(months=1)
    current_month = current_month.strftime('%Y-%m-%d')
    last_month = last_month.strftime('%Y-%m-%d')
    last_2_month = last_2_month.strftime('%Y-%m-%d')
    current_year = datetime.datetime.now().year

    requests_pmf_daily = get_pmf_daily_data(email_drop, path)
    retention_pmf_df, requests_user_weekly_pmf, cohort_df = compute_pmf_retention(requests_pmf_daily,period=period_pmf, freq=freq_pmf, invited_threshold=invited_threshold)
    retention_df, requests_user_weekly, cohort_df = compute_pmf_retention_weekly(requests_pmf_daily, invited_threshold = invited_threshold)
    avg_line_plot = avg_retention(retention_pmf_df)
    retention_chart_plot = retention_chart(retention_pmf_df, retention_threshold = 15)

    requests_user_daily_df = get_api_requests(email_drop, response_status, path, activity_threshold)
    monthly_active_users_df = compute_monthly_active_users(requests_user_daily_df, activity_threshold)
    current_month_active_user = monthly_active_users_df[monthly_active_users_df['timestamp_MS']==current_month]['active'].iloc[0]
    last_month_active_user = monthly_active_users_df[monthly_active_users_df['timestamp_MS']==last_month]['active'].iloc[0]
    last_2_month_active_user = monthly_active_users_df[monthly_active_users_df['timestamp_MS']==last_2_month]['active'].iloc[0]
    all_active_user = requests_user_daily_df['email'].nunique()
    change_pct_user_last_month = round((last_month_active_user/last_2_month_active_user -1)*100,1)

    requests_user_monthly_df = compute_users_monthly(requests_user_daily_df, activity_threshold)
    total_costs_monthly, requests_tokens_monthly, aggregated_costs_and_requests = get_monthly_cost_analysis(email_drop, response_status, requests_user_monthly_df)
    retention_revenue = calc_retention_revenue(requests_tokens_monthly)
    retention_revenue_chart_plot = retention_revenue_chart(retention_revenue, revenue_threshold = 50)

    matrix_actions, ratios = compute_pmf_engagement(cohort_df, requests_user_weekly_pmf, period=period_pmf, freq=freq_pmf)
    pmf_engagement_plot = plot_pmf_engagement(cohort_df, matrix_actions, freq_pmf)
    retention_plot = plot_pmf_retention_adj(cohort_df, retention_pmf_df)

    current_month_revenue = total_costs_monthly.loc[current_month, 'total_cost']
    last_month_revenue = total_costs_monthly.loc[last_month, 'total_cost']
    total_costs_monthly['year'] = total_costs_monthly.index.year
    payment = load_supa('users_with_payment_methods', email_drop)
    payment['created_at'] = payment['stripe_data'].apply(lambda x: np.nan if 'created' not in x else datetime.datetime.utcfromtimestamp(x['created']))
    payment['month'] = payment['created_at'].apply(lambda x: x.replace(day=1).strftime('%Y-%m-%d'))

    active_user_this_month = current_month_active_user
    active_user_last_month = last_month_active_user
    user_change_pct = (active_user_this_month/active_user_last_month -1)*100
    revenue_change_pct = (current_month_revenue/last_month_revenue -1)*100
    num_active_users = "{:,.0f}".format(all_active_user)
    current_month_revenue = "{:,.0f}".format(current_month_revenue)
    payment_user = payment.shape[0]
    add_payment_month = payment[payment['month']== current_month].shape[0]
    current_year_revenue =  "{:,.0f}".format(total_costs_monthly[total_costs_monthly['year'] == current_year]['total_cost'].sum())

    revenue_change_pct_html = color_number(f"{revenue_change_pct:.1f}%", revenue_change_pct > 0)
    user_change_pct_html = color_number(f"{user_change_pct:.1f}%", user_change_pct > 0)
    user_change_pct_last_month_html = color_number(f"{change_pct_user_last_month:.1f}%", change_pct_user_last_month > 0)
    quick_ratio_html = color_number(ratios[-1].round(2), ratios[-1]>1.5)
    quick_ratio_last_month_html = color_number(ratios[-2].round(2), ratios[-2]>1.5)
    quick_ratio_avg_html = color_number(ratios[:-1].mean().round(2), ratios[:-1].mean()> 1.5)
    paying_html = color_number(f"+{add_payment_month}", add_payment_month > 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        create_info_box('Active Users', {'Current Month': f"{active_user_this_month} ({user_change_pct_html})", 'Last Month': f"{last_month_active_user}  ({user_change_pct_last_month_html})" ,'Total Active Users': str(num_active_users)})
    with col2:
        create_info_box('Quick Ratio', {'Current Month': quick_ratio_html, 'Last Month': quick_ratio_last_month_html,'Historical Avg': quick_ratio_avg_html})
    with col3:
        create_info_box('Revenue', {'Current Month': f"{current_month_revenue} ({revenue_change_pct_html})", 'Paying Users': f"{payment_user} ({paying_html})", 'Annual Cumulated': str(current_year_revenue)})

    display_pmf_engagement(pmf_engagement_plot)
    display_retention_plot(retention_plot)
    display_avg_line(avg_line_plot)
    display_retention_chart(retention_chart_plot)
    display_retention_revenue_chart(retention_revenue_chart_plot)

if __name__ == "__main__":
    main()