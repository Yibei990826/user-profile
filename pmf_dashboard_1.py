import datetime

# Related third-party imports
# import markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')

from ixchel.functions.data import *
from ixchel.functions.plots import *
from ixchel.functions.utils import *
from ixchel.functions.processing import *
from ixchel.functions.pricing import *
# from ixchel.openai.chatgpt import *

def create_section_header(title):
    # Use Markdown for the header and an optional HTML horizontal rule for styling
    st.markdown(f"""
    ### {title}
    <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
    """, unsafe_allow_html=True)

# Function to dynamically color the numbers based on condition
def color_number(value, condition):
    color = "green" if condition else "red"
    return f"<span style='color: {color};'>{value}</span>"

# Function to create boxes with title and values
def create_info_box(title, values):
    create_section_header(title)
    for k, v in values.items():
        st.markdown(f"<b>{k}</b>: {v}", unsafe_allow_html=True)

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
            title='PMF-Engagement',
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
        title="Retention per Cohort",
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

def avg_line(average_rates):
    fig, ax = plt.subplots(figsize=(17, 3))  # Create a figure and a set of subplots
    ax.plot(average_rates.index, average_rates.values, marker='o', linestyle='-', color='#2f5293')

    ax.set_xlabel('Month')
    ax.set_ylabel('Percentage')
    ax.set_ylim(bottom=0)  # Ensure the bottom of the y-axis is zero

    # Enhance plot aesthetics with Tufte's principles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgrey')
    ax.spines['bottom'].set_color('lightgrey')

    # Annotate data points
    for i, txt in enumerate(average_rates.values):
        if pd.notna(txt):
            ax.annotate(f'{txt:.1f}%', (average_rates.index[i], average_rates.values[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.grid(False)
    return fig

def display_avg_line(avg_line_plot):
    st.markdown(
        """
        ### Average Retention Plot 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.pyplot(avg_line_plot)

def retention_chart(retention_pmf_df):
    retention_pmf_df['cohort_ym']= retention_pmf_df['cohort'].dt.strftime('%Y-%m')
    n_users = retention_pmf_df.groupby(['cohort'])['count'].max().tolist()
    melted_df = retention_pmf_df.melt(id_vars=['cohort_ym', 'month'], value_vars=['percentage'])
    pivot_df = melted_df.pivot(index='cohort_ym', columns='month', values='value').fillna('')
    pivot_df.insert(0, 'n_users', n_users)
    pivot_df.iloc[-1,1] = 100

    formatted_values = pivot_df.applymap(lambda x: f"{x:.0f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "")
    # Format 'n_users' as integers, ensuring no decimal places and no percentage sign
    formatted_values['n_users'] = pivot_df['n_users'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")
    
    # Create a larger plot object
    fig, ax = plt.subplots(figsize=(17, 5))  # Increase the size as needed
    ax.set_axis_off()  # Turn off the axis
    
    # Create a table
    table = ax.table(
        cellText=formatted_values.values,
        rowLabels=formatted_values.index,
        colLabels=formatted_values.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Set font size and scale for the table
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    # print(pivot_df)
    
    # Apply color coding based on the original data, not formatted data
    n_users_col_index = pivot_df.columns.get_loc('n_users')  # Get the column index for 'n_users'
    for (i, j), val in np.ndenumerate(pivot_df.values):
        if j == n_users_col_index:
            table[(i + 1, j)].set_facecolor('white')  # Always white for 'n_users'
            table[(i + 1, j)].set_text_props(color='black')
        elif pd.isna(val) or val == '':
            table[(i + 1, j)].set_facecolor('white')
            table[(i + 1, j)].set_text_props(color='black')
        elif val > 15:
            table[(i + 1, j)].set_facecolor('#b0d6e2')
            table[(i + 1, j)].set_text_props(color='black')
        else:
            table[(i + 1, j)].set_facecolor('#f5baa6')
            table[(i + 1, j)].set_text_props(color='black')
    
    ax.text(0.5, 0.98, 'Month', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='black', weight='bold')
    ax.text(-0.38, 0.5, 'Cohort', ha='center', va='center', rotation='vertical', transform=ax.transAxes, fontsize=16, color='black', weight='bold')
    
    return fig

def display_retention_chart(retention_chart_plot):
    st.markdown(
        """
        ### Retention Chart 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.pyplot(retention_chart_plot)

def retention_chart_revenue(df):
    # Ensure 'cohort' and 'month' are in datetime format and create 'cohort_ym'
    df['cohort'] = pd.to_datetime(df['cohort'])
    df['timestamp_W'] = pd.to_datetime(df['timestamp_W'])
    df['cohort_ym'] = df['cohort'].dt.strftime('%Y-%m')
    
    # Assuming the 'count' column represents the count of users, used to find 'n_users'
    # n_users = df.groupby(['cohort'])['count'].max().tolist()
    
    # Assuming 'percentage' column is equivalent to 'percentage_adjusted'
    melted_df = df.melt(id_vars=['cohort_ym', 'month_num'], value_vars=['percentage'])
    pivot_df = melted_df.pivot(index='cohort_ym', columns='month_num', values='value').fillna('')
    # pivot_df.insert(0, 'n_users', n_users)

    formatted_values = pivot_df.applymap(lambda x: f"{x:.0f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "")
    # formatted_values['n_users'] = pivot_df['n_users'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")
    
    fig, ax = plt.subplots(figsize=(17, 4))
    ax.set_axis_off()
    table = ax.table(
        cellText=formatted_values.values,
        rowLabels=formatted_values.index,
        colLabels=formatted_values.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    
    # n_users_col_index = pivot_df.columns.get_loc('n_users')
    for (i, j), val in np.ndenumerate(pivot_df.values):
        if pd.isna(val) or val == '':
            table[(i + 1, j)].set_facecolor('white')
            table[(i + 1, j)].set_text_props(color='black')
        elif val > 30:
            table[(i + 1, j)].set_facecolor('#b0d6e2')
            table[(i + 1, j)].set_text_props(color='black')
        else:
            table[(i + 1, j)].set_facecolor('#f5baa6')
            table[(i + 1, j)].set_text_props(color='black')
    
    ax.text(0.5, 0.95, 'Month', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='black', weight='bold')
    ax.text(-0.38, 0.5, 'Cohort', ha='center', va='center', rotation='vertical', transform=ax.transAxes, fontsize=16, color='black', weight='bold')
    
    return fig

def calculate_month_diff(row):
    return (row['timestamp_W'].year - row['cohort'].year) * 12 + (row['timestamp_W'].month - row['cohort'].month)

def display_retention_revenue_chart(retention_revenue_chart_plot):
    st.markdown(
        """
        ### Retention Revenue Chart 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.pyplot(retention_revenue_chart_plot)


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




def main():
    st.title(f'PMF Dashboard')
    email_drop = "ARRAY['ynsyilmaz34@gmail.com', 'yunus.yilmaz@pentesters.oneleet.com','testynsy+1@gmail.com', '%nixtla.io%', 'freeze1111@gmail.com', 'loama18@gmail.com', 'example_user@gmail.com','cristiani.challu@gmail.com', 'max@nixtla.io', 'gabi.weizman@gmail.com', 'fede.garza.ramirez@gmail.com', 'eduardo@panl.app','fxuanming@gmail.com','azulramirez420@gmail.com','garzaazul420@gmail.com']" 
    response_status = "ARRAY[200, 301, 302, 400, 401, 403, 404, 429, 500]"
    path = "ARRAY['/timegpt', '/timegpt_multi_series', '/timegpt_multi_series_anomalies', '/timegpt_multi_series_cross_validation', '/timegpt_multi_series_historic', '/timegpt_historic']"
    used_treshhold = 4
    activity_threshold = 4
    activity_days = 30
    invited_threshold = 4 # Number of 200 calls to determine invited
    period_pmf_w = 'W'
    freq_pmf_w = 'W-MON'
    period_pmf = 'M'
    freq_pmf = 'MS'
    tokens_per_user_monthly = fetch_data_supabase(
    f"""
    SELECT 
        email,
        DATE_TRUNC('month', created_at) AS date,
        SUM(input_tokens) AS total_input_tokens,
        SUM(output_tokens) AS total_output_tokens,
        SUM(finetune_tokens) AS total_finetune_tokens,
        COUNT(*) AS n_requests
    FROM (
        SELECT 
            created_at,
            email, 
            input_tokens, 
            output_tokens, 
            finetune_tokens
        FROM requests_with_users_info
        WHERE 
            input_tokens IS NOT NULL
            AND email IS NOT NULL
            AND email NOT LIKE ALL({email_drop})
            AND response_status = ANY ({response_status})
            AND path LIKE ANY({path})
    ) AS foo
    GROUP BY email, DATE_TRUNC('month', created_at)
    """
    )

    requests_pmf_daily = get_pmf_daily_data(email_drop, path)
    retention_pmf_df, requests_user_weekly_pmf, cohort_df = compute_pmf_retention(requests_pmf_daily,period=period_pmf, freq=freq_pmf, invited_threshold=invited_threshold)
    retention_df, requests_user_weekly, cohort_df = compute_pmf_retention_weekly(requests_pmf_daily, invited_threshold = invited_threshold)
    matrix_actions, ratios = compute_pmf_engagement(cohort_df, requests_user_weekly_pmf, period=period_pmf, freq=freq_pmf)
    pmf_engagement_plot = plot_pmf_engagement(cohort_df, matrix_actions, freq_pmf)
    # display_pmf_engagement(pmf_engagement_plot)

    retention_plot = plot_pmf_retention_adj(cohort_df, retention_pmf_df)
    # display_retention_plot(retention_plot)

    retention_pmf_df['cohort_ym']= retention_pmf_df['cohort'].dt.strftime('%Y-%m')
    n_users = retention_pmf_df.groupby(['cohort'])['count'].max().tolist()
    melted_df = retention_pmf_df.melt(id_vars=['cohort_ym', 'month'], value_vars=['percentage'])
    pivot_df = melted_df.pivot(index='cohort_ym', columns='month', values='value')
    # pivot_df.iloc[-1,0] = 100
    # print(pivot_df)

    average_rates = pivot_df.mean(axis=0)
    avg_line_plot = avg_line(average_rates)
    # display_avg_line(avg_line_plot)

    retention_chart_plot = retention_chart(retention_pmf_df)
    # display_retention_chart(retention_chart_plot)
    
    


    requests_user_daily_df = get_api_requests(email_drop, response_status, path, activity_threshold)
    requests_user_monthly = aggregate_data_by_freq(df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='MS')
    all_users = requests_user_monthly.groupby(['email'])['total_requests'].max().reset_index()

    monthly_active_users_df = compute_monthly_active_users(requests_user_daily_df, activity_threshold)
    monthly_active_users_df['timestamp_MS'] = pd.to_datetime(monthly_active_users_df['timestamp_MS'])
    first_day_current_month = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    selected_row_index = monthly_active_users_df.index[monthly_active_users_df['timestamp_MS'] == first_day_current_month].tolist()[0]
    
    active_users_df = compute_active_users(requests_user_daily_df, activity_days, activity_threshold)
    payment = load_supa('users_with_payment_methods', email_drop)
    user_with_payment_df = pd.merge(payment,active_users_df[['email']], on = 'email', how = 'inner')
    requests_user_monthly_df = compute_users_monthly(requests_user_daily_df, activity_threshold)
    total_costs_monthly, requests_tokens_monthly, aggregated_costs_and_requests = get_monthly_cost_analysis(email_drop, response_status, requests_user_monthly_df)
    revenue_list_month = requests_tokens_monthly.groupby(['timestamp'])['total_cost'].sum().tolist()
    # cost_df = apply_token_costs_to_dataframe(tokens_per_user_monthly)

    # current_year_month = datetime.datetime.now().strftime('%Y-%m')
    # current_year = datetime.datetime.now().year
    # last_month = (datetime.datetime.now() - timedelta(days=30)).strftime('%Y-%m')  # Approximation for the last month

    # # Add relevant time columns to DataFrame
    # cost_df['year_month'] = cost_df['date'].dt.strftime('%Y-%m')
    # cost_df['year'] = cost_df['date'].dt.year

    # # Filtering DataFrames for current year, current month, and last month
    # month_df = cost_df[cost_df['year_month'] == current_year_month]
    # year_df = cost_df[cost_df['year'] == current_year]
    # last_month_df = cost_df[cost_df['year_month'] == last_month]

    # Calculate revenues
    current_month_revenue = "{:,.0f}".format(revenue_list_month[-1])
    month = pd.Timestamp.now().month
    current_year_revenue = "{:,.0f}".format(sum(revenue_list_month[-month:]))
    last_month_revenue = "{:,.0f}".format(revenue_list_month[-2])

    payment['created_at'] = payment['stripe_data'].apply(lambda x: np.nan if 'created' not in x else datetime.datetime.utcfromtimestamp(x['created']))
    payment['month'] = payment['created_at'].dt.strftime('%Y-%m')
    
    # Example data
    active_user_this_month = monthly_active_users_df.loc[selected_row_index, 'active']
    active_user_last_month = monthly_active_users_df.loc[selected_row_index - 1, 'active']
    user_change_pct = (active_user_this_month/active_user_last_month -1)*100
    change_pct = (month_df['total_cost'].sum()/last_month_df['total_cost'].sum() -1)*100
    num_active_users = "{:,.0f}".format(all_users[all_users['total_requests']>= activity_threshold].shape[0])
    current_month_revenue = "{:,.0f}".format(month_df['total_cost'].sum())
    payment_user = payment.shape[0]
    add_payment_month = payment[payment['month']== current_year_month].shape[0]
    current_year_revenue =  "{:,.0f}".format(year_df['total_cost'].sum())

    change_pct_html = color_number(f"{change_pct:.1f}%", change_pct > 0)
    user_change_pct_html = color_number(f"{user_change_pct:.1f}%", user_change_pct > 0)
    quick_ratio_html = color_number(ratios[-1].round(2), ratios[-1]>1.5)
    quick_ratio_avg_html = color_number(ratios[:-1].mean().round(2), ratios[:-1].mean()> 1.5)
    paying_html = color_number(f"+{add_payment_month}", add_payment_month > 0)

    # Using columns to layout boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        create_info_box('Active Users', {'Current Month': f"{active_user_this_month} ({user_change_pct_html})", 'Total Active Users': str(num_active_users)})
    with col2:
        create_info_box('Quick Ratio', {'Current Month': quick_ratio_html, 'Historical Avg': quick_ratio_avg_html})
    with col3:
        create_info_box('Revenue', {'Current Month': f"{current_month_revenue} ({change_pct_html})", 'Paying Users': f"{payment_user} ({paying_html})", 'Annual Cumulated': str(current_year_revenue)})

    requests_user_weekly_pmf['month_num'] = requests_user_weekly_pmf.apply(calculate_month_diff, axis=1)
    cost_df['date'] = cost_df['date'].dt.tz_localize(None)
    user_with_cost = requests_user_weekly_pmf.merge(cost_df, left_on = ['email','timestamp_W'],right_on = ['email','date'], how = 'left')
    cohort_monthly_costs = user_with_cost.groupby(['cohort', 'timestamp_W'])['total_cost'].sum().reset_index()

    base_costs = cohort_monthly_costs[cohort_monthly_costs['cohort'] == cohort_monthly_costs['timestamp_W']].set_index('cohort')['total_cost']
    cohort_monthly_costs['percentage'] = cohort_monthly_costs.apply(lambda x: (x['total_cost'] / base_costs.loc[x['cohort']]) * 100, axis=1)

    current_month = datetime.datetime.now().strftime('%Y-%m-01')
    current_month = pd.to_datetime(current_month)

    # Generate all combinations for each cohort from its start date to the current month
    all_combinations = pd.concat([
        pd.DataFrame({
            'cohort': cohort,
            'timestamp_W': pd.date_range(start=cohort, end=current_month, freq='MS')
        })
        for cohort in cohort_monthly_costs['cohort'].unique()
    ], ignore_index=True)

    # Merge with existing data
    df_full = pd.merge(all_combinations, cohort_monthly_costs, on=['cohort', 'timestamp_W'], how='left')

    # Fill NaN values where no existing data was merged
    df_full['total_cost'] = df_full['total_cost'].fillna(0)
    df_full['percentage'] = df_full['percentage'].fillna(0)

    df_full['month_num'] = df_full.apply(calculate_month_diff, axis=1)
    retention_revenue_chart_plot = retention_chart_revenue(df_full[df_full['cohort']>'2023-08-01'])

    display_pmf_engagement(pmf_engagement_plot)
    display_retention_plot(retention_plot)
    display_avg_line(avg_line_plot)
    display_retention_chart(retention_chart_plot)

    display_retention_revenue_chart(retention_revenue_chart_plot)

if __name__ == "__main__":
    main()