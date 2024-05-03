import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot

from fuzzywuzzy import process
import geopandas as gpd
from geopy import geocoders
import pycountry

from .utils import aggregate_data_by_freq
from .data import fetch_data_ixchel, get_monthly_cost_analysis, fetch_data_supabase
from .processing import compute_users_monthly
from ixchel.openai.chatgpt import *


#1 ----- Data------
# load data from ixchel
def load_data(dataset):
    query = f"""
        SELECT *
        FROM {dataset} 
        """

    # Fetch the daily request data by user from the Ixchel database.
    df = fetch_data_ixchel(query=query)
    return df

# load data from supabase
def get_active_user_data(email_drop, response_status, path,invited_threshold=4):
    # Data
    query = f"""
                SELECT email, COUNT(*) AS count, DATE_TRUNC('month', created_at) AS year_month, COUNT(DISTINCT EXTRACT(DAY FROM created_at)) AS active_days, 
                    COUNT(id) AS monthly_requests
                FROM requests_with_users_info 
                WHERE email IS NOT NULL AND email NOT LIKE ALL(ARRAY['ynsyilmaz34@gmail.com', 'yunus.yilmaz@pentesters.oneleet.com','testynsy+1@gmail.com', '%nixtla.io%', 'freeze1111@gmail.com', 'loama18@gmail.com', 'example_user@gmail.com','cristiani.challu@gmail.com', 'max@nixtla.io', 'gabi.weizman@gmail.com', 'fede.garza.ramirez@gmail.com', 'fxuanming@gmail.com']) 
                    AND response_status = ANY (ARRAY[200, 301, 302, 400, 401, 403, 404, 429, 500]) 
                    AND path LIKE ANY(ARRAY['/timegpt', '/timegpt_multi_series', '/timegpt_multi_series_anomalies', '/timegpt_multi_series_cross_validation', '/timegpt_multi_series_historic', '/timegpt_historic'])
                GROUP BY email, DATE_TRUNC('month', created_at)
                """
    df_user = fetch_data_supabase(query=query)
    return df_user


#2.--------Processing -----------

# Assign roles based on user identity
def segment_role(requests_pmf_daily, signed_up):
    signed_up['domain'] = signed_up['email'].str.extract(r'@(.+)$')
    signed_up = signed_up.drop_duplicates(subset=['email'], keep='first')
    personal_domain = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com','qq.com', '163.com', 'googlemail.com','126.com',\
                   'icloud.com','protonmail.com','foxmail.com','gmx.de','live.com','proton.me','sina.com' ]
    signed_up['role'] = 'personal'
    signed_up.loc[~signed_up['domain'].isin(personal_domain), 'role'] = 'organization'
    signed_up.loc[~signed_up['org_name'].isna(), 'role'] = 'organization'
    signed_up.loc[signed_up['domain'].str.contains(r'\bedu\b|\bac\b|universi', case=False, na=False, regex=True), 'role'] = 'student'
    signed_up.loc[signed_up['org_name'].str.contains(r'universi', case=False, na=False, regex=True), 'role'] = 'student'
    requests_pmf_daily = pd.merge(requests_pmf_daily, signed_up, on = 'email', how = 'left')[['email','total_requests','timestamp','status','role']]
    requests_pmf_daily['role'].fillna('personal', inplace=True)
    return requests_pmf_daily, signed_up


# Assign roles based on user behavior
def segment_freq_user(active_user, day_cut=4, call_cut= 12):
    active_user['days_bucket'] = pd.cut(active_user['active_days'], bins=[0, day_cut, float('inf')], labels=['low', 'high'])
    active_user['calls_bucket'] = pd.cut(active_user['monthly_requests'], bins=[0, call_cut, float('inf')], labels=['low', 'high'])
    conditions = [
        (active_user['days_bucket'] == 'low') & (active_user['calls_bucket'] == 'low'),
        (active_user['days_bucket'] == 'low') & (active_user['calls_bucket'] == 'high'),
        (active_user['days_bucket'] == 'high') & (active_user['calls_bucket'] == 'low'),
        (active_user['days_bucket'] == 'high') & (active_user['calls_bucket'] == 'high')
    ]
    choices = ['Rare Users', 'Power User', 'Occasional User', 'Frequent User']
    active_user['role'] = np.select(conditions, choices, default=None)
    return active_user


# Assign roles based on user spending
def segment_user_spending(requests_tokens_monthly):
    user_cost = requests_tokens_monthly.groupby(['email','timestamp'])['total_cost'].sum().reset_index()
    user_avg_monthly_cost = user_cost.groupby(['email','timestamp'])['total_cost'].mean().reset_index()
    bins = [0, 10, 100, 1000, float('inf')]
    labels = ['0-10', '10-100', '100-1000', '1000+']
    user_avg_monthly_cost['role'] = pd.cut(user_avg_monthly_cost['total_cost'], bins=bins, labels=labels, right=False)
    user_avg_monthly_cost['role'] = user_avg_monthly_cost['role'].astype(str)
    return user_avg_monthly_cost


# Process data for retention plot
def compute_seg_retention(daily_request_seg, period = 8, invited_threshold=4):
    invited_df = daily_request_seg[daily_request_seg['status']==200]
    invited_df = invited_df.groupby('email')['total_requests'].sum().reset_index(drop=False)
    invited_df = invited_df[invited_df['total_requests']>=invited_threshold]
    invited_users = invited_df.email.unique()
    current_date = datetime.datetime.now()
    min_cohort_date = current_date - timedelta(weeks=period)

    # Segment user into monthly cohort, and set cohort to be first day of each month
    cohort_df = daily_request_seg[daily_request_seg['email'].isin(invited_users)]
    cohort_df = cohort_df.sort_values(['email','timestamp']).reset_index(drop=True)
    cohort_df['cum_requests'] = cohort_df.groupby('email')['total_requests'].cumsum()
        
    cohort_df = cohort_df[cohort_df['cum_requests']>=invited_threshold]
    cohort_df = cohort_df[['email','timestamp','role']].groupby('email').min().reset_index(drop=False)
    cohort_df['cohort'] = cohort_df['timestamp'].dt.to_period('W').dt.to_timestamp()
    cohort_df = cohort_df[['email','cohort','role']].sort_values('cohort').reset_index(drop=True)
    cohort_df = cohort_df[cohort_df['cohort'] <= min_cohort_date]
    cohorts = cohort_df.role.unique()
    cohort_sizes = cohort_df.groupby('role').count().reset_index()[['role','email']]
    cohort_sizes.columns = ['role', 'count']
    
    # Aggregate user requests to weekly level, add in cohorts
    requests_user_weekly = daily_request_seg[daily_request_seg['email'].isin(invited_users)].reset_index(drop=True)
    requests_user_weekly['timestamp_W'] = requests_user_weekly['timestamp'].dt.to_period('W').dt.start_time
    requests_user_weekly =requests_user_weekly[['email','timestamp_W','total_requests','role']].groupby(['email','timestamp_W','role']).sum().reset_index(drop=False)
    requests_user_weekly = requests_user_weekly.merge(cohort_df, on=['email','role'], how = 'left')
    requests_user_weekly = requests_user_weekly[requests_user_weekly['timestamp_W'].dt.to_period('W') >= requests_user_weekly['cohort'].dt.to_period('W')].reset_index(drop=True)
    requests_user_weekly['week'] = ((requests_user_weekly['timestamp_W'] - requests_user_weekly['cohort']).dt.days / 7).astype(int)
    requests_user_weekly = requests_user_weekly[requests_user_weekly['week']<= period]
    
    users_cohort_week = requests_user_weekly[['email','role','week']].groupby(['week','role']).count().reset_index(drop=False)

    # Calculate retention per week for each cohort
    role = requests_user_weekly['role'].unique()
    week = requests_user_weekly['week'].unique()
    index = pd.MultiIndex.from_product([role, week], names = ["role", "week"])
    retention_df = pd.DataFrame(index = index).reset_index()
    retention_df = retention_df.merge(users_cohort_week, on=['role','week'], how='left').fillna(0)

    # Calculate matrixs for retention plot
    matrix = retention_df.pivot(index='role', columns='week', values='email')
    retention_df = retention_df.merge(cohort_sizes, on='role', how = 'left')
    retention_df = retention_df.sort_values(by = ['role','week'])
    retention_df['percentage'] = 100*retention_df['email']/retention_df['count']

    return retention_df, requests_user_weekly, cohort_df

# Calculate active user matrix
def compute_monthly_active_users_seg(requests_user_daily_df, signed_up, activity_threshold = 4):
    # Aggregate user request data on a monthly frequency.
    requests_user_monthly = aggregate_data_by_freq(
        df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='MS')
    df = requests_user_monthly.merge(signed_up, on = 'email', how = 'left')[['timestamp_MS','email','total_requests','role']]
    df['role'].fillna(df['role'].mode().iloc[0], inplace = True)
    df['active'] = df['total_requests'] >= activity_threshold
    monthly_active_users = df[df['active']]
    monthly_active_users_df = monthly_active_users.groupby(['timestamp_MS','role'], as_index=False)['active'].sum()

    pivot_df = monthly_active_users_df.pivot_table(index='role', columns='timestamp_MS', values='active', fill_value=0)
    matrix = pivot_df.values.T
    
    return matrix


# Calculate active user matrix for behavior segmentation
def compute_monthly_active_users_beh(active_user_2):
    active_user_3 = active_user_2[active_user_2['monthly_requests']>= 4]
    active_user_4 = active_user_3.groupby(['year_month','role'])['email'].count().reset_index()
    pivot_df = active_user_4.pivot_table(index='role', columns='year_month', values='email', fill_value=0)
    user_matrix = pivot_df.values.T
    return user_matrix


# Calculate revenue matrix
def compute_monthly_revenue_seg(requests_tokens_monthly, signed_up_seg):
    requests_tokens_monthly = requests_tokens_monthly[requests_tokens_monthly['timestamp'] >= '2023-09-01']
    df1 = requests_tokens_monthly.merge(signed_up_seg, on = 'email', how = 'left')[['email','timestamp','total_cost','role']]
    df1['role'].fillna(df1['role'].mode().iloc[0], inplace = True)
    df2 = df1.groupby(['timestamp', 'role']).agg({'email': 'count', 'total_cost': 'sum'}).reset_index()
    matrix = df2.pivot(index='timestamp', columns='role', values='total_cost').values
    return matrix


# Calculate revenue matrix for behavior segmentation
def compute_monthly_revenue_beh(requests_tokens_monthly, active_user_2):
    requests_tokens_monthly = requests_tokens_monthly[requests_tokens_monthly['timestamp'] >= '2023-09-01']
    active_user_2['year_month'] = active_user_2['year_month'].dt.date
    requests_tokens_monthly['timestamp'] = requests_tokens_monthly['timestamp'].dt.date
    df1 = requests_tokens_monthly.merge(active_user_2, left_on = ['email','timestamp'], right_on =['email','year_month'], how = 'left')[['email','timestamp','total_cost','role']]
    df1['role'].fillna('Rare Users', inplace = True)
    df2 = df1.groupby(['timestamp', 'role']).agg({'email': 'count', 'total_cost': 'sum'}).reset_index()
    revenue_matrix = df2.pivot(index='timestamp', columns='role', values='total_cost').fillna(0).values
    return revenue_matrix


#------- Plot--------

# stack bar chart
def plot_stacked_bar_seg(matrix,plot_name, x_name, y_name, metrics, color = ['#f5baa6','#a2d2ba','#b0d6e2','#2f5293'], freq = 'M', start_month = '2023-08'):
    dates = [str(y) for y in pd.period_range(start=start_month, end=pd.Timestamp.now(), freq=freq)]
    metrics = metrics
    colors = color
    bar_data = []
    for i, metric in enumerate(metrics):
        bar_data.append(go.Bar(
                            name=metric, 
                            x=dates, 
                            y=matrix[:,i],
                            marker=dict(color=colors[i % len(colors)]),  # Cycle through colors
                            hoverinfo='all',  # Show all hover info
                        ))

    net_totals = matrix.sum(axis=1)
    
    # Convert dates to strings
    dates = [str(date) for date in dates]
    
    # Add annotations for net totals on top of each bar
    annotations = [dict(
        x=str(date), 
        y=total + 20,  # Adjust the y position for better visibility
        text=str(round(total,2)),
        showarrow=False,
        font=dict(
            family="Arial",
            size=12,
            color="black"
        )
    ) for date, total in zip(dates, net_totals)]

    layout = go.Layout(
            title=plot_name,
            xaxis=dict(
                title=x_name,
                showgrid=False,  # Reduce grid lines
                linecolor='black',  # Clean axes lines
                tickvals=dates,  # Explicitly set the tick values
                ticktext=dates,
                title_font=dict(size=14)  # Clear font for axis titles
            ),
            yaxis=dict(
                title=y_name,
                showgrid=True,  # A light grid can help read values
                gridcolor='lightgray',  # Soft color for grid
                linecolor='black',
                title_font=dict(size=14),
                zeroline=False,  # Remove zero line
            ),
            barmode='relative',
            paper_bgcolor='white',  # White background
            plot_bgcolor='white',  # White plot background for a clean design
            font=dict(
                family="Open Sans, sans-serif",  # Aesthetic font choice
                size=12,
                color="black"
            ),
            annotations = annotations,
            margin=dict(l=40, r=40, t=40, b=40),  # Tighten margins to use space
            showlegend=True,  # This line will remove the legend
            # annotations=annotations  # Add annotations for net totals
        )
    fig = go.Figure(data=bar_data, layout=layout)
    fig.show()

# pie plot
def draw_pie_plot(daily_request_seg, invited_users, colors, title):
    cohort_df = daily_request_seg[daily_request_seg['email'].isin(invited_users)]
    cohort_df = cohort_df.sort_values(['email','timestamp']).reset_index(drop=True)
    role_counts_df = cohort_df.groupby(['email','role'])['total_requests'].sum().reset_index()
    role_counts = role_counts_df['role'].value_counts()
    # Plotting a pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=140, colors = colors)
    plt.title(title)
    plt.show()

# retention plot
def plot_seg_retention(cohort_df_seg, retention_seg_df, seg):
    fig = go.Figure()
    # Customize layout according to Tufte's principles
    layout = dict(
        font=dict(size=10),
        title=f"Retention per {seg}",
        xaxis=dict(title="Week", showgrid=False, zeroline=False),
        yaxis=dict(title="Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        hovermode='x'
    )

    # Add a trace for each cohort
    for cohort in cohort_df_seg[seg].unique():
        retention_df_cohort = retention_seg_df[retention_seg_df[seg]==cohort].reset_index(drop=True)
        fig.add_trace(
            go.Scatter(
                x=retention_df_cohort.week,
                y=retention_df_cohort.percentage,
                mode='lines+markers',
                name=str(cohort),
                line=dict(width=1),
                marker=dict(size=4)
            )
        )

    fig.update_layout(layout)
    fig.show()



######## User Image ########

#------ Industrial distribution-------

# Extract industrial information from Apollo Response and signed_up, and plot
def count_industry(form_submissions):
    # Extract industry info from df
    def extract_industry(response):
        try:
            industry = response['person']['organization']['industry']
            return industry
        except (TypeError, KeyError):
            return None
    form_submissions['industry'] = form_submissions['apollo_response'].apply(extract_industry)
    form_submission_with_industry = form_submissions[form_submissions['industry'].notna()]
    # Map industry into major categories
    industry_mapping = {
        "technology & services": ["telecommunications", "information services", "information technology & services", "internet"],
        "education & research": ['e-learning',"professional training & coaching", "higher education", "research", "education management", "primary/secondary education"],
        "financial & consulting services": ['insurance',"financial services", "management consulting", "investment management", "venture capital & private equity", "banking", "capital markets", "investment banking"],
        "health & wellness": ["hospital & health care", "pharmaceuticals", "health, wellness & fitness", "medical devices", "mental health care", "medical practice"],
        "retail & consumer goods": ["packaging & containers", "food production", "retail", "consumer goods", "consumer services", "apparel & fashion"],
        "energy & utilities": ["oil & energy", "utilities", "environmental services", "renewables & environment", "building materials"],
        "engineering & manufacturing": ["mechanical or industrial engineering", "electrical/electronic manufacturing", "machinery", "semiconductors", "civil engineering", "design", "industrial automation", "automotive"],
        "marketing & media": ["marketing & advertising", "online media", "entertainment", "media production", "public relations & communications", "performing arts"],
        "travel & hospitality": ["aviation & aerospace", "transportation/trucking/railroad", "airlines/aviation", "logistics & supply chain", "leisure, travel & tourism", "hospitality"],
    }
    form_submission_with_industry['mapped_industry'] = form_submission_with_industry['industry'].apply(\
                                            lambda x: next((key for key, value in industry_mapping.items() if x in value), None))
    form_submission_with_industry['mapped_industry'].fillna('other', inplace = True)
    industry_count= pd.DataFrame(form_submission_with_industry['mapped_industry'].value_counts()).reset_index()
    return industry_count


def draw_donut_pie_plots(industry_count_potential, industry_count_active, show=6):
    # Create subplots for each pie chart
    fig, axs = plt.subplots(1, 2, figsize=(13, 3), subplot_kw=dict(aspect="equal"))

    # Iterate over the user groups
    for i, (industry_count, title) in enumerate([(industry_count_potential, 'Potential Users'),
                                                 (industry_count_active, 'Active Users')]):
        # Sorting the DataFrame by count in descending order
        industry_count_sorted = industry_count.sort_values(by='count', ascending=False)

        # Cosmetic settings
        colors = plt.cm.YlOrRd(np.linspace(0.8, 0.2, len(industry_count_sorted)))
        explode = [0.1 if i in industry_count_sorted.head(show).index else 0 for i in range(len(industry_count_sorted))]

        # Plot donut_pie chart
        wedges, texts, autotexts = axs[i].pie(industry_count_sorted['count'],
                                              autopct='',  # Do not display percentage
                                              startangle=140,
                                              colors=colors,
                                              explode=explode,
                                              textprops=dict(color="black"))
        centre_circle = plt.Circle((0, 0), 0.70, color='white')
        axs[i].add_artist(centre_circle)

        # Only display labels for the top X sectors
        for j, (text, autotext, wedge) in enumerate(zip(texts, autotexts, wedges)):
            if j < show:
                angle = np.deg2rad((wedge.theta2 - wedge.theta1) / 2 + wedge.theta1)
                y = np.sin(angle)
                x = np.cos(angle)
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(wedge.theta1)
                percent = industry_count_sorted['count'].iloc[j] / industry_count_sorted['count'].sum() * 100
                axs[i].annotate(f"{industry_count_sorted['mapped_industry'].iloc[j]} \n ({percent:.1f}%)",
                                xy=(x, y),
                                xytext=(1.1 * x, 1.1 * y),
                                horizontalalignment=horizontalalignment,
                                size=9)
        axs[i].set_title(f'{title} Industry Distribution',y=1.2)


    plt.show()

#-----Map-----
# Extract geo information and plot
def extract_geo_info(form_submissions):

    def get_best_match(name, choices):
        best_match = process.extractOne(name, choices)
        return best_match[0] if best_match is not None else None

    def handle_specific_cases(name):
        specific_cases = {'UK': 'United Kingdom',
                          '中国':"China",
                          'USA':'United States of America',
                          'US':'United States of America',
                         'Deutschland':'Germany',
                         'Usa':'United States of America',
                         'Italia':'Italy',
                         'us':'United States of America',
                         'Uk':'United Kingdom',
                         'uk':'United Kingdom',
                         'UAE':'United States of America'}
        return specific_cases.get(name, name)
    
    def standardize_country_name(name):
        try:
            return pycountry.countries.search_fuzzy(name)[0].name
        except (LookupError, AttributeError):
            return None

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    potential_geo= pd.DataFrame(form_submissions['country'].value_counts()).reset_index()
    potential_geo['Mapped_Country'] = potential_geo['country'].apply(lambda x: process.extractOne(x, world['name'])[0] if process.extractOne(x, world['name'])[1] >= 80 else None)
    potential_geo['Best_Match'] = potential_geo['country'].apply(get_best_match, choices=world['name'])
    potential_geo['Best_Match'] = potential_geo['country'].apply(handle_specific_cases)
    potential_geo['Standardized_Country'] = potential_geo['Best_Match'].apply(standardize_country_name)
    potential_geo['Country'] = potential_geo['Standardized_Country'].astype(str).apply(lambda x: process.extractOne(x, world['name'])[0] if isinstance(x, str) and process.extractOne(x, world['name'])[1] >= 80 else None)

    return potential_geo


def plot_global_distributions(potential_geo_potential, potential_geo_active):
    # Aggregate to national level for potential_geo_potential
    potential_geo_potential_agg = potential_geo_potential.groupby('Country')['count'].sum().reset_index()
    potential_geo_potential_agg['count'] = potential_geo_potential_agg['count'].apply(lambda x: min(x, 200))

    # Aggregate to national level for potential_geo_active
    potential_geo_active_agg = potential_geo_active.groupby('Country')['count'].sum().reset_index()
    potential_geo_active_agg['count'] = potential_geo_active_agg['count'].apply(lambda x: min(x, 200))

    # Load a world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge with potential_geo_potential
    merged_potential = world.merge(potential_geo_potential_agg, how='left', left_on='name', right_on='Country')
    merged_potential['count'] = merged_potential['count'].fillna(0)
    merged_active = world.merge(potential_geo_active_agg, how='left', left_on='name', right_on='Country')
    merged_active['count'] = merged_active['count'].fillna(0)

    # Cosmetic settings
    fig, axs = plt.subplots(1, 2, figsize=(16, 12))

    # Plot
    axs[0].set_title('Potential Users Geo Distributions', fontdict={'fontsize': '10', 'fontweight': '3'})
    merged_potential.plot(column='count', cmap='Blues', linewidth=0.8, ax=axs[0], edgecolor='0.8', legend=False)
    axs[0].axis('off')
    axs[1].set_title('Current Users Geo Distributions', fontdict={'fontsize': '10', 'fontweight': '3'})
    merged_active.plot(column='count', cmap='Blues', linewidth=0.8, ax=axs[1], edgecolor='0.8', legend=False)
    axs[1].axis('off')

    plt.show()


#----- User message -----
def extract_usage_info(message_df, message_column):
    
    client = OpenAI()
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')
    
    messages = message_df[message_column].tolist()
    combined_text = " ".join(map(str, messages))
    
    prompt = f"Summarize the following text:\n{combined_text}, highlighting key points on how people are using TimeGPT in different industries such as technology & services, education & research, financial & consulting services, other, engineering & manufacturing, retail & consumer goods, marketing & media, health & wellness, energy & utilities, and travel & hospitality.\nSummary:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": combined_text}
        ],
        max_tokens=200
    )
    summary = response.choices[0].message.content
    return summary
