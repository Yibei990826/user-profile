import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import calplot

from utilsforecast.plotting import plot_series
from utilsforecast.preprocessing import fill_gaps

from ixchel.functions.data import *
from ixchel.functions.utils import aggregate_data_by_freq

# import ipywidgets as widgets
# from ipywidgets.embed import embed_minimal_html
# from ipywidgets import IntSlider
# from IPython.display import display

email = 'hwang@lyft.com'
i_token = 1
email_list = f"ARRAY['{email}']"

# Assuming fetch_data_ixchel is defined elsewhere and imports are handled
def fetch_user_data(email_list):
    query_user = f"""
            SELECT *
            FROM signed_up
            where email LIKE ANY({email_list})
            """
    user_df = fetch_data_ixchel(query=query_user)

    query_user_webform = f"""
            SELECT *
            FROM form_submissions
            where email LIKE ANY({email_list})
            """
    user_webform_df = fetch_data_ixchel(query=query_user_webform)

    single_user_df = user_df.merge(user_webform_df, on='email', how='outer')
    return single_user_df

def extract_user_info(single_user_df):
    if single_user_df.empty:
        return {
                'user_name': None,
                'user_email': None,
                'linkedin_url': None,
                'use_case': None,
                'company': None,
                'company_url': None,
                'industry': None,
                'country': None,
                'message': None
        }
    else:
        first_row = single_user_df.iloc[0]
        return {
                'user_name': first_row['name_x'] if pd.notna(first_row['name_x']) else first_row['name_y'],
                'user_email': first_row['email'],
                'linkedin_url': first_row['person_linkedin_url_x'] if pd.notna(first_row['person_linkedin_url_x']) else first_row['person_linkedin_url_y'],
                'use_case': first_row['use_case'],  
                'company': first_row['org_name_x'] if pd.notna(first_row['org_name_x']) else first_row.get('company', None),
                'company_url': first_row['org_website_url_x'] if pd.notna(first_row['org_website_url_x']) else first_row['org_website_url_y'],
                'industry': first_row['org_industries_x'] if pd.notna(first_row['org_industries_x']) else first_row['org_industries_y'],
                'country': first_row['person_country_x'] if pd.notna(first_row['person_country_x']) else first_row['person_country_y'],
                'message': first_row['message']
        }

def display_user_info(user_info):
    # Using markdown to add a title with a custom style that includes a horizontal line
    st.markdown(
        """
        ### User Information
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    
    # Define the number of columns
    keys = list(user_info.keys())
    values = [user_info[key] if user_info[key] is not None else 'NA' for key in keys]
    
    # Using a container to display the user info
    with st.container():
        # Grouping keys and values into rows of 3 items each
        for i in range(0, len(keys), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(keys):
                    with col:
                        # Display each key-value pair
                        st.markdown(f"**{keys[i+j]}:** {values[i+j]}")

def aggregate_by_freq(df, id_col, time_col, freq, aggregations):
    if df[id_col].nunique()!=1:
        print ('Only 1 series is supported')
        return pd.DataFrame()
    if df.index.name != time_col:
        aggregated_df = df.set_index(time_col)
    aggregated_df = aggregated_df.resample(freq).agg(aggregations)
    aggregated_df[id_col]=df[id_col][0]
    return aggregated_df.reset_index()

def process_user_requests(email_list):
    # Fetch data
    query_requests_user = f"""
                SELECT *
                FROM metadata_requests_api
                where email LIKE ANY({email_list})
                """
    requests_df = fetch_data_ixchel(query=query_requests_user)
    requests_df['api_calls'] = 1
    requests_df['success_rate'] = 1.0 * (requests_df['response_status'] == 200)
    requests_df['n_ex_vars'] = requests_df['n_ex_vars'].fillna(0)

    # Map API paths to simpler names
    path_map = {
        '/timegpt': 'Forecast',
        '/forecast': 'Forecast',
        '/timegpt_multi_series': 'Multi_Forecast',
        '/timegpt_multi_series_anomalies': 'Anomaly',
        '/timegpt_multi_series_cross_validation': 'Cross_Validation',
        '/timegpt_multi_series_historic': 'Multi_Historical',
        '/timegpt_historic': 'Historical',
        '/historic_forecast': 'Historical',
        '/forecast_multi_series': 'Multi_Forecast',
        '/historic_forecast_multi_series': 'Multi_Historical',
        '/anomaly_detection_multi_series': 'Anomaly',
        '/cross_validation_multi_series': 'Cross_Validation'
    }
    requests_df['path_short'] = requests_df['path'].map(path_map)

    def aggregate_by_freq(df, id_col, time_col, freq, aggregations):
        if df[id_col].nunique()!=1:
            print ('Only 1 series is supported')
            return pd.DataFrame()
        if df.index.name != time_col:
            aggregated_df = df.set_index(time_col)
        aggregated_df = aggregated_df.resample(freq).agg(aggregations)
        aggregated_df[id_col]=df[id_col][0]
        return aggregated_df.reset_index()

    aggregations={'input_tokens':'sum',
                'output_tokens':'sum',
                'finetune_steps':'sum',
                'success_rate': 'mean',
                'api_calls':'sum'}
    daily_requests_df = aggregate_by_freq(requests_df,
                                        id_col='email',
                                        time_col='created_at',
                                        freq='D',
                                        aggregations=aggregations)
    daily_requests_df['created_at'] = daily_requests_df['created_at'].dt.date

    daily_requests_df = fill_gaps(daily_requests_df,
                                freq='D',
                                end=pd.Timestamp.today().date(),
                                id_col='email',
                                time_col='created_at')
    daily_requests_df = daily_requests_df.fillna(0)
    return daily_requests_df, requests_df


def display_usage_info(daily_requests_df):
    total_calls = daily_requests_df['api_calls'].sum()
    total_input_tokens = daily_requests_df['input_tokens'].sum()
    total_output_tokens = daily_requests_df['output_tokens'].sum()
    total_finetune_steps = daily_requests_df['finetune_steps'].sum()
    time_used = (daily_requests_df['created_at'].max() - daily_requests_df['created_at'].min()).days
    
    total_active_day = daily_requests_df[daily_requests_df['api_calls'] != 0]['created_at'].count()
    # Use markdown to add a title and a horizontal line for visual separation
    st.markdown(
        """
        ### Usage Information
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(4)
    labels = ["User since", "Days", "Active Days", "Total Calls"]
    values = [f"{daily_requests_df['created_at'].min().date()}", f"{time_used} days", 
              f"{total_active_day} days", f"{int(total_calls):,}"]
    
    for col, label, value in zip(cols, labels, values):
        col.markdown(f"**{label}:** <span style='font-size: 1.0em;'>{value}</span>", unsafe_allow_html=True)

    cols = st.columns(3)
    labels = ["Input Tokens", "Output Tokens", "Total Finetune Steps"]
    values = [f"{int(total_input_tokens):,}", f"{int(total_output_tokens):,}", f"{int(total_finetune_steps):,}"]
    for col, label, value in zip(cols, labels, values):
        col.markdown(f"**{label}:** <span style='font-size: 1.0em;'>{value}</span>", unsafe_allow_html=True)

    cols = st.columns(3)
    labels = ["Input Tokens per Run", "Output Tokens Per Run", "Finetune Steps per Run"]
    values = [f"{int(total_input_tokens / total_calls):,}", f"{int(total_output_tokens / total_calls):,}", 
              f"{total_finetune_steps / total_calls:.2f}"]
    
    for col, label, value in zip(cols, labels, values):
        col.markdown(f"**{label}:** <span style='font-size: 1.0em;'>{value}</span>", unsafe_allow_html=True)



def calendar_df(email_list,response_status, path):
    tokens_user_day = fetch_data_supabase(
    f"""
    SELECT 
        email,
        DATE_TRUNC('day', created_at) AS date,
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
            AND email LIKE ANY({email_list})
            AND response_status = ANY ({response_status})
            AND path LIKE ANY({path})
    ) AS foo
    GROUP BY email, DATE_TRUNC('day', created_at)
    """
    )
    tokens_user_day['date'] = pd.to_datetime(tokens_user_day['date']).dt.tz_localize(None)
    tokens_user_day['total_token'] = tokens_user_day['total_input_tokens'] + tokens_user_day['total_output_tokens'] + tokens_user_day['total_finetune_tokens']

    current_date = pd.Timestamp.now().normalize()
    date_range = pd.date_range(start=tokens_user_day['date'].min(), end=current_date, freq='D')

    tokens_user_day = fill_gaps(tokens_user_day,
                                freq='D',
                                end=pd.Timestamp.today().date(),
                                id_col='email',
                                time_col='date')
    tokens_user_day = tokens_user_day.fillna(0)
    values_requests = tokens_user_day.set_index('date').reindex(date_range)['n_requests']
    values_tokens = tokens_user_day.set_index('date').reindex(date_range)['total_token']
    return values_requests, values_tokens

def display_calendar_plots(values_requests, values_tokens):
    st.markdown(
        """
        ### Usage Time Distribution
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)  # Create two columns to display plots side by side

    with col1:  # First column for API Calls
        st.write("Number of API Calls")
        fig_requests, _ = calplot.calplot(values_requests,
                                suptitle_kws={'x': 0.5, 'y': 1.0},
                                figsize=(10, 4),
                                colorbar=True,
                                cmap='viridis')
        st.pyplot(fig_requests)

    with col2:  # Second column for Token Usage
        st.write("Number of tokens Usage")
        fig_tokens,_ = calplot.calplot(values_tokens,
                                suptitle_kws={'x': 0.5, 'y': 1.0},
                                figsize=(10, 4),
                                colorbar=True,
                                cmap='viridis'
                                )
        st.pyplot(fig_tokens)

def ts_plot_df(email_list,response_status, path, daily_requests_df):
    tokens_user = fetch_data_supabase(
    f"""
    SELECT 
        email,
        DATE_TRUNC('month', created_at) AS month,
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
            AND email LIKE ANY({email_list})
            AND response_status = ANY ({response_status})
            AND path LIKE ANY({path})
    ) AS foo
    GROUP BY email, DATE_TRUNC('month', created_at)
    """
    )
    columns = ['input_tokens', 'output_tokens', 'finetune_steps', 'success_rate', 'api_calls']
    df = []
    daily_requests_df['created_at'] = pd.to_datetime(daily_requests_df['created_at'])
    # daily_requests_df['y'] = daily_requests_df['y'].astype(float)


    for column in columns:
        # Extracting the relevant data
        df_col = daily_requests_df[['created_at', column]].reset_index(drop=True)
        
        # Creating a unique ID for the raw data
        df_col['unique_id'] = column
        df_col.columns = ['created_at', 'y', 'unique_id']
        df.append(df_col)

        # Check if column data contains zero or negative values
        if (daily_requests_df[column] <= 0).any():
            # Adjusting zero or negative values before log transformation
            adjusted_values = daily_requests_df[column] + 1
        else:
            adjusted_values = daily_requests_df[column]
        
        # Calculating log of the column values
        log_col = np.log(adjusted_values)
        df_log_col = daily_requests_df[['created_at']].copy()
        df_log_col['y'] = log_col
        df_log_col['unique_id'] = f'log_{column}'
        df_log_col.columns = ['created_at', 'y', 'unique_id']
        df.append(df_log_col)

    # Concatenating all DataFrame parts into one DataFrame for plotting
    plot_df = pd.concat(df).reset_index(drop=True)
    return tokens_user, plot_df

def display_ts_plots(plot_df):
    st.markdown(
        """
        ### Historical Usage
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    usage_plot = plot_series(plot_df, id_col='unique_id', time_col='created_at', target_col='y',plot_random = False, max_ids= 10)
    st.pyplot(usage_plot)

def cost_plot(tokens_user):
    cost_user = apply_token_costs_to_dataframe(tokens_user)
    cost_user['month'] = pd.to_datetime(cost_user['month']).dt.tz_localize(None)
    cost_df = cost_user[['month','total_cost']].rename(columns = {'month':'created_at','total_cost':'y'})
    cost_df['unique_id'] = 'monthly_cost'
    cost_df = fill_gaps(cost_df,
                        freq='M',
                        end=pd.Timestamp.today().date(),
                        id_col='unique_id',
                        time_col='created_at')
    cost_df.fillna(0, inplace = True)
    fig, ax = plt.subplots(figsize=(18, 3))  # Reduced visual span for better focus
    
    # Plotting with Seaborn
    bar_plot = sns.barplot(x='created_at', y='y', data=cost_df, ax=ax, color='skyblue')  # Consistent color to reduce visual noise
    
    # Setting labels with more clarity and concise text
    ax.set_xlabel('Month', fontsize=12)  # Clear and precise
    ax.set_ylabel('Revenue', fontsize=12)  # Corrected spelling and clear
    
    # Setting a concise and clear title
    ax.set_title('Monthly Revenue Overview', fontsize=14, fontweight='bold')
    
    # Adjusting x-ticks for better readability
    plt.xticks(rotation=45, fontsize=10)  # Moderate rotation and font size for clarity
    
    # Removing the grid lines to enhance the data ink ratio
    ax.grid(False)
    
    # Minimizing chart junk by hiding top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotating the exact values over each bar
    for p in bar_plot.patches:
        ax.annotate(format(p.get_height(), ',.0f'),  # Format for decimal places
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Position
                    ha = 'center', va = 'center',  # Center alignment
                    xytext = (0, 9),  # Distance from the top of the bar
                    textcoords = 'offset points', fontsize=10)  # Text properties

    plt.close(fig)  # Ensure the plot only shows when called to display
    return fig

def display_cost_plots(cost_df_plot):
    st.markdown(
        """
        ### Monthly Revenue 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )
    st.pyplot(cost_df_plot)

def bar_plot(data, col, title):
    counts = data[col]. value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = [col, 'counts']
    
    fig = plt.figure(figsize=(5, 3))  # Slightly larger for better readability
    bars = plt.bar(counts_df[col], counts_df['counts'], color='lightblue')
    plt.ylabel('# API Calls', fontsize=12)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Adding annotations to each bar and adjusting position slightly below the top of the bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval - (0.05 * yval), int(yval),  # Adjust yval for text positioning
                 va='top', ha='center', fontsize=10)  # Changed va to 'top' and adjusted positioning
    
    plt.xticks(fontsize=10)
    plt.grid(False)
    
    plt.tight_layout()  # Adjust layout to ensure no clipping and labels are clear
    plt.close(fig)
    return fig

def violin_plot(requests_df):
    # Create a figure and set up the grid
    fig = plt.figure(figsize=(6, 5))  # Slightly increased size for better legibility
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    
    # Create the violin plot on the main grid (bottom left)
    ax0 = fig.add_subplot(gs[1, 0])
    sns.violinplot(x='freq', y='fh', data=requests_df, ax=ax0, cut=0, color='lightblue')
    ax0.grid(False)  # Remove grid to enhance data ink ratio
    ax0.set_xlabel('Frequency', fontsize=12)
    ax0.set_ylabel('Forecast Horizon', fontsize=12)
    
    # Create the histogram on the top grid (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    hist1 = sns.histplot(x='freq', data=requests_df, ax=ax1, bins=len(requests_df['freq'].unique()), element='step', color='gray')
    ax1.xaxis.tick_top()  # Move x-axis ticks to top
    ax1.grid(False)
    ax1.set_ylabel('API Calls', fontsize=12)
    ax1.set_xlabel('')  # Remove redundant x-label

    # Annotating each bar in the top histogram
    for p in hist1.patches:
        ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=10, color='black')
    
    # Create the histogram on the right grid (bottom right)
    ax2 = fig.add_subplot(gs[1, 1])
    hist2 = sns.histplot(y='fh', data=requests_df, ax=ax2, element='step', color='gray')
    ax2.yaxis.tick_right()  # Move y-axis ticks to right
    ax2.grid(False)
    ax2.set_xlabel('API Calls', fontsize=12)
    ax2.set_ylabel('')  # Remove redundant y-label

    # Annotating each bar in the right histogram
    for p in hist2.patches:
        ax2.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), 
                     ha='left', va='center', fontsize=10, color='black')
    
    # Adding a more descriptive and centralized title
    fig.suptitle('Interaction Dynamics between Frequency and Horizon', fontsize=14, fontweight='bold')

    plt.tight_layout()  # Adjust layout to ensure no clipping of labels
    plt.close(fig)  # Use this to store plot in fig, which can be shown later if desired
    return fig

def hist_plot(data, col, title):
    # Creating a figure with two subplots (axes), one above the other
    fig, (ax_violin, ax_hist) = plt.subplots(2, 1, figsize=(5, 4), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    col_data = data[col].dropna()

    # Create a violin plot on the first (top) axis
    sns.violinplot(x=col_data, ax=ax_violin, color='lightblue')
    ax_violin.set_ylabel('Density', fontsize=10)  # Standardizing font size
    ax_violin.set_xlabel('')  # Remove x-label from violin plot for cleaner look
    ax_violin.grid(False)  # Remove grid for a cleaner look
    ax_violin.tick_params(labelsize=10)  # Ensure tick labels are readable

    # Create a histogram on the second (bottom) axis
    sns.histplot(col_data, ax=ax_hist, kde=False, color='gray', bins=30, element='step')
    ax_hist.set_xlabel(col, fontsize=10)  # Standardizing font size
    ax_hist.set_ylabel('# API Calls', fontsize=10)  # Standardizing font size
    ax_hist.grid(False)  # Consistency in grid visibility
    ax_hist.tick_params(labelsize=10)  # Ensure tick labels are readable

    # Set the title on the entire figure
    fig.suptitle(title, fontsize=12, fontweight='bold')  # Standardized and emphasized title
    plt.tight_layout()  # Adjust layout to make room for title if necessary
    plt.close(fig)  # Use this to store plot in fig. Can be shown later if desired.
    return fig

def display_usage_pattern(requests_df):
    st.markdown(
        """
        ### Usage Patterns 
        <hr style='border-top: 1px solid #ccc; margin-top: 0; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(bar_plot(requests_df, 'model', 'Distribution of Models Per API Call'))
    with col2:
        st.pyplot(bar_plot(requests_df, 'path_short', 'Distribution of Endpoints Per API Call'))
        
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(hist_plot(requests_df,'n_target_series', 'Target Series Counts Per API Call'))
    with col2:
        st.pyplot(hist_plot(requests_df,'finetune_steps', 'Fine-Tuning Steps Per API Call'))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(hist_plot(requests_df,'n_ex_vars','External Variable Counts Per API Call'))
    with col2:
        st.pyplot(violin_plot(requests_df))

# def main():
#     response_status = "ARRAY[200, 301, 302, 400, 401, 403, 404, 429, 500]"
#     path = "ARRAY['/timegpt', '/forecast', '/timegpt_multi_series', '/timegpt_multi_series_anomalies', '/timegpt_multi_series_cross_validation', '/timegpt_multi_series_historic', '/timegpt_historic', '/historic_forecast','/forecast_multi_series','/historic_forecast_multi_series','/anomaly_detection_multi_series','/cross_validation_multi_series']"
#     query_user = f" SELECT DISTINCT(email) FROM requests_with_users_info WHERE input_tokens IS NOT NULL"
#     user_df = fetch_data_supabase(query=query_user)
#     # user_email_list = ['hwang@lyft.com']
#     user_email_list = user_df.email.to_list()
#     st.title(f'User Profile')
#     user_email_list = ["hwang@lyft.com", 'scottfree.analytics@scottfreellc.com']
#     email = st.selectbox('Select an email:', user_email_list)
#     email_list = f"ARRAY['{email}']"
#     user_df = fetch_user_data(email_list)
#     user_info = extract_user_info(user_df)
#     display_user_info(user_info)
#     try:
#         daily_requests_df, requests_df = process_user_requests(email_list)
#         display_usage_info(daily_requests_df)

#         tokens_user, plot_df = ts_plot_df(email_list,response_status, path, daily_requests_df)
#         cost_df_plot = cost_plot(tokens_user)
#         display_cost_plots(cost_df_plot)
    
#         values_requests, values_tokens = calendar_df(email_list,response_status, path)
#         display_calendar_plots(values_requests, values_tokens)

#         display_ts_plots(plot_df)
#         display_usage_pattern(requests_df)
#     except:
#         print('No data in')

def main():
    st.title(f'User Profile')
    response_status = "ARRAY[200, 301, 302, 400, 401, 403, 404, 429, 500]"
    path = "ARRAY['/timegpt', '/forecast', '/timegpt_multi_series', '/timegpt_multi_series_anomalies', '/timegpt_multi_series_cross_validation', '/timegpt_multi_series_historic', '/timegpt_historic', '/historic_forecast','/forecast_multi_series','/historic_forecast_multi_series','/anomaly_detection_multi_series','/cross_validation_multi_series']"
    query_user = f"SELECT DISTINCT(email) FROM requests_with_users_info WHERE input_tokens IS NOT NULL"
    user_df = fetch_data_supabase(query=query_user)
    user_email_list_org = user_df.email.dropna().to_list()
    query_sub_user = f""" SELECT DISTINCT(email) FROM metadata_requests_api"""
    user_sub = fetch_data_ixchel(query=query_sub_user)

    user_email_subset = list(set(user_sub.email.to_list()).intersection(set(user_email_list_org)))
    default_ix = user_email_subset.index("hwang@lyft.com")
    email = st.selectbox('Select an email:', options=user_email_subset, index=default_ix)
    st.write("You selected:", email)

    email_list = f"ARRAY['{email}']"
    user_df = fetch_user_data(email_list)
    user_info = extract_user_info(user_df)
    display_user_info(user_info)
    daily_requests_df, requests_df = process_user_requests(email_list)
    display_usage_info(daily_requests_df)
    tokens_user, plot_df = ts_plot_df(email_list,response_status, path, daily_requests_df)
    cost_df_plot = cost_plot(tokens_user)
    display_cost_plots(cost_df_plot)
    
    values_requests, values_tokens = calendar_df(email_list,response_status, path)
    display_calendar_plots(values_requests, values_tokens)

    display_ts_plots(plot_df)
    display_usage_pattern(requests_df)


if __name__ == "__main__":
    main()
