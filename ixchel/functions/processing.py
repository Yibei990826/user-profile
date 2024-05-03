import numpy as np
import pandas as pd
import datetime

from .utils import aggregate_data_by_freq

def compute_daily_total_requests(requests_user_daily_df):
    # Summarize total requests for each day across all users.
    requests_total_daily_df = requests_user_daily_df.groupby('timestamp', as_index=False)['total_requests'].sum()
    return requests_total_daily_df

def compute_weekly_total_requests(requests_user_daily_df):
    # Aggregate user request data on a weekly frequency.
    requests_user_weekly = aggregate_data_by_freq(
        df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='W')
    requests_total_weekly_df = requests_user_weekly.groupby('timestamp_W', as_index=False)['total_requests'].sum()
    return requests_total_weekly_df

def compute_users_weekly(requests_user_daily_df, activity_threshold):
    # Weekly calls per user
    requests_user_weekly_df = aggregate_data_by_freq(
        df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='W')
    # Define a threshold to categorize users as active based on their weekly request count.
    requests_user_weekly_df['active'] = requests_user_weekly_df['total_requests'] >= activity_threshold
    return requests_user_weekly_df

def compute_weekly_active_users(requests_user_daily_df, activity_threshold):
    # Aggregate user request data on a weekly frequency.
    requests_user_weekly = aggregate_data_by_freq(
        df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='W')
    requests_user_weekly['active'] = requests_user_weekly['total_requests'] >= activity_threshold
    weekly_active_users = requests_user_weekly[requests_user_weekly['active']]
    weekly_active_users_df = weekly_active_users.groupby('timestamp_W', as_index=False)['active'].sum()
    return weekly_active_users_df

def compute_users_monthly(requests_user_daily_df, activity_threshold):
    # Monthly calls per user
    requests_user_monthly_df = aggregate_data_by_freq(
        df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='MS')
    requests_user_monthly_df['active'] = requests_user_monthly_df['total_requests'] >= activity_threshold
    return requests_user_monthly_df

def compute_monthly_active_users(requests_user_daily_df, activity_threshold):
    # Aggregate user request data on a monthly frequency.
    requests_user_monthly = aggregate_data_by_freq(
        df=requests_user_daily_df, unique_id='email', count_field='total_requests', date_field='timestamp', aggregation_level='MS')
    requests_user_monthly['active'] = requests_user_monthly['total_requests'] >= activity_threshold
    monthly_active_users = requests_user_monthly[requests_user_monthly['active']]
    monthly_active_users_df = monthly_active_users.groupby('timestamp_MS', as_index=False)['active'].sum()
    return monthly_active_users_df

def compute_historic_users(requests_user_daily_df, used_treshhold):
    # df_historic_users with columns: 'email', 'total_requests' (filtered by total_requests > used_threshold)
    requests_df = requests_user_daily_df.groupby('email', as_index=False)['total_requests'].sum()
    historic_users_df = requests_df[requests_df['total_requests']>=used_treshhold].reset_index(drop=True)
    return historic_users_df

def compute_active_users(requests_user_daily_df, activity_days, activity_threshold):
    # active_users_df with columns: 'email', 'last_month_requests' (filtered by last_month_requests >= activity_threshold)
    start_date = datetime.datetime.now() - datetime.timedelta(activity_days+1) # To include first date
    requests_df = requests_user_daily_df[requests_user_daily_df['timestamp']>=start_date]
    requests_df = requests_df.groupby('email', as_index=False)['total_requests'].sum()
    active_users_df = requests_df[requests_df['total_requests']>=activity_threshold].reset_index(drop=True)
    return active_users_df

def calculate_weekly_retention_df(requests_user_weekly):
    # Convert 'timestamp_W' to datetime and ensure 'email' is string
    data = requests_user_weekly
    data['timestamp_W'] = pd.to_datetime(data['timestamp_W'])
    data['email'] = data['email'].astype(str)

    # Filter the data to include only active users
    data = data[data['active'] == True]

    # Get the first activity date for each user
    user_first_activity = data.groupby('email')['timestamp_W'].min().reset_index()
    user_first_activity.columns = ['email', 'first_activity_week']

    # Merge to get the start week for each user's record
    data = data.merge(user_first_activity, on='email')

    # Calculate the cumulative count of users who have started by each week
    cumulative_users_by_week = user_first_activity.groupby('first_activity_week').size().cumsum()

    # Count active users by week
    weekly_active_users = data.groupby('timestamp_W')['email'].nunique()

    # Prepare the retention DataFrame
    retention_df = pd.DataFrame(index=pd.date_range(start=data['timestamp_W'].min(), 
                                                    end=data['timestamp_W'].max(), 
                                                    freq='W'))
    retention_df['cumulative_users'] = cumulative_users_by_week.reindex(retention_df.index, 
                                                                        method='ffill').fillna(method='bfill')
    retention_df['weekly_active_users'] = weekly_active_users.reindex(retention_df.index, fill_value=0)
    retention_df['retention_rate'] = retention_df['weekly_active_users'] / retention_df['cumulative_users']

    return retention_df

def calculate_retention_df_cohorts(data):
    data['timestamp_W'] = pd.to_datetime(data['timestamp_W'])
    data['email'] = data['email'].astype(str)

    # Filter the data to include only active users
    active_data = data[data['active'] == True]

    # Find the first activity date for each user
    user_first_activity = active_data.groupby('email')['timestamp_W'].min().reset_index()
    user_first_activity.columns = ['email', 'cohort']

    # Assign cohort week to each row
    data = data.merge(user_first_activity, on='email', how='left')

    # Initialize the DataFrame to store retention data
    cohorts = data['cohort'].drop_duplicates().sort_values().reset_index(drop=True)
    retention_matrix = pd.DataFrame()

    # For each cohort, calculate retention rate for subsequent weeks
    for cohort in cohorts:
        cohort_data = data[data['cohort'] == cohort]
        # Count the total users in the cohort
        total_users = cohort_data['email'].nunique()
        # Get the weekly retention data
        weekly_data = cohort_data.groupby('timestamp_W')['email'].nunique().reset_index()
        weekly_data['cohort'] = cohort
        weekly_data.rename(columns={'email': 'active_users'}, inplace=True)
        weekly_data['retention'] = weekly_data['active_users'].div(total_users)
        retention_matrix = retention_matrix.append(weekly_data[['cohort', 'timestamp_W', 'retention']], ignore_index=True)

    # Pivot the retention_matrix to have cohorts as columns and weeks as rows
    retention_matrix['timestamp_W'] = retention_matrix['timestamp_W'].dt.to_period('W').dt.start_time # Convert to week periods
    retention_pivot = retention_matrix.pivot_table(index='timestamp_W', columns='cohort', values='retention')
    
    return retention_pivot

def calculate_aggregated_retention_df(data):
    # Convert 'timestamp_W' to datetime and ensure 'email' is a string
    data['timestamp_W'] = pd.to_datetime(data['timestamp_W'])
    data['email'] = data['email'].astype(str)

    # Get the first activity date for each user to determine their cohort based on first use
    user_first_activity = data.groupby('email')['timestamp_W'].min().reset_index()
    user_first_activity.columns = ['email', 'first_activity_week']

    # Merge to label each activity with a user's first activity week
    data = data.merge(user_first_activity, on='email')

    # Calculate cumulative users up to each week
    weekly_new_users = user_first_activity.groupby('first_activity_week').size().cumsum()
    retention_df = pd.DataFrame(weekly_new_users, columns=['cumulative_users'])
    retention_df = retention_df.reindex(pd.date_range(start=retention_df.index.min(), 
                                                      end=data['timestamp_W'].max(), 
                                                      freq='W')).fillna(method='ffill')

    # Calculate the active users for each week
    weekly_active = data[data['active']].groupby('timestamp_W')['email'].nunique()

    # Combine the data into a single dataframe for analysis
    retention_df['weekly_active_users'] = weekly_active
    retention_df['weekly_active_users'] = retention_df['weekly_active_users'].fillna(0)

    # Calculate the retention rate as the proportion of active users out of the total cumulative users
    retention_df['retention_rate'] = retention_df['weekly_active_users'] / retention_df['cumulative_users']

    # Reset the index to have the timestamp as a column
    retention_df.reset_index(inplace=True)
    retention_df.rename(columns={'index': 'timestamp_W'}, inplace=True)

    return retention_df

# Assuming 'requests_user_weekly' is your DataFrame with the user request data
# retention_df = calculate_aggregated_retention_df(requests_user_weekly)

# Print the first few rows of the DataFrame to verify
# print(retention_df.head())

def calculate_retention_by_week_number(dataframe):
    """
    Calculate the retention rate by week number since first activity for users.

    Parameters:
    - dataframe: A pandas DataFrame with 'timestamp_W' as datetime,
                 'email' as user identifier, and 'active' indicating weekly activity.

    Returns:
    - A pandas DataFrame with 'week_number' as the number of weeks since the first activity
      and 'retention_rate' as the percentage of active users in subsequent weeks.
    """
    # Convert 'timestamp_W' to datetime if it's not already and sort by it
    dataframe['timestamp_W'] = pd.to_datetime(dataframe['timestamp_W'])
    dataframe.sort_values('timestamp_W', inplace=True)

    # Find the first activity date for each user
    first_activity = dataframe.groupby('email')['timestamp_W'].min().reset_index()
    first_activity.rename(columns={'timestamp_W': 'first_activity'}, inplace=True)
    
    # Merge to calculate the 'week_number' for each activity
    merged_data = dataframe.merge(first_activity, on='email')
    merged_data['week_number'] = ((merged_data['timestamp_W'] - merged_data['first_activity']) / 
                                  np.timedelta64(1, 'W')).astype(int)
    
    # Aggregate users by 'week_number' to find how many were active at each interval
    week_counts = merged_data.groupby('week_number')['email'].nunique().reset_index(name='active_users')
    
    # Get the total number of users who were ever active
    total_users = dataframe['email'].nunique()

    # Calculate the retention rate
    week_counts['retention_rate'] = week_counts['active_users'] / total_users

    return week_counts[['week_number', 'retention_rate']]


def compute_pmf_retention(requests_user_daily, period, freq, invited_threshold):

    invited_df = requests_user_daily[requests_user_daily['status']==200]
    invited_df = invited_df.groupby('email')['total_requests'].sum().reset_index(drop=False)
    invited_df = invited_df[invited_df['total_requests']>=invited_threshold]
    invited_users = invited_df.email.unique()

    cohort_df = requests_user_daily[requests_user_daily['email'].isin(invited_users)]
    cohort_df = cohort_df.sort_values(['email','timestamp']).reset_index(drop=True)
    cohort_df['cum_requests'] = cohort_df.groupby('email')['total_requests'].cumsum()

    cohort_df = cohort_df[cohort_df['cum_requests']>=invited_threshold]
    cohort_df = cohort_df[['email','timestamp']].groupby('email').min().reset_index(drop=False)
    cohort_df['cohort'] = cohort_df['timestamp'].dt.to_period(period).dt.start_time
    cohort_df = cohort_df[['email','cohort']].sort_values('cohort').reset_index(drop=True)
    cohorts = cohort_df.cohort.unique()
    cohort_sizes = cohort_df.groupby('cohort').count().reset_index()
    cohort_sizes.columns = ['cohort', 'count']

    requests_user_weekly = requests_user_daily[requests_user_daily['email'].isin(invited_users)].reset_index(drop=True)
    requests_user_weekly['timestamp_W'] = requests_user_weekly['timestamp'].dt.to_period(period).dt.start_time
    requests_user_weekly =requests_user_weekly[['email','timestamp_W','total_requests']].groupby(['email','timestamp_W']).sum().reset_index(drop=False)
    requests_user_weekly = requests_user_weekly.merge(cohort_df, on='email')
    requests_user_weekly = requests_user_weekly[requests_user_weekly['timestamp_W']>=requests_user_weekly['cohort']].reset_index(drop=True)
    users_cohort_week = requests_user_weekly[['email','timestamp_W','cohort']].groupby(['cohort','timestamp_W']).count().reset_index(drop=False)

    full_dates = pd.date_range(cohorts.min(), cohorts.max(), freq=freq)
    index = pd.MultiIndex.from_product([cohorts, full_dates], names = ["cohort", "timestamp_W"])
    retention_df = pd.DataFrame(index = index).reset_index()
    retention_df = retention_df[retention_df['cohort']<=retention_df['timestamp_W']].reset_index(drop=True)
    retention_df = retention_df.merge(users_cohort_week, on=['cohort','timestamp_W'], how='left').fillna(0)
    # retention_df['week'] = ((retention_df['timestamp_W'] - retention_df['cohort'])).astype(int) # / np.timedelta64(1, period)

    retention_df['month'] = retention_df.timestamp_W.dt.to_period(period) - retention_df.cohort.dt.to_period(period)
    retention_df['month'] = retention_df['month'].apply(lambda x: x.n)

    retention_df = retention_df[['cohort','month','email', 'timestamp_W']]

    #print('retention_df', retention_df)
    matrix = retention_df.pivot(index='cohort', columns='month', values='email')

    retention_df = retention_df.merge(cohort_sizes, on='cohort')
    retention_df['percentage'] = 100*retention_df['email']/retention_df['count']

    today = pd.Timestamp.today()
    day_in_month = (today + pd.offsets.MonthEnd(0)).day
    day_of_month = today.day
    month_prop = day_of_month/day_in_month

    retention_df['percentage_adjusted'] = retention_df.apply(lambda row: row['percentage'] / month_prop
                                     if row['timestamp_W'].month == today.month
                                     else row['percentage'], axis=1)
    
    return retention_df, requests_user_weekly, cohort_df

def compute_pmf_engagement(cohort_df, requests_user_weekly, period, freq):

    dates = pd.date_range(cohort_df['cohort'].min(), cohort_df['cohort'].max(), freq=freq)
    matrix_actions = np.zeros((len(dates),6)) # new, retained, expanded, resurrected, contracted, churned

    for i, week in enumerate(dates):
        # Last week and current week usage
        if i==0:
            # Empty dataset
            last_week_usage = requests_user_weekly[(requests_user_weekly['timestamp_W']=='2023-01-01')]
        else:
            last_week_usage = requests_user_weekly[(requests_user_weekly['timestamp_W']==dates[i-1])]
        last_week_usage = last_week_usage[['email','total_requests']]
        last_week_usage = last_week_usage.rename(columns={'total_requests':'last_week_requests'})
        week_usage = requests_user_weekly[requests_user_weekly['timestamp_W']==week]
        
        # All users up to that point
        all_users = cohort_df[cohort_df['cohort']<=week].reset_index(drop=True)
        all_users['timestamp_W'] = week
        all_users = all_users.merge(week_usage, how='left').fillna(0)
        all_users = all_users.merge(last_week_usage, how='left').fillna(0)

        # Cases
        all_users['new'] = (all_users['cohort']==all_users['timestamp_W'])

        all_users['retained'] = (all_users['cohort']<all_users['timestamp_W']) & (all_users['last_week_requests']>0) & \
                                (all_users['total_requests']>=0.8*all_users['last_week_requests']) & \
                                (all_users['total_requests']<=1.2*all_users['last_week_requests'])

        all_users['expanded'] = (all_users['cohort']<all_users['timestamp_W']) & (all_users['last_week_requests']>0) & \
                                (all_users['total_requests']>1.2*all_users['last_week_requests'])

        all_users['resurrected'] = (all_users['cohort']<all_users['timestamp_W']) & (all_users['last_week_requests']==0) & \
                                (all_users['total_requests']>0)

        all_users['contracted'] = (all_users['cohort']<all_users['timestamp_W']) & (all_users['last_week_requests']>0) & \
                                (all_users['total_requests']<0.8*all_users['last_week_requests']) & \
                                (all_users['total_requests']>0)

        all_users['churned'] = (all_users['total_requests']==0) & (all_users['last_week_requests']>0)

        matrix_actions[i] = np.sum(all_users.values[:, 5:],axis=0).flatten()

    matrix_actions[:,-2:] = -matrix_actions[:,-2:]

    # Ratios (retained excluded according to formula)
    positive = matrix_actions[1:,0] + matrix_actions[1:,2] + matrix_actions[1:,3]
    negative = - matrix_actions[1:, 4] - matrix_actions[1:, 5]
    ratios = positive/negative
    return matrix_actions, ratios

def calc_monthly_cost(requests_tokens_monthly):
    requests_tokens_monthly['timestamp'] = pd.to_datetime(requests_tokens_monthly['timestamp'])
    
    # Create a new column for the month extracted from the timestamp
    requests_tokens_monthly['month'] = requests_tokens_monthly['timestamp'].dt.to_period('M')
    
    # Group by 'email' and the newly created 'month' column, then calculate the sum of 'total_cost'
    monthly_costs = requests_tokens_monthly.groupby(['email', 'month'])['total_cost'].sum().reset_index()
    
    # If you want to sort the result for better readability
    monthly_costs = monthly_costs.sort_values(by=['email', 'month'])

    return monthly_costs

def calc_revenue_matrix(monthly_costs, start = '2023-08'):
    cust_record = monthly_costs.pivot_table(index='email', columns='month', values='total_cost', aggfunc='sum').fillna(0)
    months = pd.period_range(start=start, end=pd.Timestamp.now(), freq='M')

    for month_i in months:
        revenue_last_month = cust_record[month_i-1]   # The cost in last month
        revenue_month = cust_record[month_i]    # The cost in this month
        revenue_prev_cum = cust_record[[y for y in pd.period_range(start=start, end=month_i-1, freq='M')]].sum(axis=1)    # Accumulated cost previously
        #a.churn
        cust_record[f"x.churn_{month_i}"]=np.where((revenue_last_month>0) & (revenue_month==0), -revenue_last_month,0)
        #b.new_cust
        cust_record[f"x.new_{month_i}"]=np.where((revenue_prev_cum==0) & (revenue_month>0), revenue_month,0)
        #c.upsell
        cust_record[f"x.upsell_{month_i}"]=np.where((revenue_prev_cum>0) & (revenue_month>revenue_last_month),\
                                           revenue_month-revenue_last_month,0)
        #d.downsell
        cust_record[f"x.downsell_{month_i}"]=np.where((revenue_prev_cum>0) & (revenue_month>0)\
                                                     & (revenue_month<revenue_last_month),\
                                                 -(revenue_last_month-revenue_month),0)

    # Aggregate transaction information
    cond = [c for c in cust_record.columns if 'x' in str(c)]
    type_df = pd.DataFrame(cust_record[cond].sum())
    type_df.columns = ['Sum']
    type_df.reset_index(inplace = True)
    # # Generate 'year' and 'type' of aggregated results from column name
    type_df['Type'] = type_df['month'].str.extract(r'(new|churn|downsell|upsell)_\d{4}-\d{2}')[0]
    type_df['Month'] = type_df['month'].str.extract(r'(\d{4}-\d{2})')[0]
    # # Use pivot table to show result
    rst_num = pd.pivot_table(type_df, index='Type', columns='Month', values='Sum')
    rst_num.loc['Total'] = rst_num.sum(axis=0)
    rst_set = rst_num.loc[:'upsell', '2023-09':].round(2)

    matrix = np.array(rst_set.values).astype(int)
    matrix = np.round(matrix, 2)
    matrix = matrix.T
    
    return matrix


def compute_pmf_retention_weekly(requests_pmf_daily, invited_threshold):
    
    requests_pmf_daily['timestamp'] = pd.to_datetime(requests_pmf_daily['timestamp'])
    requests_pmf_daily['timestamp_2'] = requests_pmf_daily['timestamp'] + pd.offsets.MonthEnd(0)
    
    invited_threshold = 4
    invited_df = requests_pmf_daily[requests_pmf_daily['status']==200]
    invited_df = invited_df.groupby('email')['total_requests'].sum().reset_index(drop=False)
    invited_df = invited_df[invited_df['total_requests']>=invited_threshold]
    invited_users = invited_df.email.unique()
    
    # Segment user into monthly cohort, and set cohort to be first day of each month
    cohort_df = requests_pmf_daily[requests_pmf_daily['email'].isin(invited_users)]
    cohort_df = cohort_df.sort_values(['email','timestamp']).reset_index(drop=True)
    cohort_df['cum_requests'] = cohort_df.groupby('email')['total_requests'].cumsum()
    
    cohort_df = cohort_df[cohort_df['cum_requests']>=invited_threshold]
    cohort_df = cohort_df[['email','timestamp']].groupby('email').min().reset_index(drop=False)
    cohort_df['cohort'] = cohort_df['timestamp'].dt.to_period('M').dt.to_timestamp()
    cohort_df = cohort_df[['email','cohort']].sort_values('cohort').reset_index(drop=True)
    cohorts = cohort_df.cohort.unique()
    cohort_sizes = cohort_df.groupby('cohort').count().reset_index()
    cohort_sizes.columns = ['cohort', 'count']

    # Aggregate individual transaction into weekly data
    requests_user_weekly = requests_pmf_daily[requests_pmf_daily['email'].isin(invited_users)].reset_index(drop=True)
    full_dates = pd.date_range(requests_user_weekly['timestamp'].min(), requests_user_weekly['timestamp'].max(), freq='W-MON')
    full_dates = [pd.Timestamp(date).tz_localize(None) for date in full_dates]
    
    requests_user_weekly['timestamp_W'] = requests_user_weekly['timestamp'].dt.to_period('W').dt.start_time
    requests_user_weekly =requests_user_weekly[['email','timestamp_W','total_requests']].groupby(['email','timestamp_W']).sum().reset_index(drop=False)
    requests_user_weekly = requests_user_weekly.merge(cohort_df, on='email', how = 'left')
    requests_user_weekly = requests_user_weekly[requests_user_weekly['timestamp_W'].dt.to_period('M') >requests_user_weekly['cohort'].dt.to_period('M')].reset_index(drop=True)
    users_cohort_week = requests_user_weekly[['email','timestamp_W','cohort']].groupby(['cohort','timestamp_W']).count().reset_index(drop=False)

    # Calculate retention per week for each cohort
    index = pd.MultiIndex.from_product([cohorts, full_dates], names = ["cohort", "timestamp_W"])
    retention_df = pd.DataFrame(index = index).reset_index()
    retention_df['cohort_year_month'] = retention_df['cohort'].dt.to_period('M')
    retention_df['timestamp_W_year_month'] = retention_df['timestamp_W'].dt.to_period('M')
    retention_df = retention_df[retention_df['cohort_year_month'] < retention_df['timestamp_W_year_month']].reset_index(drop=True)
    retention_df.drop(columns=['cohort_year_month', 'timestamp_W_year_month'], inplace=True)
    
    retention_df = retention_df.merge(users_cohort_week, on=['cohort','timestamp_W'], how='left').fillna(0)
    retention_df['cohort_next_month'] = retention_df['cohort'] + pd.DateOffset(months=1)
    retention_df['week'] = (retention_df['timestamp_W'].dt.to_period('W') - retention_df['cohort_next_month'].dt.to_period('W')).apply(lambda x: x.n)
    retention_df.drop(columns=['cohort_next_month'], inplace=True)
    retention_df = retention_df[retention_df['week']>0]
    retention_df = retention_df[['cohort','week','email', 'timestamp_W']]
    
    matrix = retention_df.pivot(index='cohort', columns='week', values='email')
    retention_df = retention_df.merge(cohort_sizes, on='cohort')
    retention_df['percentage'] = 100*retention_df['email']/retention_df['count']
    zero_week_data = pd.DataFrame({'cohort': matrix.index, 'week': 0, 'percentage': 100})
    retention_df = pd.concat([zero_week_data, retention_df], ignore_index=True)
    
    return retention_df, requests_user_weekly, cohort_df