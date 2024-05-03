import os
import pandas as pd
import psycopg2
import datetime

from .pricing import apply_token_costs_to_dataframe

# Details:
# - fetch_data_ixchel and fetch_data_supabase use psycopg2 to connect to their respective databases and execute the provided SQL query.
# - get_messages_from_last_7_days uses fetch_data_ixchel to run a specific query designed to retrieve messages from the past week.
# - Both fetch_data functions are general-purpose and can accept any SQL query with or without parameters, while get_messages_from_last_7_days is specific to a task and has no parameters.


def fetch_data_ixchel(query, params=None):
    # fetch_data_ixchel(query, params=None): Fetches data from the Ixchel database into a DataFrame based on the SQL query and optional parameters.

    DB_HOST = 'aws-0-us-east-1.pooler.supabase.com'
    DB_NAME = "postgres"
    DB_USER = 'yibei_ixchel_user.izcrvwzuspcqevjhjlrj'
    DB_PASSWORD = 'ystZczWiBiviLGMm'
    try:
        # Using 'with' to ensure that the connection is closed automatically
        with psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            options="-c statement_timeout=200000",
        ) as connection:
            df = pd.read_sql_query(
                query, connection, params=params
            )  # Use the params argument
            return df
    except Exception as e:
        print(e)
        raise


def fetch_data_supabase(query, params=None):
    # fetch_data_supabase(query, params=None): Fetches data from the Supabase database into a DataFrame based on the SQL query and optional parameters.

    DB_HOST = 'aws-0-us-east-1.pooler.supabase.com'
    DB_NAME = "postgres"
    DB_USER='yibei_dev_portal_user.helnftllsvskaqzkxyyy'
    DB_PASSWORD = 'h7O/7TK1DnSvcCem'
    try:
        # Using 'with' to ensure that the connection is closed automatically
        with psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            options="-c statement_timeout=200000",
        ) as connection:
            df = pd.read_sql_query(
                query, connection, params=params
            )  # Use the params argument
            return df
    except Exception as e:
        print(e)
        raise


def load_supa(database, email_drop):
    # load tables from supabase
    query = f"""
            SELECT *
            FROM {database} 
            WHERE 
                email IS NOT NULL 
                AND email NOT LIKE ALL({email_drop}) 
            """
    df = fetch_data_supabase(query=query)
    return df


def get_messages_from_last_7_days():
    # get_messages_from_last_7_days(): Fetches the last 7 days' worth of messages from public.form_submissions and returns it as a DataFrame.

    # Define the query to fetch the data.
    # We use NOW() - INTERVAL '7 days' to get the date range for the last 7 days.
    query = """
    SELECT
        name,
        email,
        company,
        message,
        country,
        person_linkedin_url,
        person_title,
        org_industry,
        org_estimated_num_employees,
        org_industries,
        org_annual_revenue
    FROM
        public.form_submissions
    WHERE
        date >= NOW() - INTERVAL '7 days';
    """

    # Use the fetch_data_ixchel_depre function to get the data.
    try:
        df = fetch_data_ixchel(query)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")


def get_monthly_cost_analysis(email_drop, response_status, requests_user_monthly_df):
    # Define the SQL query to fetch daily request data by user, grouped by day.
    query_tokens_user_monthly = f"""
        SELECT 
            u.email,
            COUNT(*) AS total_requests,
            DATE_TRUNC('month', r.created_at) as timestamp,
            COALESCE(SUM(r.input_tokens),0) AS total_input_tokens,
            COALESCE(SUM(r.output_tokens),0) AS total_output_tokens,
            COALESCE(SUM(r.finetune_tokens),0) AS total_finetune_tokens
        FROM 
            users u
            left join users_teams ut on ut."user" = u.id
            left join teams t on t.id = ut.team
            left join keys k on k.team = t.id
            left join requests r on r.key = k.id
        WHERE 
            u.email IS NOT NULL 
            AND u.email NOT LIKE ALL({email_drop}) 
            AND r.response_status = ANY ({response_status}) 
        GROUP BY 
            u.email, timestamp
        ORDER BY 
            timestamp, u.email;
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


def get_company_info(df, email_column="email"):
    # Define the SQL query template
    query_template = """
    SELECT
        email,  -- Make sure 'email' is the field name in your DB
        company,
        message,
        country,
        person_linkedin_url,
        person_title,
        org_industry,
        org_estimated_num_employees,
        org_industries,
        org_annual_revenue
    FROM
        public.form_submissions
    WHERE
        email IN %s
    """

    # Prepare a tuple of email addresses to use in the SQL query's "IN" clause
    email_tuple = tuple(df[email_column].unique())

    # If there are no emails, return the original dataframe without changes
    if not email_tuple:
        return df

    # Fetch additional data for these emails
    additional_info_df = fetch_data_ixchel(query_template, params=(email_tuple,))

    # Merge the original outlier DataFrame with the additional info DataFrame
    # Perform a left join on the email column
    merged_df = pd.merge(df, additional_info_df, on=email_column, how="left")

    return merged_df


# Usage
# email_column should be passed if the email column in outliers_df is named differently.


def get_company_info_min(df, email_column="email"):
    # Define the SQL query template
    query_template = """
    SELECT
        email,  
        company,
        message
    FROM
        public.form_submissions
    WHERE
        email IN %s
    """

    # Prepare a tuple of email addresses to use in the SQL query's "IN" clause
    email_tuple = tuple(df[email_column].unique())

    # If there are no emails, return the original dataframe without changes
    if not email_tuple:
        return df

    # Fetch additional data for these emails
    additional_info_df = fetch_data_ixchel(query_template, params=(email_tuple,))

    # Merge the original outlier DataFrame with the additional info DataFrame
    # Perform a left join on the email column
    merged_df = pd.merge(df, additional_info_df, on=email_column, how="left")

    return merged_df


def get_webform_data():
    # Returns :
    # sub_df with columns: 'submission_date', 'email'
    # daily_sub_count with columns: 'submission_date', 'daily_count'
    # total_sub: Int, total submissions
    # weekly_percent_change_submissions: Float, percentage change in submissions from last week to this week
    # monthly_percent_change_submissions: Float, percentage change in submissions from last week to this week

    # Web Flow (aka Web Form) data from nixtla.io submit interest form.
    query_form_submissions = """
            SELECT date as submission_date, email
            FROM form_submissions
            """
    sub_df = fetch_data_ixchel(query=query_form_submissions)
    sub_df["submission_date"] = pd.to_datetime(sub_df["submission_date"]).dt.date
    daily_df = (
        sub_df.groupby("submission_date")
        .count()
        .reset_index()
        .rename(columns={"email": "daily_count"})
    )
    daily_df = daily_df.sort_values(by="submission_date").reset_index()
    return sub_df, daily_df


def get_signups_data():
    # DataFrames Created:
    # sign_df with columns: 'name', 'email', 'registration_date', 'count'
    # daily_sign_df with columns: 'registration_date', 'count'
    # total_signs: Integer, total number of completed registrations
    # weekly_percent_change: Float, percentage change in registrations week over week

    # Signed up data. Signup means completed registration in dashboard.nixtla.io
    query_signed_up = """
                SELECT name, email, created_at as registration_date
                FROM signed_up
                """
    sign_df = fetch_data_ixchel(query=query_signed_up)
    sign_df["count"] = 1
    sign_df["registration_date"] = sign_df["registration_date"].dt.date
    daily_df = sign_df.groupby("registration_date").count().reset_index(drop=False)
    return sign_df, daily_df


def get_api_requests(email_drop, response_status, path, activity_threshold):
    # DataFrames Created:
    # requests_user_daily: Columns 'email', 'total_requests', 'timestamp' (individual user requests, grouped by day)
    # requests_total_daily_df: Columns 'timestamp', 'total_requests' (sum of all user requests, grouped by day)

    # Define the SQL query to fetch daily request data by user, grouped by day.
    query_requests_user_daily = f"""
            SELECT email, COUNT(*) AS total_requests, DATE_TRUNC('DAY', created_at) as timestamp
            FROM requests_with_users_info 
            WHERE email IS NOT NULL and email NOT LIKE ALL({email_drop}) AND response_status = ANY ({response_status}) AND path LIKE ANY({path})
            GROUP BY email, timestamp
            """

    # Fetch the daily request data by user from the Supabase database.
    requests_user_daily = fetch_data_supabase(query=query_requests_user_daily)
    return requests_user_daily


def get_avg_user_data(email_drop, response_status, path):
    # Data
    query = f"""
                SELECT email, COUNT(*) AS count, MIN(created_at) as first_request, MAX(created_at) as last_request
                FROM requests_with_users_info 
                WHERE email IS NOT NULL and email NOT LIKE ALL({email_drop}) AND response_status = ANY ({response_status}) AND path LIKE ANY({path})
                GROUP BY email
                """
    df_user_avg = fetch_data_supabase(query=query)
    df_user_avg["first_request"] = pd.to_datetime(df_user_avg["first_request"])
    df_user_avg["first_request"] = df_user_avg["first_request"].dt.tz_localize(None)
    today_ = pd.Timestamp.now().tz_localize(None)
    df_user_avg["days"] = (
        today_.normalize() - df_user_avg["first_request"]
    ).dt.days + 1
    df_user_avg["calls"] = df_user_avg["count"] / df_user_avg["days"]
    df_user_avg = df_user_avg[df_user_avg["calls"] >= 1]
    return df_user_avg


def get_mail_funnel_data(signup_df, historic_users_df, active_users_df, payment_df):
    # Invited
    query_email_flow = """
            SELECT email, flow_name
            FROM events_email_flows
            """
    df_emails_flow = fetch_data_ixchel(query=query_email_flow)

    # Calculate the unique counts of emails at each stage
    total_emails_sent = df_emails_flow[
        "email"
    ].nunique()  # Count unique emails that were sent

    # Merge and count unique for signed up users
    emails_signed_up = pd.merge(
        df_emails_flow, signup_df[["email"]], on="email", how="inner"
    )
    signed_up_count = emails_signed_up[
        "email"
    ].nunique()  # Ensure this is a unique count

    # Merge and count unique for users who used the API
    emails_used_api = pd.merge(
        df_emails_flow, historic_users_df[["email"]], on="email", how="inner"
    )
    used_api_count = emails_used_api["email"].nunique()  # Ensure this is a unique count

    # Merge and count unique for active users
    emails_active = pd.merge(
        df_emails_flow, active_users_df[["email"]], on="email", how="inner"
    )
    active_count = emails_active["email"].nunique()  # Ensure this is a unique count

    # Merge and count unique for users added payment methods
    added_payment = pd.merge(
        df_emails_flow, payment_df[["email"]], on="email", how="inner"
    )
    payment_count = added_payment["email"].nunique()

    # Build funnel_values list with integer values
    funnel_stages_email = ["Email Sent", "SignedUp", "Used API", "Active","Added Payment","Billed"]
    funnel_values_email = [
        int(total_emails_sent),
        int(signed_up_count),
        int(used_api_count),
        int(active_count),
        int(payment_count),
        0
    ]
    return funnel_stages_email, funnel_values_email


def get_pmf_daily_data(email_drop, path):
    query_requests_user_daily_pmf = f"""
        SELECT email, COUNT(*) AS total_requests, DATE_TRUNC('DAY', created_at) as timestamp, response_status as status
        FROM requests_with_users_info 
        WHERE email IS NOT NULL and email NOT LIKE ALL({email_drop}) AND path LIKE ANY({path})
        GROUP BY email, status, timestamp
        """

    # Fetch the daily request data by user from the Supabase database.
    requests_pmf_daily = fetch_data_supabase(query=query_requests_user_daily_pmf)
    requests_pmf_daily = requests_pmf_daily[
        requests_pmf_daily["timestamp"] >= "2023-08-01"
    ].reset_index(drop=True)
    return requests_pmf_daily
