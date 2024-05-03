import pandas as pd
from IPython.core.display import HTML
from IPython.display import display, Markdown

def aggregate_data_by_freq(df, unique_id, count_field, date_field, aggregation_level):
    """
    Aggregates data in the DataFrame by the given time frequency on the 'date_field',
    and optionally also by 'unique_id' if it is provided and exists in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data to be aggregated.
    - unique_id: (optional) string representing the column name for unique identifiers.
                 If None or not present in df, aggregation won't be by unique identifier.
    - count_field: string representing the column name for count values.
    - date_field: string representing the column name for date values.
    - aggregation_level: string representing the pandas frequency for aggregation (e.g., 'D', 'W', 'M').
    
    Returns:
    - Aggregated pandas DataFrame with informative column names.
    """

    # Ensure the date_field is of datetime type
    df[date_field] = pd.to_datetime(df[date_field])

    # Ensure the date_field is of datetime type and normalize to midnight, remove timezone if present
    df[date_field] = pd.to_datetime(df[date_field]).dt.normalize().dt.tz_localize(None)

    # Check if unique identifier is provided and if it exists in the DataFrame
    if unique_id and unique_id in df.columns:
        # Perform the aggregation including unique_id
        df_agg = df.groupby([pd.Grouper(key=date_field, freq=aggregation_level), unique_id])[count_field].sum().reset_index()
    else:
        # Perform the aggregation without unique_id
        df_agg = df.groupby(pd.Grouper(key=date_field, freq=aggregation_level))[count_field].sum().reset_index()

    # Sort the aggregated data
    df_agg.sort_values(by=date_field, inplace=True)

    # Rename columns for clarity
    df_agg.columns = [f'timestamp_{aggregation_level}' if col == date_field else col for col in df_agg.columns]

    return df_agg

def find_outliers_iqr(data_frame, column):
    """
    Identifies outliers using the IQR method and returns them.

    Parameters:
    - data_frame: pandas DataFrame in which to find outliers.
    - column: The name of the column to check for outliers.

    Returns:
    - outliers: DataFrame containing the outliers.
    """

    # Calculate IQR
    Q1 = data_frame[column].quantile(0.25)
    Q3 = data_frame[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outlier_condition = (data_frame[column] < lower_bound) | (data_frame[column] > upper_bound)
    outliers = data_frame[outlier_condition]

    return outliers

def dataframe_to_html(df):
    # Define a function to create an HTML string for DataFrame display
    style = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    """
    
    # Start the HTML string with the table tag and headers
    html = style + '<table>'
    html += '<tr><th>Email</th><th>Company</th><th>Message</th></tr>'
    
    # Add rows for each item in the DataFrame
    for _, row in df.iterrows():
        # Handle non-available company information
        company_info = row.get('company', 'N/A') if pd.notnull(row.get('company')) else 'N/A'
        html += f"<tr><td>{row['email']}</td><td>{company_info}</td><td>{row['message']}</td></tr>"
    
    # Close the table tag
    html += '</table>'
    return html

def display_top_users(df, status, sort_column, num_results=3):
    # Modify the function to use displayHTML to show DataFrame
    # Filter and sort users based on status and sort_column
    filtered_users = df[df['active'] == status].sort_values(by=sort_column, ascending=False)
    
    # Get the top 'num_results' users from the sorted DataFrame
    top_users = filtered_users.head(num_results)
    
    # Print the top users using an HTML table
    print(f"Top {num_results} {'Active' if status else 'Inactive'} Users by {sort_column}:")
    return dataframe_to_html(top_users[['email', 'company', 'message']])
