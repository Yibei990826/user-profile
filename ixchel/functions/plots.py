import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot


class FunnelChartError(Exception):
    """Custom exception for funnel chart errors."""
    pass

def plot_funnel_chart(funnel_stages, funnel_values, title='Conversion Funnel'):
    """
    Plots a funnel chart with the given stages and values.

    Parameters:
    - funnel_stages: A list of strings representing the stages of the funnel.
    - funnel_values: A list of integers or floats representing the values at each stage.

    Returns:
    - A Plotly figure object if successful.
    
    Raises:
    - FunnelChartError: If any validation check fails.
    """
    
    # Validation checks
    if not isinstance(funnel_stages, list) or not isinstance(funnel_values, list):
        raise FunnelChartError("Both funnel_stages and funnel_values should be lists.")
    if len(funnel_stages) != len(funnel_values):
        raise FunnelChartError("The length of funnel_stages and funnel_values must be the same.")
    if not all(isinstance(item, (int, float)) for item in funnel_values):
        raise FunnelChartError("All items in funnel_values must be numbers (int or float).")
    if not all(isinstance(item, str) for item in funnel_stages):
        raise FunnelChartError("All items in funnel_stages must be strings.")
    
    # Create the figure
    fig = go.Figure(go.Funnel(
        name='Funnel Chart',
        y=funnel_stages,
        x=funnel_values,
        textinfo="value+percent initial",
        opacity=0.7,
        marker={
            "color": ["deepskyblue", "lightseagreen", "gold", "lightcoral", "mediumpurple", "pink"],
            "line": {"width": [0, 0.5, 0.5, 0.5, 0.5, 0]}
        },
    ))

    # Clean up the layout
    fig.update_layout(
        title=title,
        showlegend=False,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        funnelmode="stack",
        funnelgap=0.05
    )
    
    # Show the figure
    #fig.show()
    return fig

def plot_historic_bars(df, date_col, count_col, window_size=7, title=''):
    """
    Plots a bar chart with a trend line for daily registrations, excluding the percentage change.

    Parameters:
    - df: pandas DataFrame containing the registration data.
    - date_col: string name of the DataFrame column with registration dates.
    - count_col: string name of the DataFrame column with daily registration counts.
    - window_size: integer size of the rolling window for trend calculation.
    - title: string title for the chart, which will also include the total count.

    Returns:
    - A Plotly figure object.
    """
    
    # Check if required columns are in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"The DataFrame does not contain the required date column: {date_col}")
    if count_col not in df.columns:
        raise ValueError(f"The DataFrame does not contain the required count column: {count_col}")
    
    # Calculate the total count for the title
    total_count = df[count_col].sum()
    mean_count = df[count_col].mean().round()
    
    # Calculate the trend line with a rolling window
    df['trend'] = df[count_col].rolling(window_size).mean()
    
    # Visualization: Bar Chart for Daily, Weekly, etc. Registrations
    fig_reg = go.Figure()

    # Add bar chart for registrations with values on top of each bar
    fig_reg.add_trace(go.Bar(
        x=df[date_col], 
        y=df[count_col], 
        marker_color='rgba(26, 118, 255, 0.7)',  
        text=df[count_col],
        textposition='outside',
        name=''
    ))

    # Add the trend line to the chart
    fig_reg.add_trace(go.Scatter(
        x=df[date_col],
        y=df['trend'],
        name='Trend line',
        mode='lines',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Update the chart's appearance and layout
    enhanced_title = f"{title} (Total: {total_count:,} | Mean: {mean_count:,})"
    fig_reg.update_layout(
        title_text=enhanced_title,
        xaxis_tickfont_size=14,
        yaxis=dict(
            title="Count",
            showgrid=False,
            showline=False,
            showticklabels=True,
        ),
        # The yaxis2 dictionary has been removed since we no longer have a secondary axis
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)'
    )

    # Remove any traces of the percentage change from the layout if they exist
    fig_reg.layout.pop('yaxis2', None)

    # Drop the prev. created column
    df.drop('trend', axis=1, inplace=True)
    return fig_reg

def create_box_plot(data_frame, column, title, y_axis_title, show_outliers=False, y_axis_range=None):
    """
    Creates and displays a box plot for the specified column in the provided dataframe, with options to exclude outliers
    or set a specific y-axis range.

    Parameters:
    - data_frame: A pandas DataFrame containing the data.
    - column: The name of the column in the dataframe to be plotted.
    - title: The title of the plot.
    - y_axis_title: The title of the Y-axis.
    - show_outliers: Boolean, whether to show outliers in the plot or not.
    - y_axis_range: Tuple or list with two elements (min, max) to define the y-axis range.

    Returns:
    - fig: Plotly Figure object representing the box plot.
    """

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=data_frame[column],
        boxpoints='outliers' if show_outliers else False,  # show only outliers, if allowed
        jitter=0.3,  # spread of data points
        pointpos=-1.8 if show_outliers else None  # position of data points; disable if outliers are hidden
    ))

    fig.update_layout(
        title_text=title,
        yaxis_title=y_axis_title,
        yaxis=dict(
            range=y_axis_range,  # set y-axis range if specified
            showgrid=True,
            zeroline=False,
            showline=True,
            showticklabels=True,
        ),
        xaxis=dict(
            showline=False,
            showticklabels=False,
            showgrid=False
        ),
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)'
    )

    #fig.show()

    return fig

def create_stacked_bar_chart(df, title):

    df['company'] = df['company'].fillna('').astype(str)
    df['message'] = df['message'].fillna('').astype(str)

    email_to_company = df.drop_duplicates('email').set_index('email')['company'].to_dict()
    message_to_company = df.drop_duplicates('email').set_index('email')['message'].to_dict()

    # Convert 'timestamp' to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    # Create a new column 'month_year' as key to sort
    df['month_year'] = df['timestamp'].dt.strftime('%Y-%m')  
    
    # Create a new column with the formatted 'Month Year' for display
    df['display_month_year'] = df['timestamp'].dt.strftime('%B %Y')  # For display

    # Group by 'timestamp' and 'email' to calculate total cost
    monthly_costs = df.groupby(['month_year', 'email'], as_index=False)['total_cost'].sum()
    
    # Sort the values to determine the stack order
    sorted_emails = monthly_costs.sort_values(by=['total_cost'], ascending=False)['email'].unique()

    # Pivot table creation
    pivot_df = monthly_costs.pivot(index='month_year', columns='email', values='total_cost').fillna(0)
    
    # Ensure columns are in sorted order of emails
    pivot_df = pivot_df[sorted_emails]  # Ensure correct column order
    
    # Reindex pivot_df for display using the readable month-year format
    display_index = df.drop_duplicates('month_year').set_index('month_year')['display_month_year']
    pivot_df.index = pivot_df.index.map(display_index)


    
    # Truncate the message to n characters and replace newlines with spaces
    max_length = 150
    truncated_messages = {email: (message.replace('\n', ' ')[:max_length] + '...') if len(message) > max_length else message
                          for email, message in message_to_company.items()}

    # Define your elegant color palette
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']


    # Create the bar data, stacking by user, following the sorted_emails order, including hover info
    bar_data = [go.Bar(
                    name=email, 
                    x=pivot_df.index, 
                    y=pivot_df[email],
                    marker=dict(color=colors[idx % len(colors)]),  # Cycle through colors
                    text=[email for _ in pivot_df.index],
                    textposition = 'inside',
                    hoverinfo='all',  # Show all hover info
                    hovertext=[f"{email}<br>Company: {email_to_company.get(email,'N/A')}<br>Msg: {truncated_messages.get(email,'')}" for _ in pivot_df.index],  # Hover info
                    hovertemplate="%{hovertext}<br>Total Cost: %{y:$.2f}<extra></extra>",  # Custom hover template
                ) for idx, email in enumerate(sorted_emails)]
    
    # Calculate grand total
    grand_total = df['total_cost'].sum()
    monthly_totals = pivot_df.sum(axis=1)

    # Modify the title to include grand total using formatted string literal for comma notation
    grand_total_formatted = f"{grand_total:,.2f}"  # Format the grand total with comma for thousands
    # chart_title = f"Total Cost by User and Month - Grand Total: ${grand_total_formatted}"
    chart_title = f"{title} - Grand Total: ${grand_total_formatted}"

    # Create layout with Tufte's principles in mind
    layout = go.Layout(
        title=chart_title,
        xaxis=dict(
            title="Month",
            showgrid=False,  # Reduce grid lines
            linecolor='black',  # Clean axes lines
            title_font=dict(size=14)  # Clear font for axis titles
        ),
        yaxis=dict(
            title="Total Cost",
            showgrid=True,  # A light grid can help read values
            gridcolor='lightgray',  # Soft color for grid
            linecolor='black',
            title_font=dict(size=14),
            zeroline=False,  # Remove zero line
        ),
        barmode='stack',
        paper_bgcolor='white',  # White background
        plot_bgcolor='white',  # White plot background for a clean design
        font=dict(
            family="Open Sans, sans-serif",  # Aesthetic font choice
            size=12,
            color="black"
        ),
        margin=dict(l=40, r=40, t=40, b=40),  # Tighten margins to use space
        showlegend=False  # This line will remove the legend
    )

    # Create figure and plot
    fig = go.Figure(data=bar_data, layout=layout)

    # Add totals as text on the bars
    for i, total in enumerate(monthly_totals):
        fig.add_annotation(
            x=pivot_df.index.astype(str)[i],
            y=total,
            text="{:,.2f}".format(total),  # Format the text to be rounded (or as you prefer)
            showarrow=False,
            yshift=10  # Shift the text up a little so it sits above the bar
        )

    fig.update_traces(marker_line_width=0.5, marker_line_color='black')  # Thin lines for each bar for definition
    fig.update_xaxes(tickangle=-45)  # Angle x labels to avoid crowding
    fig.show()

def create_scatter_quadrants(df, x_column, y_column, hover_column, title):
    """
    Creates a Tufte-style interactive scatter plot.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    x_column : str
        The name of the column to be used for the x-axis.
    y_column : str
        The name of the column to be used for the y-axis.
    hover_column : str
        The name of the column to be displayed when hovering over points.
    title : str
        The title of the plot.
    """
    # Create the figure with the necessary aesthetics for Tufte principles
    fig = px.scatter(df, x=x_column, y=y_column, hover_data=[hover_column],
                     title=title, template="simple_white")

    # Add jitter to the x-axis points
    fig.update_traces(x=df[x_column]+np.random.uniform(-0.01, 0.01, size=len(df)),marker=dict(size=8, opacity=0.8, line=dict(width=0)))

    v_line_at = 0.8
    h_line_at = 1000

    # Assuming `v_line_at` and `h_line_at` are correctly computed medians from the correct df.
    # Add vertical line at the median of the x axis
    fig.add_shape(type="line", 
                x0=v_line_at, y0=df[y_column].min(), x1=v_line_at, y1=df[y_column].max(),
                line=dict(color="Red", width=2, dash="dot"))

    # Add horizontal line at the median of the y axis
    fig.add_shape(type="line", 
                x0=df[x_column].min(), y0=h_line_at, x1=df[x_column].max(), y1=h_line_at,
                line=dict(color="Red", width=2, dash="dot"))

    # Format the axes and grid
    fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='LightGrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey', 
                     title=x_column)
    fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='LightGrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey', 
                     title=y_column)

    # Calculate the positions for quadrant annotations based on fixed positions
    max_x = df[x_column].max()
    min_x = df[x_column].min()
    max_y = df[y_column].max()
    min_y = df[y_column].min()

    # Calculate middle positions for annotation in each quadrant
    q1_x = (max_x + v_line_at) / 2
    q1_y = (max_y + h_line_at) / 2
    q2_x = (min_x + v_line_at) / 2
    q2_y = q1_y
    q3_x = q2_x
    q3_y = (min_y + h_line_at) / 2
    q4_x = q1_x
    q4_y = q3_y

    # Add annotations for each quadrant
    fig.add_annotation(x=q1_x, y=q1_y, text="Cash Cows 🤑", showarrow=False, font=dict(color="black"))
    fig.add_annotation(x=q2_x, y=q2_y, text="Deserters 🤔", showarrow=False, font=dict(color="black"))
    fig.add_annotation(x=q3_x, y=q3_y, text="LowLow 😒", showarrow=False, font=dict(color="black"))
    fig.add_annotation(x=q4_x, y=q4_y, text="Loyal Base 🏟️", showarrow=False, font=dict(color="black"))


    # Format the x-axis tick format if it's a percentage
    if 'percent' in x_column.lower() or 'activity' in x_column.lower():
        fig.update_xaxes(tickformat=".0%")

    # Hide the legend as Tufte prefers labeling directly on the data when possible
    fig.update_layout(showlegend=False)

    # Tufte's designs prefer clear, thin fonts, so we may opt for a sans-serif family
    fig.update_layout(font=dict(family="Arial, sans-serif", size=10, color="Black"))

    # Minimal coloring for high data-ink ratio
    fig.update_traces(marker_color='rgba(0, 0, 0, 0.6)')

    # Tufte encourages the use of micro/macro readings, you can enable points to be interactive for details
    fig.update_traces(hoverinfo="all")

    # Adding hover label configurations
    fig.update_traces(hoverlabel=dict(bgcolor="white", 
                                      font_size=12, 
                                      font_family="Arial, sans-serif"))

    # Add an interactive hovermode for more information
    fig.update_layout(hovermode='closest')

    # Show the plot
    fig.show()


#################################### PMF ###################################
############################################################################
def plot_pmf_retention(cohort_df, retention_pmf_df):
    fig = go.Figure()
    # Customize layout according to Tufte's principles
    layout = dict(
        font=dict(size=10),
        title="Retention per Cohort",
        xaxis=dict(title="Month", showgrid=False, zeroline=False),
        yaxis=dict(title="Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        hovermode='x'
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
    fig.show()

def plot_pmf_retention_adj(cohort_df, retention_pmf_df):
    fig = go.Figure()
    # Customize layout according to Tufte's principles
    layout = dict(
        font=dict(size=10),
        title="Retention per Cohort",
        xaxis=dict(title="Month", showgrid=False, zeroline=False),
        yaxis=dict(title="Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        hovermode='x'
    )

    # Add a trace for each cohort
    for cohort in cohort_df['cohort'].unique():
        retention_df_cohort = retention_pmf_df[retention_pmf_df.cohort==cohort].reset_index(drop=True)
        fig.add_trace(
            go.Scatter(
                x=retention_df_cohort.month,
                y=retention_df_cohort.percentage_adjusted,
                mode='lines+markers',
                name=str(cohort),
                line=dict(width=1),
                marker=dict(size=4)
            )
        )

    fig.update_layout(layout)
    fig.show()

def plot_average_retention(retention_pmf_df):
    # Calculate the average retention per week across cohorts
    average_retention = retention_pmf_df.groupby('week')['percentage'].mean().reset_index()

    # Create the figure
    fig = go.Figure()

    # Customize layout according to Tufte's principles
    layout = dict(
        font=dict(size=10),
        title="Average Retention Per Week Across Cohorts",
        xaxis=dict(title="Week", showgrid=False, zeroline=False),
        yaxis=dict(title="Average Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        hovermode='x'
    )

    # Format the percentage labels
    labels = ['{:.1f}%'.format(pct) for pct in average_retention['percentage']]

    # Add the average retention trace with text labels
    fig.add_trace(
        go.Scatter(
            x=average_retention['week'],
            y=average_retention['percentage'],
            mode='lines+markers+text',
            name="Average Retention",
            line=dict(width=1),
            marker=dict(size=4),
            text=labels,  # Add formatted percentage labels
            textposition="top center"  # Position the labels above the markers
        )
    )

    fig.update_layout(layout)
    fig.show()

def plot_pmf_engagement(cohort_df, matrix_actions, freq):
    dates = pd.date_range(cohort_df['cohort'].min(), cohort_df['cohort'].max(), freq=freq)
    metrics = ['new', 'retained', 'expanded', 'resurrected', 'contracted', 'churned']
    colors = ['#a2d2ba', '#327556', '#b0d6e2', '#2f5293', '#f5baa6', '#ab3c33']
    bar_data = []
    for i, metric in enumerate(metrics):
        bar_data.append(go.Bar(
                            name=metric, 
                            x=dates, 
                            y=matrix_actions[:,i],
                            marker=dict(color=colors[i % len(colors)]),  # Cycle through colors
                            hoverinfo='all',  # Show all hover info
                        ))

    layout = go.Layout(
            title='PMF-Engagement',
            xaxis=dict(
                title="Time",
                showgrid=False,  # Reduce grid lines
                linecolor='black',  # Clean axes lines
                title_font=dict(size=14)  # Clear font for axis titles
            ),
            yaxis=dict(
                title="Number of actions",
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
            margin=dict(l=40, r=40, t=40, b=40),  # Tighten margins to use space
            showlegend=True  # This line will remove the legend
        )
    fig = go.Figure(data=bar_data, layout=layout)
    fig.show()

def plot_incremental_revenue(matrix, freq, start_month = '2023-09'):
    dates = [str(y) for y in pd.period_range(start=start_month, end=pd.Timestamp.now(), freq=freq)]
    metrics = ['churn', 'downsell', 'new', 'upsell']
    colors = ['#ab3c33','#f5baa6', '#b0d6e2', '#2f5293']
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
        x=str(date),  # Convert date to string
        y=total + 500,  # Adjust the y position for better visibility
        text=str(round(total, 2)),  # Format the text to two decimal places
        showarrow=False,
        font=dict(
            family="Arial",
            size=12,
            color="black"
        )
    ) for date, total in zip(dates, net_totals)]

    layout = go.Layout(
            title='Incremental Revenue',
            xaxis=dict(
                title="Time",
                showgrid=False,  # Reduce grid lines
                linecolor='black',  # Clean axes lines
                tickvals=dates,  # Explicitly set the tick values
                ticktext=dates,
                title_font=dict(size=14)  # Clear font for axis titles
            ),
            yaxis=dict(
                title="Revenue",
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
            margin=dict(l=40, r=40, t=40, b=40),  # Tighten margins to use space
            showlegend=True,  # This line will remove the legend
            # annotations=annotations  # Add annotations for net totals
        )
    fig = go.Figure(data=bar_data, layout=layout)
    fig.show()

def plot_pmf_retention_week(cohort_df, retention_pmf_df):
    fig = go.Figure()
    # Customize layout according to Tufte's principles
    layout = dict(
        font=dict(size=10),
        title="Retention per Cohort",
        xaxis=dict(title="Week", showgrid=False, zeroline=False),
        yaxis=dict(title="Percentage", showgrid=True, gridcolor='lightgray', zeroline=False, range=[0, 100]),
        plot_bgcolor='white',
        hovermode='x'
    )

    # Add a trace for each cohort
    for cohort in cohort_df['cohort'].unique():
        retention_df_cohort = retention_pmf_df[retention_pmf_df.cohort==cohort].reset_index(drop=True)
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

