import numpy as np
from openai import OpenAI


class DashboardAgent:
    def __init__(self):
        # defaults tclo getting the key using os.environ.get("OPENAI_API_KEY")
        # if you saved the key under a different environment variable name, you can do something like:
        # client = OpenAI(
        #   api_key=os.environ.get("CUSTOM_ENV_NAME"),
        # )
        self.client = OpenAI()

    def analyze_webform(self, daily_df):
        # Compute the sum of submissions for relevant weeks and months
        self.total_sub = daily_df['daily_count'].sum()
        self.this_week_sub = daily_df.tail(7)['daily_count'].sum()
        self.last_week_sub = daily_df.iloc[-14:-7]['daily_count'].sum()
        self.this_month_sub = daily_df.tail(30)['daily_count'].sum()
        self.last_month_sub = daily_df.iloc[-60:-30]['daily_count'].sum()

        # Percentage changes
        #TODO: add divide no-nan function
        if self.last_week_sub != 0:
            self.weekly_percent_sub = 100*(self.this_week_sub-self.last_week_sub)/self.last_week_sub
        else:
            self.weekly_percent_sub = float('inf')
        if self.last_month_sub != 0:
            self.monthly_percent_sub = 100*(self.this_month_sub-self.last_month_sub)/self.last_month_sub
        else:
            self.monthly_percent_sub = float('inf')

        self.kpi_text_submissions = f"""
        #### **Web Form Submissions**
        ---
        **Total Submissions:** {self.total_sub}

        **This Week's Submissions (last 7 days):** {self.this_week_sub} 
            - _({self.weekly_percent_sub:.2f}% change from last week)_

        **This Months Submissions (last 30 days):** {self.this_month_sub} 
            - _({self.monthly_percent_sub:.2f}% change from last month)_

        **Today's Submissions:** {daily_df['daily_count'].iloc[-1]}

        ---
        """

    def analyze_signups(self, daily_df):
        self.total_signs = daily_df['count'].sum()
        self.today_signs = daily_df.tail(1)['count'].sum()
        self.this_week_signs = daily_df.iloc[-8:-1]['count'].sum()
        self.last_week_signs = daily_df.iloc[-15:-8]['count'].sum()
        self.weekly_percent_sings = 100*(self.this_week_signs-self.last_week_signs)/self.last_week_signs

        # Display the key metrics as formatted text
        self.kpi_text_reg = f"""
        #### **New Registered Users Metrics**
        ---
        - **Total Registrations:** {self.total_signs}
        - **This Week's Registrations:** {self.this_week_signs} ({self.weekly_percent_sings:.2f}% change from last week)
        - **Today's Registrations:** {self.today_signs}

        ---

        """

    def analyze_requests(self,requests_daily_df, requests_weekly_df):
        # Calculations for API requests metrics
        self.latest_requests = requests_weekly_df['total_requests'].iloc[-2]
        previous_requests = requests_weekly_df['total_requests'].iloc[-3]
        self.requests_percent_change = ((self.latest_requests - previous_requests) / previous_requests) * 100
        requests_weekly_df['growth_rate'] = requests_weekly_df['total_requests'].pct_change() * 100
        self.average_growth_rate_requests = requests_weekly_df['growth_rate'][1:].mean()  # Skip the first NaN value

        # Display the key metrics as formatted text
        self.kpi_text_api_requests = f"""
        #### **API Requests Metrics**
        ---
        - **Latest Week's Total API Requests:** {self.latest_requests}
        - **Change from Previous Week:** {self.requests_percent_change:.2f}%
        - **Average Weekly Growth Rate:** {self.average_growth_rate_requests:.2f}%
        - **Total API Requests:** {requests_daily_df['total_requests'].sum():.2f}
        - **Mean Daily API Requests:** {requests_daily_df['total_requests'].mean():.2f}

        ---
        """

    def analyze_mau(self, monthly_active_users_df):
        # Calculations for MAU metrics
        self.latest_mau = monthly_active_users_df['active'].iloc[-2]
        previous_mau = monthly_active_users_df['active'].iloc[-3]
        self.mau_percent_change = ((self.latest_mau - previous_mau) / previous_mau) * 100
        monthly_active_users_df['growth_rate'] = monthly_active_users_df['active'].pct_change() * 100
        self.average_growth_rate_mau = monthly_active_users_df['growth_rate'][1:].mean()  # Skip the first NaN value

        # Display the key metrics as formatted text
        self.kpi_text_mau = f"""
        #### **Monthly Active Users Metrics**
        ---
        - **Latest Month's MAU:** {self.latest_mau}
        - **Change from Previous Month:** {self.mau_percent_change:.2f}%
        - **Average Monthly Growth Rate:** {self.average_growth_rate_mau:.2f}%
        """

    def analyze_wau(self, weekly_active_users_df):
        # Calculations for WAU metrics
        self.latest_wau = weekly_active_users_df['active'].iloc[-2]
        previous_wau = weekly_active_users_df['active'].iloc[-3]
        self.wau_percent_change = ((self.latest_wau - previous_wau) / previous_wau) * 100
        weekly_active_users_df['growth_rate'] = weekly_active_users_df['active'].pct_change() * 100
        self.average_growth_rate_wau = weekly_active_users_df['growth_rate'][1:].mean()  # Skip the first NaN value

        # Display the key metrics as formatted text
        self.kpi_text_wau = f"""
        #### **Weekly Active Users Metrics**
        ---
        - **Latest Week's WAU:** {self.latest_wau}
        - **Change from Previous Week:** {self.wau_percent_change:.2f}%
        - **Average Weekly Growth Rate:** {self.average_growth_rate_wau:.2f}%

        ---
        """

    def analyze_user_avg(self, df_user_avg):
        # Key indicators
        self.average_daily_calls_per_user = df_user_avg['calls'].mean()
        self.median_daily_calls_per_user = df_user_avg['calls'].median()
        self.ninety_percentile_daily_calls = df_user_avg['calls'].quantile(0.9)

        # Display the key metrics as formatted text
        self.kpi_text_user_avg_api = f"""
        #### **Average Daily API Calls Per User Metrics**
        ---
        - **Average Daily Calls/User:** {self.average_daily_calls_per_user:.2f}
        - **Median Daily Calls/User:** {self.median_daily_calls_per_user:.2f}
        - **90th Percentile Daily Calls/User:** {self.ninety_percentile_daily_calls:.2f}

        ---
        """

    def analyze_web_conversion(self, funnel_stages_web, funnel_values_web):
        # Calculate conversion rates
        conversion_rates_web = [funnel_values_web[i+1] / funnel_values_web[i] * 100 for i in range(len(funnel_values_web) - 1)]

        self.kpi_text_conversion_web = """
        **Cumulative Conversion Web Form Rates**
        ---
        """

        for i, stage in enumerate(funnel_stages_web[:-1]):
            # if stage in ['Web Form', 'SignedUp']:
            #     pass
            # else:
            self.kpi_text_conversion_web += f"**From {stage} to {funnel_stages_web[i+1]}:** {conversion_rates_web[i]:.2f} ({funnel_values_web[i]}->{funnel_values_web[i+1]})%\n"
            self.kpi_text_conversion_web += """
        """

    def analyze_email_conversion(self, funnel_stages_email, funnel_values_email):
        # Calculate conversion rates ensuring to skip division by zero
        conversion_rates_email = [(funnel_values_email[i+1] / funnel_values_email[i] * 100) if funnel_values_email[i] > 0 else 0 for i in range(len(funnel_values_email) - 1)]

        self.kpi_text_conversion_email = f"""
        **Cumulative Conversion Rates Email Campaign**
        Total emails sent: {funnel_values_email[0]:.2f}
        ---
        """

        for i, stage in enumerate(funnel_stages_email[:-1]):
            self.kpi_text_conversion_email += f"**From {stage} to {funnel_stages_email[i+1]}:** {conversion_rates_email[i]:.2f}%\n"
            self.kpi_text_conversion_email += """
        """
    
    def analyze_costs(self, total_costs_monthly):
        # Ensure that the data is sorted by timestamp if not already:
        total_costs_monthly = total_costs_monthly.sort_values(by='timestamp')

        # Get the latest month costs data (assuming the data is ordered with the most recent month last)
        latest_month_costs = total_costs_monthly.iloc[-2]  # The second last row with the latest data
        previous_month_costs = total_costs_monthly.iloc[-3]  # The second to last row for previous month data

        # Calculate the percent change for each cost type
        latest_month_costs['cost_input_change'] = ((latest_month_costs['cost_input'] - previous_month_costs['cost_input']) / previous_month_costs['cost_input']) * 100 if previous_month_costs['cost_input'] > 0 else np.nan
        latest_month_costs['cost_output_change'] = ((latest_month_costs['cost_output'] - previous_month_costs['cost_output']) / previous_month_costs['cost_output']) * 100 if previous_month_costs['cost_output'] > 0 else np.nan
        latest_month_costs['cost_finetune_change'] = ((latest_month_costs['cost_finetune'] - previous_month_costs['cost_finetune']) / previous_month_costs['cost_finetune']) * 100 if previous_month_costs['cost_finetune'] > 0 else np.nan
        latest_month_costs['total_cost_change'] = ((latest_month_costs['total_cost'] - previous_month_costs['total_cost']) / previous_month_costs['total_cost']) * 100 if previous_month_costs['total_cost'] > 0 else np.nan

        # Now, let's prepare the KPI text:
        self.kpi_text_cost = f"""
        #### **Monthly Theoretical Income in USD (Latest Month)**
        ---
        - **Income Input Tokens:** {latest_month_costs['cost_input']:.2f} (Change: {latest_month_costs['cost_input_change']:.2f}%)
        - **Income Output Tokens:** {latest_month_costs['cost_output']:.2f} (Change: {latest_month_costs['cost_output_change']:.2f}%)
        - **Income Finetuning:** {latest_month_costs['cost_finetune']:.2f} (Change: {latest_month_costs['cost_finetune_change']:.2f}%)
        - **Total income:** {latest_month_costs['total_cost']:.2f} (Change: {latest_month_costs['total_cost_change']:.2f}%)

        ---
        """
    
    def build_summary(self):
        # Combining and formatting all the KPI textual displays
        self.summary = f"""
        ---
        {self.kpi_text_submissions}
        ---
        {self.kpi_text_reg}
        ---
        {self.kpi_text_mau}
        ---
        {self.kpi_text_wau}
        ---
        {self.kpi_text_api_requests}
        ---
        {self.kpi_text_user_avg_api}
        ---
        {self.kpi_text_cost}
        ---
        {self.kpi_text_conversion_web}
        ---
        {self.kpi_text_conversion_email}
        """

    def generate_report(self):

        few_shot_input ='''
        "\n---\n\n#### **Web Form Submissions**\n---\n**Total Submissions:** 2068\n\n**This Week's Submissions (last 7 days):** 230 \n    - _(-33.33% change from last week)_\n\n**This Months Submissions (last 30 days):** 892 \n    - _(166.27% change from last month)_\n\n**Today's Submissions:** 5\n\n---\n\n---\n\n#### **New Registered Users Metrics**\n---\n- **Total Registrations:** 2023\n- **This Week's Registrations:** 451 (-1.96% change from last week)\n- **Today's Registrations:** 5\n\n---\n\n\n---\n\n#### **Monthly Active Users Metrics**\n---\n- **Latest Month's MAU:** 74\n- **Change from Previous Month:** 94.74%\n- **Average Monthly Growth Rate:** 490.77%\n\n---\n\n#### **Weekly Active Users Metrics**\n---\n- **Latest Week's WAU:** 29\n- **Change from Previous Week:** -21.62%\n- **Average Weekly Growth Rate:** 37.40%\n\n---\n\n---\n\n#### **API Requests Metrics**\n---\n- **Latest Week's Total API Requests:** 8574\n- **Change from Previous Week:** 70.69%\n- **Average Weekly Growth Rate:** 157.12%\n- **Total API Requests:** 30627.00\n- **Mean Daily API Requests:** 273.46\n\n---\n\n---\n\n#### **Average Daily API Calls Per User Metrics**\n---\n- **Average Daily Calls/User:** 13.54\n- **Median Daily Calls/User:** 3.34\n- **90th Percentile Daily Calls/User:** 24.08\n\n---\n\n---\n\n#### **Monthly Theoretical Income (Latest Month)**\n---\n- **Income Input Tokens:** 13172.78 (Change: 575.83%)\n- **Income Output Tokens:** 3302.56 (Change: 268.43%)\n- **Income Finetuning:** 446.63 (Change: -84.13%)\n- **Total Input:** 16921.97 (Change: 198.94%)\n\n---\n\n---\n\n**Cumulative Conversion Web Form Rates**\n---\n**From Web Form to SignedUp:** 97.82%\n\n**From SignedUp to Used API:** 5.88%\n\n**From Used API to Active:** 57.98%\n\n**From Active to Closed:** 1.45%\n\n\n---\n\n**Cumulative Conversion Rates Email Campaign**\n---\n**From Email Sent to SignedUp:** 73.15%\n\n**From SignedUp to Used API:** 44.30%\n\n**From Used API to Active:** 91.43%\n\n\n"
        '''

        few_shot_output ='''
        <div style="width:100%; font-family:arial; margin: 0 auto; color:#333;">\n\n<div style="text-align:center; padding:20px; background-color: #f8f8f8;">\n    <h2 style="margin:0;">Dashboard Overview</h2>\n</div>\n\n<div style="display:flex; justify-content: space-between; padding:20px; background-color: #fff;">\n\n    <div style="width:25%; border:1px solid #ccc; background-color: #f8f8f8; padding:20px; text-align:center;">\n        <h3>Web Form Submissions</h3>\n        <hr>\n        <p><strong>Total:</strong> 2,068</p>\n        <p><strong>This Week:</strong> 230 (-33.33%)</p>\n        <p><strong>This Month:</strong> 892 (+166.27%)</p>\n        <p><strong>Today:</strong> 5</p>\n    </div>\n\n    <div style="width:25%; border:1px solid #ccc; background-color: #f8f8f8; padding:20px; text-align:center;">\n        <h3>User Metrics</h3>\n        <hr>\n        <p><strong>Total Registrations:</strong> 2,023</p>\n        <p><strong>This Week\'s Registrations:</strong> 451 (-1.96%)</p>\n        <p><strong>Today\'s Registrations:</strong> 5</p>\n        <p><strong>Monthly Active Users:</strong> 74 (+94.74%)</p>\n        <p><strong>Weekly Active Users:</strong> 29 (-21.62%)</p>\n    </div>\n\n    <div style="width:25%; border:1px solid #ccc; background-color: #f8f8f8; padding:20px; text-align:center;">\n        <h3>API Requests</h3>\n        <hr>\n        <p><strong>Total API Requests:</strong> 30,627</p>\n        <p><strong>Last Week API Requests:</strong> 8,574 (+70.69%)</p>\n        <p><strong>Daily API Requests:</strong> 273</p>\n        <p><strong>Daily Calls/User:</strong> 13.54</p>\n    </div>\n\n    <div style="width:25%; border:1px solid #ccc; background-color: #f8f8f8; padding:20px; text-align:center;">\n        <h3>Theoretical Income (Latest Month)</h3>\n        <hr>\n        <p><strong>Input Tokens:</strong> 13,172.78 (+575.83%)</p>\n        <p><strong>Outcome Tokens:</strong> 3,302.56 (+268.43%)</p>\n        <p><strong>Finetuning:</strong> 446.63 (-84.13%)</p>\n        <p><strong>Total Input:</strong> 16,921.97 (+198.94)</p>\n    </div>\n\n</div>\n\n<div style="display:flex; justify-content: space-between; padding:20px; background-color: #fff;">\n\n    <div style="width:50%; border:1px solid #ccc; background-color: #f8f8f8; padding:20px; text-align:center;">\n        <h3>Cumulative Conversion Web Form Rates</h3>\n        <hr>\n        <p><strong>Web Form to SignedUp:</strong> 97.82%</p>\n        <p><strong>SignedUp to Used API:</strong> 5.88%</p>\n        <p><strong>Used API to Active:</strong> 57.98%</p>\n        <p><strong>Active to Closed:</strong> 1.45%</p>\n    </div>\n\n    <div style="width:50%; border:1px solid #ccc; background-color: #f8f8f8; padding:20px; text-align:center;">\n        <h3>Cumulative Conversion Rates Email Campaign</h3>\n        <hr>\n        <p><strong>Email Sent to SignedUp:</strong> 73.15%</p>\n        <p><strong>SignedUp to Used API:</strong> 44.30%</p>\n        <p><strong>Used API to Active:</strong> 91.43%</p>\n    </div>\n    \n</div>\n\n</div>
        '''

        system = """You receive a report containing key perfomance metrics and you you produce insightfull, clear and well formated HTML Dashboards. You use the necessary CSS to make it look great. The output should work in Databricks with displayHTML(). You only output the HTML, no further comments or code blocks. No need to begin with ```hmtl. You always make sure that your outpout works. You are concise and number oriented.  You change formating, titles and arrange info as you see fit to make if easier to follow and read. But you don't make any numbers up. You use the whole widht of the page, if necessary. You can use tables effectively. You also format numbers in such a way that they are easy to read: comma notation, positive growth in green, negative in red, etc.  Your style is elegant but stylish."""

        user = f"""Here is my report. Please create an insightfull and clear Dashboard. Remeber to use green for positive growth and red for negative growth.:  
        -----
        {self.summary}
        -----
        """

        #gpt-4-1106-preview	
        kpi_report = self.client.chat.completions.create(
                    model="gpt-4-32k",
                    messages=[
                        {'role': 'system', 'content': f'{system}'},
                        {'role':  'user', 'content': f'{few_shot_input}'},
                        {'role': 'assistant', 'content': f'{few_shot_output}'},
                        {'role': 'user', 'content': f'{user}'}
                        ],
                    max_tokens=5_000,
                )

        system = """You act as a seasoned entrepreneur with expertise in steering technology companies that offer APIs as services to data scientists. Your role involves receiving a weekly HTML report outlining key performance indicators (KPIs) and your task is to provide a no-nonsense, succinct analysis of these KPIs, cutting through the fluff to evaluate performance accurately. You keep your answers below 250 words. You never make stuff up and only use the numbers presentet in the report. Where possible, you should draw on industry benchmarks for comparison, giving context to your analysis and guiding strategic decision-making. You output in well formated and great looking HTML. You make numbers stand out by hihglighting them. And if necessary you underline important statements. For the moment you ignore and don't say anything about Cumulative Conversion Web Form Rates"""

        user = f"""Here is my weekly report:   
        -----
        {kpi_report.choices[0].message.content}
        -----
        """

        weekly_analysis = self.client.chat.completions.create(
                    model="gpt-4-32k",
                    messages=[
                        {'role': 'system', 'content': f'{system}'},
                        {'role': 'user', 'content': f'{user}'}
                        ],
                    max_tokens=5_000,
                )
        return kpi_report, weekly_analysis

    def agent_webform_analysis(self, df_last_7_days):
        csv_string = df_last_7_days[['company', 'message', 'org_industries']].to_csv(index=False)
        prompt_summary_messages = 'I will provide a table of my show interest web form. Please tell me the most important companies in the list. Also, Im particuallry interested in tech and famous companies like Uber, Brex, Stripe, etc. Also, summarize the different use cases contained in the column messages. Addtionaly, analyze and tell if any interesting things you can think of. Ommit comments about NAs. Please start with the title: Analysis of Last Weeks submissions. Please ouput well formated markdown. Here is the data:  ' +  csv_string

        webform_analysis = self.client.chat.completions.create(
                    model="gpt-4-32k",
                    messages=[
                        {'role': 'user', 'content': f'{prompt_summary_messages}'},
                        ],
                    max_tokens=5_000,
                )
        return webform_analysis
