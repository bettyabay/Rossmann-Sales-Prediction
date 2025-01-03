""" 1. Setting Up Logging
Before diving into each task, initialize logging for the project: """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import holidays
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Adjust level as needed 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()  # Optional: Print logs to console as well
    ]
)

""" 2. Logging for Each Task
Task 1.1: Check for Promotion Distribution in Training and Test Sets +"""

def plot_promo_distribution(train_data, test_data):
    try:
        promo_counts_train = train_data['Promo'].value_counts()
        promo_counts_test = test_data['Promo'].value_counts()

        plt.bar(['Train - No Promo', 'Train - Promo'], promo_counts_train, label='Train', color='blue', alpha=0.7)
        plt.bar(['Test - No Promo', 'Test - Promo'], promo_counts_test, label='Test', color='orange', alpha=0.7)
        plt.title("Promotion Distribution in Training and Test Sets")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

        logging.info("Promotion distribution plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting promotion distribution: {e}")

# Task 1.2: Compare sales behavior before, during, and after holidays  +
def analyze_sales_during_holidays(data):
    logging.info("Analyzing sales behavior before, during, and after holidays...")
    
    # Analyze the number of times stores were closed for different reasons
    closed_total = len(data[data['Open'] == 0])
    closed_school_holiday = len(data[(data['Open'] == 0) & (data['SchoolHoliday'] == 1)])
    closed_state_holiday = len(data[(data['Open'] == 0) & 
                                    ((data['StateHoliday'] == 'a') | 
                                     (data['StateHoliday'] == 'b') | 
                                     (data['StateHoliday'] == 'c'))])
    closed_no_reason = len(data[(data['Open'] == 0) & 
                                (data['StateHoliday'] == "0") & 
                                (data['SchoolHoliday'] == 0)])

    # Log the findings
    logging.info(f"In the data, stores were closed {closed_total} times on given days.")
    logging.info(f"Out of those, {closed_school_holiday} times it was closed due to school holidays.")
    logging.info(f"Stores were closed {closed_state_holiday} times due to state holidays (bank, Easter, or Christmas).")
    logging.info(f"However, {closed_no_reason} times stores were closed with no apparent reason (no holidays announced).")

    # Analyze sales before, during, and after holidays
    data = data.sort_values(by='Date')

    # Filter data for holidays
    holiday_sales = data[data['StateHoliday'].isin(['a', 'b', 'c'])]
    before_holiday_sales = data[data['Date'].isin(holiday_sales['Date'] - pd.Timedelta(days=1))]
    after_holiday_sales = data[data['Date'].isin(holiday_sales['Date'] + pd.Timedelta(days=1))]

    # Aggregate sales
    holiday_sales_mean = holiday_sales['Sales'].mean()
    before_holiday_sales_mean = before_holiday_sales['Sales'].mean()
    after_holiday_sales_mean = after_holiday_sales['Sales'].mean()

    # Log sales findings
    logging.info(f"Average sales during holidays: {holiday_sales_mean}")
    logging.info(f"Average sales before holidays: {before_holiday_sales_mean}")
    logging.info(f"Average sales after holidays: {after_holiday_sales_mean}")

    # Plot the comparison
    plot_holiday_sales_comparison(before_holiday_sales_mean, holiday_sales_mean, after_holiday_sales_mean)

def plot_holiday_sales_comparison(before, during, after):
    # Bar chart for comparison
    categories = ['Before Holiday', 'During Holiday', 'After Holiday']
    sales = [before, during, after]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, sales, color=['orange', 'red', 'green'])
    plt.title('Sales Behavior Before, During, and After Holidays')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig("holiday_sales_comparison.png")
    logging.info("Holiday sales comparison plot saved as 'holiday_sales_comparison.png'.")
    plt.show()


#Task 1.3: Find seasonal purchase behaviors +
def plot_weekly_sales(df):
    logging.info("Plotting weekly sales...")
        # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the Date column as the index
    df.set_index('Date', inplace=True)

    weekly_sales = df['Sales'].resample('W').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(weekly_sales.index, weekly_sales)
    plt.title('Weekly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

def plot_monthly_sales(df):
    logging.info("Plotting monthly sales...")
        # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the Date column as the index
    df.set_index('Date', inplace=True)
    
    monthly_sales = df['Sales'].resample('M').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_sales.index, monthly_sales)
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

    #Task 1.4: Correlation between sales and number of customers +
def plot_sales_vs_customers(df):
    logging.info("Plotting sales vs customers scatter plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

# Task 1.5: Promo effect on sales and customers
def plot_promo_effect(df):
    logging.info("Plotting promo effect over time...")
    monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
    monthly_promo_sales.columns = ['No Promo', 'Promo']

    monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()

# Task 1.6: Determine effective promo deployment
def effective_promo_deployment(df, store_col, promo_col, sales_col):
    logging.info("Analyzing effective promo deployment strategies.")
    promo_sales = data.groupby(store_col)[[promo_col, sales_col]].mean()
    logging.debug("Average promo and sales per store:\n%s", promo_sales)


# Task 1.7: Trends in customer behavior during store opening/closing +
def plot_opening_closing_trends(data):
    try:
       sns.lineplot(data=data[data['Open'] == 1], x='Date', y='Customers', label='Open')
       sns.lineplot(data=data[data['Open'] == 0], x='Date', y='Customers', label='Closed')
       plt.legend()

       logging.info("Trends during store opening and closing times plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting trends during store opening and closing times: {e}")


# Task 1.8: Impact of assortment type
def plot_weekday_weekend_sales(data):
    try:
        sns.boxplot(data=data, x='Weekend', y='Sales') 
        plt.title("Sales of Stores Open on Weekdays vs Weekends")
        plt.ylabel("Sales")
        plt.show()

        logging.info("Weekday and weekend sales plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting weekday and weekend sales: {e}")


# Task 1.9: Competitor distance effect
def plot_assortment_effect(data):
    try:
        sns.boxplot(data=data, x='Assortment', y='Sales')
        plt.title("Effect of Assortment Type on Sales")
        plt.xlabel("Assortment Type")
        plt.ylabel("Sales")
        plt.show()

        logging.info("Assortment type effect on sales plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting assortment type effect on sales: {e}")

# Task 1.10. Competitor Distance Effect on Sales
def plot_competitor_distance_effect(data):
    try:
        sns.scatterplot(data=data, x='CompetitionDistance', y='Sales')
        plt.title("Effect of Competitor Distance on Sales")
        plt.xlabel("Distance to Competitor")
        plt.ylabel("Sales")
        plt.show()

        logging.info("Competitor distance effect on sales plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting competitor distance effect on sales: {e}")


# Task 1.11 Effect of Store and Competitor Proximity in City Centers
def analyze_city_center_proximity(df):
    try:
        # Assuming city center stores are identified by a boolean column 'CityCenter'
        city_center_stores = df[df['CityCenter'] == 1]
        logging.debug(f"City center stores data: {city_center_stores.head()}")

        # Group by competitor proximity
        city_center_sales = city_center_stores.groupby('CompetitionDistance')['Sales'].mean().reset_index()
        logging.debug(f"Sales in city center stores with nearby competitors: {city_center_sales}")

        # Plot
        sns.lineplot(df=city_center_sales, x='CompetitionDistance', y='Sales')
        plt.title("Effect of Store and Competitor Proximity in City Centers")
        plt.xlabel("Distance to Competitor (meters)")
        plt.ylabel("Average Sales")
        plt.show()

        logging.info("City center proximity analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error analyzing city center proximity: {e}")


# Task 1.12 Effect of Opening/Reopening of Competitors
def analyze_competitor_reopening(df):
    try:
        # Identify stores with competitor reopening data
        reopening_stores = df[df['CompetitionOpenSinceYear'].notnull()]
        reopening_stores['ReopeningEffect'] = reopening_stores['Sales'].pct_change()  # Example effect calculation
        logging.debug(f"Reopening effect data: {reopening_stores.head()}")

        # Plot
        sns.boxplot(data=reopening_stores, x='ReopeningEffect', y='Sales')
        plt.title("Effect of Opening/Reopening of Competitors on Sales")
        plt.xlabel("Reopening Effect (Percentage Change)")
        plt.ylabel("Sales")
        plt.show()

        logging.info("Competitor reopening effect analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error analyzing competitor reopening effects: {e}")


def plot_acf_pacf(df):
    logging.info("Plotting ACF and PACF...")
    monthly_sales = df['Sales'].resample('M').sum()
    n_lags = len(monthly_sales) // 3
    acf_values = acf(monthly_sales.dropna(), nlags=n_lags)
    pacf_values = pacf(monthly_sales.dropna(), nlags=n_lags)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    ax1.stem(range(len(acf_values)), acf_values, use_line_collection=True)
    ax1.axhline(y=0, linestyle='--', color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax1.set_title('Autocorrelation Function')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')

    ax2.stem(range(len(pacf_values)), pacf_values, use_line_collection=True)
    ax2.axhline(y=0, linestyle='--', color='gray')
    ax2.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax2.set_title('Partial Autocorrelation Function')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Correlation')

    plt.tight_layout()
    plt.show()


def plot_sales_heatmap(df):
    logging.info("Plotting sales heatmap by day of week and month...")
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    sales_heatmap = df.pivot_table(values='Sales', index='DayOfWeek', columns='Month', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(sales_heatmap, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Average Sales by Day of Week and Month')
    plt.xlabel('Month')
    plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
    plt.show()