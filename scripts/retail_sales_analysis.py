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
Task 1.1: Check for Promotion Distribution in Training and Test Sets """

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

"""Task 1.2: Compare sales behavior before, during, and after holidays
python"""
def plot_sales_behavior(holiday_sales):
    try:
        sns.lineplot(df=holiday_sales, x='Date', y='Sales', hue='Period')  # Period: Before, During, After
        plt.title("Sales Behavior Before, During, and After Holidays")
        plt.ylabel("Sales")
        plt.show()

        logging.info("Sales behavior before, during, and after holidays plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting sales behavior: {e}")


"""Task 1.3: Find seasonal purchase behaviors"""
def plot_weekly_sales(df):
    logging.info("Plotting weekly sales...")
    weekly_sales = df['Sales'].resample('W').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(weekly_sales.index, weekly_sales)
    plt.title('Weekly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

def plot_monthly_sales(df):
    logging.info("Plotting monthly sales...")
    monthly_sales = df['Sales'].resample('M').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_sales.index, monthly_sales)
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

    """Task 1.4: Correlation between sales and number of customers"""
def plot_sales_vs_customers(df):
    logging.info("Plotting sales vs customers scatter plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

"""Task 1.5: Promo effect on sales and customers"""
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

"""Task 1.6: Determine effective promo deployment"""
def effective_promo_deployment(df, store_col, promo_col, sales_col):
    logging.info("Analyzing effective promo deployment strategies.")
    promo_sales = data.groupby(store_col)[[promo_col, sales_col]].mean()
    logging.debug("Average promo and sales per store:\n%s", promo_sales)


"""Task 1.7: Trends in customer behavior during store opening/closing"""
def plot_opening_closing_trends(data):
    try:
        sns.lineplot(data=data, x='Time', y='Customers')  # Replace 'Time' with actual time column
        plt.title("Customer Trends During Store Opening and Closing Times")
        plt.xlabel("Time")
        plt.ylabel("Number of Customers")
        plt.show()

        logging.info("Trends during store opening and closing times plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting trends during store opening and closing times: {e}")


"""Task 1.8: Impact of assortment type"""
def plot_weekday_weekend_sales(data):
    try:
        sns.boxplot(data=data, x='Weekend', y='Sales')  # Replace 'Weekend' with a column indicating weekends
        plt.title("Sales of Stores Open on Weekdays vs Weekends")
        plt.ylabel("Sales")
        plt.show()

        logging.info("Weekday and weekend sales plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting weekday and weekend sales: {e}")


"""Task 1.9: Competitor distance effect"""
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

"""Task 1.10. Competitor Distance Effect on Sales"""
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