"""
Oil Price Predictor
Study.com CS 311 Project #1

This program downloads historical Brent crude oil futures data and trains a
Random Forest Regression model to predict the next trading day's closing price.

Commodity ticker used: BZ=F
Data source: Yahoo Finance through the yfinance Python package

Notes:
This is not meant to be a real trading tool. The point of the project is to show
the full machine learning workflow: get data, clean it, create features, train a
model, check the results, and make a simple prediction.
"""

# pandas is the main data handling library here.
# It gives us a DataFrame, which is basically a spreadsheet-style table in Python.
# I use it to hold the oil price history, add new feature columns, remove missing
# rows, and save the cleaned data back out to a CSV file.
import pandas as pd

# numpy is used for some math operations.
# In this project I mainly use it for the square root calculation when converting
# mean squared error into root mean squared error.
import numpy as np

# yfinance pulls historical market data from Yahoo Finance.
# For this assignment, it saves me from manually downloading a CSV file, although
# the rest of the program still treats the downloaded data like a normal dataset.
import yfinance as yf

# matplotlib creates the chart at the end of the program.
# The chart compares the model's predictions against the actual Brent crude oil
# prices, so there is a visual result for the report.
import matplotlib.pyplot as plt

# RandomForestRegressor is the machine learning model used in this project.
# It builds a group of decision trees and averages their predictions. That makes
# it useful for this kind of tabular data without needing a neural network.
from sklearn.ensemble import RandomForestRegressor

# These metrics give a quick way to judge the model.
# MAE tells the average dollar error, RMSE penalizes bigger errors more, and R2
# gives a rough measure of how well the model explains the test data.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def download_oil_data(ticker="BZ=F", period="10y"):
    """
    Download historical Brent crude oil futures data.

    Parameters:
        ticker: Yahoo Finance ticker symbol for Brent crude oil futures.
        period: Amount of historical data to download.

    Returns:
        A pandas DataFrame containing the historical price data.
    """

    # Pull the historical price data from Yahoo Finance.
    # BZ=F is the Yahoo Finance ticker for Brent crude oil futures.
    # progress=False keeps the terminal output cleaner.
    # auto_adjust=False keeps the normal Open, High, Low, Close, and Volume fields.
    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)

    # If the download fails or returns no rows, there is no point continuing.
    # This usually means the internet connection failed or the ticker symbol is wrong.
    if data.empty:
        raise ValueError("No data was downloaded. Check the ticker symbol or internet connection.")

    # Depending on the yfinance version, the returned columns can sometimes be
    # multi-level columns instead of plain column names. That makes basic column
    # selection more annoying, so this flattens the column names if needed.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Yahoo Finance returns Date as the index by default.
    # I want Date to be a regular column because it makes plotting and saving the
    # cleaned dataset easier to understand later.
    data = data.reset_index()

    # Return the downloaded data to the next step in the pipeline.
    return data


def prepare_features(data):
    """
    Clean the dataset and create the machine learning features.

    The target value is the next trading day's closing price, so the Close column
    is shifted up by one row. That means each row uses today's data to predict
    the next trading day's closing price.

    Parameters:
        data: Raw historical price data.

    Returns:
        A cleaned DataFrame with model features and the target column.
    """

    # Make a copy so the original downloaded data is not changed directly.
    # This is not strictly required, but it helps avoid weird side effects later.
    data = data.copy()

    # Keep only the columns needed for this project.
    # Adjusted Close is not used because the assignment is easier to explain with
    # the normal open, high, low, close, and volume fields.
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Drop rows with missing values in the core market data fields.
    # Machine learning models do not handle blank cells well, so I remove those
    # rows before creating the rest of the features.
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Create a feature for the prior trading day's closing price.
    # This gives the model recent price context without using the future target.
    data["Previous_Close"] = data["Close"].shift(1)

    # Create moving average features to capture short-term price trends.
    # A 5-day average is roughly one trading week.
    data["MA_5"] = data["Close"].rolling(window=5).mean()

    # A 10-day average gives a slightly longer short-term trend.
    data["MA_10"] = data["Close"].rolling(window=10).mean()

    # A 20-day average is roughly one trading month.
    data["MA_20"] = data["Close"].rolling(window=20).mean()

    # Create the daily percentage return feature.
    # This is the day-to-day percentage move in the closing price.
    data["Daily_Return"] = data["Close"].pct_change()

    # Create a five-day rolling volatility feature based on daily returns.
    # This gives the model a simple view of whether the market has been jumpy or calm.
    data["Volatility_5"] = data["Daily_Return"].rolling(window=5).std()

    # Create the prediction target: tomorrow's closing price.
    # shift(-1) moves the next row's Close value up into the current row.
    # That is what turns this into a supervised learning problem.
    data["Target"] = data["Close"].shift(-1)

    # Drop rows that now have missing values because of rolling calculations or shifting.
    # The first few rows lose data because moving averages need enough history.
    # The last row loses data because there is no next-day close yet.
    data = data.dropna()

    # Reset the row numbers after dropping missing rows.
    # This is mostly for cleaner CSV output and easier reading.
    data = data.reset_index(drop=True)

    # Return the cleaned dataset with all model features included.
    return data


def train_and_evaluate_model(data):
    """
    Train the Random Forest model and evaluate it on test data.

    This uses a time-based split instead of a random split because market data
    has an order. Older rows are used for training, and newer rows are used for
    testing. That is closer to how forecasting works in the real world.

    Parameters:
        data: Cleaned DataFrame with features and target.

    Returns:
        The trained model, test data, predictions, metrics, and feature list.
    """

    # These are the input columns the model is allowed to use.
    # I am intentionally not using Date as a feature because the model needs price
    # behavior, not just a calendar value.
    feature_columns = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Previous_Close",
        "MA_5",
        "MA_10",
        "MA_20",
        "Daily_Return",
        "Volatility_5",
    ]

    # X contains the input data used to make predictions.
    # In machine learning terms, these are the features.
    X = data[feature_columns]

    # y contains the value the model is trying to predict.
    # Here, that value is the next trading day's closing price.
    y = data["Target"]

    # Use the first 80 percent of rows for training.
    # The remaining 20 percent will be used for testing.
    split_index = int(len(data) * 0.80)

    # Training data comes from the earlier part of the time series.
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    # Testing data comes from the later part of the time series.
    # This is more realistic than mixing old and new rows together randomly.
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Create the Random Forest Regression model.
    # n_estimators=200 means the model builds 200 decision trees.
    # random_state=42 makes the results repeatable when the same data is used.
    # min_samples_leaf=3 helps keep each tree from getting too sensitive to noise.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3,
    )

    # Train the model using the training data.
    # This is where the model learns the relationship between the features and target.
    model.fit(X_train, y_train)

    # Use the trained model to predict prices for the test data.
    # These predictions can be compared against the real test values.
    predictions = model.predict(X_test)

    # Calculate the mean absolute error.
    # This is the average size of the model's error in dollars.
    mae = mean_absolute_error(y_test, predictions)

    # Calculate the root mean squared error.
    # RMSE also measures error in dollars, but it punishes larger misses more heavily.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Calculate the R-squared score.
    # Higher is generally better, but it should not be treated as proof that the
    # model can predict future oil shocks or news events.
    r2 = r2_score(y_test, predictions)

    # Store the metrics in a dictionary so they are easy to print and reuse.
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    # Return everything needed for the results section and the chart.
    return model, X_test, y_test, predictions, metrics, feature_columns


def predict_next_day_price(model, data, feature_columns):
    """
    Predict the next trading day's Brent crude oil closing price.

    Parameters:
        model: Trained Random Forest model.
        data: Cleaned DataFrame with the latest feature values.
        feature_columns: List of features used by the model.

    Returns:
        A single predicted price value.
    """

    # Select the most recent row of feature data.
    # This represents the latest available market information in the dataset.
    latest_features = data[feature_columns].tail(1)

    # Predict the next trading day's closing price.
    # model.predict returns an array, so [0] pulls out the single number.
    predicted_price = model.predict(latest_features)[0]

    # Return the prediction as a number.
    return predicted_price


def save_results_plot(data, y_test, predictions, output_file="actual_vs_predicted_brent_crude_oil.png"):
    """
    Save a chart comparing actual and predicted prices.

    Parameters:
        data: Cleaned DataFrame with dates.
        y_test: Actual target values from the test set.
        predictions: Model predictions for the test set.
        output_file: Name of the saved chart image.
    """

    # Match the test dates to the y_test index positions.
    # This keeps the chart lined up with the actual dates from the dataset.
    test_dates = data.loc[y_test.index, "Date"]

    # Create the chart area.
    plt.figure(figsize=(12, 6))

    # Plot the actual next-day closing prices.
    plt.plot(test_dates, y_test.values, label="Actual Price")

    # Plot the model's predicted next-day closing prices.
    plt.plot(test_dates, predictions, label="Predicted Price")

    # Add a clear title.
    plt.title("Actual vs. Predicted Brent Crude Oil Closing Prices")

    # Label the horizontal axis.
    plt.xlabel("Date")

    # Label the vertical axis.
    plt.ylabel("Price in U.S. Dollars")

    # Add a legend so the two lines are easy to identify.
    plt.legend()

    # Rotate the dates so they are easier to read.
    plt.xticks(rotation=45)

    # Tight layout helps prevent labels from being cut off.
    plt.tight_layout()

    # Save the chart as an image file.
    plt.savefig(output_file)

    # Close the figure so the program does not hold it open in memory.
    plt.close()


def save_cleaned_dataset(data, output_file="cleaned_brent_crude_oil_data.csv"):
    """
    Save the cleaned dataset used by the model.

    Parameters:
        data: Cleaned DataFrame.
        output_file: Name of the CSV output file.
    """

    # Save the cleaned dataset to a CSV file without the row index.
    # This gives the assignment reviewer a way to inspect the final dataset.
    data.to_csv(output_file, index=False)


def run_full_workflow():
    """
    Run the full oil price prediction workflow.

    This keeps the machine learning process in one place so the menu can call it.
    """

    # Download the Brent crude oil futures data.
    raw_data = download_oil_data()

    # Clean the data and create the model features.
    model_data = prepare_features(raw_data)

    # Save the cleaned dataset so it can be reviewed later.
    save_cleaned_dataset(model_data)

    # Train the model and calculate performance metrics.
    model, X_test, y_test, predictions, metrics, feature_columns = train_and_evaluate_model(model_data)

    # Predict the next trading day's Brent crude oil closing price.
    next_day_prediction = predict_next_day_price(model, model_data, feature_columns)

    # Save the actual vs. predicted chart.
    save_results_plot(model_data, y_test, predictions)

    # Return the important results so the menu can reuse them.
    return {
        "model_data": model_data,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "next_day_prediction": next_day_prediction,
    }


def print_menu():
    """
    Print the user menu.

    This makes the program feel more like a small application instead of only
    a script that runs once from top to bottom.
    """

    print()
    print("Oil Price Predictor Menu")
    print("1. Run full prediction workflow")
    print("2. Show dataset information")
    print("3. Show model performance")
    print("4. Show next trading day prediction")
    print("5. Exit")
    print()


def main():
    """
    Main program with a simple interactive menu.
    """

    print("Oil Price Predictor")
    print("Brent Crude Oil Futures Next-Day Closing Price Prediction")

    # Store workflow results after the model has been run.
    # At the start, this is None because no model has been trained yet.
    results = None

    # Keep showing the menu until the user chooses to exit.
    while True:
        print_menu()

        # Ask the user to choose a menu option.
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            print()
            print("Running full prediction workflow...")

            # Run the full machine learning workflow.
            results = run_full_workflow()

            print()
            print("Workflow complete.")
            print("Files Created")
            print("cleaned_brent_crude_oil_data.csv")
            print("actual_vs_predicted_brent_crude_oil.png")

        elif choice == "2":
            print()

            # Make sure the workflow has been run before showing results.
            if results is None:
                print("Please run option 1 first so the dataset can be loaded and processed.")
            else:
                model_data = results["model_data"]
                feature_columns = results["feature_columns"]

                print("Dataset Information")
                print(f"Rows used after preprocessing: {len(model_data)}")
                print(f"Features used by model: {len(feature_columns)}")

        elif choice == "3":
            print()

            # Make sure model metrics exist before printing them.
            if results is None:
                print("Please run option 1 first so the model can be trained.")
            else:
                metrics = results["metrics"]

                print("Model Performance")
                print(f"Mean Absolute Error: ${metrics['MAE']:.2f}")
                print(f"Root Mean Squared Error: ${metrics['RMSE']:.2f}")
                print(f"R-squared Score: {metrics['R2']:.4f}")

        elif choice == "4":
            print()

            # Make sure the prediction exists before printing it.
            if results is None:
                print("Please run option 1 first so the next-day price can be predicted.")
            else:
                next_day_prediction = results["next_day_prediction"]

                print("Next-Day Prediction")
                print(f"Predicted next trading day closing price: ${next_day_prediction:.2f}")

        elif choice == "5":
            print()
            print("Exiting Oil Price Predictor.")
            break

        else:
            print()
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


# This makes sure main() runs only when this file is executed directly.
if __name__ == "__main__":
    main()
