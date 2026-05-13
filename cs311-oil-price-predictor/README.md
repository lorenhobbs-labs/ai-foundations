# CS311 Oil Price Predictor

This folder contains my Study.com CS 311 Project #1 machine learning application for predicting Brent crude oil prices.

## Project Summary

Oil Price Predictor is a Python machine learning application that predicts the next trading day closing price of Brent crude oil futures. The project uses historical Brent crude oil futures data, creates price-based features, trains a Random Forest Regression model, evaluates the model, and saves a chart comparing actual and predicted prices.

## Commodity

* **Commodity:** Brent crude oil futures
* **Yahoo Finance ticker:** BZ=F
* **Prediction target:** next trading day closing price

## Files

* `oil_price_predictor.py`: Python source code
* `Oil_Price_Predictor_Report_Brent.docx`: Assignment report draft
* `requirements.txt`: Required Python libraries
* `README.md`: Project instructions

After the program runs, it also creates:

* `cleaned_brent_crude_oil_data.csv`
* `actual_vs_predicted_brent_crude_oil.png`

## How to Run

1. Install Python 3.
2. Open a terminal in this project folder.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the program:

```bash
python oil_price_predictor.py
```

5. Review the terminal output and the generated chart file.

## Notes

This is a course project, not a trading tool. The model uses historical price data to make a short-term prediction, but real oil prices can change because of supply, demand, geopolitical events, inventory reports, and other market conditions that are not fully captured by historical prices alone.
