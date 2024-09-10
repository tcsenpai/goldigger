import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, clone_model as keras_clone_model
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import Callback
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
import os
import tensorflow as tf
from tabulate import tabulate
from scipy.stats import randint, uniform
import sklearn.base
import argparse

# Suppress warnings and TensorFlow logging
def suppress_warnings_method():
    # Filter out warnings
    warnings.filterwarnings('ignore')
    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Suppress TensorFlow verbose logging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Custom progress bar for Keras model training
class TqdmProgressCallback(Callback):
    def __init__(self, epochs, description):
        super().__init__()
        # Initialize progress bar
        self.progress_bar = tqdm(total=epochs, desc=description, leave=False)

    def on_epoch_end(self, epoch, logs=None):
        # Update progress bar at the end of each epoch
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=f"{logs['loss']:.4f}", val_loss=f"{logs['val_loss']:.4f}")

    def on_train_end(self, logs=None):
        # Close progress bar at the end of training
        self.progress_bar.close()

# Fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Add technical indicators to the stock data
def add_technical_indicators(data):
    """
    Add technical indicators to the dataset.
    """
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.volatility.bollinger_hband_indicator(data['Close']), ta.volatility.bollinger_mavg(data['Close']), ta.volatility.bollinger_lband_indicator(data['Close'])
    return data

# Prepare data for model training by scaling and creating sequences
def prepare_data(data, look_back=60):
    """
    Prepare data for model training.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predicting the 'Close' price
    
    return np.array(X), np.array(y), scaler

# Create an LSTM model for time series prediction
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create a GRU model for time series prediction
def create_gru_model(input_shape):
    model = Sequential([
        GRU(units=50, return_sequences=True, input_shape=input_shape),
        GRU(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate a model using time series cross-validation
def train_and_evaluate_model(model, X, y, n_splits=5, model_name="Model"):
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    oof_predictions = np.zeros_like(y)
    
    # Iterate through each fold
    with tqdm(total=n_splits, desc=f"Training {model_name}", leave=False) as pbar:
        for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
            # Split data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Handle different model types (sklearn models vs Keras models)
            if isinstance(model, (RandomForestRegressor, XGBRegressor)):
                # For sklearn models
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_val_2d = X_val.reshape(X_val.shape[0], -1)
                cloned_model = sklearn.base.clone(model)
                cloned_model.fit(X_train_2d, y_train)
                val_pred = cloned_model.predict(X_val_2d)
                oof_predictions[val_index] = val_pred
            elif isinstance(model, Sequential):
                # For Keras models (LSTM and GRU)
                cloned_model = keras_clone_model(model)
                cloned_model.compile(optimizer='adam', loss='mean_squared_error')
                cloned_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0, 
                                callbacks=[TqdmProgressCallback(100, f"{model_name} Epoch {fold}/{n_splits}")])
                val_pred = cloned_model.predict(X_val)
                oof_predictions[val_index] = val_pred.flatten()
            else:
                # Raise error for unsupported model types
                raise ValueError(f"Unsupported model type: {type(model)}")
            
            # Calculate and store the score for this fold
            score = r2_score(y_val, val_pred)
            scores.append(score)
            pbar.update(1)
    
    # Calculate overall score and return results
    overall_score = r2_score(y, oof_predictions)
    return np.mean(scores), np.std(scores), overall_score, oof_predictions

# Make predictions using an ensemble of models
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        if isinstance(model, (RandomForestRegressor, XGBRegressor)):
            pred = model.predict(X.reshape(X.shape[0], -1))
        else:
            pred = model.predict(X)
        predictions.append(pred.flatten())  # Flatten the predictions
    return np.mean(predictions, axis=0)

# Calculate risk metrics (Sharpe ratio and max drawdown)
def calculate_risk_metrics(returns):
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming daily returns
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    return sharpe_ratio, max_drawdown

# Predict future stock prices using a trained model
def predict_future(model, last_sequence, scaler, days):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        if isinstance(model, (RandomForestRegressor, XGBRegressor)):
            prediction = model.predict(current_sequence.reshape(1, -1))
            future_predictions.append(prediction[0])  # prediction is already a scalar
        else:
            prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
            future_predictions.append(prediction[0][0])  # Take only the first (and only) element
        
        # Update the sequence for the next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = prediction  # Use the full prediction for updating
    
    return np.array(future_predictions)

# Split data into training and testing sets, respecting temporal order
def time_based_train_test_split(X, y, test_size=0.2):
    """
    Split the data into training and testing sets, respecting the temporal order.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

# Tune hyperparameters for Random Forest model
def tune_random_forest(X, y, quick_test=False):
    # Define parameter distribution based on quick_test flag
    if quick_test:
        print("Quick test mode: Performing simplified Random Forest tuning...")
        param_dist = {
            'n_estimators': randint(10, 50),
            'max_depth': randint(3, 10)
        }
        n_iter = 5
    else:
        print("Full analysis mode: Performing comprehensive Random Forest tuning...")
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
        n_iter = 20

    # Initialize Random Forest model
    rf = RandomForestRegressor(random_state=42)
    # Perform randomized search for best parameters
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=n_iter, cv=3 if quick_test else 5, 
                                   verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X.reshape(X.shape[0], -1), y)
    print(f"Best Random Forest parameters: {rf_random.best_params_}")
    return rf_random.best_estimator_

# Tune hyperparameters for XGBoost model
def tune_xgboost(X, y, quick_test=False):
    # Define parameter distribution based on quick_test flag
    if quick_test:
        print("Quick test mode: Performing simplified XGBoost tuning...")
        param_dist = {
            'n_estimators': randint(10, 50),
            'max_depth': randint(3, 6)
        }
        n_iter = 5
    else:
        print("Full analysis mode: Performing comprehensive XGBoost tuning...")
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4)
        }
        n_iter = 20

    # Initialize XGBoost model
    xgb = XGBRegressor(random_state=42)
    # Perform randomized search for best parameters
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, 
                                    n_iter=n_iter, cv=3 if quick_test else 5, 
                                    verbose=2, random_state=42, n_jobs=-1)
    xgb_random.fit(X.reshape(X.shape[0], -1), y)
    print(f"Best XGBoost parameters: {xgb_random.best_params_}")
    return xgb_random.best_estimator_

# Main function to analyze stock data and make predictions
def analyze_and_predict_stock(symbol, start_date, end_date, future_days=30, suppress_warnings=False, quick_test=False):
    # Suppress warnings if flag is set
    if suppress_warnings:
        suppress_warnings_method()

    print(f"Starting analysis for {symbol}...")

    # Fetch and prepare stock data
    data = fetch_stock_data(symbol, start_date, end_date)
    data = add_technical_indicators(data)
    data.dropna(inplace=True)

    if quick_test:
        # Use only the last 100 data points for quick testing
        data = data.tail(100)

    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']
    X, y, scaler = prepare_data(data[features])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=0.2)

    # Train and evaluate models
    print("\nStarting model training and hyperparameter tuning...")
    models = [
        ("LSTM", create_lstm_model((X.shape[1], X.shape[2]))),
        ("GRU", create_gru_model((X.shape[1], X.shape[2]))),
        ("Random Forest", tune_random_forest(X, y, quick_test)),
        ("XGBoost", tune_xgboost(X, y, quick_test))
    ]

    results = {}
    oof_predictions = {}
    model_stats = []
    with tqdm(total=len(models), desc="Overall Progress", position=0) as pbar:
        for name, model in models:
            print(f"\nTraining and evaluating {name} model...")
            cv_score, cv_std, overall_score, oof_pred = train_and_evaluate_model(model, X, y, n_splits=3 if quick_test else 5, model_name=name)
            print(f"{name} model results:")
            print(f"  Cross-validation R² score: {cv_score:.4f} (±{cv_std:.4f})")
            print(f"  Overall out-of-fold R² score: {overall_score:.4f}")
            
            # Retrain on full dataset
            if isinstance(model, (RandomForestRegressor, XGBRegressor)):
                model.fit(X.reshape(X.shape[0], -1), y)
                train_score = model.score(X.reshape(X.shape[0], -1), y)
            else:
                history = model.fit(X, y, epochs=100, batch_size=32, verbose=0)
                train_score = 1 - history.history['loss'][-1]  # Use final training loss as a proxy for R²
            
            results[name] = model
            oof_predictions[name] = oof_pred
            
            # Calculate overfitting score (difference between train and cv scores)
            overfitting_score = train_score - overall_score
            
            model_stats.append({
                'Model': name,
                'CV R² Score': cv_score,
                'CV R² Std': cv_std,
                'OOF R² Score': overall_score,
                'Train R² Score': train_score,
                'Overfitting Score': overfitting_score
            })
            
            pbar.update(1)

    # Create a DataFrame with model statistics
    stats_df = pd.DataFrame(model_stats)
    stats_df = stats_df.sort_values('OOF R² Score', ascending=False).reset_index(drop=True)
    
    # Add overfitting indicator
    stats_df['Overfit'] = stats_df['Overfitting Score'].apply(lambda x: 'Yes' if x > 0.05 else 'No')

    # Print the table
    print("\nModel Performance Summary:")
    print(tabulate(stats_df, headers='keys', tablefmt='pretty', floatfmt='.4f'))

    # Use out-of-fold predictions for ensemble
    ensemble_predictions = np.mean([oof_predictions[name] for name in results.keys()], axis=0)
    
    # Predict future data
    future_predictions = []
    for model in results.values():
        future_pred = predict_future(model, X[-1], scaler, future_days)
        future_predictions.append(future_pred)
    future_predictions = np.mean(future_predictions, axis=0)
    
    # Inverse transform the predictions (only for 'Close' price)
    close_price_scaler = MinMaxScaler(feature_range=(0, 1))
    close_price_scaler.fit(data['Close'].values.reshape(-1, 1))
    ensemble_predictions = close_price_scaler.inverse_transform(ensemble_predictions.reshape(-1, 1))
    future_predictions = close_price_scaler.inverse_transform(future_predictions.reshape(-1, 1))

    # Calculate returns and risk metrics
    actual_returns = data['Close'].pct_change().dropna()
    predicted_returns = pd.Series(ensemble_predictions.flatten()).pct_change().dropna()

    sharpe_ratio, max_drawdown = calculate_risk_metrics(actual_returns)
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")

    # Plot results
    plt.figure(figsize=(20, 16))  # Increased figure height
    
    # Price prediction plot
    plt.subplot(2, 1, 1)
    plot_data = data.iloc[-len(ensemble_predictions):]
    future_dates = pd.date_range(start=plot_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    
    plt.plot(plot_data.index, plot_data['Close'], label='Actual Price', color='blue')
    plt.plot(plot_data.index, ensemble_predictions, label='Predicted Price', color='red', linestyle='--')
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='--')
    
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Model performance summary table
    plt.subplot(2, 1, 2)
    plt.axis('off')
    table = plt.table(cellText=stats_df.values,
                      colLabels=stats_df.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Lower the title and add more space between plot and table
    plt.title('Model Performance Summary', pad=60)

    plt.tight_layout()
    plt.savefig(f'{symbol}_prediction_with_stats.png', dpi=300, bbox_inches='tight')
    print(f"Plot with statistics saved as '{symbol}_prediction_with_stats.png'")
    plt.show()

    print(f"\nFuture predictions for the next {future_days} days:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.date()}: ${price[0]:.2f}")

    print("\nAnalysis and prediction completed successfully.")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Stock Price Prediction and Analysis Tool')
    
    parser.add_argument('-s', '--symbol', type=str, default='MSFT',
                        help='Stock symbol to analyze (default: MSFT)')
    
    parser.add_argument('-sd', '--start_date', type=str, default='2018-01-01',
                        help='Start date for historical data (default: 2018-01-01)')
    
    parser.add_argument('-fd', '--future_days', type=int, default=30,
                        help='Number of days to predict into the future (default: 30)')
    
    parser.add_argument('-q', '--quick_test', action='store_true',
                        help='Run in quick test mode (default: False)')
    
    parser.add_argument('-sw', '--suppress_warnings', action='store_true',
                        help='Suppress warnings (default: False)')

    args = parser.parse_args()

    # Validate start_date
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        parser.error("Incorrect start date format, should be YYYY-MM-DD")

    return args

# Main execution block
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    symbol = args.symbol
    start_date = args.start_date
    end_date = datetime.now().strftime("%Y-%m-%d")
    future_days = args.future_days
    quick_test_flag = args.quick_test
    suppress_warnings_flag = args.suppress_warnings

    # Print analysis parameters
    print(f"Analyzing {symbol} from {start_date} to {end_date}")
    print(f"Predicting {future_days} days into the future")
    print(f"Quick test mode: {'Enabled' if quick_test_flag else 'Disabled'}")
    print(f"Warnings suppressed: {'Yes' if suppress_warnings_flag else 'No'}")

    # Run the stock analysis and prediction
    analyze_and_predict_stock(symbol, start_date, end_date, future_days, 
                              suppress_warnings=suppress_warnings_flag, 
                              quick_test=quick_test_flag)
