import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, clone_model as keras_clone_model
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
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
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from tensorflow.keras.regularizers import l1_l2
from matplotlib.dates import num2date


# Suppress warnings and TensorFlow logging
def suppress_warnings_method():
    # Filter out warnings
    warnings.filterwarnings("ignore")
    # Suppress TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Suppress TensorFlow verbose logging
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
    data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
    data["SMA_50"] = ta.trend.sma_indicator(data["Close"], window=50)
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
    data["MACD"] = ta.trend.macd_diff(data["Close"])
    data["BB_upper"], data["BB_middle"], data["BB_lower"] = (
        ta.volatility.bollinger_hband_indicator(data["Close"]),
        ta.volatility.bollinger_mavg(data["Close"]),
        ta.volatility.bollinger_lband_indicator(data["Close"]),
    )
    # Advanced indicators
    data["EMA_20"] = ta.trend.ema_indicator(data["Close"], window=20)
    data["ATR"] = ta.volatility.average_true_range(
        data["High"], data["Low"], data["Close"]
    )
    data["ADX"] = ta.trend.adx(data["High"], data["Low"], data["Close"])
    data["Stoch_K"] = ta.momentum.stoch(data["High"], data["Low"], data["Close"])
    data["Volatility"] = data["Close"].rolling(window=20).std()
    data["Price_Change"] = data["Close"].pct_change()
    data["Volume_Change"] = data["Volume"].pct_change()
    data["High_Low_Range"] = (data["High"] - data["Low"]) / data["Close"]
    return data


# Prepare data for model training by scaling and creating sequences
def prepare_data(data, look_back=60):
    """
    Prepare data for model training.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data) - 1):  # Note the -1 here
        X.append(scaled_data[i - look_back : i])
        y.append(scaled_data[i + 1, 0])  # Predicting the next 'Close' price

    return np.array(X), np.array(y), scaler


# Create an LSTM model for time series prediction
def create_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(
                units=64,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            ),
            Dropout(0.2),  # Add dropout layer
            LSTM(units=32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.2),  # Add dropout layer
            Dense(
                units=16, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            ),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Create a GRU model for time series prediction
def create_gru_model(input_shape):
    model = Sequential(
        [
            GRU(
                units=64,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            ),
            Dropout(0.2),  # Add dropout layer
            GRU(units=32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.2),  # Add dropout layer
            Dense(
                units=16, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            ),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Train and evaluate a model using time series cross-validation
def train_and_evaluate_model(model, X, y, n_splits=5, model_name="Model"):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_predictions = []
    all_true_values = []

    with tqdm(total=n_splits, desc=f"Training {model_name}", leave=False) as pbar:
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if isinstance(model, (RandomForestRegressor, XGBRegressor)):
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                model.fit(X_train_2d, y_train)
                predictions = model.predict(X_test_2d)
            elif isinstance(model, Sequential):
                early_stopping = EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
                with tqdm(total=100, desc="Epochs", leave=False) as epoch_pbar:

                    class EpochProgressCallback(Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            epoch_pbar.update(1)

                    model.fit(
                        X_train,
                        y_train,
                        epochs=100,
                        batch_size=32,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[early_stopping, EpochProgressCallback()],
                    )
                predictions = model.predict(X_test, verbose=0).flatten()

            all_predictions.extend(predictions)
            all_true_values.extend(y_test)
            pbar.update(1)

    score = r2_score(all_true_values, all_predictions)
    return score, 0, score, np.array(all_predictions)


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


def weighted_ensemble_predict(models, X, weights):
    predictions = []
    for model, weight in zip(models, weights):
        if isinstance(model, (RandomForestRegressor, XGBRegressor)):
            pred = model.predict(X.reshape(X.shape[0], -1))
        else:
            pred = np.array(
                [model.predict(X[i : i + 1], verbose=0)[0][0] for i in range(len(X))]
            )
        predictions.append(weight * pred)
    return np.sum(predictions, axis=0)


# Calculate risk metrics (Sharpe ratio and max drawdown)
def calculate_risk_metrics(returns):
    sharpe_ratio = (
        np.mean(returns) / np.std(returns) * np.sqrt(252)
    )  # Assuming daily returns
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    return sharpe_ratio, max_drawdown


# Predict future stock prices using a trained model
def predict_future(model, last_sequence, scaler, days):
    future_predictions = []
    current_sequence = last_sequence.copy()

    with tqdm(total=days, desc="Predicting future", leave=False) as pbar:
        for _ in range(days):
            if isinstance(model, (RandomForestRegressor, XGBRegressor)):
                prediction = model.predict(current_sequence.reshape(1, -1))
                future_predictions.append(
                    prediction[0]
                )  # prediction is already a scalar
            else:
                prediction = model.predict(
                    current_sequence.reshape(
                        1, current_sequence.shape[0], current_sequence.shape[1]
                    ),
                    verbose=0,
                )
                future_predictions.append(
                    prediction[0][0]
                )  # Take only the first (and only) element

            # Update the sequence for the next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = prediction  # Use the full prediction for updating
            pbar.update(1)

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
        param_dist = {"n_estimators": randint(10, 50), "max_depth": randint(3, 10)}
        n_iter = 5
    else:
        print("Full analysis mode: Performing comprehensive Random Forest tuning...")
        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(5, 50),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        n_iter = 20

    # Initialize Random Forest model
    rf = RandomForestRegressor(random_state=42)
    # Perform randomized search for best parameters
    tscv = TimeSeriesSplit(n_splits=3 if quick_test else 5)
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",  # Change to MSE
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(X.reshape(X.shape[0], -1), y)
    print(f"Best Random Forest parameters: {rf_random.best_params_}")
    return rf_random.best_estimator_


# Tune hyperparameters for XGBoost model
def tune_xgboost(X, y, quick_test=False):
    # Define parameter distribution based on quick_test flag
    if quick_test:
        print("Quick test mode: Performing simplified XGBoost tuning...")
        param_dist = {"n_estimators": randint(10, 50), "max_depth": randint(3, 6)}
        n_iter = 5
    else:
        print("Full analysis mode: Performing comprehensive XGBoost tuning...")
        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 1.0),
            "colsample_bytree": uniform(0.6, 1.0),
            "gamma": uniform(0, 5),
        }
        n_iter = 20

    # Initialize XGBoost model
    xgb = XGBRegressor(random_state=42)
    # Perform randomized search for best parameters
    tscv = TimeSeriesSplit(n_splits=3 if quick_test else 5)
    xgb_random = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",  # Change to MSE
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    xgb_random.fit(X.reshape(X.shape[0], -1), y)
    print(f"Best XGBoost parameters: {xgb_random.best_params_}")
    return xgb_random.best_estimator_


def implement_trading_strategy(actual_prices, predicted_prices, threshold=0.01):
    returns = []
    position = 0  # -1: short, 0: neutral, 1: long
    for i in range(1, len(actual_prices)):
        predicted_return = (predicted_prices[i] - actual_prices[i - 1]) / actual_prices[
            i - 1
        ]
        if predicted_return > threshold and position <= 0:
            position = 1  # Buy
        elif predicted_return < -threshold and position >= 0:
            position = -1  # Sell
        actual_return = (actual_prices[i] - actual_prices[i - 1]) / actual_prices[i - 1]
        returns.append(position * actual_return)
    return np.array(returns)


def select_features_rfe(X, y, n_features_to_select=10):
    if isinstance(X, np.ndarray) and len(X.shape) == 3:
        X_2d = X.reshape(X.shape[0], -1)
    else:
        X_2d = X

    rfe = RFE(
        estimator=RandomForestRegressor(random_state=42),
        n_features_to_select=n_features_to_select,
    )
    X_selected = rfe.fit_transform(X_2d, y)
    selected_features = rfe.support_
    return X_selected, selected_features


def calculate_ensemble_weights(models, X, y):
    weights = []
    for name, model in models:
        _, _, score, _ = train_and_evaluate_model(
            model, X, y, n_splits=5, model_name=name
        )
        weights.append(max(score, 0))  # Ensure non-negative weights

    if sum(weights) == 0:
        # If all weights are zero, use equal weights
        return [1 / len(weights)] * len(weights)
    else:
        return [w / sum(weights) for w in weights]  # Normalize weights


def augment_data(X, y, noise_level=0.01):
    X_aug = X.copy()
    y_aug = y.copy()
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug += noise
    return X_aug, y_aug


# Main function to analyze stock data and make predictions
def analyze_and_predict_stock(
    symbol,
    start_date,
    end_date,
    future_days=30,
    suppress_warnings=False,
    quick_test=False,
    models_to_run=["LSTM", "GRU", "Random Forest", "XGBoost"],
):
    # Suppress warnings if flag is set
    if suppress_warnings:
        suppress_warnings_method()

    print(f"Starting analysis for {symbol}...")

    print(f"Fetching stock data for {symbol}...")
    data = fetch_stock_data(symbol, start_date, end_date)
    print(f"Adding technical indicators...")
    data = add_technical_indicators(data)
    data.dropna(inplace=True)

    if quick_test:
        print("Quick test mode: Using only the last 100 data points.")
        data = data.tail(100)

    print("Preparing data for model training...")
    features = [
        "Close",
        "Volume",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "BB_upper",
        "BB_middle",
        "BB_lower",
        "Volatility",
        "Price_Change",
        "Volume_Change",
        "High_Low_Range",
    ]
    X, y, scaler = prepare_data(data[features])

    print("Augmenting data...")
    X_aug, y_aug = augment_data(X, y)
    X = np.concatenate((X, X_aug), axis=0)
    y = np.concatenate((y, y_aug), axis=0)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=0.2)

    print("\nStarting model training and hyperparameter tuning...")
    models = []
    if "LSTM" in models_to_run:
        models.append(("LSTM", create_lstm_model((X.shape[1], X.shape[2]))))
    if "GRU" in models_to_run:
        models.append(("GRU", create_gru_model((X.shape[1], X.shape[2]))))
    if "Random Forest" in models_to_run:
        models.append(("Random Forest", tune_random_forest(X, y, quick_test)))
    if "XGBoost" in models_to_run:
        models.append(("XGBoost", tune_xgboost(X, y, quick_test)))

    results = {}
    oof_predictions = {}
    model_stats = []
    with tqdm(total=len(models), desc="Training Models", position=0) as pbar:
        for name, model in models:
            print(f"\nTraining and evaluating {name} model...")
            cv_score, cv_std, overall_score, oof_pred = train_and_evaluate_model(
                model, X, y, n_splits=3 if quick_test else 5, model_name=name
            )
            print(f" {name} model results:")
            print(f"  Cross-validation R² score: {cv_score:.4f} (±{cv_std:.4f})")
            print(f"  Overall out-of-fold R² score: {overall_score:.4f}")

            print(f"Retraining {name} model on full dataset...")
            if isinstance(model, (RandomForestRegressor, XGBRegressor)):
                model.fit(X.reshape(X.shape[0], -1), y)
                train_score = model.score(X.reshape(X.shape[0], -1), y)
            else:
                with tqdm(total=100, desc="Epochs", leave=False) as epoch_pbar:

                    class EpochProgressCallback(Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            epoch_pbar.update(1)

                    history = model.fit(
                        X,
                        y,
                        epochs=100,
                        batch_size=32,
                        verbose=0,
                        callbacks=[EpochProgressCallback()],
                    )
                train_score = (
                    1 - history.history["loss"][-1]
                )  # Use final training loss as a proxy for R²

            results[name] = model
            oof_predictions[name] = oof_pred

            overfitting_score = train_score - overall_score

            model_stats.append(
                {
                    "Model": name,
                    "CV R² Score": cv_score,
                    "CV R² Std": cv_std,
                    "OOF R² Score": overall_score,
                    "Train R² Score": train_score,
                    "Overfitting Score": overfitting_score,
                }
            )

            pbar.update(1)

    # Create a DataFrame with model statistics
    stats_df = pd.DataFrame(model_stats)
    stats_df = stats_df.sort_values("OOF R² Score", ascending=False).reset_index(
        drop=True
    )

    # Add overfitting indicator
    stats_df["Overfit"] = stats_df["Overfitting Score"].apply(
        lambda x: "Yes" if x > 0.05 else "No"
    )

    # Print the table
    print("\nModel Performance Summary:")
    print(tabulate(stats_df, headers="keys", tablefmt="pretty", floatfmt=".4f"))

    print("\nCalculating ensemble weights...")
    ensemble_weights = calculate_ensemble_weights(models, X_test, y_test)
    print(f"Ensemble weights: {ensemble_weights}")

    print("Making ensemble predictions...")
    ensemble_predictions = weighted_ensemble_predict(
        [model for _, model in models], X, ensemble_weights
    )

    print(f"Predicting future data for the next {future_days} days...")
    future_predictions = []
    for name, model in models:
        print(f"  Making future predictions with {name} model...")
        future_pred = predict_future(model, X[-1], scaler, future_days)
        future_predictions.append(future_pred)
    future_predictions = np.mean(future_predictions, axis=0)

    print("Inverse transforming predictions...")
    close_price_scaler = MinMaxScaler(feature_range=(0, 1))
    close_price_scaler.fit(data["Close"].values.reshape(-1, 1))
    ensemble_predictions = close_price_scaler.inverse_transform(
        ensemble_predictions.reshape(-1, 1)
    )
    future_predictions = close_price_scaler.inverse_transform(
        future_predictions.reshape(-1, 1)
    )

    # Ensure ensemble_predictions matches the length of the actual data
    ensemble_predictions = ensemble_predictions[-len(data) :]

    print("Plotting results...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 24))

    # Price prediction plot
    plot_data = data.iloc[-len(ensemble_predictions) :]
    future_dates = pd.date_range(
        start=plot_data.index[-1] + pd.Timedelta(days=1), periods=future_days
    )

    ax1.plot(plot_data.index, plot_data["Close"], label="Actual Price", color="blue")
    ax1.plot(
        plot_data.index,
        ensemble_predictions,
        label="Predicted Price",
        color="red",
        linestyle="--",
    )
    ax1.plot(
        future_dates,
        future_predictions,
        label="Future Predictions",
        color="green",
        linestyle="--",
    )

    # Add price indications for every day (initially invisible)
    annotations = []
    for i, (date, price) in enumerate(zip(plot_data.index, ensemble_predictions)):
        ann = ax1.annotate(
            f"${price[0]:.2f}",
            (date, price[0]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            alpha=0.7,
            visible=False,
        )
        annotations.append(ann)

    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        ann = ax1.annotate(
            f"${price[0]:.2f}",
            (date, price[0]),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            alpha=0.7,
            visible=False,
        )
        annotations.append(ann)

    ax1.set_title(f"{symbol} Stock Price Prediction")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()

    # Add hover annotation
    hover_annot = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    hover_annot.set_visible(False)

    def update_hover_annot(event):
        vis = hover_annot.get_visible()
        if event.inaxes == ax1:
            x, y = event.xdata, event.ydata
            date = num2date(x).strftime("%Y-%m-%d")
            hover_annot.xy = (x, y)
            hover_annot.set_text(f"Date: {date}\nPrice: ${y:.2f}")
            hover_annot.set_visible(True)
            fig.canvas.draw_idle()
        elif vis:
            hover_annot.set_visible(False)
            fig.canvas.draw_idle()

    # Connect the hover event
    fig.canvas.mpl_connect("motion_notify_event", update_hover_annot)

    # Add zoom event handler
    def on_zoom(event):
        ax1 = event.inaxes
        if ax1 is None:
            return
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()

        # Calculate the zoom level based on the x-axis range
        zoom_level = (plot_data.index[-1] - plot_data.index[0]).days / (
            xlim[1] - xlim[0]
        ).days

        # Adjust annotation visibility based on zoom level
        for ann in annotations:
            ann.set_visible(
                zoom_level > 5
            )  # Show annotations when zoomed in more than 5x

        fig.canvas.draw_idle()

    # Connect the zoom event handler
    fig.canvas.mpl_connect("motion_notify_event", on_zoom)

    # Model performance summary table
    ax2.axis("off")
    table = ax2.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Lower the title and add more space between plot and table
    ax2.set_title("Model Performance Summary", pad=60)

    # Implement trading strategy
    strategy_returns = implement_trading_strategy(
        plot_data["Close"].values, ensemble_predictions.flatten()
    )
    strategy_sharpe_ratio = (
        np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    )
    print(f"Trading Strategy Sharpe Ratio: {strategy_sharpe_ratio:.4f}")

    # Calculate cumulative returns of the trading strategy
    cumulative_returns = (1 + strategy_returns).cumprod() - 1

    # Add new subplot for trading strategy performance
    ax3.plot(
        plot_data.index[-len(cumulative_returns) :],
        cumulative_returns,
        label="Strategy Cumulative Returns",
        color="purple",
    )
    ax3.set_title(f"{symbol} Trading Strategy Performance")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Cumulative Returns")
    ax3.legend()

    # Add strategy Sharpe ratio as text on the plot
    ax3.text(
        0.05,
        0.95,
        f"Strategy Sharpe Ratio: {strategy_sharpe_ratio:.4f}",
        transform=ax3.transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(
        f"{symbol}_prediction_with_stats_and_strategy.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"Plot with statistics and strategy performance saved as '{symbol}_prediction_with_stats_and_strategy.png'"
    )
    plt.show()

    print(f"\nFuture predictions for the next {future_days} days:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.date()}: ${price[0]:.2f}")

    print("\nAnalysis and prediction completed successfully.")


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Stock Price Prediction and Analysis Tool"
    )

    parser.add_argument(
        "-s",
        "--symbol",
        type=str,
        default="MSFT",
        help="Stock symbol to analyze (default: MSFT)",
    )

    parser.add_argument(
        "-sd",
        "--start_date",
        type=str,
        default="2018-01-01",
        help="Start date for historical data (default: 2018-01-01)",
    )

    parser.add_argument(
        "-fd",
        "--future_days",
        type=int,
        default=30,
        help="Number of days to predict into the future (default: 30)",
    )

    parser.add_argument(
        "-q",
        "--quick_test",
        action="store_true",
        help="Run in quick test mode (default: False)",
    )

    parser.add_argument(
        "-sw",
        "--suppress_warnings",
        action="store_true",
        help="Suppress warnings (default: False)",
    )

    args = parser.parse_args()

    # Validate start_date
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
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
    analyze_and_predict_stock(
        symbol,
        start_date,
        end_date,
        future_days,
        suppress_warnings=suppress_warnings_flag,
        quick_test=quick_test_flag,
    )
