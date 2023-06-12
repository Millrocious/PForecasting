import io
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from xgboost import DMatrix

# Load the saved XGBoost model
model_filename = "xgboost_model.bin"
loaded_model = xgb.Booster()
loaded_model.load_model(model_filename)

# Load the required data and define future dates
pjme_all = pd.read_csv('PJME_hourly.csv', index_col=[0], parse_dates=[0])

# Initialize the FastAPI app
app = FastAPI()


# Define the prediction endpoint
@app.get("/predict")
def predict_usage(start_date: str, end_date: str):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Convert input data to a DataFrame
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    future_features = create_features(pd.DataFrame(index=future_dates))
    predictions = loaded_model.predict(DMatrix(future_features))

    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(20)

    # Plot the original data
    _ = pjme_all[['PJME_MW']].plot(ax=ax, style='-', linewidth=3)

    # Plot the predictions for future dates
    plt.plot(future_dates, predictions, color='r', linestyle='-', linewidth=3)

    # Set the plot boundaries and labels
    ax.set_xbound(lower=start_date, upper=end_date)
    ax.set_ylim(10000, 60000)
    plt.title('Future forecast')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend(['Original Data', 'Predictions'])

    # Save the plot as an image in memory
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()  # Close the plot to release resources
    img_buffer.seek(0)

    # Return the image as a streaming response
    return StreamingResponse(img_buffer, media_type="image/png")


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth']]
    if label:
        y = df[label]
        return X, y
    return X


# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)