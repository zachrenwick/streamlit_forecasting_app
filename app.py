import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

st.title('📈 Automated Time Series Forecasting')

"""
This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵 

**In beta mode**

Created by Zach Renwick: https://twitter.com/zachrenwick

Code available here: https://github.com/zachrenwick/streamlit_forecasting_app
"""

"""
### Step 1: Import Data
"""
df = st.file_uploader('Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.', type='csv', encoding='auto', key=None)

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)

"""
### Step 2: Select Forecast Horizon

Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)

"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)

"""
### Step 4: Analyze Forecast Performance (Optional)

Advanced users can view the Mean Average Percentage Error (MAPE), Mean Average Error (MAE), and Median Absolute Percentage Error (MdAPE) for your selected periods and horizon below.

If you'd like to learn more about the initial period, periods, and horizon inputs, read more here: https://facebook.github.io/prophet/docs/diagnostics.html
"""

initial_period = st.text_input(label = 'Initial Period (i.e. 365 days) The initial period should be long enough to capture all of the components of the model, in particular seasonalities and extra regressors: at least a year for yearly seasonality, at least a week for weekly seasonality, etc.', value='', max_chars=10, key=None, type='default')
period_input = st.text_input(label = 'Period (i.e. 5 days) This input is for the spacing between cutoff dates', value='', max_chars=10, key=None, type='default')

if st.button('Run Forecast Metrics'):
    with st.spinner('Crunching forecast metrics... this may take a few minutes. In the meantime you can still save the forecast from the below link...'):             
        df_cv = cross_validation(m, initial= initial_period, period = period_input, horizon = periods_input)
        st.write(df_cv.head())
        
        df_p = performance_metrics(df_cv)
        
        selected_df_p = df_p[['horizon','mae', 'mape', 'mdape']]
        st.write(selected_df_p)

        #fig_csv = plot_cross_validation_metric(df_cv, metric='mape')
        #st.write(fig_csv)

    st.success('Done!')
else:
    st.write('')


"""
### Step 5: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""

if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)