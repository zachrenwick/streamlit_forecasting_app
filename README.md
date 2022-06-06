# 📈 Automated Time Series Forecasting

**Background:** This MVP data web app uses the Streamlit framework and Facebook's Prophet forecasting package to generate a dynamic forecast from your own data. 

Try it out here: https://share.streamlit.io/zachrenwick/streamlit_forecasting_app/app.py 

You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵

**In beta mode**

Created by Zach Renwick: https://twitter.com/zachrenwick.

Code available here: https://github.com/zachrenwick/streamlit_forecasting_app

![Screenshot1](/images/screenshot1.jpg)
![Screenshot2](/images/screenshot2.jpg)
![Screenshot3](/images/screenshot3.jpg)
![Screenshot4](/images/screenshot4.jpg)

## Docker
* Build with `docker build -t ts-forecast-app .` (takes some time!)
* Run with `docker run -p 8501:8501 ts-forecast-app:latest`
* Open [http://localhost:8501/](http://localhost:8501/)

## Example data
The Peyton Manning data [from the prophet quickstart](https://facebook.github.io/prophet/docs/quick_start.html#python-api) is included in the `example_data` folder