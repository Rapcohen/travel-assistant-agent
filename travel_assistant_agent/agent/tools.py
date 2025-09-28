import logging
import os
from typing import Any, Callable, List

import requests
from langchain_core.tools import tool

_WEATHER_API_BASE_URL = 'http://api.weatherapi.com/v1'


@tool
def get_weather_forecast(location: str) -> str:
    """
    Get the current weather forecast for a given location using a public weather API.
    Can only give the forecast for the next 14 days.
    :param location: The location to get the forecast for.
    """
    api_key = os.getenv('WEATHER_API_KEY')
    url = f'{_WEATHER_API_BASE_URL}/forecast.json?key={api_key}&q={location}&days=14'

    try:
        response = requests.get(url)
        response.raise_for_status()
        forecast = response.json()
    except Exception as e:
        logging.debug(e)
        return 'Current weather data is unavailable at the moment.'

    return '\n'.join([
        '[{date}] High: {high}°C, Low: {low}°C, Chance of rain: {chance_of_rain}'.format(
            date=forecast_day['date'],
            high=forecast_day['day']['maxtemp_c'],
            low=forecast_day['day']['mintemp_c'],
            chance_of_rain=forecast_day['day']['daily_chance_of_rain']
        )
        for forecast_day in forecast['forecast']['forecastday']
    ])


TOOLS: List[Callable[..., Any]] = [get_weather_forecast]
