import requests
import json
import time

API_URL = "http://127.0.0.1:5000/predict"

def send_prediction_request():
    payload = {
        "data": [
            {
                "country": "Indonesia",
                "date": "2022-01-10",
                "total_vaccinations": 250000000,
                "people_vaccinated": 150000000,
                "people_fully_vaccinated": 100000000,
                "total_boosters": 5000000,
                "daily_vaccinations_raw": 850000,
                "daily_vaccinations_per_million": 3200,
                "people_vaccinated_per_hundred": 55.3,
                "people_fully_vaccinated_per_hundred": 38.6,
                "total_vaccinations_per_hundred": 93.9,
                "population": 273523621
            },
            {
                "country": "Malaysia",
                "date": "2022-01-10",
                "total_vaccinations": 35000000,
                "people_vaccinated": 26000000,
                "people_fully_vaccinated": 25000000,
                "total_boosters": 1000000,
                "daily_vaccinations_raw": 120000,
                "daily_vaccinations_per_million": 3700,
                "people_vaccinated_per_hundred": 80.2,
                "people_fully_vaccinated_per_hundred": 77.1,
                "total_vaccinations_per_hundred": 107.5,
                "population": 32700000
            }
        ]
    }

    print(f"Sending POST request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response Content:")
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Failed to connect to the server: {e}")

if __name__ == "__main__":
    # Wait for server to be ready in typical run scenarios
    print("Simulating inference requests to serving API...")
    send_prediction_request()
