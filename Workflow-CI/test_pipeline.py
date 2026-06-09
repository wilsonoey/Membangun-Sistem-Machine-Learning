import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def create_sample_csv(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        [
            {
                "country": "Indonesia",
                "date": "2022-01-01",
                "total_vaccinations": 100,
                "people_vaccinated": 80,
                "people_fully_vaccinated": 20,
                "total_boosters": 5,
                "daily_vaccinations_raw": 10,
                "daily_vaccinations_per_million": 1.2,
                "people_vaccinated_per_hundred": 0.5,
                "people_fully_vaccinated_per_hundred": 0.2,
                "total_vaccinations_per_hundred": 0.7,
                "population": 1000,
                "daily_vaccinations": 10,
            },
            {
                "country": "Malaysia",
                "date": "2022-01-02",
                "total_vaccinations": 200,
                "people_vaccinated": 150,
                "people_fully_vaccinated": 50,
                "total_boosters": 10,
                "daily_vaccinations_raw": 20,
                "daily_vaccinations_per_million": 1.5,
                "people_vaccinated_per_hundred": 0.8,
                "people_fully_vaccinated_per_hundred": 0.4,
                "total_vaccinations_per_hundred": 1.2,
                "population": 1200,
                "daily_vaccinations": 20,
            },
            {
                "country": "Thailand",
                "date": "2022-01-03",
                "total_vaccinations": 300,
                "people_vaccinated": 240,
                "people_fully_vaccinated": 60,
                "total_boosters": 15,
                "daily_vaccinations_raw": 30,
                "daily_vaccinations_per_million": 2.0,
                "people_vaccinated_per_hundred": 1.0,
                "people_fully_vaccinated_per_hundred": 0.5,
                "total_vaccinations_per_hundred": 1.5,
                "population": 1500,
                "daily_vaccinations": 30,
            },
            {
                "country": "Vietnam",
                "date": "2022-01-04",
                "total_vaccinations": 400,
                "people_vaccinated": 320,
                "people_fully_vaccinated": 80,
                "total_boosters": 20,
                "daily_vaccinations_raw": 40,
                "daily_vaccinations_per_million": 2.5,
                "people_vaccinated_per_hundred": 2.0,
                "people_fully_vaccinated_per_hundred": 0.7,
                "total_vaccinations_per_hundred": 2.0,
                "population": 1800,
                "daily_vaccinations": 40,
            },
            {
                "country": "Philippines",
                "date": "2022-01-05",
                "total_vaccinations": 500,
                "people_vaccinated": 400,
                "people_fully_vaccinated": 100,
                "total_boosters": 25,
                "daily_vaccinations_raw": 50,
                "daily_vaccinations_per_million": 3.0,
                "people_vaccinated_per_hundred": 2.5,
                "people_fully_vaccinated_per_hundred": 0.9,
                "total_vaccinations_per_hundred": 2.5,
                "population": 2100,
                "daily_vaccinations": 50,
            },
        ]
    )
    csv_path = tmp_path / "sample_vaccinations.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


def test_training_runs_without_error(tmp_path):
    csv_path = create_sample_csv(tmp_path)
    result = subprocess.run(
        [sys.executable, "MLProject/modelling.py", "--input_csv", str(csv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path("models/latest").exists()
    assert Path("models/model_info.json").exists()


def test_health_endpoint_metadata_file_exists():
    path = Path("models/model_info.json")
    if not path.exists():
        pytest.skip("Metadata model belum tersedia karena training belum dijalankan.")
    content = json.loads(path.read_text(encoding="utf-8"))
    assert "metrics" in content
    assert "target_column" in content or "best_alpha" in content
