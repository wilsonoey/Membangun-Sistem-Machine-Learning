# Panduan Monitoring

Dokumen ini menjelaskan cara menjalankan stack monitoring untuk model serving yang telah dibuat.

## Komponen

- **Flask API** sebagai penyedia endpoint inferensi
- **Prometheus** untuk scraping metrik dari endpoint `/metrics`
- **Grafana** untuk visualisasi metrik observabilitas

## Menjalankan Monitoring

Pastikan API sudah berjalan di port `5000`.

```bash
docker compose up -d
```

## Endpoint Penting

- API health: `http://127.0.0.1:5000/health`
- API metrics: `http://127.0.0.1:5000/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## Metrik yang Dipantau

- `prediction_requests_total`
- `prediction_errors_total`
- `prediction_latency_seconds`

## Import Dashboard Grafana

1. Buka Grafana.
2. Login menggunakan akun default admin/admin.
3. Tambahkan data source Prometheus ke `http://prometheus:9090` jika memakai Docker network.
4. Import file `grafana_dashboard.json`.

## Bukti Screenshot

Jika submission membutuhkan bukti visual, gunakan placeholder berikut:

- `Petunjuk_prometheus-dashboard.txt`
- `Petunjuk_grafana-dashboard.txt`

## Interpretasi Dashboard

Dashboard dirancang untuk menunjukkan volume request, jumlah error, dan estimasi latency rata-rata. Ketiga metrik ini penting untuk mendeteksi kegagalan layanan, lonjakan trafik, dan penurunan performa model serving.