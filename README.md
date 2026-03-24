# sepsis-prediction-project
Early Sepsis Prediction system using Machine Learning on the PhysioNet 2019 dataset. This project follows the KDD process (Integration, Cleaning, Transformation, and Mining) to identify clinical patterns in ICU time-series data. Built with Python, Pandas, and Scikit-learn to enable early medical intervention through data-driven insights.

## Dashboard app

The dashboard now runs as two local processes:

```powershell
python apps/sepsis_dashboard/backend/main.py
```

```powershell
cd apps/sepsis_dashboard/frontend
npm run dev
```

By default the frontend proxies `/api/*` requests to `http://127.0.0.1:8000`.

Optional environment variables:

- `SEPSIS_BACKEND_HOST` and `SEPSIS_BACKEND_PORT` to change the Python server bind address.
- `VITE_BACKEND_ORIGIN` to change the Vite dev proxy target.
- `VITE_API_BASE_URL` to point the built frontend at a non-default backend base URL.
