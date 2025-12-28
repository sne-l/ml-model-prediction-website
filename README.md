# ml-model-prediction-website

A small Flask web app that exposes ML model prediction and summarization features through a simple website. This repository contains the application server, HTML templates, static assets, and a notebook for summarization experiments.

**Quick summary:**
- **What:** Flask-based demo site for running/presenting ML model predictions and a text summarizer.
- **Where:** The app entry point is `app.py`. Templates are in `templates/` and static assets are in `static/` and `instance/`.


**Repository layout**

- `app.py`: Flask application entry point.
- `templates/`: HTML templates used by the Flask app (e.g. `home.html`, `signup.html`).
- `static/` and `instance/`: CSS and other static assets.
- `summarizer.ipynb`: Jupyter notebook with summarization experiments.
- `LICENSE`: Project license.

**Development notes**

- To add or update model code, modify `app.py` or create a new module and import it.
- Use the `templates/` folder for HTML changes and `static/` or `instance/` for CSS and assets.
- If you add heavy dependencies (PyTorch, TensorFlow), consider using a separate environment and documenting steps in `requirements.txt`.

**Running the summarizer notebook**

Open `summarizer.ipynb` in Jupyter Notebook or JupyterLab and run the cells. Install notebook dependencies into the same virtual environment before running.

**License & Contact**

This project is provided under the Apache License in `LICENSE`.
