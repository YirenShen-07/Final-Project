# Electronics Sales & Customer Analytics Dashboard

An interactive Streamlit prototype for exploring electronics sales, customer purchasing behavior, product performance, and 30-day sales forecasts.

## Business questions
- Which products contribute the most order value?
- How do customer spending and repeat purchasing vary?
- Which customers are in the top spending decile?
- How do daily sales trends change over time?

## Implementation
Python, Pandas, Streamlit, Plotly, Matplotlib, Seaborn, Prophet, scikit-learn, SHAP, and LIME.
The notebook includes missing-value handling, date conversion, category standardization, customer aggregation, interactive filters, KMeans segmentation, and a 3D sales visualization. Saved data output contains 20,000 transaction rows; the app also removes duplicate rows.

## Explore
- [Project notebook](FinalProject.ipynb)
- [Example dataset](Electronic_sales_Sep2023-Sep2024.csv)
- [Introduction video](https://www.youtube.com/watch?v=tO_OpAjtfkQ)

## Run the prototype
1. Open the notebook in Google Colab.
2. Adjust the initial dataset path to your own copy of the included CSV, or skip the exploratory Drive-dependent cells.
3. Run the installation cells and the cell that writes `app.py`.
4. Configure your own ngrok token securely if using a tunnel.
5. Run the Streamlit launch cell and open its temporary URL.
6. Upload the included CSV in the application.

A standalone app and dependency file are not yet included. The tunnel URL is temporary.

## Interpretation and limitations
- Current aggregates include cancelled orders: they represent recorded order value, not verified realized revenue.
- The 30-day Prophet forecast has not been evaluated against a held-out time period or baseline.
- SHAP currently explains numeric KMeans cluster IDs, whose numbering is arbitrary; this is an experimental visualization, not a validated explanation of customer value.
- LIME explains a decision-tree surrogate trained to reproduce cluster labels, not KMeans directly.
- No measured sales uplift, retention improvement, or production deployment is claimed.

## Project context
Developed by Yiren Shen for Duke's AIPI 590 Explainable AI course. The repository is presented as an analytics portfolio prototype.
