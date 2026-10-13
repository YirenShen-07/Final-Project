"""Produce real evaluation artifacts and descriptive charts for the bundled data."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from analytics import DATA,clean_data,revenue_summary,completed_orders,customer_table,daily_revenue,backtest,fit_forecast,segment_customers,explain_cluster_distance,surrogate_explanation
ROOT=Path(__file__).parent
def run():
    raw=pd.read_csv(DATA);data=clean_data(raw);out=ROOT/'reports';out.mkdir(exist_ok=True)
    series=daily_revenue(data);metrics,details=backtest(series)
    metrics.to_csv(out/'forecast_metrics.csv');details.to_csv(out/'backtest_predictions.csv')
    forecast=fit_forecast(series);forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(out/'forecast.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4));details.plot(ax=ax);ax.set_ylabel('Completed merchandise revenue ($)');ax.set_title('Chronological 30-day holdout — no future observations in training');fig.tight_layout();fig.savefig(out/'backtest.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(series.index,series.values,color='#aaa',label='Actual');f=forecast.loc[forecast.ds>series.index.max()];ax.plot(f.ds,f.yhat,color='#244b69',label='Forecast');ax.fill_between(f.ds,f.yhat_lower,f.yhat_upper,color='#244b69',alpha=.2,label='80% model interval');ax.set_ylabel('Completed revenue ($)');ax.set_title('30-day forecast after fitting all observed dates');ax.legend();fig.tight_layout();fig.savefig(out/'forecast.png',dpi=160);plt.close(fig)
    products=completed_orders(data).groupby('Product Type')['Total Price'].sum().sort_values();products.to_csv(out/'product_revenue.csv')
    fig,ax=plt.subplots(figsize=(8,4));products.plot.barh(ax=ax,color='#244b69');ax.set_xlabel('Completed merchandise revenue ($)');ax.set_title('Product performance — cancelled orders excluded');fig.tight_layout();fig.savefig(out/'product_revenue.png',dpi=160);plt.close(fig)
    customers=customer_table(data);segmented,scaled,model=segment_customers(customers)
    segmented.groupby('Cluster').mean(numeric_only=True).to_csv(out/'segment_profiles.csv')
    values=explain_cluster_distance(scaled,model,0)
    import shap
    fig,ax=plt.subplots(figsize=(8,4));shap.plots.bar(values,show=False,ax=ax);ax.set_title('SHAP: squared distance to Cluster 0 center');fig.tight_layout();fig.savefig(out/'shap_distance.png',dpi=160);plt.close(fig)
    fidelity,weights,cluster=surrogate_explanation(scaled,model)
    pd.DataFrame(weights,columns=['Feature condition','Weight']).to_csv(out/'lime_surrogate.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));w=pd.DataFrame(weights,columns=['Feature condition','Weight']);ax.barh(w['Feature condition'],w.Weight,color='#244b69');ax.set_title(f'LIME: decision-tree surrogate probability for Cluster {cluster}');fig.tight_layout();fig.savefig(out/'lime_surrogate.png',dpi=160);plt.close(fig)
    revenue=revenue_summary(data)
    run={'raw_rows':len(raw),'clean_rows':len(data),'customers':len(customers),'repeat_completed_customer_share':float((customers.Transaction_Count>1).mean()),
         'revenue_definitions':revenue,'date_min':str(series.index.min().date()),'date_max':str(series.index.max().date()),
         'holdout_start':str(details.index.min().date()),'holdout_end':str(details.index.max().date()),'training_days':len(series)-30,'test_days':30,
         'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'surrogate_fidelity':float(fidelity),
         'SHAP_target':'squared distance to fixed cluster center 0, standardized features','SHAP_sample_rows':len(values.values),'SHAP_background_rows':40}
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n');print(metrics.round(2).to_string());print(json.dumps(revenue,indent=2));return metrics
if __name__=='__main__':run()
