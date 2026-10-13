"""Business definitions, chronological forecast evaluation, and bounded XAI."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from prophet import Prophet

ROOT=Path(__file__).parent
DATA=ROOT/'Electronic_sales_Sep2023-Sep2024.csv'
CUSTOMER_FEATURES=['Total_Spend','Transaction_Count','Avg_Rating','Age','Purchase_Frequency','Customer_Lifetime_Days']
REQUIRED={'Customer ID','Purchase Date','Order Status','Total Price','Product Type','Rating','Age','Loyalty Member'}

def clean_data(data):
    missing=REQUIRED-set(data.columns)
    if missing:raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    d=data.copy().drop_duplicates()
    d['Purchase Date']=pd.to_datetime(d['Purchase Date'],errors='coerce').dt.normalize()
    d['Order Status']=d['Order Status'].astype('string').str.strip().str.lower()
    for c in ['Total Price','Rating','Age']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.loc[d['Purchase Date'].notna() & d['Total Price'].notna() & (d['Total Price']>=0)].copy()
    d['Product Type']=d['Product Type'].fillna('Unknown')
    return d.sort_values('Purchase Date',kind='stable')

def revenue_summary(data):
    completed=data.loc[data['Order Status']=='completed']
    return {'Recorded order value':float(data['Total Price'].sum()),
            'Completed merchandise revenue':float(completed['Total Price'].sum()),
            'Cancelled order value':float(data.loc[data['Order Status']=='cancelled','Total Price'].sum()),
            'Completed orders':len(completed),'Cancelled orders':int((data['Order Status']=='cancelled').sum())}

def completed_orders(data):
    return data.loc[data['Order Status']=='completed'].copy()

def customer_table(data):
    d=completed_orders(data).dropna(subset=['Customer ID'])
    if d.empty:raise ValueError('No completed orders with customer IDs')
    c=d.groupby('Customer ID').agg(Total_Spend=('Total Price','sum'),Transaction_Count=('Total Price','size'),
        Avg_Transaction=('Total Price','mean'),Avg_Rating=('Rating','mean'),Age=('Age','first'),
        Current_Loyalty=('Loyalty Member','last'),First_Purchase=('Purchase Date','min'),Last_Purchase=('Purchase Date','max'))
    c['Customer_Lifetime_Days']=(c['Last_Purchase']-c['First_Purchase']).dt.days
    c['Purchase_Frequency']=c['Transaction_Count']/(c['Customer_Lifetime_Days']+1)
    return c

def daily_revenue(data):
    d=completed_orders(data)
    if d.empty:raise ValueError('No completed orders')
    dates=pd.date_range(data['Purchase Date'].min(),data['Purchase Date'].max(),freq='D')
    return d.groupby('Purchase Date')['Total Price'].sum().reindex(dates,fill_value=0).rename('Revenue').rename_axis('Date')

def fit_forecast(series, days=30):
    model=Prophet(weekly_seasonality=True,yearly_seasonality=False,daily_seasonality=False,interval_width=.8)
    model.fit(series.rename_axis('ds').reset_index(name='y'))
    future=model.make_future_dataframe(periods=days)
    forecast=model.predict(future)
    forecast['yhat']=forecast['yhat'].clip(lower=0)
    forecast['yhat_lower']=forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper']=forecast['yhat_upper'].clip(lower=0)
    return forecast

def backtest(series, days=30):
    if len(series)<days+60:raise ValueError('Need at least 90 daily observations for a 30-day holdout')
    train=series.iloc[:-days];test=series.iloc[-days:]
    forecast=fit_forecast(train,days).set_index('ds').loc[test.index]
    predictions={'Trailing 7-day mean':np.full(days,train.iloc[-7:].mean()),
                 'Weekly seasonal naive':np.resize(train.iloc[-7:].to_numpy(),days),
                 'Prophet':forecast['yhat'].to_numpy()}
    rows=[];details=pd.DataFrame({'Actual':test})
    for name,p in predictions.items():
        rows.append({'Model':name,'MAE':mean_absolute_error(test,p),'RMSE':np.sqrt(mean_squared_error(test,p))})
        details[name]=p
    return pd.DataFrame(rows).set_index('Model'),details

def segment_customers(customers,n_clusters=4):
    if len(customers)<n_clusters:raise ValueError('Too few customers for selected cluster count')
    raw=customers[CUSTOMER_FEATURES]
    imputer=SimpleImputer(strategy='median');imputed=imputer.fit_transform(raw)
    scaler=StandardScaler();scaled=scaler.fit_transform(imputed)
    model=KMeans(n_clusters=n_clusters,n_init=10,random_state=42).fit(scaled)
    result=customers.copy();result['Cluster']=model.labels_
    return result,scaled,model

def explain_cluster_distance(scaled,model,cluster=0):
    """SHAP explains squared distance to one fixed center, never cluster-ID numbers."""
    import shap
    rng=np.random.default_rng(42)
    background=scaled[rng.choice(len(scaled),min(40,len(scaled)),replace=False)]
    sample=scaled[rng.choice(len(scaled),min(24,len(scaled)),replace=False)]
    def distance(x):return np.square(np.asarray(x)-model.cluster_centers_[cluster]).sum(axis=1)
    explainer=shap.Explainer(distance,background,feature_names=CUSTOMER_FEATURES,algorithm='exact')
    values=explainer(sample)
    return values

def surrogate_explanation(scaled,model):
    """LIME explains a decision-tree surrogate; held-out accuracy measures fidelity."""
    from lime.lime_tabular import LimeTabularExplainer
    labels=model.predict(scaled)
    counts=np.bincount(labels)
    stratify=labels if np.all(counts[counts>0]>=2) else None
    a,b,ya,yb=train_test_split(scaled,labels,test_size=.2,random_state=42,stratify=stratify)
    tree=DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,random_state=42).fit(a,ya)
    fidelity=accuracy_score(yb,tree.predict(b))
    class_names=[f'Cluster {i}' for i in tree.classes_]
    explainer=LimeTabularExplainer(a,feature_names=CUSTOMER_FEATURES,class_names=class_names,mode='classification',random_state=42)
    predicted_index=int(np.argmax(tree.predict_proba(b[:1])[0]))
    explanation=explainer.explain_instance(b[0],tree.predict_proba,num_features=len(CUSTOMER_FEATURES),labels=(predicted_index,),num_samples=1000)
    return fidelity,explanation.as_list(label=predicted_index),int(tree.classes_[predicted_index])
