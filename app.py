"""Run with: streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics import DATA,clean_data,revenue_summary,completed_orders,customer_table,daily_revenue,backtest,fit_forecast,segment_customers,explain_cluster_distance,surrogate_explanation,CUSTOMER_FEATURES
st.set_page_config(page_title='Sales Analytics',page_icon='📊',layout='wide')

@st.cache_data
def load(source):return clean_data(pd.read_csv(source))
@st.cache_data
def forecast_outputs(series):return backtest(series),fit_forecast(series)

def main():
    st.title('Electronics Sales & Customer Analytics')
    st.caption('Explore product performance, customer behavior, and evaluated sales forecasts.')
    uploaded=st.sidebar.file_uploader('Upload a sales CSV',type=['csv'])
    page=st.sidebar.selectbox('Analysis',['Overview','Customers','Forecast','Products','Explainability','3D Sales'])
    try:data=clean_data(pd.read_csv(uploaded)) if uploaded else load(str(DATA))
    except (ValueError,KeyError,pd.errors.ParserError) as exc:st.error(str(exc));return
    summary=revenue_summary(data)
    st.sidebar.caption('Revenue = Total Price on completed orders. Add-ons are excluded. Cancelled orders remain in the recorded-order total.')
    if page=='Overview':
        st.subheader('Order value and realized-status revenue')
        cols=st.columns(3)
        for col,key in zip(cols,['Recorded order value','Completed merchandise revenue','Cancelled order value']):col.metric(key,f'${summary[key]:,.0f}')
        st.caption('Completed order status is used as a revenue proxy; refunds and cash settlement are not available.')
        selection=st.multiselect('Product types',sorted(data['Product Type'].unique()),default=sorted(data['Product Type'].unique()))
        chosen=completed_orders(data);chosen=chosen.loc[chosen['Product Type'].isin(selection)]
        product=chosen.groupby('Product Type')['Total Price'].sum().reset_index()
        st.plotly_chart(px.bar(product,x='Product Type',y='Total Price',labels={'Total Price':'Completed merchandise revenue ($)'},color_discrete_sequence=['#244b69']),width='stretch')
        st.dataframe(data.head(20),width='stretch')
    elif page=='Customers':
        customers=customer_table(data);st.subheader('Completed-order customer behavior')
        cols=st.columns(3);cols[0].metric('Customers',f'{len(customers):,}');cols[1].metric('Repeat purchaser share',f'{(customers.Transaction_Count>1).mean():.1%}');cols[2].metric('Top-decile spend threshold',f'${customers.Total_Spend.quantile(.9):,.0f}')
        monthly=completed_orders(data).groupby(pd.Grouper(key='Purchase Date',freq='MS'))['Total Price'].sum().reset_index()
        st.plotly_chart(px.line(monthly,x='Purchase Date',y='Total Price',markers=True,labels={'Total Price':'Completed merchandise revenue ($)'}),width='stretch')
        st.dataframe(customers.sort_values('Total_Spend',ascending=False).head(20),width='stretch')
        st.caption('Repeat purchases are counted only on completed orders. Most recent loyalty status is selected chronologically. High spend is descriptive, not a prediction of future value.')
    elif page=='Forecast':
        st.subheader('30-day sales forecast with chronological backtest')
        series=daily_revenue(data)
        if len(series)<90:st.warning('At least 90 days of data are required.');return
        with st.spinner('Fitting forecast and evaluating the last 30 days…'):
            ((metrics,details),forecast)=forecast_outputs(series)
        st.dataframe(metrics.round(2),width='stretch')
        st.caption('Models are fitted before the final 30 days; that holdout is used only for evaluation. Settings are fixed. Forecasts are exploratory and are not guaranteed to beat simple baselines.')
        long=details.reset_index().melt(id_vars='Date',var_name='Series',value_name='Revenue')
        st.plotly_chart(px.line(long,x='Date',y='Revenue',color='Series',title='Held-out actuals and forecasts'),width='stretch')
        future=forecast.loc[forecast.ds>series.index.max()]
        fig=go.Figure();fig.add_scatter(x=series.index,y=series.values,name='Completed revenue');fig.add_scatter(x=future.ds,y=future.yhat,name='30-day forecast');fig.add_scatter(x=future.ds,y=future.yhat_upper,line={'width':0},showlegend=False);fig.add_scatter(x=future.ds,y=future.yhat_lower,fill='tonexty',line={'width':0},name='80% model interval')
        st.plotly_chart(fig,width='stretch');st.caption('Missing dates are treated as zero orders, assuming the uploaded period has complete coverage. Annual seasonality is disabled because the example covers only about one year.')
    elif page=='Products':
        st.subheader('Product performance on completed orders')
        d=completed_orders(data);p=d.groupby('Product Type').agg(Revenue=('Total Price','sum'),Orders=('Total Price','size'),Average_Rating=('Rating','mean')).reset_index()
        st.dataframe(p,width='stretch');metric=st.selectbox('Metric',['Revenue','Orders','Average_Rating']);st.plotly_chart(px.bar(p,x='Product Type',y=metric),width='stretch')
        st.caption('Orders counts transactions; it is not the number of unique SKUs. Ratings alone do not establish product quality.')
    elif page=='Explainability':
        st.subheader('Customer segments and transparent explanation targets')
        customers=customer_table(data)
        if len(customers)<2:st.warning('At least two customers are required.');return
        k=st.slider('Clusters',2,min(8,len(customers)),min(4,len(customers)))
        segmented,scaled,model=segment_customers(customers,k)
        st.dataframe(segmented.groupby('Cluster')[CUSTOMER_FEATURES].mean().round(2),width='stretch')
        cluster=st.selectbox('Explain distance to cluster',list(range(k)))
        st.caption('Cluster IDs are arbitrary labels. SHAP below explains squared Euclidean distance to the selected center in standardized feature space: positive values mean farther away, negative values mean closer.')
        if st.button('Compute bounded SHAP and LIME explanations'):
            with st.spinner('Computing explanations for a small fixed sample…'):
                values=explain_cluster_distance(scaled,model,cluster)
                import shap
                fig,ax=plt.subplots(figsize=(8,4));shap.plots.bar(values,show=False,ax=ax);st.pyplot(fig);plt.close(fig)
                fidelity,weights,predicted=surrogate_explanation(scaled,model)
                st.metric('Held-out surrogate agreement',f'{fidelity:.1%}')
                st.caption(f'LIME explains the surrogate decision tree probability for Cluster {predicted}, not KMeans itself. Agreement measures imitation of this segmentation, not business effectiveness. Continuous perturbations can be off-distribution.')
                w=pd.DataFrame(weights,columns=['Feature condition','Weight']);st.plotly_chart(px.bar(w,x='Weight',y='Feature condition',orientation='h'),width='stretch')
        st.caption('Clustering is descriptive and fitted to this dataset. SHAP uses an interventional background and can break feature dependencies; explanations are not causal.')
    else:
        st.subheader('Monthly product revenue surface')
        d=completed_orders(data);matrix=d.pivot_table(index='Product Type',columns=pd.Grouper(key='Purchase Date',freq='MS'),values='Total Price',aggfunc='sum',fill_value=0)
        fig=go.Figure(go.Surface(z=matrix.values,x=list(range(len(matrix.columns))),y=list(range(len(matrix.index)))))
        fig.update_layout(scene={'xaxis':{'title':'Month','tickvals':list(range(len(matrix.columns))),'ticktext':[x.strftime('%Y-%m') for x in matrix.columns]},'yaxis':{'title':'Product','tickvals':list(range(len(matrix.index))),'ticktext':matrix.index.tolist()},'zaxis':{'title':'Completed revenue ($)'}})
        st.plotly_chart(fig,width='stretch')
if __name__=='__main__':main()
