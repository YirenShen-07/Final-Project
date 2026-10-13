"""Regression checks for revenue definitions, chronology, and explanation targets."""
import unittest
import numpy as np
import pandas as pd
from analytics import clean_data,revenue_summary,customer_table,segment_customers,explain_cluster_distance

class AnalyticsTests(unittest.TestCase):
    def example(self):
        return pd.DataFrame({'Customer ID':[1,1,1],'Purchase Date':['2024-03-03','2024-03-01','2024-03-02'],
          'Order Status':['Completed','Completed','Cancelled'],'Total Price':[30,10,20],'Product Type':['Laptop']*3,
          'Rating':[3,4,1],'Age':[25]*3,'Loyalty Member':['Yes','No','No']})
    def test_cancelled_orders_are_not_revenue(self):
        r=revenue_summary(clean_data(self.example()))
        self.assertEqual(r['Recorded order value'],60)
        self.assertEqual(r['Completed merchandise revenue'],40)
        self.assertEqual(r['Cancelled order value'],20)
    def test_latest_loyalty_is_chronological(self):
        c=customer_table(clean_data(self.example()))
        self.assertEqual(c.loc[1,'Current_Loyalty'],'Yes')
        self.assertEqual(c.loc[1,'Transaction_Count'],2)
    def test_shap_explains_distance_not_numeric_cluster_ids(self):
        rng=np.random.default_rng(42)
        d=pd.DataFrame(rng.normal(size=(30,6)),columns=['Total_Spend','Transaction_Count','Avg_Rating','Age','Purchase_Frequency','Customer_Lifetime_Days'])
        _,scaled,model=segment_customers(d,2)
        values=explain_cluster_distance(scaled,model,0)
        expected=np.square(values.data-model.cluster_centers_[0]).sum(axis=1)
        np.testing.assert_allclose(values.base_values+values.values.sum(axis=1),expected,atol=1e-8)
if __name__=='__main__':unittest.main()
