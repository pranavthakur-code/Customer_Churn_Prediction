import numpy as np, pandas as pd
rng=np.random.default_rng(42); n=3000
tenure=np.minimum(rng.gamma(1.4,22,n).astype(int),72); tenure[rng.choice(n,8,replace=False)]=0
contract=rng.choice(['Month-to-month','One year','Two year'],n,p=[.55,.21,.24])
inet=rng.choice(['DSL','Fiber optic','No'],n,p=[.34,.44,.22])
pay=rng.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],n,p=[.34,.23,.22,.21])
def svc(): 
    v=rng.choice(['Yes','No'],n,p=[.4,.6]); return np.where(inet=='No','No internet service',v)
sec,tech=svc(),svc()
base=np.select([inet=='DSL',inet=='Fiber optic'],[45,80],20)
monthly=np.clip(base+rng.normal(0,8,n)+5*(sec=='Yes')+5*(tech=='Yes'),18.25,118.75).round(2)
total=(monthly*tenure*(1+rng.normal(0,.02,n))).round(2)
senior=rng.choice([0,1],n,p=[.84,.16])
z=-1+1.2*(contract=='Month-to-month')-.8*(contract=='Two year')+.7*(inet=='Fiber optic')-.03*tenure+.01*(monthly-65)+.5*(pay=='Electronic check')+.3*senior
churn=np.where(rng.random(n)<1/(1+np.exp(-z)),'Yes','No')
df=pd.DataFrame({'customerID':[f'{i:04d}-{rng.integers(1000,9999)}' for i in range(n)],
 'gender':rng.choice(['Male','Female'],n),'SeniorCitizen':senior,'Partner':rng.choice(['Yes','No'],n),
 'tenure':tenure,'Contract':contract,'InternetService':inet,'OnlineSecurity':sec,'TechSupport':tech,
 'PaymentMethod':pay,'MonthlyCharges':monthly,'TotalCharges':total.astype(str),'Churn':churn})
df.loc[df.tenure==0,'TotalCharges']=' '
df.loc[rng.choice(n,90,replace=False),'MonthlyCharges']=np.nan
df.loc[rng.choice(n,45,replace=False),'PaymentMethod']=np.nan
df.loc[rng.choice(n,6,replace=False),'MonthlyCharges']=999.99
i=rng.choice(n,40,replace=False); df.loc[i,'Contract']=df.loc[i,'Contract'].str.lower()
i=rng.choice(n,25,replace=False); df.loc[i,'Contract']=' '+df.loc[i,'Contract']+' '
df=pd.concat([df,df.sample(35,random_state=1)],ignore_index=True).sample(frac=1,random_state=3).reset_index(drop=True)
df.to_csv('data/telco_churn_sample.csv',index=False); print(df.shape)
