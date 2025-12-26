import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

#Page Config
st.set_page_config("Simple Linear Regression",layout="centered")
#pip install streamlit seaborn scikit-learn matplotlib pandas"""
#streamlit run app_linear_regression.py

#load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
#streamlit doesnot deafultly allow html tabs but we are making true using unsafe_allowed_html
#Title
st.markdown("""
<div class="card">
            <h1>Simple Linear Regression</h1>
            <p>Predict <b> Tip amount </b> from <b> Total Bill </b> using Simple Linear Regression.</p>
            </div>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()
#Dataset Preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head(15))
st.markdown('</div>',unsafe_allow_html=True)
#Prepare Data

x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#Train Model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
#Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)
#visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"])
ax.plot(df["total_bill"],model.predict(scaler.transform(df[["total_bill"]])),color="black")
ax.set_xlabel("Total Bill($) ")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)
#Performance metrics
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader('Model Performance')
c1,c2=st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R2",f"{adj_r2:.3f}")
st.markdown('</div>',unsafe_allow_html=True)

#m and c
st.markdown(f"""
            <div class="card">
            <h3>Model Interception </h3>
            <p><b>Co-efficient:</b>{model.coef_[0]:.3f}<br>
            <b>Intercept:</b>{model.intercept_:.3f}</p>
            </div>
            """,unsafe_allow_html=True)

#Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")
bill=st.slider("Enter Total Bill Amount ($)",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)