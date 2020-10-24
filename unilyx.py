#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,LassoLars,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.feature_selection import RFE
import ppscore as pps
import random
import base64
pd.options.display.float_format = "{:.2f}".format
st.beta_set_page_config(page_title="Unilyx ML App", page_icon="zap", layout='centered', initial_sidebar_state='auto')
st.set_option('deprecation.showPyplotGlobalUse', False)


# In[ ]:


h1="<img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKUAAAA2CAYAAACm2Vk3AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAYESURBVHhe7ZuNVes4EEbpADqADqAD6AA6gA6gA+gAOoAOoAPogBIogRKye/MiVpl8sjWSWcd6uufM2beJo5/Rp5E0MgerTmfP6KLs7B1SlB8fH2v7TT4/P9d1fH9/bz75ewm+6PxhR5QXFxerg4ODtfHv3+Dx8fGnjpOTk9XX19fmm2VCf66urlZ3d3fuSXZ9ff3ji7Ozsz5J/2VLlK+vrz8OCsZnU2PruL+/33yzPGJRYUdHR9mTjOfi32LPz8+bb/9etkSJOKyTphbM+/v7Th3n5+ebb5cFy67tC4ZQc1C+WPIEnYouygpUXzz96aLUdFFWoJZfjD1mDl2Umi7KSuJDG3Z6epp9WOmi1HRRTgB7S8RJ3zx0UWq6KGeki1LTRTkjXZSaLsoZ6aLUNCdKDhkvLy/rGxZuSEIdJLW5oeLWhT3glHAKp05vubWifHt7W9tU5JTHZQrP1N480fdUXc2IEic9PDysxWfLV4ZAaUstCDGu8/b2dvPNODWijK+Db25uNp+WwyQO5aWuOxmnsWdyuLy8/ClHtb0JUSKMOCp6DCHXgAhtmbmUilL9ruZ6UuVbVa7VPlPyboTSmGXxorSRqsRqIk0cPYLlUipKIpT9HS+2lGLv7zH1zoPqq8d3qr/Hx8ebb/9j0aJkhg8Jkg5TdjD1TLDSfqpycykVJaixKomWSuBKKEAAODw83Hk+p17qUWOlxL9oUXJ7YssK5alDB46hP8qxGG3zMpco6YvtR0m09IpbvUmGjflO+Sn14spiRYnjbDlYzr0zglWCZl/qZS5RgldQChu9EPoYyveUowIBqHbi/xSLFSVLjC3Hc/Jl6VcR0/v+6JyirI2WSly59at9qDqRqz7SZvyfYpGiZEbaMlL7oCHUoOS+CxmYU5Sgxiw3WiJg+1srqiFU3+MTOWXl7iNjFilK+2YO5l22AjbS4EQPc4sS7KqREy1V3d4JiejUNiicyJVvclazRYpStXNoORgiTuQG87APolQRf2ySxsn3YCU+TJ3IVfm547xIUdYKKUb1uXYJy2UqUYInWiK++FnMOwYxajtlDeHm+nWRoqwRgkX1mTbmsi+i9ERLdUjx9Fmh6o/NU34XZSOiBLu/U9GSaBU/g5UcEhWpvLEnKwJbHlQHiP9DlN5Gd1FqVHk2Wqr+ju0/c1DlBuPw6NmvbnlwCsGMoW4EvAPRRZnGtsdGy5Jk+RipW57YPG8VbXlQbVhz0gse1H7GO1O7KNOoMoN/1b6vtj40o3KR6nKD1+Ny2PGgOt57BmkIZorqgDcV0UU5TCpa8t/4c8yTabDwW/XKINtAvlPCdOcpIff6qASSqrbsoTvQFK2IEp/a304hSrXiqXH1JsstKjUXl5nKYY6tjDseVDksrFaY6hCFlWyyWxEl2N/i5ylQIrRWkiwPKL8RYKxGSt4qkh4kxKqCcBjq90Aj+bsYVV5JlISWRKnSKJ76U6SCSzDaXYoSGhExJXK1l2Ubl9JS0oOpnBOW88dXNPDp6UnuY7ChTozRkijVCsKAsdXhTzXwYekKNRQtS4XPuKtzwVh5qi2p1TfpQR4eEiZG47jj5FSFAzH+n8rU88EQpDfixrQkSvys9l2x5Z5aLaloWZMsV5pgYuWgfBW/VRQY9CAOUwXVWK0goSVRglrerJWitmIl+3hQ7fQcllKBzvo7q7cM3NhszjE6oMK1F+vompmvlk8PVtT4qYQhYdb0z26fStsHtq8IzDueBCSrJft+ZfYIUDmNUrmnMUOMpXsYBW0J7aCDtlNe4miXuxQF4tlPW0qjEOAjm2ahbaUrixI6Y1hK7Hf6XNquWJj0z1K0LlAog0fEolBrOJbOIxbvTPJAO6Yqn7JKD177inqn8TfHYyrKNyudvYaoawXp2f/NSRdlo6gUzFJWgi7KBlGpILZVS6GLskFUGmjKg+Zv00XZGBxk7I1LTUppDrooG0PlXWvSVHPQRdkYUybL56KLsiHU2zs1yfK56KJsCLV0LyFZbumibAgESOonCJJT+BLpomwQrkyxpdJF2dk7uig7e0cXZWfPWK3+AYeY9rX452rNAAAAAElFTkSuQmCC' class='img-fluid'>"
st.markdown(h1,unsafe_allow_html=True)
#st.title('Unilyx')
st.subheader(f":zap: A blazingly fast, intuitive ML app")
credits = 'Created with :heart: by <a href="https://www.linkedin.com/in/sohamkulkarni98/" target="_blank">Soham Kulkarni</a>'
st.markdown(credits,unsafe_allow_html=True)
my_placeholder=st.empty()
intro_text = """     
             Unilyx is an interactive app that enables on the fly data exploration
             and lets you run various ML models within a browser. It runs a python backend 
             and employs streamlit API for ease of use.

             The best thing about Unilyx is that you don't need any coding skills/prior ML experience. Simply upload your data, select the variables and you're good to go!
             """
my_placeholder.info(intro_text)
my_placeholder_1=st.empty()
usage_guidelines="""
<details>
<summary><b>Usage Guidelines</b></summary>

**Kindly reload the tab once before uploading a file. (Right-click > Reload)**

- File Format: The app supports only .csv format. Max upload size is capped at 200 MB.

- Left Panel: Left Panel will populate after the upload is complete. While the app is capable of simultaneously executing the multiple left panel selections, using them one-by-one will be memory-efficient and will generally yield faster results.

- Visualization: Most of the plots are interactive. Click on the upper right corner
to view in fullscreen. To save a plot: Right-click > Save image as.

- Running a model: Make sure to choose the correct set of predictors and label before running a model. Only NUMERICAL predictors can be selected while the label can either be numeric or string. The model will run only if the data doesn't contain NULL/Nan values.

- Saving model results: Every screen that runs a model has a 'Download CSV file' option at the bottom (right-click and save as <some_name>.csv). Please note that download functionality currently works only for relatively small files.

</details>
"""
about_pps="""
<details>
<summary><b>Predictive Power Score</b></summary>

**The Predictive Power Score (PPS) finds every relationship that the correlation can — and more. Thus, the PPS matrix can be used to detect and understand linear or nonlinear patterns in your data.**

- The score always ranges from 0 to 1 and is data-type agnostic.

- A score of 0 means that the column x cannot predict the column y better than a naive baseline model.

- A score of 1 means that the column x can perfectly predict the column y given the model.

- A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.

- For more information on PPS, please visit-   https://pypi.org/project/ppscore/0.0.2/#calculation-of-the-pps

</details>
"""
path_warning = ":exclamation: Export unsuccessful. Make sure you've entered the correct path and file name"
my_placeholder_1.markdown(usage_guidelines, unsafe_allow_html=True)
my_placeholder_2=st.empty()
my_placeholder_2.markdown("")


# In[ ]:


graph_loding_messages = ["Please hang around for a while. The Elders of the Internet are contemplating your request....",
                         "Please wait.... let the magic potion boil.",
                         ":rainbow: Specialis Revelio....",
                         "Please wait while the little elves do the job you requested :sparkles: :sparkles:",
                         ":hammer_and_wrench: Yes there really are magic elves with an abacus working frantically in here, give'em some time",
                         "Please wait.... Spinning the wheel of fortune....",
                         "Patience! This is difficult, you know... :wink:"
                         "Please wait.... convincing AI not to turn evil....",
                         "Discovering new ways of making you wait....",
                         "Well.... Still loads faster than the Windows update :stuck_out_tongue_winking_eye:",
                         "Just a moment.... I swear it's almost done.",
                         "Baking cake...er...I mean loading, yeah loading...",
                         ":zzz: :zzz:......... Oh shit, you were waiting for me to do something? Oh okay, well then.",
                         "Please wait.... Entangling superstrings"]

actual_graph_loding_messages = ["Please wait while the little elves draw the plot you requested :bar_chart: :chart_with_upwards_trend: :chart_with_downwards_trend:",
                               ":hammer_and_wrench: Yes there really are magic elves with an abacus working frantically in here, give'em some time",
                               "Loading.... Crossing t's and dotting i's :bar_chart: :chart_with_upwards_trend: :chart_with_downwards_trend:",
                               "Feel free to spin in your chair while the plot loads"]

download_note="""
                <small><i>
                 Note: Download works only for relatively small files
                </i></small>
              """

pred_label_same_warning = ':exclamation: predictors and label cannot be same. Please check your selections.'


# In[ ]:


uploaded_file = st.file_uploader(label="Upload a data file",type="csv")
if uploaded_file is not None:
    my_placeholder.empty()
    my_placeholder_1.empty()
    my_placeholder_2.empty()
    @st.cache 
    def get_data():
        return pd.read_csv(uploaded_file)
    df = get_data()
    rowcol = f"The uploaded file has {df.shape[0]} rows and {df.shape[1]} columns."
    st.write(rowcol)
    default_pred_list=list(df.select_dtypes(include='number').columns)
    predictors=st.sidebar.multiselect('Choose the NUMERICAL predictors (independent variables)',options=default_pred_list,default=default_pred_list)
    label=st.sidebar.selectbox('Choose the label (target/dependent variable)',df.columns)
    Numerical_Columns=list(df.select_dtypes(include='number').columns)
    Categorical_Columns=list(df.select_dtypes(include='object').columns)
    Null_Checker=sum(df.isnull().sum())
    pred_list = list(predictors)
    pred_label_checker = [value for value in pred_list if value in label] 
    if st.sidebar.checkbox('Run data validation checks'):
        if len(pred_label_checker)==0:
            if st.checkbox('Show the complete data'):
                with st.spinner(':exclamation: Please wait. This may take a while to load'):
                    st.dataframe(df)
                st.markdown("---")        
            if st.checkbox('Check the data type of the columns'):
                num_col=list(df.select_dtypes(include='number').columns)
                data_types=pd.DataFrame(df.dtypes,columns=['data type'])
                data_types['variable type']=data_types['data type'].apply(lambda x: 'Numeric' if x in 
                                     ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64','float_', 'float16', 'float32', 'float64'] else 'Category' if x == 'object' else 'Datetime' if x == 'datetime64' else '')
                st.write('Data type of the columns',data_types)
                st.markdown("---") 
            if st.checkbox('Check if the data contains any Null/Nan values'):
                if Null_Checker == 0:
                    st.markdown("**The uploaded data doesn't contain any NULL/Nan values**")
                elif Null_Checker > 0:
                    st.write("Null count per column")
                    null_count_col=pd.DataFrame(df.isnull().sum(),columns=['Null Count'])
                    st.write(null_count_col.sort_values(by='Null Count',ascending=False))
                    st.markdown("") 
                    st.markdown("")
                    st.write('Red bars indicate location of the Null values, might not be visible for relatively small number of Nulls')
                    plt.subplots_adjust(left=0.1,bottom=0.35)
                    sns.heatmap(df.isnull(),cbar=False,cmap='YlOrRd').set(title='Location of Null values',)
                    st.pyplot()
                else:
                    pass
        else:
            st.warning(pred_label_same_warning)        
    if st.sidebar.checkbox('Perform exploratory data analysis'):
        if len(pred_label_checker)==0:
            st.markdown("")
            st.write('Basic statistical summary',df.describe().transpose())
            st.markdown("")
            st.markdown("")
            if st.checkbox('Pivot table'):
                ind=st.multiselect('index',df.columns,default=None)
                col=st.multiselect('column',df.columns,default=None)
                val=st.multiselect('values',df.columns,default=None)
                if val is not None:
                    agg=st.selectbox('aggregation function',['average','sum','min','max'])
                    if agg =='average':
                        agg1=np.mean
                    elif agg == 'sum':
                        agg1=np.sum
                    elif agg == 'min':
                        agg1=np.min
                    else:
                        agg1=np.max
                if st.checkbox('Display the pivot table'):
                    pivot=pd.pivot_table(df, values=val, index=ind,columns=col,aggfunc=agg1,margins=True,margins_name='Total')
                    st.dataframe(pivot)
                st.markdown("---")
            if st.checkbox('Correlation'):
                st.write('Correlation Table',df.corr())
                plt.figure(figsize=(15,10))
                plt.title('Correlation Heatmap')
                plt.xticks(rotation=45)
                plt.yticks(rotation=45)
                plt.subplots_adjust(left=0.3,bottom=0.35)
                sns.heatmap(df.corr(),annot=True,cmap='Blues',linewidths=0.1,vmax=1.0,linecolor='white')
                st.pyplot()
                st.markdown("---")
            if st.checkbox('Predictive Power Score'):
                st.markdown(about_pps, unsafe_allow_html=True)
                with st.spinner("Well... This is hard, you know... This may take a while"):
                    pps_fig=round(pps.matrix(df),4)
                    plt.figure(figsize=(round(len(df.columns)*1.25,1),round(len(df.columns)*0.8,1)))
                    plt.subplots_adjust(left=0.3,bottom=0.35)
                    sns.heatmap(pps_fig,cmap='Blues',annot=True)
                    st.pyplot()
                    st.markdown("---")
            if st.checkbox('Visualizations'):
                if Null_Checker>0:
                    st.warning(":exclamation: Some visualizations may throw an error since the data contains Null/Nan values. It is recommanded to remove/impute those values for accurate results")
                else:
                    pass
                graph_options=['Parallel Category Diagram','Box plot','Histogram','Count plot','Scatter plot','Scatter matrix','Joint plot','Categorical plot','Line chart','Bar plot']
                graph_options.sort()
                graph=st.radio('Select an option:',graph_options)
                if graph=='Histogram':
                    xhist=st.selectbox('variable',df.columns)
                    if st.checkbox('Display the Histogram'):
                        n_bins=st.number_input('number of bins',min_value=10,max_value=100,step=5)
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            hist=px.histogram(df, x=xhist, nbins=n_bins)
                            st.plotly_chart(hist)
                elif graph == 'Parallel Category Diagram':
                    if st.checkbox('Display the Parallel Category Diagram'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            if label in Numerical_Columns:
                                cat_color=label
                            else:
                                ne=LabelEncoder()
                                numlabelcolor=ne.fit_transform(df[label].astype('str'))
                                cat_color=numlabelcolor
                        pcatdim=px.parallel_categories(data_frame=df,dimensions=Categorical_Columns,color=cat_color,labels={"color": label})
                        st.plotly_chart(pcatdim)            
                elif graph == 'Box plot':
                    ybox=st.selectbox('variable',Numerical_Columns)
                    if st.checkbox('Display the Box plot'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            if len(df[label].unique())<6:
                                if st.checkbox('Split by label'):
                                    box_color=label
                                else:
                                    box_color=None
                            else:
                                box_color=None
                            boxplot=px.box(df, y=ybox, color=box_color)
                            st.plotly_chart(boxplot)            
                elif graph == 'Count plot':
                    xcountplot=st.selectbox('variable',df.columns)
                    if st.checkbox('Display the count plot'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            sns.countplot(x=xcountplot,data=df,hue=label)
                            st.pyplot()
                elif graph=='Scatter plot':
                    xscatter=st.selectbox('x variable',df.columns)
                    yscatter=st.selectbox('y variable',df.columns)
                    if st.checkbox('Display the scatter plot'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            scatterchart=px.scatter(data_frame=df,x=xscatter,y=yscatter,color=label,hover_data=predictors)
                            st.plotly_chart(scatterchart)
                elif graph=='Scatter matrix':
                    scatter_width=len(pred_list)*250
                    if st.checkbox('Display the scatter matrix'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            scattermatrics=px.scatter_matrix(data_frame=df,dimensions=pred_list,color=label,hover_data=predictors,width=scatter_width,height=scatter_width)
                            scattermatrics.update_layout(font=dict(family="Arial",size=8,color="#0E0D0D"))    
                            st.plotly_chart(scattermatrics)            
                elif graph=='Categorical plot':
                    xcatplot=st.selectbox('categorical variable',df.columns)
                    ycatplot=st.selectbox('other variable',df.columns)
                    if st.checkbox('Display the categorical plot'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            sns.catplot(x=xcatplot, y=ycatplot, kind="swarm",data=df,hue=label)
                            st.pyplot()
                elif graph=='Line chart':
                    xline=st.selectbox('x variable',df.columns)
                    yline=st.selectbox('y variable',df.columns)
                    if st.checkbox('Display the line chart'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            linechart=px.line(data_frame=df,x=xline,y=yline,color=label,hover_data=predictors)
                            st.plotly_chart(linechart)
                elif graph=='Bar plot':
                    xbar=st.selectbox('x variable',df.columns)
                    ybar=st.selectbox('y variable',df.columns)
                    if st.checkbox('Display the bar chart'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            barchart=px.bar(data_frame=df,x=xbar,y=ybar,color=label,hover_data=predictors)
                            st.plotly_chart(barchart)
                elif graph=='Joint plot':
                    xjoint=st.selectbox('x variable',Numerical_Columns)
                    yjoint=st.selectbox('y variable',Numerical_Columns)
                    if st.checkbox('Show me the jointplots'):
                        with st.spinner(random.choice(actual_graph_loding_messages)):
                            sns.jointplot(x=xjoint,y=yjoint,data=df)
                            st.pyplot()
                else:
                    st.write('please make a selection')
        else:
            st.warning(pred_label_same_warning)          
    if st.sidebar.checkbox('Run ML models'):
        if len(pred_label_checker)==0:
            X=df[predictors]
            y=df[label]
            Null_Checker_Model=sum(X.isnull().sum())
            null_df_model=pd.DataFrame(X.isnull().sum(),columns=['Null Count'])
            if Null_Checker_Model == 0:
                choice=st.radio('Make a selection:',['Split the existing file into train/test and run the model','Use a different validation file','Make predictions for file with unknown values/classes'])
                if choice == 'Split the existing file into train/test and run the model':
                    st.info('The default train/test split is 70/30')
                    modeltype = st.selectbox('Choose a model type:',['Regression','Classification'])
                    if modeltype == 'Regression':
                        regression_model=st.selectbox("Choose a regression model",['Linear Regression','Ridge Regression','LassoLARS'])
                        if regression_model == 'Linear Regression':
                            if st.checkbox("Run the Linear Regression"):
                                X=df[predictors]
                                y=df[label]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98)
                                scaler=StandardScaler()
                                X_train1=scaler.fit_transform(X_train)
                                X_test1=scaler.transform(X_test)
                                linr=LinearRegression()
                                linr.fit(X_train1,y_train)
                                prediction_linr=linr.predict(X_test1)
                                if st.checkbox('Print Coeffecient'):
                                    coeffecients = pd.DataFrame(linr.coef_,X_train.columns)
                                    coeffecients.columns = ['Coeffecient']
                                    st.write(coeffecients)
                                if st.checkbox('Evaluate Results'):
                                    st.write('Mean Aboslute Error:', round(metrics.mean_absolute_error(y_test, prediction_linr),2))
                                    st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, prediction_linr),2))
                                    st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, prediction_linr)),2))
                                    scatter_linr=px.scatter(x=y_test,y=prediction_linr,title="Prediction vs Actual",labels={'x':'Actual', 'y':'Prediction'})
                                    st.plotly_chart(scatter_linr)
                                if st.checkbox('Display & Export Results'):
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_linr
                                    st.dataframe(X_test_new)
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        elif regression_model == 'Ridge Regression':
                            if st.checkbox("Run the Ridge Regression"):
                                X=df[predictors]
                                y=df[label]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98)
                                scaler=StandardScaler()
                                X_train1=scaler.fit_transform(X_train)
                                X_test1=scaler.transform(X_test)
                                alpha_value=st.number_input(label='alpha value',min_value=0.05,max_value=0.95,value=0.5,step=0.1)
                                rr=Ridge(alpha=alpha_value)
                                rr.fit(X_train1,y_train)
                                prediction_rr=rr.predict(X_test1)
                                if st.checkbox('Print Coeffecient'):
                                    coeffecients = pd.DataFrame(rr.coef_,X_train.columns)
                                    coeffecients.columns = ['Coeffecient']
                                    st.write(coeffecients)
                                if st.checkbox('Evaluate Results'):
                                    st.write('Mean Aboslute Error:', round(metrics.mean_absolute_error(y_test, prediction_rr),2))
                                    st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, prediction_rr),2))
                                    st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, prediction_rr)),2))
                                    scatter_rr=px.scatter(x=y_test,y=prediction_rr,title="Prediction vs Actual",labels={'x':'Actual', 'y':'Prediction'})
                                    st.plotly_chart(scatter_rr)
                                if st.checkbox('Display & Export Results'):
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_rr
                                    st.dataframe(X_test_new)  
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        elif regression_model == 'LassoLARS':
                            if st.checkbox("Run the LassoLARS"):
                                X=df[predictors]
                                y=df[label]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98)
                                scaler=StandardScaler()
                                X_train1=scaler.fit_transform(X_train)
                                X_test1=scaler.transform(X_test)
                                alpha_value=st.number_input(label='alpha value',min_value=0.05,max_value=0.95,value=0.5,step=0.1)
                                ll=LassoLars(alpha=alpha_value)
                                ll.fit(X_train1,y_train)
                                prediction_ll=ll.predict(X_test1)
                                if st.checkbox('Print Coeffecient'):
                                    coeffecients = pd.DataFrame(ll.coef_,X_train.columns)
                                    coeffecients.columns = ['Coeffecient']
                                    st.write(coeffecients)
                                if st.checkbox('Evaluate Results'):
                                    st.write('Mean Aboslute Error:', round(metrics.mean_absolute_error(y_test, prediction_ll),2))
                                    st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, prediction_ll),2))
                                    st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, prediction_ll)),2))
                                    scatter_ll=px.scatter(x=y_test,y=prediction_ll,title="Prediction vs Actual",labels={'x':'Actual', 'y':'Prediction'})
                                    st.plotly_chart(scatter_ll)
                                if st.checkbox('Display & Export Results'):
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_ll
                                    st.dataframe(X_test_new)  
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        else:
                            pass
                    else:
                        classification_model=st.selectbox("Choose a classification model",['RandomForestClassifier','Logistic Regression','Support Vector Machine','Naive Bayes Classifier','Multilayer perceptron'])
                        if classification_model == 'RandomForestClassifier':
                            if st.checkbox("Run the RandomForestClassifier and evaluate the results"):
                                with st.spinner(random.choice(graph_loding_messages)):
                                    X=df[predictors]
                                    y=df[label]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98, stratify=y)
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    rf=RandomForestClassifier()
                                    rf.fit(X_train1,y_train)
                                    prediction_rf=rf.predict(X_test1)
                                    cm_rf=confusion_matrix(y_test,prediction_rf)
                                    cr_rf = classification_report(y_test,prediction_rf, output_dict=True)
                                    report_rf = pd.DataFrame(cr_rf).transpose()
                                    st.write("Classification Report",report_rf)
                                    acc_rf=accuracy_score(y_test,prediction_rf)
                                    st.write('Accuracy Score:',round(acc_rf,2))
                                    if len(df[label].unique())==2:
                                        TP = cm_rf[1][1]
                                        TN = cm_rf[0][0]
                                        FP = cm_rf[0][1]
                                        FN = cm_rf[1][0]
                                        Sensitivity = (TP/float(TP + FN))
                                        Specificity = (TN/float(TN + FP))
                                        st.write('Sensitivity:',round(Sensitivity,2))
                                        st.write('Specificity:',round(Specificity,2))
                                    else:
                                        pass
                                    st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_rf)
                                    plt.title('Confusion Matrix')
                                    sns.heatmap(cm_rf,annot=True,cmap='Blues',fmt="d")
                                    plt.ylabel('True class')
                                    plt.xlabel('Predicted class')
                                    st.pyplot()
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_rf
                                    st.dataframe(X_test_new)
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        elif classification_model == 'Logistic Regression':
                            if st.checkbox("Run the Logistic Regression and evaluate the results"):
                                with st.spinner(random.choice(graph_loding_messages)):
                                    X=df[predictors]
                                    y=df[label]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98,stratify=y)
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    lr=LogisticRegression()
                                    lr.fit(X_train1,y_train)
                                    prediction_lr=lr.predict(X_test1)
                                    cm_lr=confusion_matrix(y_test,prediction_lr)
                                    cr_lr = classification_report(y_test,prediction_lr, output_dict=True)
                                    report_lr = pd.DataFrame(cr_lr).transpose()
                                    st.write("Classification Report",report_lr)
                                    acc_lr=accuracy_score(y_test,prediction_lr)
                                    st.write('Accuracy Score:',round(acc_lr,2))
                                    if len(df[label].unique())==2:
                                        TP = cm_lr[1][1]
                                        TN = cm_lr[0][0]
                                        FP = cm_lr[0][1]
                                        FN = cm_lr[1][0]
                                        Sensitivity = (TP/float(TP + FN))
                                        Specificity = (TN/float(TN + FP))
                                        st.write('Sensitivity:',round(Sensitivity,2))
                                        st.write('Specificity:',round(Specificity,2))
                                    else:
                                        pass
                                    st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_lr)
                                    plt.title('Confusion Matrix')
                                    sns.heatmap(cm_lr,annot=True,cmap='Blues',fmt="d")
                                    plt.ylabel('True class')
                                    plt.xlabel('Predicted class')
                                    st.pyplot()
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_lr
                                    st.dataframe(X_test_new)
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        elif classification_model == 'Support Vector Machine':
                            if st.checkbox("Run the Support Vector Classifier and evaluate the results"):
                                with st.spinner(random.choice(graph_loding_messages)):
                                    X=df[predictors]
                                    y=df[label]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98,stratify=y)
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    svm = SVC()
                                    svm.fit(X_train1,y_train)
                                    prediction_svm=svm.predict(X_test1)
                                    cm_svm=confusion_matrix(y_test,prediction_svm)
                                    cr_svm = classification_report(y_test,prediction_svm, output_dict=True)
                                    report_svm = pd.DataFrame(cr_svm).transpose()
                                    st.write("Classification Report",report_svm)
                                    acc_svm=accuracy_score(y_test,prediction_svm)
                                    st.write('Accuracy Score:',round(acc_svm,2))
                                    if len(df[label].unique())==2:
                                        TP = cm_svm[1][1]
                                        TN = cm_svm[0][0]
                                        FP = cm_svm[0][1]
                                        FN = cm_svm[1][0]
                                        Sensitivity = (TP/float(TP + FN))
                                        Specificity = (TN/float(TN + FP))
                                        st.write('Sensitivity:',round(Sensitivity,2))
                                        st.write('Specificity:',round(Specificity,2))
                                    else:
                                        pass
                                    st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_svm)
                                    plt.title('Confusion Matrix')
                                    sns.heatmap(cm_svm,annot=True,cmap='Blues',fmt="d")
                                    plt.ylabel('True class')
                                    plt.xlabel('Predicted class')
                                    st.pyplot()
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_svm
                                    st.dataframe(X_test_new)
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        elif classification_model == 'Naive Bayes Classifier':
                            if st.checkbox("Run the Naive Bayes Classifier and evaluate the results"):
                                with st.spinner(random.choice(graph_loding_messages)):
                                    X=df[predictors]
                                    y=df[label]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98,stratify=y)
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    nbc = GaussianNB()
                                    nbc.fit(X_train1,y_train)
                                    prediction_nbc=nbc.predict(X_test1)
                                    cm_nbc=confusion_matrix(y_test,prediction_nbc)
                                    cr_nbc = classification_report(y_test,prediction_nbc, output_dict=True)
                                    report_nbc = pd.DataFrame(cr_nbc).transpose()
                                    st.write("Classification Report",report_nbc)
                                    acc_nbc=accuracy_score(y_test,prediction_nbc)
                                    st.write('Accuracy Score:',round(acc_nbc,2))
                                    if len(df[label].unique())==2:
                                        TP = cm_nbc[1][1]
                                        TN = cm_nbc[0][0]
                                        FP = cm_nbc[0][1]
                                        FN = cm_nbc[1][0]
                                        Sensitivity = (TP/float(TP + FN))
                                        Specificity = (TN/float(TN + FP))
                                        st.write('Sensitivity:',round(Sensitivity,2))
                                        st.write('Specificity:',round(Specificity,2))
                                    else:
                                        pass
                                    st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_nbc)
                                    plt.title('Confusion Matrix')
                                    sns.heatmap(cm_nbc,annot=True,cmap='Blues',fmt="d")
                                    plt.ylabel('True class')
                                    plt.xlabel('Predicted class')
                                    st.pyplot()
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_nbc
                                    st.dataframe(X_test_new)
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        elif classification_model == 'Multilayer perceptron':
                            if st.checkbox("Run the Multilayer perceptron and evaluate the results"):
                                with st.spinner(random.choice(graph_loding_messages)):
                                    st.warning(':point_right: This is a neural network based classifier. The parameters must be correctly chosen to get the relevent results.')
                                    X=df[predictors]
                                    y=df[label]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98,stratify=y)
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    nn=st.number_input(label='Number of neutrons', min_value=1,value=X_train.shape[1],max_value=X_train.shape[1]*2, step=1)
                                    epochs=st.number_input(label='Number of epochs', min_value=100, max_value=50000, step=100)
                                    mlp = MLPClassifier(hidden_layer_sizes=(nn,nn,nn), activation='relu', solver='adam', max_iter=epochs)
                                    mlp.fit(X_train1,y_train)
                                    prediction_mlp = mlp.predict(X_test1)
                                    cm_mlp=confusion_matrix(y_test,prediction_mlp)
                                    cr_mlp = classification_report(y_test,prediction_mlp, output_dict=True)
                                    report_mlp = pd.DataFrame(cr_mlp).transpose()
                                    st.write("Classification Report",report_mlp)
                                    acc_mlp=accuracy_score(y_test,prediction_mlp)
                                    st.write('Accuracy Score:',round(acc_mlp,2))
                                    if len(df[label].unique())==2:
                                        TP = cm_mlp[1][1]
                                        TN = cm_mlp[0][0]
                                        FP = cm_mlp[0][1]
                                        FN = cm_mlp[1][0]
                                        Sensitivity = (TP/float(TP + FN))
                                        Specificity = (TN/float(TN + FP))
                                        st.write('Sensitivity:',round(Sensitivity,2))
                                        st.write('Specificity:',round(Specificity,2))
                                    else:
                                        pass
                                    st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_mlp)
                                    plt.title('Confusion Matrix')
                                    sns.heatmap(cm_mlp,annot=True,cmap='Blues',fmt="d")
                                    plt.ylabel('True class')
                                    plt.xlabel('Predicted class')
                                    st.pyplot()
                                    st.markdown('**Results**')
                                    st.write('Last column contains the predicted values')
                                    X_test_new=X_test
                                    X_test_new['True Values']=y_test
                                    X_test_new['Predicted Values']=prediction_mlp
                                    st.dataframe(X_test_new)
                                    exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                        else:
                            st.write("Make a selection")
                elif choice == 'Use a different validation file':
                    test_file = st.file_uploader('Choose a new validation file. The existing file will be treated as train data', type="csv")
                    if test_file is not None:
                        @st.cache
                        def get_validation_data():
                            return pd.read_csv(test_file)
                        test_df = get_validation_data()
                        
                        trainrowcol=f"The train data has {df.shape[0]} rows and {df.shape[1]} columns."
                        testrowcol = f"The test data has {test_df.shape[0]} rows and {test_df.shape[1]} columns."
                        st.write(trainrowcol)
                        st.write(testrowcol)
                        modeltype = st.selectbox('Choose a model type:',['Regression','Classification'])
                        if modeltype == 'Regression':
                            regression_model=st.selectbox("Choose a regression model",['Linear Regression','Ridge Regression','LassoLARS'])
                            if regression_model == 'Linear Regression':
                                if st.checkbox("Run the Linear Regression"):
                                    X_train=df[predictors]
                                    X_test=test_df[predictors]
                                    y_train=df[label]
                                    y_test=test_df[label]
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    linr=LinearRegression()
                                    linr.fit(X_train1,y_train)
                                    prediction_linr=linr.predict(X_test1)
                                    if st.checkbox('Print Coeffecient'):
                                        coeffecients = pd.DataFrame(linr.coef_,X_train.columns)
                                        coeffecients.columns = ['Coeffecient']
                                        st.write(coeffecients)
                                    if st.checkbox('Evaluate Results'):
                                        st.write('Mean Aboslute Error:', round(metrics.mean_absolute_error(y_test, prediction_linr),2))
                                        st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, prediction_linr),2))
                                        st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, prediction_linr)),2))
                                        scatter_linr=px.scatter(x=y_test,y=prediction_linr,title="Prediction vs Actual",labels={'x':'Actual', 'y':'Prediction'})
                                        st.plotly_chart(scatter_linr)
                                    if st.checkbox('Display & Export Results'):
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_linr
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            X_test_new.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                            elif regression_model == 'Ridge Regression':
                                if st.checkbox("Run the Ridge Regression"):
                                    X_train=df[predictors]
                                    X_test=test_df[predictors]
                                    y_train=df[label]
                                    y_test=test_df[label]
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    alpha_value=st.number_input(label='alpha value',min_value=0.05,max_value=0.95,value=0.5,step=0.1)
                                    rr=Ridge(alpha=alpha_value)
                                    rr.fit(X_train1,y_train)
                                    prediction_rr=rr.predict(X_test1)
                                    if st.checkbox('Print Coeffecient'):
                                        coeffecients = pd.DataFrame(rr.coef_,X_train.columns)
                                        coeffecients.columns = ['Coeffecient']
                                        st.write(coeffecients)
                                    if st.checkbox('Evaluate Results'):
                                        st.write('Mean Aboslute Error:', round(metrics.mean_absolute_error(y_test, prediction_rr),2))
                                        st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, prediction_rr),2))
                                        st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, prediction_rr)),2))
                                        scatter_rr=px.scatter(x=y_test,y=prediction_rr,title="Prediction vs Actual",labels={'x':'Actual', 'y':'Prediction'})
                                        st.plotly_chart(scatter_rr)
                                    if st.checkbox('Display & Export Results'):
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_rr
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)   
                            elif regression_model == 'LassoLARS':
                                if st.checkbox("Run the LassoLARS"):
                                    X_train=df[predictors]
                                    X_test=test_df[predictors]
                                    y_train=df[label]
                                    y_test=test_df[label]
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    alpha_value=st.number_input(label='alpha value',min_value=0.05,max_value=0.95,value=0.5,step=0.1)
                                    ll=LassoLars(alpha=alpha_value)
                                    ll.fit(X_train1,y_train)
                                    prediction_ll=ll.predict(X_test1)
                                    if st.checkbox('Print Coeffecient'):
                                        coeffecients = pd.DataFrame(ll.coef_,X_train.columns)
                                        coeffecients.columns = ['Coeffecient']
                                        st.write(coeffecients)
                                    if st.checkbox('Evaluate Results'):
                                        st.write('Mean Aboslute Error:', round(metrics.mean_absolute_error(y_test, prediction_ll),2))
                                        st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, prediction_ll),2))
                                        st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, prediction_ll)),2))
                                        scatter_ll=px.scatter(x=y_test,y=prediction_ll,title="Prediction vs Actual",labels={'x':'Actual', 'y':'Prediction'})
                                        st.plotly_chart(scatter_ll)
                                    if st.checkbox('Display & Export Results'):
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_ll
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)         
                        else:
                            classification_model=st.selectbox("Choose a classification model",['RandomForestClassifier','Logistic Regression','Support Vector Machine','Naive Bayes Classifier','Multilayer perceptron'])
                            if classification_model == 'RandomForestClassifier':
                                if st.checkbox("Run the RandomForestClassifier and evaluate the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        y_test=test_df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        rf=RandomForestClassifier()
                                        rf.fit(X_train1,y_train)
                                        prediction_rf=rf.predict(X_test1)
                                        cm_rf=confusion_matrix(y_test,prediction_rf)
                                        cr_rf = classification_report(y_test,prediction_rf, output_dict=True)
                                        report_rf = pd.DataFrame(cr_rf).transpose()
                                        st.write("Classification Report",report_rf)
                                        acc_rf=accuracy_score(y_test,prediction_rf)
                                        st.write('Accuracy Score:',round(acc_rf,2))
                                        if len(df[label].unique())==2:
                                            TP = cm_rf[1][1]
                                            TN = cm_rf[0][0]
                                            FP = cm_rf[0][1]
                                            FN = cm_rf[1][0]
                                            Sensitivity = (TP/float(TP + FN))
                                            Specificity = (TN/float(TN + FP))
                                            st.write('Sensitivity:',round(Sensitivity,2))
                                            st.write('Specificity:',round(Specificity,2))
                                        else:
                                            pass
                                        st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_rf)
                                        plt.title('Confusion Matrix')
                                        sns.heatmap(cm_rf,annot=True,cmap='Blues',fmt="d")
                                        plt.ylabel('True class')
                                        plt.xlabel('Predicted class')
                                        st.pyplot()
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_rf
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Logistic Regression':
                                if st.checkbox("Run the Logistic Regression and evaluate the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        y_test=test_df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        lr=LogisticRegression()
                                        lr.fit(X_train1,y_train)
                                        prediction_lr=lr.predict(X_test1)
                                        cm_lr=confusion_matrix(y_test,prediction_lr)
                                        cr_lr = classification_report(y_test,prediction_lr, output_dict=True)
                                        report_lr = pd.DataFrame(cr_lr).transpose()
                                        st.write("Classification Report",report_lr)
                                        acc_lr=accuracy_score(y_test,prediction_lr)
                                        st.write('Accuracy Score:',round(acc_lr,2))
                                        if len(df[label].unique())==2:
                                            TP = cm_lr[1][1]
                                            TN = cm_lr[0][0]
                                            FP = cm_lr[0][1]
                                            FN = cm_lr[1][0]
                                            Sensitivity = (TP/float(TP + FN))
                                            Specificity = (TN/float(TN + FP))
                                            st.write('Sensitivity:',round(Sensitivity,2))
                                            st.write('Specificity:',round(Specificity,2))
                                        else:
                                            pass
                                        st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_lr)
                                        plt.title('Confusion Matrix')
                                        sns.heatmap(cm_lr,annot=True,cmap='Blues',fmt="d")
                                        plt.ylabel('True class')
                                        plt.xlabel('Predicted class')
                                        st.pyplot()
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_lr
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Support Vector Machine':
                                if st.checkbox("Run the Support Vector Classifier and evaluate the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        y_test=test_df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        svm = SVC()
                                        svm.fit(X_train1,y_train)
                                        prediction_svm=svm.predict(X_test1)
                                        cm_svm=confusion_matrix(y_test,prediction_svm)
                                        cr_svm = classification_report(y_test,prediction_svm, output_dict=True)
                                        report_svm = pd.DataFrame(cr_svm).transpose()
                                        st.write("Classification Report",report_svm)
                                        acc_svm=accuracy_score(y_test,prediction_svm)
                                        st.write('Accuracy Score:',round(acc_svm,2))
                                        if len(df[label].unique())==2:
                                            TP = cm_svm[1][1]
                                            TN = cm_svm[0][0]
                                            FP = cm_svm[0][1]
                                            FN = cm_svm[1][0]
                                            Sensitivity = (TP/float(TP + FN))
                                            Specificity = (TN/float(TN + FP))
                                            st.write('Sensitivity:',round(Sensitivity,2))
                                            st.write('Specificity:',round(Specificity,2))
                                        else:
                                            pass
                                        st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_svm)
                                        plt.title('Confusion Matrix')
                                        sns.heatmap(cm_svm,annot=True,cmap='Blues',fmt="d")
                                        plt.ylabel('True class')
                                        plt.xlabel('Predicted class')
                                        st.pyplot()
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_svm
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Naive Bayes Classifier':
                                if st.checkbox("Run the Naive Bayes Classifier and evaluate the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        y_test=test_df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        nbc = GaussianNB()
                                        nbc.fit(X_train1,y_train)
                                        prediction_nbc=nbc.predict(X_test1)
                                        cm_nbc=confusion_matrix(y_test,prediction_nbc)
                                        cr_nbc = classification_report(y_test,prediction_nbc, output_dict=True)
                                        report_nbc = pd.DataFrame(cr_nbc).transpose()
                                        st.write("Classification Report",report_nbc)
                                        acc_nbc=accuracy_score(y_test,prediction_nbc)
                                        st.write('Accuracy Score:',round(acc_nbc,2))
                                        if len(df[label].unique())==2:
                                            TP = cm_nbc[1][1]
                                            TN = cm_nbc[0][0]
                                            FP = cm_nbc[0][1]
                                            FN = cm_nbc[1][0]
                                            Sensitivity = (TP/float(TP + FN))
                                            Specificity = (TN/float(TN + FP))
                                            st.write('Sensitivity:',round(Sensitivity,2))
                                            st.write('Specificity:',round(Specificity,2))
                                        else:
                                            pass
                                        st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_nbc)
                                        plt.title('Confusion Matrix')
                                        sns.heatmap(cm_nbc,annot=True,cmap='Blues',fmt="d")
                                        plt.ylabel('True class')
                                        plt.xlabel('Predicted class')
                                        st.pyplot()
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_nbc
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Multilayer perceptron':
                                st.warning(':point_right: This is a neural network based classifier. The parameters must be correctly chosen to get the relevent results.')
                                if st.checkbox("Run the Multilayer perceptron and evaluate the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        y_test=test_df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        nn=st.number_input(label='Number of neutrons', min_value=1,value=X_train.shape[1],max_value=X_train.shape[1]*2, step=1)
                                        epochs=st.number_input(label='Number of epochs', min_value=100, max_value=50000, step=100)
                                        mlp = MLPClassifier(hidden_layer_sizes=(nn,nn,nn), activation='relu', solver='adam', max_iter=epochs)
                                        mlp.fit(X_train1,y_train)
                                        prediction_mlp = mlp.predict(X_test1)
                                        cm_mlp=confusion_matrix(y_test,prediction_mlp)
                                        cr_mlp = classification_report(y_test,prediction_mlp, output_dict=True)
                                        report_mlp = pd.DataFrame(cr_mlp).transpose()
                                        st.write("Classification Report",report_mlp)
                                        acc_mlp=accuracy_score(y_test,prediction_mlp)
                                        st.write('Accuracy Score:',round(acc_mlp,2))
                                        if len(df[label].unique())==2:
                                            TP = cm_mlp[1][1]
                                            TN = cm_mlp[0][0]
                                            FP = cm_mlp[0][1]
                                            FN = cm_mlp[1][0]
                                            Sensitivity = (TP/float(TP + FN))
                                            Specificity = (TN/float(TN + FP))
                                            st.write('Sensitivity:',round(Sensitivity,2))
                                            st.write('Specificity:',round(Specificity,2))
                                        else:
                                            pass
                                        st.write('Confusion Matrix: (cols- predicted class, rows- True class)',cm_mlp)
                                        plt.title('Confusion Matrix')
                                        sns.heatmap(cm_mlp,annot=True,cmap='Blues',fmt="d")
                                        plt.ylabel('True class')
                                        plt.xlabel('Predicted class')
                                        st.pyplot()
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['True Values']=y_test
                                        X_test_new['Predicted Values']=prediction_mlp
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                    else:
                        text1 = """                        Upload a validation file to proceed. The existing data file will be treated as train data.
                        Everything is run locally and the data you upload never leaves your system.
                        """
                        st.info(text1)
                else:
                    test_file = st.file_uploader('Choose a new file to make predictions. The existing file will be treated as train data', type="csv")
                    if test_file is not None:
                        @st.cache
                        def get_unknown_data():
                            return pd.read_csv(test_file)
                        test_df = get_unknown_data()
                        
                        trainrowcol=f"The train data has {df.shape[0]} rows and {df.shape[1]} columns."
                        testrowcol = f"The prediction data has {test_df.shape[0]} rows and {test_df.shape[1]} columns."
                        st.write(trainrowcol)
                        st.write(testrowcol)
                        modeltype = st.selectbox('Choose a model type:',['Regression','Classification'])
                        if modeltype == 'Regression':
                            regression_model=st.selectbox("Choose a regression model",['Linear Regression','Ridge Regression','LassoLARS'])
                            if regression_model == 'Linear Regression':
                                if st.checkbox("Run the Linear Regression"):
                                    X_train=df[predictors]
                                    X_test=test_df[predictors]
                                    y_train=df[label]
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    linr=LinearRegression()
                                    linr.fit(X_train1,y_train)
                                    prediction_linr=linr.predict(X_test1)
                                    if st.checkbox('Print Coeffecient'):
                                        coeffecients = pd.DataFrame(linr.coef_,X_train.columns)
                                        coeffecients.columns = ['Coeffecient']
                                        st.write(coeffecients)
                                    if st.checkbox('Display & Export Results'):
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_linr
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            if regression_model == 'Ridge Regression':
                                if st.checkbox("Run the Ridge Regression"):
                                    X_train=df[predictors]
                                    X_test=test_df[predictors]
                                    y_train=df[label]
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    alpha_value=st.number_input(label='alpha value',min_value=0.05,max_value=0.95,value=0.5,step=0.1)
                                    rr=Ridge(alpha=alpha_value)
                                    rr.fit(X_train1,y_train)
                                    prediction_rr=rr.predict(X_test1)
                                    if st.checkbox('Print Coeffecient'):
                                        coeffecients = pd.DataFrame(rr.coef_,X_train.columns)
                                        coeffecients.columns = ['Coeffecient']
                                        st.write(coeffecients)
                                    if st.checkbox('Display & Export Results'):
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_rr
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            if regression_model == 'LassoLARS':
                                if st.checkbox("Run the LassoLARS"):
                                    X_train=df[predictors]
                                    X_test=test_df[predictors]
                                    y_train=df[label]
                                    scaler=StandardScaler()
                                    X_train1=scaler.fit_transform(X_train)
                                    X_test1=scaler.transform(X_test)
                                    alpha_value=st.number_input(label='alpha value',min_value=0.05,max_value=0.95,value=0.5,step=0.1)
                                    ll=Ridge(alpha=alpha_value)
                                    ll.fit(X_train1,y_train)
                                    prediction_ll=ll.predict(X_test1)
                                    if st.checkbox('Print Coeffecient'):
                                        coeffecients = pd.DataFrame(ll.coef_,X_train.columns)
                                        coeffecients.columns = ['Coeffecient']
                                        st.write(coeffecients)
                                    if st.checkbox('Display & Export Results'):
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_ll
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)            
                        else:
                            classification_model=st.selectbox("Choose a classification model",['RandomForestClassifier','Logistic Regression','Support Vector Machine','Naive Bayes Classifier','Multilayer perceptron'])
                            if classification_model == 'RandomForestClassifier':
                                if st.checkbox("Run the RandomForestClassifier and display the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        rf=RandomForestClassifier()
                                        rf.fit(X_train1,y_train)
                                        prediction_rf=rf.predict(X_test1)
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_rf
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Logistic Regression':
                                if st.checkbox("Run the Logistic Regression and display the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        lr=LogisticRegression()
                                        lr.fit(X_train1,y_train)
                                        prediction_lr=lr.predict(X_test1)
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_lr
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Support Vector Machine':
                                if st.checkbox("Run the Support Vector Classifier and display the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        svm = SVC()
                                        svm.fit(X_train1,y_train)
                                        prediction_svm=svm.predict(X_test1)
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_svm
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Naive Bayes Classifier':
                                if st.checkbox("Run the Naive Bayes Classifier and display the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        nbc = GaussianNB()
                                        nbc.fit(X_train1,y_train)
                                        prediction_nbc=nbc.predict(X_test1)
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_nbc
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)
                            elif classification_model == 'Multilayer perceptron':
                                st.warning(':point_right: This is a neural network based classifier. The parameters must be correctly chosen to get the relevent results.')
                                if st.checkbox("Run the Multilayer perceptron and display the results"):
                                    with st.spinner(random.choice(graph_loding_messages)):
                                        X_train=df[predictors]
                                        X_test=test_df[predictors]
                                        y_train=df[label]
                                        scaler=StandardScaler()
                                        X_train1=scaler.fit_transform(X_train)
                                        X_test1=scaler.transform(X_test)
                                        nn=st.number_input(label='Number of neutrons', min_value=1,value=X_train.shape[1],max_value=X_train.shape[1]*2, step=1)
                                        epochs=st.number_input(label='Number of epochs', min_value=100, max_value=50000, step=100)
                                        mlp = MLPClassifier(hidden_layer_sizes=(nn,nn,nn), activation='relu', solver='adam', max_iter=epochs)
                                        mlp.fit(X_train1,y_train)
                                        prediction_mlp = mlp.predict(X_test1)
                                        st.markdown('**Results**')
                                        st.write('Last column contains the predicted values')
                                        X_test_new=X_test
                                        X_test_new['Predicted Values']=prediction_mlp
                                        st.dataframe(X_test_new)
                                        exp_path=st.text_input('Export Path','Desktop/results.xlsx')
                                        st.markdown("")
                                        if st.button('Export'):
                                            try:
                                                X_test_new.to_excel(exp_path,index=False)
                                                st.success("Export Successful!")
                                                st.write(f"Exported to  {exp_path}")
                                            except:
                                                st.error(path_warning)          
                    else:
                        text2 = """                        Upload a file to proceed for which you want to make predictions. The existing file will be treated as train data.
                        Everything is run locally and the data you upload never leaves your system.
                        
                        Obviously, you won't be able to evaluate results for this case
                        """
                        st.info(text2)
            else:
                text3 = """                        :exclamation: The model won't run since the data contains NULL/Nan values.
                        
                        """
                st.warning(text3)
                st.write(null_df_model.sort_values(by='Null Count',ascending=False))
        else:
            st.warning(pred_label_same_warning)            
    if st.sidebar.checkbox('Evaluate feature importance'):
        if len(pred_label_checker)==0:
            X=df[predictors]
            y=df[label]
            Null_Checker_Feature_Importance=sum(X.isnull().sum())
            null_df_feat_imp=pd.DataFrame(X.isnull().sum(),columns=['Null Count'])
            if Null_Checker_Feature_Importance == 0:
                if label in Numerical_Columns:
                    eval_type=st.radio("My label(target/dependent variable) has",['continuous values','discrete classes'])
                    if eval_type == 'discrete classes':
                        modeltype_0 = st.selectbox('Select a method to calculate feature importance:',['Random Forest Regressor','Recursive Feature Elimination'])
                        if modeltype_0 == 'Random Forest Regressor':
                            if st.checkbox("Find feature importance using Random Forest Regressor"):
                                with st.spinner(":hammer_and_wrench: Yes there really are magic elves with an abacus working frantically in here, give'em some time"):
                                    X=df[predictors]
                                    y=df[label]
                                    model_0 = RandomForestRegressor(random_state=98, max_depth=10)
                                    model_0.fit(X,y)
                                    features = X.columns
                                    importances = model_0.feature_importances_
                                    indices = np.argsort(importances)
                                    plt.subplots_adjust(left=0.3)
                                    plt.title('Feature Importance')
                                    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                                    plt.xlabel('Relative Importance')
                                    st.pyplot()
                        if modeltype_0 == 'Recursive Feature Elimination':
                            if st.checkbox("Find feature importance using Recursive Feature Elimination"):
                                with st.spinner(":hammer_and_wrench: Yes there really are magic elves with an abacus working frantically in here, give'em some time"):
                                    X=df[predictors]
                                    y=df[label]
                                    model_0 = RandomForestClassifier()
                                    rfe = RFE(model_0, 1)
                                    fit = rfe.fit(X, y)
                                    ad=pd.DataFrame(fit.ranking_,columns=['Rank'])
                                    af=pd.DataFrame(X.columns,columns=['Feature'])
                                    aaaa=pd.concat([af,ad],axis=1)
                                    bbbb=aaaa.sort_values(by=['Rank'])
                                    st.write('Feature ranks by relative importance')
                                    st.dataframe(bbbb)
                                    exp_path=st.text_input('Export Path','Desktop/ranks.xlsx')
                                    st.markdown("")
                                    if st.button('Export'):
                                        try:
                                            bbbb.to_excel(exp_path,index=False)
                                            st.success("Export Successful!")
                                            st.write(f"Exported to  {exp_path}")
                                        except:
                                            st.error(path_warning)
                    if eval_type == 'continuous values':
                        if st.checkbox("Find feature importance using Random Forest Regressor"):
                            with st.spinner(":hammer_and_wrench: Yes there really are magic elves with an abacus working frantically in here, give'em some time"):
                                X=df[predictors]
                                y=df[label]
                                model_0 = RandomForestRegressor(random_state=98, max_depth=10)
                                model_0.fit(X,y)
                                features = X.columns
                                importances = model_0.feature_importances_
                                indices = np.argsort(importances)
                                plt.subplots_adjust(left=0.3)
                                plt.title('Feature Importance')
                                plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                                plt.yticks(range(len(indices)), [features[i] for i in indices])
                                plt.xlabel('Relative Importance')
                                st.pyplot()
                else:
                    st.warning(" :point_right: The label(target/independent variable) needs to be **NUMERICAL** to evaluate the feature importance")                
            else:
                text4 = """                        :exclamation: The feature importance can't be evaluated since the data contains NULL/Nan values.
                        
                        """
                st.warning(text4)
                st.write(null_df_feat_imp.sort_values(by='Null Count',ascending=False))
        else:
            st.warning(pred_label_same_warning)        
    if st.sidebar.checkbox('Perform data dimensionality reduction'):
        if len(pred_label_checker)==0:
            X=df[predictors]
            y=df[label]
            Null_Checker_Dimensionality_Reduction=sum(X.isnull().sum())
            null_df_dim_red=pd.DataFrame(X.isnull().sum(),columns=['Null Count'])
            if Null_Checker_Dimensionality_Reduction == 0: 
                modeltype_1 = st.selectbox('Choose a dimensionality reduction technique:',['PCA','LDA'])
                if modeltype_1 == 'PCA':
                    if st.checkbox("Run the Principal Component Analysis and display the results"):
                        X=df[predictors]
                        y=df[label]
                        scaler = StandardScaler()
                        X_scaled=scaler.fit_transform(X)
                        max_dim=(len(X.columns)-1)
                        dim=st.number_input(label='Number of dimensions',min_value=2,max_value=max_dim,value=int(round((len(X.columns)/2),0)+1),step=1)
                        no_of_dim = f"The data will be reduced to {dim} dimensions."
                        st.markdown(no_of_dim)
                        pca = PCA(n_components=dim)
                        PrincipalComponents = pca.fit_transform(X_scaled)
                        pca_evr=pd.DataFrame(pca.explained_variance_ratio_,columns=['exp_var_ratio'])
                        st.write('explained variance ratio:',pca_evr)
                        evr_info=f"The {dim} principal components together contain {round((sum(pca.explained_variance_ratio_)*100),2)}% information of the original data. Increase the number of dimensions to retain more information."
                        st.info(evr_info)
                        col_list=[]
                        for i in range(1,(dim + 1)):
                            col_list.append('PComp_'+str(i))
                        pca_df=pd.DataFrame(data=PrincipalComponents,columns=col_list)
                        reduced_dim_df=pd.concat([pca_df,y],axis=1)
                        st.write('Download the results and use those as input data for any model you wish to run')
                        st.dataframe(reduced_dim_df)
                        exp_path=st.text_input('Export Path','Desktop/principal_components.xlsx')
                        st.markdown("")
                        if st.button('Export'):
                            try:
                                reduced_dim_df.to_excel(exp_path,index=False)
                                st.success("Export Successful!")
                                st.write(f"Exported to  {exp_path}")
                            except:
                                st.error(path_warning)
                elif modeltype_1 == 'LDA':
                    st.warning(":point_right: Make sure that the label contains **discrete values/classes**, else the LDA won't run")
                    if st.checkbox("Run the Linear Discriminant Analysis and display the results"):
                        X=df[predictors]
                        y=df[label]
                        scaler = StandardScaler()
                        X_scaled=scaler.fit_transform(X)
                        max_dim=min(len(X.columns), y.nunique() - 1)
                        dim=st.number_input(label='Number of dimensions',min_value=1,max_value=max_dim,value=max_dim,step=1)
                        no_of_dim = f"The data will be reduced to {dim} dimension(s)."
                        st.markdown(no_of_dim)
                        lda = LinearDiscriminantAnalysis(n_components=dim)
                        LDAComponents = lda.fit_transform(X_scaled,y)
                        lda_evr=pd.DataFrame(lda.explained_variance_ratio_,columns=['exp_var_ratio'])
                        st.write('explained variance ratio:',lda_evr)
                        evr_info=f"The {dim} components together contain {round((sum(lda.explained_variance_ratio_)*100),2)}% information of the original data."
                        st.info(evr_info)
                        col_list=[]
                        for i in range(1,(dim + 1)):
                            col_list.append('Comp_'+str(i))
                        lda_df=pd.DataFrame(data=LDAComponents,columns=col_list)
                        reduced_dim_df=pd.concat([lda_df,y],axis=1)
                        st.write('Download the results and use those as input data for any model you wish to run')
                        st.dataframe(reduced_dim_df)
                        exp_path=st.text_input('Export Path','Desktop/lda.xlsx')
                        st.markdown("")
                        if st.button('Export'):
                            try:
                                reduced_dim_df.to_excel(exp_path,index=False)
                                st.success("Export Successful!")
                                st.write(f"Exported to  {exp_path}")
                            except:
                                st.error(path_warning)
            else:
                text5 = """                        :exclamation: The dimensionality reduction can't be performed since the data contains NULL/Nan values.
                        
                        """
                st.warning(text5)
                st.write(null_df_dim_red.sort_values(by='Null Count',ascending=False))
        else:
            st.warning(pred_label_same_warning)        
else:
    st.info("Don't worry! Everything is run locally and the data you upload never leaves your system.")


# In[ ]:
