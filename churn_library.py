# import libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)

    # Encode Churn dependent variable : 0 = Did not churn ; 1 = Churned
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Drop redundant Attrition_Flag variable (replaced by Churn response variable)
    dataframe.drop('Attrition_Flag', axis=1, inplace=True)

    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder

    input:
        dataframe: pandas dataframe

    output:
        None
    '''

    # Ensure images directory exists
    os.makedirs("./images/eda", exist_ok=True)

    # Analyze categorical features and plot distribution
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    for cat_column in cat_columns:
        plt.figure(figsize=(7, 4))
        (dataframe[cat_column]
            .value_counts('normalize')
            .plot(kind='bar', rot=45, title=f'{cat_column} - % Churn')
         )
        plt.savefig(os.path.join("./images/eda", f'{cat_column}.png'), bbox_inches='tight')
        plt.show()
        plt.close()

    # Analyze Numeric features
    num_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Churn']

    dataframe_eda_num = dataframe[num_columns] 

    plt.figure(figsize=(10, 5))
    dataframe_eda_num['Churn'].plot(kind='hist', title='Distribution Churn')
    plt.savefig(os.path.join("./images/eda", 'Churn_hist.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    dataframe_eda_num['Customer_Age'].plot(kind='hist', title='Distribution of Customer Age')
    plt.savefig(os.path.join("./images/eda", 'Customer_Age.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(dataframe_eda_num['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join("./images/eda", 'Total_Trans_Ct.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    # plot correlation matrix
    plt.figure(figsize=(15, 7))
    sns.heatmap(dataframe_eda_num.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(15, 7))
    dataframe_eda_num.plot(x='Total_Trans_Amt', y='Total_Trans_Ct', kind='scatter', title='Correlation analysis between 2 features')
    plt.savefig(os.path.join("./images/eda", 'Total_Trans_Amt_vs_Total_Trans_Ct.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category

    input:
        dataframe: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name

    output:
        dataframe: pandas dataframe with new columns for encoded categorical features
    '''

    for category in category_lst:
        category_means = dataframe.groupby(category).mean()[response]
        new_feature = category + '_' + response
        dataframe[new_feature] = dataframe[category].map(category_means)

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    Converts remaining categorical using one-hot encoding adding the response
    str prefix to new columns Then generate train and test datasets

    input:
        dataframe: pandas dataframe
        response: string of response name

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')
    y = dataframe[response]
    X = dataframe.drop(response, axis=1)
    X = X['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_classification_report(model_name, y_train, y_test, y_train_preds, y_test_preds):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
        model_name: (str) name of the model, ie 'Random Forest'
        y_train: training response values
        y_test:  test response values
        y_train_preds: training predictions from model_name
        y_test_preds: test predictions from model_name

    output:
        None
    '''
    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties='monospace')

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties='monospace')

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(os.path.join("./images/results", fig_name), bbox_inches='tight')

    # Display figure
    plt.show()
    plt.close()


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder using plot_classification_report
    helper function

    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    plot_classification_report('Logistic Regression', y_train, y_test, y_train_preds_lr, y_test_preds_lr)
    plot_classification_report('Random Forest', y_train, y_test, y_train_preds_rf, y_test_preds_rf)


def feature_importance_plot(model, X_data, model_name, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')
    plt.show()
    plt.close()


def confusion_matrix(model, model_name, X_test, y_test):
    '''
    Display confusion matrix of a model on test data

    input:
        model: trained model
        X_test: X testing data
        y_test: y testing data
    output:
        None
    '''
    class_names = ['Not Churned', 'Churned']
    plt.figure(figsize=(15, 5))
    ax = plt.gca()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, xticks_rotation='horizontal', colorbar=False, ax=ax)
    ax.grid(False)
    plt.title(f'{model_name} Confusion Matrix on test data')
    plt.savefig(os.path.join("./images/results", f'{model_name}_Confusion_Matrix.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(os.path.join("./images/results", 'ROC_curves.png'), bbox_inches='tight')
    plt.close()

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    for model, model_type in zip([cv_rfc.best_estimator_, lrc], ['Random_Forest', 'Logistic_Regression']):
        confusion_matrix(model, model_type, X_test, y_test)

    feature_importance_plot(cv_rfc.best_estimator_, X_train, 'Random_Forest', "./images/results")


if __name__ == "__main__":
    dataset = import_data("./data/bank_data.csv")
    print('Dataset successfully loaded...Now conducting data exploration')
    perform_eda(dataset)
    X_train, X_test, y_train, y_test = perform_feature_engineering(dataset, response='Churn')
    print('Start training the data...please wait')
    train_models(X_train, X_test, y_train, y_test)
    print('Training completed. Best model weights + performance metrics saved')
