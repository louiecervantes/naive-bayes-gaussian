#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    
    st.title('Naive Bayes Classifier')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
    st.write('The titanic dataset contains information about multiple \
    people like their ages, sexes, sibling counts, number of parent \
    or children companions on aboard, embarkment points and \
    whether or not they survived the disaster. Based on these \
    features, you have to predict if an arbitrary passenger on \
    Titanic would survive the sinking. The question of interest \
    for this natural dataset is how survival relates to the \
    other attributes. There is obviously no practical need to \
    predict survival, so the real interest is in interpretation, \
    but success at prediction would appear to be closely related \
    to the discovery of interesting features of the relationship. \
    To simplify the processing, the datarows with missing values \
    so this dataset is not the original dataset available at the \
    machine learning websites.')

    if st.button('Start'):
        df = pd.read_csv('data_decision_trees.csv', header=None)
        # st.dataframe(df, use_container_width=True)  
        
        # display the dataset
        st.dataframe(df, use_container_width=True)  

        #load the data and the labels
        X = df.values[:,0:-1]
        y = df.values[:,-1]          
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)

        # Create the logistic regression 
        clf = GaussianNB()

        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)

        cmNB = confusion_matrix(y_test, y_test_pred)
        st.text(cmNB)
        
        # Test the classifier on the testing set
        accuracy = clf.score(X_test, y_test_pred)
        st.write('accuracy = ' + str(accuracy))
        st.text(classification_report(y_test, y_test_pred))
        visualize_classifier(clf, X, y)

def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)

    # Create a plot
    plt.figure()

    # Specify the title
    plt.title(title)

    # Choose a color scheme for the plot 
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
    st.pyplot()
    
#run the app
if __name__ == "__main__":
    app()
