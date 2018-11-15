# Applying Machine Learning Algorithms on Free Music Archive Dataset

import pandas as pd
import numpy as np
from sklearn import tree
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------
# Model Evaluation

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

#-----------------------------------------------------------------------------------------------------------
# Apply Data Preprocessing
def apply_data_split_preprocessing(raw_dataset,labels):

	# Perform Basic Preprocessing and Train, Validation Split
	X_train, X_test, y_train, y_test = train_test_split(raw_dataset,labels,test_size=0.30,random_state=42)
	
	# Apply Basic Preprocessing Steps

	print 'Applying Data Preprocessing'

	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(X_train)

	X_test_minmax = min_max_scaler.fit_transform(X_test)

	return X_train_minmax, X_test_minmax, y_train, y_test


#-----------------------------------------------------------------------------------------------------------
# Apply Decision_Tree

def apply_decision_tree(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Decision Tree'

	# Training the classifier
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train_preprocessed, y_train)

	# Testing the classifier on Test Data
	y_test_pred = clf.predict(X_test_preprocessed) 

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Decision Tree Model is: ', acc

#----------------------------------------------------------------------------------------------------------
# Apply Random Forest

def apply_random_forest(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Random Forest'

	# Training the classifier
	classifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
	classifier = classifier.fit(X_train_preprocessed,y_train)
	
	# Testing the classifier on Test Data
	y_test_pred = classifier.predict(X_test_preprocessed) 

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Random ForestClassifier Model is: ', acc


#-------------------------------------------------------------------------------------------------------------
# Apply Multi-Class SVM. Make sure for probabilities we use Platts Scaling

def apply_multi_class_svm(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Multi-Class SVM'

	pass

#-------------------------------------------------------------------------------------------------------------

def main():

	echonest_full_dataset_path = '/home/ayush/FOML_Project/echonest_full_data.csv'
	echonest_full_dataset = pd.read_csv(echonest_full_dataset_path,index_col=[0,1],sep=',')
	echonest_full_dataset = echonest_full_dataset.reset_index()

	echonest_full_dataset = echonest_full_dataset.dropna(subset=['parent_genre_title','parent_genre_id','track_id'])
	echonest_full_dataset = echonest_full_dataset.dropna(axis=1,how='any')
	#print echonest_full_dataset.shape

	parent_genre_id = np.array(list(echonest_full_dataset['parent_genre_id']),dtype=np.float64)
	parent_genre_title = list(echonest_full_dataset['parent_genre_title'])
	track_id_list = list(echonest_full_dataset['track_id'])

	echonest_full_dataset = echonest_full_dataset.drop(axis=1,columns=['parent_genre_title','track_id','parent_genre_id'])
	echonest_full_dataset = echonest_full_dataset.drop(echonest_full_dataset.select_dtypes(['object']), axis=1)

	echonest_full_dataset.to_csv('fma_clean_data.csv',sep=',',index=False)

	# Store DataFrame as Numpy
	echonest_full_data_array = echonest_full_dataset.values

	# Preprocess data
	X_train_preprocessed, X_test_preprocessed, y_train, y_test = apply_data_split_preprocessing(echonest_full_data_array,parent_genre_id)

	# Apply Decision Tree
	apply_decision_tree(X_train_preprocessed,X_test_preprocessed,y_train,y_test)

	# Apply Random Forest
	apply_random_forest(X_train_preprocessed,X_test_preprocessed,y_train,y_test)


if __name__ == '__main__':
	main()