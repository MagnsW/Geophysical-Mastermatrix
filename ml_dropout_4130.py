
# coding: utf-8

# Dropout statistics data analyis
# =====
# This notebook summarizes the efforts made in the analysis of statistical dropout data for the standard 4130cu. in. array with Bolt LLXT guns. Other arrays or dropout rules may easily be analyzed by saving this Jupyter Notebook with a new name and feed in the raw statistics (dan) files for the relavant source. Dropout rules are specified in this notebook under the Heading called Dropout Rule.

# ## Motivation
# Dropout specs and modeling has been a topic for discussion for years:
# The tolerance specs are sharp, resulting in a red or green box in a dropout matrix, but the matrix is based on a set of assumptions that are quite inaccurate.
# 
# <li>Source modeling code and calibration: old vs new matters</li>
# 
# <li>Temperature sensitivity</li>
# 
# <li>We are assuming a static and 100% correctly deployed source.</li>
# 
# <li>Dropouts are done on vertical farfields, even if relevant farfields are mostly non-vertical</li>
# 
# <li>Origin of specs (operational more than geophysical, “nice round numbers”).</li>
# 
# <li>Etc.</li>
# 
# Reshoots due to dropouts are costly. A revision is in line with a streamlining/cost-cutting philosophy
# There is no guarantee that a reshoot results in improved data quality
# Current practice of bespoke dropout matrix per project is error prone and resource demanding. And triggers discussion if new modeling should be used to allow for more dropped guns.
# 
# Shot-to-shot designature: Some possibility to correct for a range of bad shots.

# Include perl script

# 
# 

# # Importing modules for plotting

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ## Setting some parameters for reading the input
# 

# Below, the header of two dan-files (broadband and conventional/cross-correlation) are listed to confirm the modeling parameters used in Nucleus. First specify the path to the dan files for the dataset to analyze:

# In[2]:


path = "./batch_4130/"


# In[3]:


file = "4130T__040_2000_080_5.dan"
f = open(path + "bb_" + file)
lines = f.read().splitlines()
f.close()
for i, line in enumerate(lines):
    if i < 56:
        print(i, line)


# In[4]:


file = "4130T__040_2000_080_5.dan"
f = open(path + "cc_" + file)
lines = f.read().splitlines()
f.close()
for i, line in enumerate(lines):
    if i < 56:
        print(i, line)


# Setting the numbers of lines to skip. This is for the 4130 array.

# In[5]:


#bb statistics
onegunskip = 43
twogunskip = 74
spareskip = 539
eof = 557

#conventional statistics
onegunskipcc = 55
twogunskipcc = 86
spareskipcc = 551
eofcc = 569

arrayvol = '4130T'
arraydepth = [4, 5, 6, 7, 8, 9]
subsep = [8, 10]
temp = [5, 10, 15, 20, 25]
prefix = ['bb', 'cc']


# Setting file name

# In[6]:


filenames = []

for a in arraydepth:
	for s in subsep:
		for t in temp:
			if s < 10: 
				filenames.append(arrayvol + '__0' + str(a) + '0_2000_0' + str(s) + '0_' + str(t) + '.dan')
			else:
				filenames.append(arrayvol + '__0' + str(a) + '0_2000_' + str(s) + '0_' + str(t) + '.dan')


# Just checking the file names:

# In[7]:


count = 0
for filename in filenames:
    if count < 10:
        print(filename)
    count += 1
print(len(filenames))


# ## Modeling Parameters
# This report analyse statistical values of the following dropout scenarios:
# <li>Temperatures: 5, 10, 15, 20, 25</li>
# <li>Depths 4-9m; 1m increment</li>
# <li>Subarray separation: 8 and 10m</li>
# <li>Bolt and GII guns</li>
# <li>Broadband (refl coeff zero) and conventional (refl coeff -1)</li>
# Total: 123 360 combinations
# <li>60 source/temperature combinations</li>
# <li>2 modelings per source (broadband and conv)</li>
# <li>2 gun types</li>
# <li>All single and two gun dropouts with spare gun substitution</li>
# <li>514 dropout combinations per source (nominal source always used as reference)</li>
# 
# The datasets are analyzed separately for Gun type to detect systematic differences.
# 

# Defining the column names:

# In[8]:


columns_one_gun_bb_raw = ['droparray1', 'dropgun1', 'gunvolume1', 'AvgdB', 'MaxdB', 'MaxPhase']
columns_two_gun_bb_raw = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'AvgdB', 'MaxdB', 'MaxPhase']
columns_spare_gun_bb_raw = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'AvgdB', 'MaxdB', 'MaxPhase']	
columns_one_gun_cc_raw = ['droparray1', 'dropgun1', 'gunvolume1', 'Peak', 'Peakch','PtoB','PtoBch', 'x-corr', 'AvgdB', 'MaxdB']
columns_two_gun_cc_raw = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'Peak', 'Peakch', 'PtoB', 'PtoBch', 'x-corr', 'AvgdB', 'MaxdB']
columns_spare_gun_cc_raw = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'Peak', 'Peakch', 'PtoB', 'PtoBch', 'x-corr', 'AvgdB', 'MaxdB']


# Statistical data defined as panda dataframes:

# In[9]:


stat_one_gun_bb_raw = pd.DataFrame()
stat_two_gun_bb_raw = pd.DataFrame()
stat_spare_gun_bb_raw = pd.DataFrame()
stat_one_gun_cc_raw = pd.DataFrame()
stat_two_gun_cc_raw = pd.DataFrame()
stat_spare_gun_cc_raw = pd.DataFrame()


# Reading data from files into dataframes:

# In[10]:


#path = "./batch_4130/"
for filename in filenames:
    temp_stat_one_gun_bb_raw = pd.read_csv(path + 'bb_' + filename, names=columns_one_gun_bb_raw, skiprows=onegunskip, delim_whitespace=True, nrows=twogunskip-onegunskip)
    temp_stat_one_gun_bb_raw['filename'] = filename
    temp_stat_one_gun_bb_raw['Depth'] = int(filename[7:10])
    temp_stat_one_gun_bb_raw['Subsep'] = int(filename[16:19])
    temp_stat_one_gun_bb_raw['Temperature'] = int(filename[20:22].replace('.',''))
    stat_one_gun_bb_raw = stat_one_gun_bb_raw.append(temp_stat_one_gun_bb_raw)
    temp_stat_one_gun_bb_raw = []
    
    temp_stat_two_gun_bb_raw = pd.read_csv(path + 'bb_' + filename, names=columns_two_gun_bb_raw, skiprows=twogunskip, delim_whitespace=True, nrows=spareskip-twogunskip)
    temp_stat_two_gun_bb_raw['filename'] = filename
    temp_stat_two_gun_bb_raw['Depth'] = int(filename[7:10])
    temp_stat_two_gun_bb_raw['Subsep'] = int(filename[16:19])
    temp_stat_two_gun_bb_raw['Temperature'] = int(filename[20:22].replace('.',''))
    stat_two_gun_bb_raw = stat_two_gun_bb_raw.append(temp_stat_two_gun_bb_raw)
    temp_stat_two_gun_bb_raw = []
    
    temp_stat_spare_gun_bb_raw = pd.read_csv(path + 'bb_' + filename, names=columns_spare_gun_bb_raw, skiprows=spareskip, delim_whitespace=True, nrows=eof-spareskip)
    temp_stat_spare_gun_bb_raw['filename'] = filename
    temp_stat_spare_gun_bb_raw['Depth'] = int(filename[7:10])
    temp_stat_spare_gun_bb_raw['Subsep'] = int(filename[16:19])
    temp_stat_spare_gun_bb_raw['Temperature'] = int(filename[20:22].replace('.',''))
    stat_spare_gun_bb_raw = stat_spare_gun_bb_raw.append(temp_stat_spare_gun_bb_raw)
    temp_stat_spare_gun_bb_raw = []
    
    temp_stat_one_gun_cc_raw = pd.read_csv(path + 'cc_' + filename, names=columns_one_gun_cc_raw, skiprows=onegunskipcc, delim_whitespace=True, nrows=twogunskipcc-onegunskipcc)
    temp_stat_one_gun_cc_raw['filename'] = filename
    temp_stat_one_gun_cc_raw['Depth'] = int(filename[7:10])
    temp_stat_one_gun_cc_raw['Subsep'] = int(filename[16:19])
    temp_stat_one_gun_cc_raw['Temperature'] = int(filename[20:22].replace('.',''))
    stat_one_gun_cc_raw = stat_one_gun_cc_raw.append(temp_stat_one_gun_cc_raw)
    temp_stat_one_gun_cc_raw = []
    
    temp_stat_two_gun_cc_raw = pd.read_csv(path + 'cc_' + filename, names=columns_two_gun_cc_raw, skiprows=twogunskipcc, delim_whitespace=True, nrows=spareskipcc-twogunskipcc)
    temp_stat_two_gun_cc_raw['filename'] = filename
    temp_stat_two_gun_cc_raw['Depth'] = int(filename[7:10])
    temp_stat_two_gun_cc_raw['Subsep'] = int(filename[16:19])
    temp_stat_two_gun_cc_raw['Temperature'] = int(filename[20:22].replace('.',''))
    stat_two_gun_cc_raw = stat_two_gun_cc_raw.append(temp_stat_two_gun_cc_raw)
    temp_stat_two_gun_cc_raw = []
    
    temp_stat_spare_gun_cc_raw = pd.read_csv(path + 'cc_' + filename, names=columns_spare_gun_cc_raw, skiprows=spareskipcc, delim_whitespace=True, nrows=eofcc-spareskipcc)
    temp_stat_spare_gun_cc_raw['filename'] = filename
    temp_stat_spare_gun_cc_raw['Depth'] = int(filename[7:10])
    temp_stat_spare_gun_cc_raw['Subsep'] = int(filename[16:19])
    temp_stat_spare_gun_cc_raw['Temperature'] = int(filename[20:22].replace('.',''))
    stat_spare_gun_cc_raw = stat_spare_gun_cc_raw.append(temp_stat_spare_gun_cc_raw)
    temp_stat_spare_gun_cc_raw = []


# The follwing is a lambda function that converts to string and adds zeros to a total length of 2 characters to the input x (which could be an int). Is later used to convert all gun numbers from int to string. (3 to "03", for example)

# In[11]:


add_zero = lambda x: str(int(x)).zfill(2)


# Adding all the string representations for gun numbering in a dedicated set of columns. Is later used to make array_gun_number index with a logical order. 

# In[12]:


stat_one_gun_bb_raw['dropgun1_str'] = stat_one_gun_bb_raw['dropgun1'].apply(add_zero)
stat_two_gun_bb_raw['dropgun1_str'] = stat_two_gun_bb_raw['dropgun1'].apply(add_zero)
stat_two_gun_bb_raw['dropgun2_str'] = stat_two_gun_bb_raw['dropgun2'].apply(add_zero)
stat_spare_gun_bb_raw['dropgun1_str'] = stat_spare_gun_bb_raw['dropgun1'].apply(add_zero)
stat_spare_gun_bb_raw['dropgun2_str'] = stat_spare_gun_bb_raw['dropgun2'].apply(add_zero)
stat_one_gun_cc_raw['dropgun1_str'] = stat_one_gun_cc_raw['dropgun1'].apply(add_zero)
stat_two_gun_cc_raw['dropgun1_str'] = stat_two_gun_cc_raw['dropgun1'].apply(add_zero)
stat_two_gun_cc_raw['dropgun2_str'] = stat_two_gun_cc_raw['dropgun2'].apply(add_zero)
stat_spare_gun_cc_raw['dropgun1_str'] = stat_spare_gun_bb_raw['dropgun1'].apply(add_zero)
stat_spare_gun_cc_raw['dropgun2_str'] = stat_spare_gun_bb_raw['dropgun2'].apply(add_zero)


# Printing some statistics to check that dataframes are ok (Remove hash for what you want to print)

# In[13]:


#print('stat_one_gun_bb_raw: ')
#print(stat_one_gun_bb_raw.head())
#print(stat_one_gun_bb_raw.head())
#print(stat_one_gun_bb_raw.tail())
#print(stat_one_gun_cc_raw)
#print(stat_two_gun_bb_raw.info())
#print('stat_spare_gun')

#print(stat_one_gun_cc_raw.head())
#print(stat_one_gun_cc_raw.tail())
#print('stat_two_gun_cc_raw: ')
#print(stat_two_gun_cc_raw.head())
#print(stat_two_gun_cc_raw.info())
# print('stat_spare_gun_cc_raw')
print(stat_spare_gun_cc_raw.tail())
# print(stat_spare_gun_cc_raw.info())
#print(len(stat_one_gun_bb_raw), len(stat_two_gun_bb_raw), len(stat_spare_gun_bb_raw))
#print(len(stat_one_gun_cc_raw), len(stat_two_gun_cc_raw), len(stat_spare_gun_cc_raw))


# Next step is to copy dataframes to identical copy with "all" postfix. Then later, the mastermatrix filter will be applied to the original dataframes. The "all" dataframes will contain the unfiltered data material.

# In[14]:


print(len(stat_two_gun_bb_raw))


# In[15]:


stat_two_gun_bb_raw['legal'] = stat_two_gun_bb_raw.apply(lambda row: 1 if (row['AvgdB'] < 0.85) & 
                                                         (row['MaxdB'] < 3) & 
                                                         (row['MaxPhase'] < 20) 
                                                         else 0, axis=1)


# In[16]:


stat_two_gun_bb_raw.legal.value_counts()
# 1 means legal (55%), 0 means illegal (45%)


# In[17]:


stat_two_gun_bb_raw.dtypes


# In[18]:


stat_two_gun_bb_raw['gun1_unique'] = stat_two_gun_bb_raw.apply(lambda row: str(row['droparray1']) + '.' + row['dropgun1_str'], axis=1)
stat_two_gun_bb_raw['gun2_unique'] = stat_two_gun_bb_raw.apply(lambda row: str(row['droparray2']) + '.' + row['dropgun2_str'], axis=1)


# In[19]:


#stat_two_gun_bb_raw


# In[20]:


ml_data_two_gun_bb = stat_two_gun_bb_raw[['filename', 'gun1_unique', 'gun2_unique', 'Depth', 'Subsep', 'Temperature', 'legal']]


# In[21]:


#ml_data_two_gun_bb


# In[22]:


def valuemapping(values_to_map):
    values = values_to_map.unique()
    mapping = {}
    for i, value in enumerate(values, 1):
        mapping[value] = i
    return mapping  


# In[23]:


gun_mapping = {'1.01': 1, 
               '1.02': 2, 
               '1.03': 3, 
               '1.04': 4, 
               '1.05': 5, 
               '1.07': 6, 
               '1.09': 7, 
               '1.11': 9, 
               '1.12': 10, 
               '1.13': 11, 
               '1.14': 12, 
               '2.01': 13, 
               '2.02': 14, 
               '2.03': 15, 
               '2.05': 16, 
               '2.07': 17, 
               '2.09': 18, 
               '2.11': 19, 
               '2.12': 20, 
               '2.13': 21, 
               '3.01': 23, 
               '3.02': 24, 
               '3.03': 25, 
               '3.05': 27, 
               '3.07': 28, 
               '3.09': 29, 
               '3.10': 30, 
               '3.11': 31, 
               '3.12': 32, 
               '3.13': 33, 
               '3.14': 34}
#gun_mapping1 = valuemapping(ml_data_two_gun_bb.gun1_unique)
#gun_mapping2 = valuemapping(ml_data_two_gun_bb.gun2_unique)
print(gun_mapping)
ml_data_two_gun_bb['gun1_num'] = ml_data_two_gun_bb.gun1_unique.map(gun_mapping)
ml_data_two_gun_bb['gun2_num'] = ml_data_two_gun_bb.gun2_unique.map(gun_mapping)


# In[24]:


ml_data_two_gun_bb = ml_data_two_gun_bb[['filename', 'gun1_num', 'gun2_num', 'Depth', 'Subsep', 'Temperature', 'legal']]


# In[25]:


#ml_data_two_gun_bb


# In[26]:


def plotmatrix(x, y, legal):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=legal)
    plt.xlim(0, 36)
    plt.ylim(0, 36)
    plt.title('Matrix')
    plt.xlabel("Gun 1")
    plt.ylabel("Gun 2")
    plt.gca().invert_yaxis()
    plt.show()


# In[27]:


examplefile = '4130T__060_2000_080_15.dan'


# In[28]:


ml_data_test = ml_data_two_gun_bb[(ml_data_two_gun_bb['filename'] == examplefile)]


# In[29]:


plotmatrix(ml_data_test.gun1_num, ml_data_test.gun2_num, ml_data_test.legal)


# In[30]:


#ml_data_two_gun_bb.corr()


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import itertools


# In[32]:


def normalize_data(feature_data):
    x = feature_data.values.astype('float64')
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    return feature_data


# In[33]:


def normalize_custom(feature_data):
    feature_data['Depth'] = feature_data.apply(lambda row: row['Depth']/10, axis=1)
    feature_data['Subsep'] = feature_data.apply(lambda row: row['Subsep']/20, axis=1)
    feature_data['Temperature'] = feature_data.apply(lambda row: row['Temperature']/5, axis=1)
    return feature_data


# In[34]:


def normalize_custom_inverse(feature_data):
    feature_data['Depth'] = feature_data.apply(lambda row: row['Depth']*10, axis=1)
    feature_data['Subsep'] = feature_data.apply(lambda row: row['Subsep']*20, axis=1)
    feature_data['Temperature'] = feature_data.apply(lambda row: row['Temperature']*5, axis=1)
    return feature_data


# In[35]:


def knn(training_data, training_labels, validation_data, validation_labels, maxk, avg):
    normalize_custom(training_data)
    normalize_custom(validation_data)
    k_list = []
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    for k in range(1, maxk, 2):
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(training_data, training_labels)
        print(k, ': ', classifier.score(validation_data, validation_labels))
        k_list.append(k)
        y_predict = classifier.predict(validation_data)
        accuracies.append(metrics.accuracy_score(validation_labels, y_predict))
        recalls.append(metrics.recall_score(validation_labels, y_predict, average=avg))
        precisions.append(metrics.precision_score(validation_labels, y_predict, average=avg))
        f1s.append(metrics.f1_score(validation_labels, y_predict, average=avg))
    normalize_custom_inverse(training_data)
    normalize_custom_inverse(validation_data)
    return k_list, y_predict, accuracies, recalls, precisions, f1s


# In[36]:


prediction_features = ml_data_two_gun_bb
prediction_labels = ml_data_two_gun_bb


# In[37]:


training_data, test_data, training_labels, test_labels = train_test_split(prediction_features, prediction_labels, test_size = 0.2, random_state = 100)
print(len(training_data), len(training_labels))
print(len(test_data), len(test_labels))


# In[38]:


#training_data


# In[39]:


training_one_config = training_data[(training_data['filename'] == examplefile)]


# In[40]:


plotmatrix(training_one_config.gun1_num, training_one_config.gun2_num, training_one_config.legal)


# In[41]:


test_one_config = test_data[(test_data['filename'] == examplefile)]


# In[42]:


plotmatrix(test_one_config.gun1_num, test_one_config.gun2_num, test_one_config.legal)


# In[43]:


training_data = training_data.drop(["legal", "filename"], axis=1)
test_data_with_filename = test_data.drop(["legal"], axis=1)
test_data = test_data_with_filename.drop(["filename"], axis=1)
training_labels = training_labels["legal"]
test_labels = test_labels["legal"]


# In[44]:


training_data.head()


# In[45]:


k_list, prediction, accuracies, recalls, precisions, f1 = knn(training_data, training_labels, test_data, test_labels, 31, 'binary')


# k=3 is the sweetspot without normalization
# k=5 is the sweetspot with costum normalization

# In[46]:


training_data.head()


# In[47]:


def plotaccuracy(k_list, accuracies, recalls, precision, f1, title):
    plt.figure(figsize=(10, 8))
    plt.plot(k_list, accuracies, label='accuracy')
    plt.plot(k_list, recalls, label='recall')
    plt.plot(k_list, precision, label='precision')
    plt.plot(k_list, f1, label='f1')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.show()


# In[48]:


plotaccuracy(k_list, accuracies, recalls, precisions, f1, "Legal/illegal Classifier Accuracy")


# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples. The best value is 1 and the worst value is 0.
# 
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.
# 
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
# 
# F1 = 2 (precision recall) / (precision + recall)

# In[49]:


training_data.head()


# In[50]:


test_data.head()


# In[81]:


#Running classification and prediction with best k
normalize_custom(training_data)
normalize_custom(test_data)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(training_data, training_labels)
print(classifier.score(test_data, test_labels))
y_predict = classifier.predict(test_data)
normalize_custom_inverse(training_data)
normalize_custom_inverse(test_data);


# In[52]:


training_data.head()


# In[53]:


test_data.head()


# In[54]:


#Running classification and prediction with best k
#classifier = KNeighborsClassifier(n_neighbors = 3)
#classifier.fit(training_data_norm, training_labels)
#print(classifier.score(test_data_norm, test_labels))
#y_predict = classifier.predict(test_data_norm)


# In[55]:


cnf_matrix = metrics.confusion_matrix(test_labels, y_predict)


# In[56]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[57]:


class_names = [0, 1]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[58]:


len(y_predict)


# In[59]:


def prediction1(list):
    df = pd.DataFrame([list])
    return classifier.predict(df)


# In[60]:


predicted_matrix = test_data_with_filename
predicted_matrix['legal'] = test_data.apply(lambda row: prediction1([row['gun1_num'], row['gun2_num'], row['Depth'], row['Subsep'], row['Temperature']]), axis=1)


# In[61]:


#predicted_matrix


# In[62]:


predicted_matrix_example = predicted_matrix[(predicted_matrix['filename'] == examplefile)]


# In[63]:


plotmatrix(test_one_config.gun1_num, test_one_config.gun2_num, test_one_config.legal)

plotmatrix(predicted_matrix_example.gun1_num, predicted_matrix_example.gun2_num, predicted_matrix_example.legal)


# In[64]:


def plotmatrix_combine(x1, y1, legal1, x2, y2, legal2):
    plt.figure(figsize=(8, 6))
    plt.scatter(x1, y1, c=legal1)
    plt.scatter(x2, y2, c=legal2)
    plt.xlim(0, 36)
    plt.ylim(0, 36)
    plt.title('Combined Matrix')
    plt.xlabel("Gun 1")
    plt.ylabel("Gun 2")
    plt.gca().invert_yaxis()
    plt.show()


# In[65]:


plotmatrix_combine(training_one_config.gun1_num, training_one_config.gun2_num, training_one_config.legal, predicted_matrix_example.gun1_num, predicted_matrix_example.gun2_num, predicted_matrix_example.legal)
plotmatrix(ml_data_test.gun1_num, ml_data_test.gun2_num, ml_data_test.legal)


# In[72]:


test_data.head()


# In[74]:


test1 = [7, 12, 60, 80, 15]
df = pd.DataFrame([test1])
df = df.rename(columns={0: 'gun1_num', 1: 'gun2_num', 2: 'Depth', 3: 'Subsep', 4: 'Temperature'})


# In[82]:


df


# In[76]:


normalize_custom(df)


# In[78]:


print(classifier.predict(df))


# In[83]:


normalize_custom_inverse(df)

