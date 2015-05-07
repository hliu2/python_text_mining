import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics


#load the data 
movie_reviews_data_folder = 'C:/Users/HAI LIU/ipythonnotebook/doc/tutorial/text_analytics/data/movie_reviews/txt_sentoken'
dataset = load_files(movie_reviews_data_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))


# split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)


#exploer how the min_df and max_df change the number of features we get 
import numpy as np
value_range=np.arange(0.01,0.99,0.01)
y1=[TfidfVectorizer(min_df=x).fit_transform(docs_train).shape[1] for x in value_range]
y2=[TfidfVectorizer(max_df=x).fit_transform(docs_train).shape[1] for x in value_range]
from ggplot import *
print qplot(value_range,y=y1,geom='line')+xlab('min_df')+ylab('features')
print qplot(value_range,y=y2,geom='line')+xlab('max_df')+ylab('features')


#exploer how the ngram_range change the number of features we get 
x=[1 for i in range(10)]
y=np.arange(10)+1
parameter=zip(x,y)
y3=[TfidfVectorizer(ngram_range=i).fit_transform(docs_train).shape[1] for i in parameter]
fig=plt.figure(figsize=(8,6))
plt.plot([1,2,3,4,5,6,7,8,9,10],y3,'b--o')
plt.xlabel('ngram')
plt.ylabel('features')


#setting max_df and n_gram_range as default, we choose min_df in [1,0.2,0.4,0.6,0.8] seperately, 
#and store the corresponding Xtrain and Xtest into min_df_data array.
min_df_data=[(TfidfVectorizer(min_df=i).fit_transform(docs_train).toarray(),
TfidfVectorizer(min_df=i).fit(docs_train).transform(docs_test).toarray()) for i in [1,2,3,4]]
             

#setting min_df and n_gram_range as default, we choose max_df in [1,0.2,0.4,0.6,0.8] seperately, 
#and store the corresponding Xtrain and Xtest into max_df_data array.
max_df_data=[(TfidfVectorizer(max_df=i).fit_transform(docs_train).toarray(),
TfidfVectorizer(max_df=i).fit(docs_train).transform(docs_test).toarray()) for i in [0.85,0.90,0.95,1.0]]
             

#setting min_df and max_df as default, we choose ngram_range in [1,0.2,0.4,0.6,0.8] seperately, 
#and store the corresponding Xtrain and Xtest into ngram_range_data array.
ngram_range_data=[(TfidfVectorizer(ngram_range=i).fit_transform(docs_train),
TfidfVectorizer(ngram_range=i).fit(docs_train).transform(docs_test)) for i in [(1,1),(1,2),(1,3)]]
             

# explore parameters in tfidf for both linear SVC and KNN
param_grid = [
  {'C': [100, 1000]},
   ]
grid_search = GridSearchCV(LinearSVC(), param_grid, n_jobs=-1, verbose=1)
min_df_fit=[grid_search.fit(i[0],y_train).predict(i[1]) for i in min_df_data ]
max_df_fit=[grid_search.fit(i[0],y_train).predict(i[1]) for i in max_df_data ]
ngram_range_fit=[grid_search.fit(i[0],y_train).predict(i[1]) for i in ngram_range_data]
min_df_svc_score=[metrics.accuracy_score(min_df_fit[i],y_test) for i in range(4)]
max_df_svc_score=[metrics.accuracy_score(max_df_fit[i],y_test) for i in range(4)]
ngram_range_svc_score=[metrics.accuracy_score(ngram_range_fit[i],y_test) for i in range(3)]

from sklearn.neighbors import KNeighborsClassifier
param_grid = [
  {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]},
   ]
grid_search1 = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, verbose=1)
min_df_fit1=[grid_search1.fit(i[0],y_train).predict(i[1]) for i in min_df_data ]
max_df_fit1=[grid_search1.fit(i[0],y_train).predict(i[1]) for i in max_df_data ]
ngram_range_fit1=[grid_search1.fit(i[0],y_train).predict(i[1]) for i in ngram_range_data]
min_df_knn_score=[metrics.accuracy_score(min_df_fit1[i],y_test) for i in range(4)]
max_df_knn_score=[metrics.accuracy_score(max_df_fit1[i],y_test) for i in range(4)]
ngram_range_knn_score=[metrics.accuracy_score(ngram_range_fit1[i],y_test) for i in range(3)]

min_df_svc_score=[metrics.accuracy_score(min_df_fit[i],y_test) for i in range(4)]
max_df_svc_score=[metrics.accuracy_score(max_df_fit[i],y_test) for i in range(4)]
ngram_range_svc_score=[metrics.accuracy_score(ngram_range_fit[i],y_test) for i in range(3)]


# plot these data to see how min_df, max_df and ngram change the accuracy of knn and linearSVC 
import matplotlib.pyplot as plt 
fig=plt.figure(figsize=(8,6))
plt.plot([1,2,3,4], min_df_svc_score, 'bo--',label='svm')
plt.plot([1,2,3,4], min_df_knn_score, 'ro--',label='knn')
plt.legend(loc='best')
plt.xlabel('min_df')
plt.ylabel('score')
plt.ylim(0.6,0.9)

fig=plt.figure(figsize=(8,6))
plt.plot([0.85,0.9,0.95,1.0], max_df_svc_score, 'bo--',label='svm')
plt.plot([0.85,0.9,0.95,1.0], max_df_knn_score, 'ro--',label='knn')
plt.legend(loc='best')
plt.xlabel('max_df')
plt.ylabel('score')

fig=plt.figure(figsize=(8,6))
plt.plot([1,2,3], ngram_range_svc_score, 'bo--',label='svm')
plt.plot([1,2,3], ngram_range_knn_score, 'ro--',label='knn')
plt.legend(loc='best')
plt.xlabel('ngram')
plt.ylabel('score')


#explore parameters in classifier to see how these parameters change the testing accuracy. 
data=[TfidfVectorizer().fit_transform(docs_train).toarray(), TfidfVectorizer().fit(docs_train).transform(docs_test).toarray()]
k=[1,2,3,4,5,6,7,8,9,10]
predicted_knn=[KNeighborsClassifier(n_neighbors=i).fit(data[0],y_train).predict(data[1]) for i in k]
score_knn=[metrics.accuracy_score(predicted_knn[i],y_test) for i in range(10)]

C=[1,10,100,1000,10000]
predicted_svm=[LinearSVC(C=i).fit(data[0],y_train).predict(data[1]) for i in C]
score_svm=[metrics.accuracy_score(predicted_svm[i],y_test) for i in range(5)]

fig=plt.figure(figsize=(8,6))
plt.plot([1,10,100,1000,10000], score_svm, 'bo--',label='svm')
plt.xlabel('C')
plt.ylabel('score')
plt.xlim(-1000,10000)
fig=plt.figure(figsize=(8,6))
plt.plot([1,2,3,4,5,6,7,8,9,10], score_knn, 'bo--',label='svm')
plt.xlabel('K')
plt.ylabel('score')


# select examples where the prediction was incorrect
x_train=TfidfVectorizer().fit_transform(docs_train)
x_test=TfidfVectorizer().fit(docs_train).transform(docs_test)
predicted=LinearSVC(C=1000).fit(x_train,y_train).predict(x_test)
predicted!=y_test


#select first two wrong prediction to conjecture on why the classifier made a mistake for this prediction 
print("\n".join(docs_test[8].split("\n")))
print("\n".join(docs_test[18].split("\n")))


#try pca to find a two dimensional plot in which the positive and negative reviews are separated
data=TfidfVectorizer().fit_transform(docs_train).toarray()
from sklearn.decomposition import PCA
X_r=PCA(n_components=2).fit(data).transform(data)
plt.figure()
for c, i in zip("rg", [0, 1]):
    plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], c=c)
    

#try factor analysis to find a two dimensional plot in which the positive and negative reviews are separated
data=TfidfVectorizer().fit_transform(docs_train).toarray()
from sklearn.decomposition import FactorAnalysis
X_r=FactorAnalysis(n_components=2).fit(data).transform(data)
fit=plt.figure()
for c, i in zip("rg", [0, 1]):
    plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], c=c)
fit=plt.figure()
for c, i in zip("rg", [0, 1]):
    plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], c=c)
    plt.xlim((-2.62132758e-02,-0.0262114831))
    plt.ylim(-0.0254559094,sort(X_r[:,1])[-4])
