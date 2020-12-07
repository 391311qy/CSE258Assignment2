from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

def accuracy(y_test, y_pred):
	C = confusion_matrix(y_test, y_pred)
	TN = C[0,0]
	FN = C[1,0]
	TP = C[1,1] 
	FP = C[1,0]
	acc = (TP + TN) / (TN + FN + TP + FP)
	return acc

def linear_regression(X, y, X_test, y_test, C = 1.0):
	clf = linear_model.Ridge(C, fit_intercept=False) # MSE + 1.0 l2
	clf.fit(X, y)
	predictions = clf.predict(X_test)
	#predictions_train = clf.predict(X)
	#print("mse of prediction train is "+ str(mean_squared_error(y ,predictions_train)))
	print("mse of linear regression prediction is "+ str(mean_squared_error(y_test ,predictions)))
	print("mae of linear regression prediction is "+ str(mean_absolute_error(y_test ,predictions)))

def bayesianRidge_regression(X, y, X_test, y_test, t = 1e-6):
    clf_Bayes = linear_model.BayesianRidge(tol=t, fit_intercept=False, compute_score=True)
    clf_Bayes.fit(X, y)
    y_pred_bayes = clf_Bayes.predict(X_test)
    print("MSE of bayes ridge regression prediction is "+ str(mean_squared_error(y_test ,y_pred_bayes)))
    print("mae of bayes ridge regression prediction is "+ str(mean_absolute_error(y_test ,y_pred_bayes)))

def softmax_classification(X, y, X_test, y_test, iter = 1000, random_state = 42):
	clf_lr = linear_model.LogisticRegression(solver='sag', max_iter=iter, random_state=random_state,
								multi_class="multinomial", class_weight="balanced").fit(X, y)
	y_pred_lr = clf_lr.predict(X_test)
	acc = accuracy(y_pred_lr, y_test)
	print("MSE of softmax prediction is "+ str(mean_squared_error(y_test ,y_pred_lr)))
	print("BER of softmax prediction is "+ str(1 - balanced_accuracy_score(y_test, y_pred_lr, adjusted=False)))
	print("accuracy of softmax is " + str(acc))

def random_forest_classification(X, y, X_test, y_test, n_estimators = 100):
    clf_rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=2, random_state=0)
    clf_rf.fit(X, y)
    y_pred_rf = clf_rf.predict(X_test)
    #acc_rf = accuracy(y_test, y_pred_rf)
    print("MSE of random forest prediction is "+ str(mean_squared_error(y_test ,y_pred_rf)))
    print("BER of random forest prediction is "+ str(1 - balanced_accuracy_score(y_test, y_pred_rf, adjusted=False)))
    #print("accuracy of random forest is " + str(acc_rf))