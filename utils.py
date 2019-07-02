from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



class metrics:
	def __init__(self,cm):
		self.cm=np.array(cm)
		self.classes=len(cm)
	def precision(self,i):
		true_positive=self.cm[i][i]
		sum_axis_0=np.sum(self.cm, axis=0)
		prec=true_positive/(sum_axis_0[i]+0.0)
		return prec

	def recall(self,i):
		true_positive=self.cm[i][i]
		sum_axis_1=np.sum(self.cm, axis=1)
		recall=true_positive/(sum_axis_1[i]+0.0)
		return recall

	def f_measure(self,i):
		f_mes=(2*self.precision(i)*self.recall(i))/(self.precision(i)+self.recall(i))
		return f_mes

def print_confusion_matrix(cm,score,size):
    plt.figure(figsize=(size,size))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)




def evaluation_metrics(cf,z):
    gbt_metrics_test = metrics(cf)
    print ('<<<<<<<<<<<<'+z+' METRICS >>>>>>>>>>>>>')
    print ('accuracy : ',(cf[0][0]+cf[1][1])/np.sum(cf))
    #print '\n'
    print ('=================class_0==================')
    print ('fMeasure_of_class_0 : ',gbt_metrics_test.f_measure(0))
    print ('precision_of_class_0: ',gbt_metrics_test.precision(0))
    print ('recall_of_class_0:    ',gbt_metrics_test.recall(0))

    #print '\n'
    print ('=================class_1==================')
    print ('fMeasure_of_class_1 : ',gbt_metrics_test.f_measure(1))
    print ('precision_of_class_1: ',gbt_metrics_test.precision(1))
    print ('recall_of_class_1:    ',gbt_metrics_test.recall(1))

    print_confusion_matrix(cf,'cf',3)