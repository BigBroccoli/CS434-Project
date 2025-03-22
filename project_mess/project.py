import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score, cross_val_predict

def encode(frame):
    holder = frame.copy(deep=True)
    for feature in holder.columns:
        encoder = LabelEncoder() 
        holder[feature] = encoder.fit_transform(holder[feature])

    return holder

class Regressor: #past life
    def __init__(self, data, train, test, rnd, itr, eta):
        
        self.data = data
        self.itr = itr
        self.eta = eta

        self.accuracy = None
        self.precision = None
        self.recall = None

        model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=rnd, max_iter=itr)

    
    def stats(self):
        tp = np.sum((self.targets == 1) & (self.fx == 1))
        fp = np.sum((self.targets == -1) & (self.fx == 1))
        fn = np.sum((self.targets == 1) & (self.fx == -1))
        tn = np.sum((self.targets == -1) & (self.fx == -1))

        total = len(self.targets)

        self.accuracy = (tp + tn) / total if total > 0 else None
        self.precision = (tp / (tp + fp)) if (tp+fp) > 0 else None
        self.recall = (tp / (tp + fn)) if (tp+fn) > 0 else None

    def model_state(self):
        print("Weights shape:", self.weights)
        print("Bias:", self.bias)
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)


# def main(args):
def main():
    # print(f"Model: {args.model}")
    # print(f"Maximum Iterations: {args.iter}")
    # print(f"Learning Rate: {args.lr}")

    train = pd.read_csv('train.csv')
    test = pd.read_csv('right_test.csv')

    train_labels = train.columns 
    test_labels = test.columns

    # Method 1 - items to encode
    # categorical = train.select_dtypes(include=['object']).columns.tolist()
    # numeric = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Method 2 - items to encode
    catagorical = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    # Method 1
    train_X = train.iloc[:, :-1] #everything but last column of training data
    train_y = train.iloc[:, -1]  #last column only of training data

    test_X = test.iloc[:, :-1] #everything but last column of test data
    test_y = test.iloc[:, -1] #last column only of test data

    # Method 2 -- probably the worse of the two
    # train_y = train.pop(train_labels[-1]) #everything but last column of training data
    # train_X = train  #last column only of training data

    # test_y = test.pop(test_labels[-1]) #everything but last column of test data
    # test_X = test #last column only of test data

    matched_data = sum(train_labels==test_labels)
    if(len(train_labels)!=matched_data or len(test_labels)!=matched_data):
        raise Exception("Data Labels are not a match; fix data.")

    print(train_X.shape)
    print(test_X.shape)

    print(train_y.shape)
    print(test_y.shape)



    # print(train_labels)
    # print(test_labels)
    # print(train_types)
    # print(test_types)
    
    # train.to_excel('train.xlsx', index=False)
    # test.to_excel('test.xlsx', index=False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Perceptron Classifier")
    # parser.add_argument("--model", type=str, default="iris", help="['iris','mush','musk']")
    # parser.add_argument("--iter", type=int, default=100, help="Number of epochs")
    # parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    # parser.add_argument("--map_1", type=str, default="rnd", help="Feature 1 (to map)")
    # parser.add_argument("--map_2", type=str, default="rnd", help="Feature 2 (to map)")
    # you need rnd for logarithic model
    main()
    # args = parser.parse_args()
    # main(args)

# Model is marginally better because it includes variables  (personal  attributes  of  customer  like  age,  purpose,  credit  history,  credit amount, credit duration, etc.)
# other than checking account information (which shows wealth of a customer) that should be taken into account to calculate the probability of default on loan correctly. 
# Therefore, by using a logistic regression approach, the right customers to be targeted for granting loan can be easily detected by evaluating their likelihood of default on loan. 
# The model concludes that a bank should not only target the rich customers for granting loans, but it should assess the other attributes of a customer as well which play a very important part in credit granting decisions and predicting the loan defaulters.
# The results of the different machine learning algorithms are based on accuracy, precision, recall, and F1-score. The ROC curve needs to be plotted based on the confusion matrix

# In addition to report the above specific steps of the machine learning pipeline, the team project  also  needs  to  specify  which  machine  learning  techniques  or  deep  learning method is used or developedto predict the loan’s approval or decline decisions with high  accuracy,  precision,  recall  and  AUC  scores?
# What  are  the  state-of-the-art machine  learning  based  techniques  and  algorithms?
# How  can  you  develop  a  new method or algorithm to improve the prediction performance or make it efficient?
# How do you quickly detect these characteristics in your model so that you can ensure a high predictionperformance?
# In order to answer these technical questions, you need to
# (1)Briefly  summarize  the  existing  breakthrough  in  this  topic  and  summarize  the challenges facing in existing studies.
# (2)Draw an overall flowchart to show how your model works on such kind of data.
# (3)Report the average performance over different fold in the test sets using cross validation.
# (4)Report the confusion matrix and AUC curves and scores with different fold of cross validation.
# (5)Draw the three comparative curves of training, validation, and test with different iterations or epochs, and show how do you address underfitting and overfitting problems.
# (6)Report the running time of your model or algorithm.
# (7)Summarize how your team worked together and make this team project complete, what are the efforts of each teammate.
# (8)Summarize your contributions to existing studies.
