#######################################################################################################################
#   Author  :Koushik Modayur Chandramouleeswaran
#   ModuleName  : DecisionTree.py
#   Descriprion : Decision Tree implementation
#   Date    : 11/15/2015
#######################################################################################################################
import csv
import math
import operator


training_data = []                      #Global list for storing training data
tree= {}
count = 0
cntr = 0
def load_train_data(file_name):
    # Function to read the input file and store it in a list
    try:
        train = open(file_name)
        data = csv.reader(train)
        for rec in data:
            temp_list = []
            for cols in rec:
                temp_list.append(cols)

            training_data.append(temp_list)

        print "Training data loaded"
    except IOError:
        print "Training File not available. Please check the filename"
        exit()

def get_test_data(test_file):
    #Function to load test data
    try:
        test_data = []
        test = open(test_file)
        data = csv.reader(test)
        for rec in data:
            temp_list = []
            for cols in rec:
                temp_list.append(cols)

            test_data.append(temp_list)
        return test_data

        print "Test data loaded"
    except IOError:
        print "Test File not available. Please check the filename"
        exit()


def majorityCnt(classlist):
#Function to find the majority class
    classcount={}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote] += 1
    sortedClassCount=sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def calcentropy(feature,data):
    #Function to calculate Entropy

    feature_pos = data[0].index(feature)
    feature_count = {}
    unique_features_counts = {}
    length = len(data[1:])

    for rec in data[1:]:

        if (rec[feature_pos],rec[0]) in feature_count:
            feature_count[(rec[feature_pos],rec[0])] += 1
        else:
            feature_count[(rec[feature_pos],rec[0])] = 1

        if rec[feature_pos] in unique_features_counts:
            unique_features_counts[rec[feature_pos]] +=1
        else:
            unique_features_counts[rec[feature_pos]] = 1


    Entropy = {}
    overall_Entropy = 0.0
    for key in unique_features_counts.keys():
        for keys in feature_count.keys():
            if key in keys:
                if key in Entropy:
                    Entropy[key] += -(float(feature_count[keys])/unique_features_counts[key])*math.log(float(feature_count[keys])/unique_features_counts[key],2)
                else:
                    Entropy[key] = -(float(feature_count[keys])/unique_features_counts[key])*math.log(float(feature_count[keys])/unique_features_counts[key],2)
    #print Entropy

    for key,value in Entropy.iteritems():
        overall_Entropy += value * (float(unique_features_counts[key])/length)

    return overall_Entropy

def getbestfeature(featurelist,data):
    #Function to determine best feature for the next level

    feature_Entropy = {}
    bestfeat = None
    for features in featurelist:
        if features not in feature_Entropy:

            feature_Entropy[features] = calcentropy(features,data)

    if len(feature_Entropy) > 0 :
        bestfeat = sorted(feature_Entropy.items(), key=operator.itemgetter(1),reverse = False)[0][0]


    return  bestfeat


def getdata(data,feature,val):
    #split the data for next node of the tree
    new_data = []

    position = data[0].index(feature)

    for rec in data:
        if len(rec[0])>1:
            record = rec[:]
            del(record[position])
            new_data.append(record)
        if rec[position] == val:
            dat = rec[:]
            del(dat[position])
            new_data.append(dat)

    return new_data

def getmaxprobclass(classlist):
    #Determine the class with maximum probability
    max_class = {}

    for classes in classlist:
        max_class.update({classes:classlist.count(classes)})


    return  sorted(max_class.items(), key=operator.itemgetter(1),reverse = True)[0][0]



def buildtree(featurelist,inpdata):
    #Build decision tree
    pos = None
    classlist=[ x[0] for x in inpdata[1:]]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(classlist)==1:
        return majorityCnt(classlist)
    else:

        unique_val = set()
        node= getbestfeature(featurelist,inpdata)

        tree={node:{}}

        if node!= None :
            pos = inpdata[0].index(node)

            featurelist.remove(node)


            for rec in inpdata:
                if  rec[pos]!=node:
                    unique_val.add(rec[pos])

            for val in unique_val:
                sl = featurelist[:]

                if node in tree.keys():
                    stree = buildtree(sl,getdata(inpdata,node,val))
                    tree[node].update({val:stree})


        else:
            #print featurelist,classlist
            out = getmaxprobclass(classlist)

            return out

    return tree

def predict(model,features,test):
    #Function to predict the class of a test data using the decision tree(model)
    root = model.keys()[0]
    next_level = model[root]
    output = "p"
    feature_pos = features.index(root)
    for key in next_level.keys():
        if test[feature_pos] == key:
            if type(next_level[key]).__name__ == 'dict':
                output = predict(next_level[key],features,test)
            else:
                output = next_level[key]

    return output

if __name__ == "__main__" :
    load_train_data("mushroom_train.csv")
    feature_list = training_data[0][1:]
    tree = buildtree(feature_list,training_data)
    print tree
    test_features = ["shape","surface","color","bruises"]

    test_data = get_test_data("mushroom_test.csv")

    real_class = []
    pred_class = []
    for testrec in test_data:
        real_class.append(testrec[0])
        pred_class.append(predict(tree,test_features,testrec[1:]))

    pred_count = 0
    for i in range(0,len(real_class)):
        if pred_class[i] == real_class[i]:
            pred_count+=1

    print "output = "
    print pred_class
    print "Accuracy for test data = " + str((float(pred_count)/len(real_class))*100) + "%"


