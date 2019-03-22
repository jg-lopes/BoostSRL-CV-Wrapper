import os
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold

# SET THESE VARIABLES ACCORDING TO THE PROBLEM

num_folds = 5


# FINDS THE TARGET NAME FROM THE POSITIVE EXAMPLES

with open('./data/pos.txt', mode='r') as f:
	target_example = f.readline()

pattern = re.compile(r"^([\w]*)")
matches = pattern.findall(target_example)

target = matches[0]

# CLEARS LINGERING FILES FROM PAST EXECUTIONS
os.system("rm -f -r folds")
os.system("rm -f precomputed.txt")
os.system("rm -f recomputed.txt")
os.system("rm -f results.txt")

# CREATING FOLDERS TO STORE THE EXECUTIONS FOR EACH FOLD
for count in range(1,num_folds+1):
	try:
		os.makedirs("./folds/fold_" + str(count) + "/test")
	except FileExistsError:
		pass

	try:
		os.makedirs("./folds/fold_" + str(count) + "/train")
	except FileExistsError:
		pass
	
	
# - COPYING FACTS AND CREATING BK

bk_location = "import: \"../../background.txt\"."

for count in range(1,num_folds+1):
	
	# COPYING FACTS 
	os.system("cp ./data/facts.txt ./folds/fold_" + str(count) + "/train/train_facts.txt")
	os.system("cp ./data/facts.txt ./folds/fold_" + str(count) + "/test/test_facts.txt")

	# COPYING BACKGROUND
	os.system("cp ./data/background.txt ./folds/background.txt")

	# WRITING BACKGROUND LOCATION
	with open('folds/fold_'+ str(count) + '/train/train_bk.txt', 'w') as f:
		f.write(bk_location)
	with open('folds/fold_'+ str(count) + '/test/test_bk.txt', 'w') as f:
		f.write(bk_location)

# - DEALING WITH POSITIVE EXAMPLES

# Opening and reading
f_pos = open('./data/pos.txt', mode='r')
pos_list = f_pos.read().split("\n")
f_pos.close()

# Creating samples to feed to stratifiedKfold function
X = np.array(pos_list)
y = np.array(np.zeros(len(pos_list)))

# Randomizes list
np.random.shuffle(X)

skf = StratifiedKFold(n_splits=num_folds)
skf.get_n_splits(X, y)

# Iterating over the multiple folds
count = 1
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]

	# Writing a fold to their respective archives 
	with open('folds/fold_'+ str(count) + '/train/train_pos.txt', 'w') as f:
		for item in X_train:
			f.write("%s\n" % item)
	with open('folds/fold_'+ str(count) + '/test/test_pos.txt', 'w') as f:
		for item in X_test:
			f.write("%s\n" % item)
	count += 1


# - DEALING WITH NEGATIVE EXAMPLES

# Opening and reading
f_neg = open('./data/neg.txt', mode='r')
neg_list = f_neg.read().split("\n")
f_neg.close()

# Creating samples to feed to stratifiedKfold function
X = np.array(neg_list)
y = np.array(np.zeros(len(neg_list)))

# Randomizes list
np.random.shuffle(X)

skf = StratifiedKFold(n_splits=num_folds)
skf.get_n_splits(X, y)

# Iterating over the multiple folds
count = 1
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]

	# Writing a fold to their respective archives 
	with open('folds/fold_'+ str(count) + '/train/train_neg.txt', 'w') as f:
		for item in X_train:
			f.write("%s\n" % item)
	with open('folds/fold_'+ str(count) + '/test/test_neg.txt', 'w') as f:
		for item in X_test:
			f.write("%s\n" % item)
	count += 1

for count in range(1,num_folds+1):
    os.system("java -jar BoostSRL.jar -l -train folds/fold_" + str(count) + "/train/ -target " + target)

for count in range(1,num_folds+1):
    os.system("java -jar BoostSRL.jar -i -model folds/fold_" + str(count) + "/train/models -test folds/fold_"
         + str(count) + "/test/ -target " + target + " -aucJarPath .")


AUC_ROC = []
AUC_PR = []
CLL = []
Threshold = []
Precision = []
Recall = []
F1 = []

results_list = [AUC_ROC, AUC_PR, CLL, Threshold, Precision, Recall, F1]

for count in range(1, num_folds+1):
    f = open("./folds/fold_" + str(count) + "/test/test_infer_dribble.txt")
    lines = f.readlines()

    results = lines[-8:-1]

    results  = "".join(results)

    pattern = re.compile(r"= (-?[\d\.\w]*)")

    matches = pattern.findall(results)

    for i, match in enumerate(matches):
        if match == "NaN":
            match = "-99999"
        results_list[i].append(match)

for i, l in enumerate(results_list):
    results_list[i] = [float(i) for i in l]

def list_average(l):
    average = sum(l)/float(len(l))

    if average < -100:
        return "NaN"

    return str(sum(l)/float(len(l)))

print("Average Results")
print("AUC ROC = " + list_average(results_list[0]))
print("AUC PR = " + list_average(results_list[1]))
print("CLL = " + list_average(results_list[2]))
print("Precision = " + list_average(results_list[3]) + " at average threshold " + list_average(results_list[4]))
print("Recall = " + list_average(results_list[5]))
print("F1 = " + list_average(results_list[6]))

with open('results.txt', 'w') as f:
    f.write("Average Results")
    f.write("\nAUC ROC = " + list_average(results_list[0]))
    f.write("\nAUC PR = " + list_average(results_list[1]))
    f.write("\nCLL = " + list_average(results_list[2]))
    f.write("\nPrecision = " + list_average(results_list[3]) + " at average threshold " + list_average(results_list[4]))
    f.write("\nRecall = " + list_average(results_list[5]))
    f.write("\nF1 = " + list_average(results_list[6]))
