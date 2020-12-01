import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score


#gt=pd.read_csv('SAT_GT.csv')
gt=pd.read_csv('MRA_ref_sent_comb.csv')
MIMIC=pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')

baselines=pd.read_csv('baselines1.csv')

#cand=pd.read_csv('SAT_cand_sent.csv')
cand=pd.read_csv('MRA_cand_word_comb.csv')



#####################################################################
# Chexpert accuracies for MRA and SAT
################################################################################
# print(gt.shape)
#
# print(cand.shape)


gt=gt.fillna(0)

#print(gt.head)

gt=gt.replace(-1,0)

#print(gt.head)

cand=cand.fillna(0)

cand=cand.fillna(0)
cand=cand.replace(-1,0)

#print(cand['No Finding'])

classes=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']

Accuracy_class_wise=[]
F1_class_wise=[]

for c in classes:
    Accuracy_class_wise.append(accuracy_score(cand[c],gt[c]))
    F1_class_wise.append(f1_score(cand[c],gt[c]))

# print(Accuracy_class_wise)
# print(np.mean(Accuracy_class_wise[1:]))
#
print(np.mean(F1_class_wise))

Accuracy_row_wise=[]
F1_row_wise=[]
prec_row_wise=[]
recl_row_wise=[]

#print(cand.shape[0])

for r in range(0,(cand.shape[0])):

    g=list(gt.iloc[r,1:])
    #print(g)
    c=list(cand.iloc[r,1:])
    #print(c)
    Accuracy_row_wise.append(accuracy_score(c,g))
    F1_row_wise.append(f1_score(c, g))
    prec_row_wise.append((precision_score(c,g)))
    recl_row_wise.append(recall_score(c,g))


print(np.mean(Accuracy_row_wise))
print(np.mean(prec_row_wise))
print(np.mean(recl_row_wise))
print(len(F1_row_wise))
f1=pd.DataFrame([])
#f1=pd.read_csv('f1_chexprt.csv')

f1['sent']=F1_row_wise

f1.to_csv('f1_chexprt_word_MRA.csv')

#####################################################################
# Chexpert accuracies for baselines IU Xray and MIMIC-CXR
################################################################################

print(baselines.shape)

baselines=baselines.fillna(0)
#
# #print(gt.head)
#
baselines=baselines.replace(-1,0)


Accuracy_row_wise=[]
F1_row_wise=[]
prec_row_wise=[]
recl_row_wise=[]

#print(cand.shape[0])

for r in range(0,(gt.shape[0])-1):

    g=list(gt.iloc[r,1:])

    c=list(baselines.iloc[4,1:])
    #print(c)
    Accuracy_row_wise.append(accuracy_score(c,g))
    F1_row_wise.append(f1_score(c, g))
    prec_row_wise.append((precision_score(c, g)))
    recl_row_wise.append(recall_score(c, g))

print(np.mean(Accuracy_row_wise))

print(np.mean(prec_row_wise))
print(np.mean(recl_row_wise))

Accuracy_row_wise=[]
F1_row_wise=[]
prec_row_wise=[]
recl_row_wise=[]

for r in range(0,(gt.shape[0])-1):

    g=list(gt.iloc[r,1:])

    c=list(baselines.iloc[5,1:])
    #print(c)
    Accuracy_row_wise.append(accuracy_score(c,g))
    F1_row_wise.append(f1_score(c, g))
    prec_row_wise.append((precision_score(c, g)))
    recl_row_wise.append(recall_score(c, g))

print(np.mean(Accuracy_row_wise))

print(np.mean(prec_row_wise))
print(np.mean(recl_row_wise))
print(np.mean(F1_row_wise))


####################################################

#MIMC -CXR

################################################

# print(MIMIC.shape)
#
# MIMIC=MIMIC.fillna(0)
# #
# # #print(gt.head)
# #
# MIMIC=MIMIC.replace(-1,0)
#
#
# Accuracy_row_wise=[]
# F1_row_wise=[]
#
# #print(cand.shape[0])
# #print(MIMIC.columns[2:])
# baselines=baselines[MIMIC.columns[2:]]
# #print(baselines.columns)
# for r in range(0,(MIMIC.shape[0])-1):
#     print(r)
#
#     g=list(MIMIC.iloc[r,2:])
#
#     # print(len(g))
#     # print(g)
#     c=list(baselines.iloc[2,0:])
#     #print(baselines.iloc[2,0:])
#     # print(len(c))
#     # print(c)
#     Accuracy_row_wise.append(accuracy_score(c,g))
#     F1_row_wise.append(f1_score(c, g))
#
# print(np.mean(Accuracy_row_wise))
#
# for r in range(0,(MIMIC.shape[0])-1):
#
#     g=list(MIMIC.iloc[r,2:])
#
#     c=list(baselines.iloc[3,0:])
#     #print(c)
#     Accuracy_row_wise.append(accuracy_score(c,g))
#     F1_row_wise.append(f1_score(c, g))
#
# print(np.mean(Accuracy_row_wise))