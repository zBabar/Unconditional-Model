import numpy as np
import pandas as pd
import baselines as b
import pickle as pkl



with open('/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/SAT_Findings/word/Sample1/train/train.annotations.pkl', 'rb') as f:
    reports = pkl.load(f,encoding='latin-1')

with open('/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/SAT_Findings/word/Sample1/test/test.annotations.pkl', 'rb') as f:

    test_reports = pkl.load(f,encoding='latin-1')

with open('/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/word/impression_first/Sample1/test/test.annotations.pkl', 'rb') as f1:
    test_reports = pkl.load(f1,encoding='latin-1')

#print(test_reports)
""""
scores=[]

max_score=0
data=pd.DataFrame([])
for r in list(reports['caption']):

    data['cand']=[r]*reports.shape[0]
    data['reports']=list(reports['caption'])
    bleus,meteor, rouge, ciders=b.all_scores(data['reports'],data['cand'])

    if bleus[3]>max_score:
        max_score=bleus[3]
        print(bleus,meteor, rouge, ciders)
        print(r)

print(reports['caption'][0])


report,idx,bleus=b.find_best_report_bleu(list(reports['caption']),list(reports['caption']))
print(report,bleus)
data=pd.DataFrame([])
data['cand']=[report]*reports.shape[0]

data['reports']=list(reports['caption'])
print(b.all_scores(data['reports'],data['cand']))
"""
# report=b.optimal_report_bleu2(reports['caption'])
# report='of the heart size and pulmonary xxxx are clear there is in the lungs are within normal limits no pneumothorax no focal airspace disease of the cardiomediastinal silhouette is no acute cardiopulmonary abnormality consolidation pleural effusion or mediastinal'
# #report=' '.join(report)
# print(report)
# data=pd.DataFrame([])
# data['cand']=[report]*test_reports.shape[0]
#
# data['reports']=list(test_reports['caption'])
# print(b.all_scores(data['reports'],data['cand']))




################# MIMIC CXR

import re

data=pd.read_csv('/home/zaheer/pythonCode/MIMIC_CXR/processed_data.csv')

print(type(data.findings[0]))

data=data['findings'].to_list()

print(len(data))


random_lst=[]
pattern = '[0-9]'
for inst in data:

    if isinstance(inst, str) :
    #print(type(inst))
    #print(inst)

        inst=inst.lower()
        inst1= inst.replace(',', '').replace("'", "").replace('"', '').replace('.', '')
        inst1 = inst1.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace('___','')
        inst1=re.sub(pattern, '', inst1)

        random_lst.append(inst1)



random_lst=random_lst[:20000]



report='The heart is normal in size. There is a moderate hiatal hernia. The mediastinal and hilar contours appear otherwise unremarkable. The lungs appear clear. There is no pleural effusion or pneumothorax. Severe rightward convex curvature is centered along the mid thoracic spine.'
#report= 'lung volumes are low  this results in crowding of the bronchovascular structures  there may be mild pulmonary vascular congestion  the heart size is borderline enlarged.  The mediastinal and hilar contours are relatively unremarkable innumerable nodules are demonstrated in both lungs, more pronounced in the left upper and lower lung fields compatible with metastatic disease  No new focal consolidation, pleural effusion or pneumothorax is seen, with chronic elevation of right hemidiaphragm again seen  the patient is status post right lower lobectomy  rib deformities within the right hemithorax is compatible with prior postsurgical changes'
report=report.lower()
report= report.replace(',', '').replace("'", "").replace('"', '').replace('.', '')
report = report.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace('___','')
report=re.sub(pattern, '', report)

print(report)
data=pd.DataFrame([])

#data['cand']=[report]*len(random_lst)

#data['reports']=random_lst
#print(b.all_scores(data['reports'],data['cand']))
print(b.all_scores(random_lst,report))