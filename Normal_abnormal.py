import numpy as np
import pandas
import baselines as b

#to_id={k:i for i,k in enumerate(data['report_id'].to_list())}
data_with_tags = np.load('/home/zaheer/pythonCode/DCM/Two_Images/Multi/Data_with_tags.npy',allow_pickle=True)

data_sat = pandas.read_json('/home/zaheer/pythonCode/DCM/Two_Images/word_sent_tags_MRA_new.json')
generated_sat_words = data_sat['Cand_word'].to_list()




reference_sat = [r[3:-3] for r in data_sat['Ref_word'].to_list()]
tags_sat = [data_with_tags[()][r] for r in data_sat['report_id']]
is_normal_sat = np.array(['normal' in t for t in tags_sat])

print(b.all_scores(reference_sat, generated_sat_words))
print(b.all_scores(np.array(reference_sat)[is_normal_sat], np.array(generated_sat_words)[is_normal_sat]))
print(b.all_scores(np.array(reference_sat)[np.logical_not(is_normal_sat)], np.array(generated_sat_words)[np.logical_not(is_normal_sat)]))

