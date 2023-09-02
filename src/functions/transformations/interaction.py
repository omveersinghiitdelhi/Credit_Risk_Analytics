"""This function creates interactions of numeric vars"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def interaction_variables_creation(input_df,cols_to_interact):
    """ Creation of pairwise interactions between numeric variabbles"""
    poly=PolynomialFeatures(interaction_only=True)
    output_nparray=poly.fit_transform(input_df[cols_to_interact])
    powers_nparray=poly.powers_
    input_feature_names=cols_to_interact
    target_feature_name=['']
    for feature_distillation in powers_nparray[1:]:
        intermediary_lebel=""
        final_label=""
        for i in range(len(input_feature_names)):
            if feature_distillation[i]==0:
                continue
            else:
                variable=input_feature_names[i]
                power=feature_distillation[i]
                intermediary_lebel=variable
                if final_label=="":                      # if the final lebel isnt yet specified
                    final_label=intermediary_lebel[0:30] # limit max number of charactors
                else:
                    final_label=final_label+'_x_'+intermediary_lebel
                    final_label=final_label[0:40]  # limit the number of charactor

        target_feature_name.append(final_label)
    output_df=pd.DataFrame(output_nparray,columns=target_feature_name)
    output_df.drop('', axis=1, inplace=True)
    nonit=list(set(input_df.columns).difference(cols_to_interact))
    output_df=pd.merge(input_df[nonit],output_df,left_on=input_df[nonit].index,right_on=output_df.index)
    output_df.drop('key_0',axis=1,inplace=True)
    return output_df