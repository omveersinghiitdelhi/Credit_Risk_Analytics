config={
    'dev_sample':'train', # Development Sample
    'val_samples':['test','oot1','oot2'], # Validation Samples
    'target':'DEF_OUTCOME', # name of the target variables
    'target_type':'binary',
    'weight':'wgt',
    'excludelist':['ACCOUNT_NO','CUSTOMER_ID','DATE'],
    'random_seed':12345 # random seed for replication
}