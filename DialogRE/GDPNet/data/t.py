import json

with open("pedc_type12.json",'r') as f:
    all = json.load(f)
with open("train.json", 'w') as train_f:
    json.dump(all[0:90],train_f)
with open("dev.json", 'w') as val_f:
    json.dump(all[90:105],val_f)
with open("test.json", 'w') as test_f:
    json.dump(all[105:-1],test_f)
