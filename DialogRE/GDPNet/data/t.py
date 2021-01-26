import json
import random
random.seed(0)

def divide():
    with open("./data/pedc.json",'r') as f:
        all = json.load(f)
        random.shuffle(all)

    total=len(all)
    section=total//6

    with open("./data/train.json", 'w') as train_f:
        json.dump(all[:section*4],train_f)
    with open("./data/dev.json", 'w') as val_f:
        json.dump(all[section*4:section*5],val_f)
    with open("./data/test.json", 'w') as test_f:
        json.dump(all[section*5:],test_f)
