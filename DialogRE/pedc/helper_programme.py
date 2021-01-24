import json

with open("data/dev.json",'r',encoding='utf8') as json_file:
    
    data = json.load(json_file)
    start = len(data) // 2 + 30 #starting position
    for i in range(0, min(130,len(data) - start)):  # 130: we want to annotate 130 conversatiosn in total
        annotated_data = []
        if i != 0:  #if first time， use blank data. (it will wipe out the previous data inside data_generated.json)
            with open("data_generated.json") as outfile:
                annotated_data = json.load(outfile)
        conversation = data[start + i] 
        trigger_word_data = []
        for j in range(0, len(conversation[0])):  #loopiong through each line within a conversation
            print(conversation[0][j])
            print("trigger_word:")
            trigger_word = input()
            if (trigger_word):
                trigger_word_data.append({"trigger_word": trigger_word, "sent_id": j})
        conversation[1] = trigger_word_data
        print(conversation[1])
        annotated_data.append(conversation)
        with open("data_generated.json", "w") as outfile: #save the file each time we finished annotating 1 conversation, 
                                                        #if you can't finish annotating 130 conversations, just quit and the data will be saved
            json.dump(annotated_data, outfile, indent=1)

