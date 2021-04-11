sents1 = []
sents2 = []
with open("src/train_vnf", 'r', encoding='utf-8', errors='ignore') as f1, open('src/train_pos', 'r', encoding='utf-8', errors='ignore') as f2:
    for sent1, sent2 in zip(f1, f2):
        # if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
        # if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
        sents1.append(sent1.strip())
        sents2.append(sent2.strip())

print(sents1)