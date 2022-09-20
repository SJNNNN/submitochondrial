import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from tqdm import tqdm
model_dir = Path('test-cache')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=-1)
labels=[]
sequence=[]
vec_lst=[]
np.set_printoptions(threshold=np.inf)
f = open("./SM424-18/outer/outer.txt", 'r', encoding="utf-8")
f1 = open("./SM424-18/outer/outer_result.txt",'w', encoding="utf-8")
# f1 = open("test-cache/FastText_result.txt", 'w', encoding="utf-8")
lines = f.readlines()
for line in lines:
       sequence.append(line.split(' ')[0])
       labels.append(line.split(' ')[1].strip())
f.close()
for i in tqdm(range(len(labels)), desc='elmo'):
    start = i * 1  # 批量embedding
    batch = sequence[start:start+1]
    if len(batch) == 0:
        continue
    vec = embedder.embed_sentence(batch)
    a=np.array(vec).sum(axis=0).mean(axis=0)
    b=np.reshape(a,(1,1024))
    f1.writelines(str(b.tolist()).replace('[','').replace(']','').replace(',',' ')+'\n')
f1.close()
f3 = open("./SM424-18/outer/outer_result.txt", 'r', encoding="utf-8")
f4 = open("./SM424-18/outer/outer_seqvec_end.txt",'w', encoding="utf-8")
lines = f3.readlines()
for i,line in enumerate(lines):
    f4.writelines(line.strip()+" "+labels[i]+'\n')
f4.close()
