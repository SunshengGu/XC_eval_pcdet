import glob
import numpy

paths = glob.glob('../201*/**/*.bin', recursive=True)
samples = numpy.array([[x.split('/')[-6], x.split('/')[-5], x.split('/')[-1].rstrip('.bin')] for x in paths])
  
indices = numpy.random.permutation(samples.shape[0])
split = int(samples.shape[0] * 0.8)
training_idx, val_idx = indices[:split], indices[split:]
training, val = samples[training_idx], samples[val_idx]

with open("train.txt", "w") as f:
  for idx in training:
    f.write("%s %s %s\n" % (idx[0], idx[1], idx[2]))
    
with open("val.txt", "w") as f:
  for idx in val:
    f.write("%s %s %s\n" % (idx[0], idx[1], idx[2]))
    