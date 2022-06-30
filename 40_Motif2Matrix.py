# this script summarizes the motifs into a matrix,
# each row is a gene, each column is a motif
# the data in matrix is the number of motif occurance
# 
# to run this script, execute outside the MotifSummary folder. 
# $ python Motif2Matrix_02202022.py
#
# output:
# MotifMat_02202022.csv

import pandas as pd
import os
import sys

os.chdir('MotifSummary')
motiffiles = os.listdir('.')

# step 1. go through all the *.MotifSummary.csv files,
# find all the gene names and all the motif names
# 
motiflist = [] # motif name list
genelist = [] # gene name list
for eachm in motiffiles:
    if not eachm.find('MotifSummary.csv')>-1:
        continue
    motifname = eachm.split('.')[0] # get motif name
    motiflist.append(motifname) # add motif name to existing list
    #print(motifname)
    eachmf = open(eachm,'r') # open the motifSummary file and check gene names.
    for line in eachmf:
        ct = line.split(',')
        genename = ct[1] # get gene name
        genelist.append(genename) # add gene name to the list
    
    eachmf.close()
    
genelist = sorted(list(set(genelist))) # sort gene name such that genes from same species go together.

#print(genelist)
#print(len(genelist))

#print(motiflist)
#print(len(motiflist))

genelistset = set(genelist) # create set to ensure uniqueness
motiflistset = set(motiflist) # create set to ensure uniqueness


# create empty data from to save the data. 
# rows are genes
# columns are motifs
# fill in 0 for all entries to begin with.
df = pd.DataFrame(0, columns = motiflist, 
                   index = genelist)

# go through the motif list and populate the matrix. 

for eachm in motiffiles:
    if not eachm.find('MotifSummary.csv')>-1:
        continue
    motifname = eachm.split('.')[0]
    #print(motifname)
    eachmf = open(eachm,'r')
    for line in eachmf:
        ct = line.split(',')
        genename = ct[1]
        count = int(ct[2])
        df.at[genename, motifname] = count

print(df)
        
# save the output
df.to_csv('MotifMat_02202022.csv',sep=',')

