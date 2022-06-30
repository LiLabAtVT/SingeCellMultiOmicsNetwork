# this script summarizes the motif mapping results. 
# there are three for loops that each loop navigates one layer of the folder structure produced by motif search by FIMO
# the first folder contains the following sub-folders:
#.     ABI3VP1_tnt, AP2EREBP_tnt, ARF_tnt, ...
# the second folder layer, inside each of the above folders: 
#      AT5G18090_col_a, AT5G25475_col_a, ... each folder is a gene in the gene family
#
# the third folder layer, inside each of the gene folder:
#.    NGA4_col_a_Arabidopsis_upstream_out, NGA4_col_a_Sisymbrium_upstream_out
# each folder is a result for one species.
#
#
# To execute the script, go in to the folder that contains "ABI3VP1_tnt, AP2EREBP_tnt, ARF_tnt"
# command:
# $ python SummaryGene2Motif_02202022.v3.py
#
# The output of this script are motif summary files, such as:
#.     AT5G18090_col_a.MotifSummary.csv
# this is a comma separated three column file:
#
# ABI3VP1_tnt.FUS3_col_a_m1,5p1k_SpY|Sp1g00010,1
# ABI3VP1_tnt.FUS3_col_a_m1,5p1k_SpY|Sp1g00020,1
# ABI3VP1_tnt.FUS3_col_a_m1,5p1k_SpY|Sp1g00050.b,1
# ABI3VP1_tnt.FUS3_col_a_m1,5p1k_SpY|Sp1g00070.a.x,1
#
# Column 1: motif name, Column 2: promotername|genename, Column 3, number of motif occurance.
# 
# this output of these files can be copied to a folder for downstream summary by
# find . -type f -name *.MotifSummary.csv | xargs -I {} cp {} ../SummaryFolder
# 
# downstream processing use script:
# Motif2Matrix_02202022.py
#

import pandas as pd
import os
import sys

ppfolders = os.listdir('.') # folder contains all motifs

for eachppf in ppfolders: 
    if not os.path.isdir(eachppf): # skip non folder
        continue
    if not eachppf.find('_tnt')>-1: # skip ecoli folders
        continue
    os.chdir(eachppf) # enter the motif-gene-family folder.
    pfolders = os.listdir('.')

    for eachpf in pfolders:
        if not os.path.isdir(eachpf):
            continue

        print(eachpf)
        os.chdir(eachpf) # enter the specific gene motif folder

        folders=os.listdir('.')
        print(folders)

        # select only the folders that are useful 
        ftokeep = ['Arabidopsis_upstream','Camelina_upstream','Eutrema_upstream','Sisymbrium_upstream','upstream_Scp']
        for eachfolder in folders:
            if not any(x in eachfolder for x in ftokeep): # only work with the selected folders. 
                continue 
            #print(eachfolder)
            fimofile = eachfolder+'/'+'fimo.tsv'
            if os.path.exists(fimofile):
                try:
                    motiffile = pd.read_csv(fimofile,sep='\t', comment='#')
                except pd.errors.EmptyDataError: # handle empty files
                    continue
            else:
                continue
            #print(motiffile.iloc[1:3,1:4])
            motifsummary = motiffile['sequence_name'].value_counts()
            test1 = motiffile.groupby(['motif_id','sequence_name'])['sequence_name'].count()
            test1.to_csv('../'+eachpf+'.'+'MotifSummary.csv',mode='a', header=False)
    
        os.chdir('..')
    os.chdir('..')
#alldf.to_csv('test.csv')
