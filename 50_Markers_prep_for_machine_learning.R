library(tidyverse)

## Load and rename the matrix ##

matrix_1 <- read.csv("MotifMat_02202022.csv")
head(matrix_1)
matrix_1_Arath <- matrix_1 %>%
  filter(grepl("_AtY|AT", X)) 

sample_ath <- matrix_1[1:20, 1:5]
head(sample_ath)
matrix_renamed <- matrix_1_Arath %>%
  mutate(tmp = gsub("5p1k_AtY\\|", "", X)) %>%
  mutate(Gene = gsub("\\..*", "", tmp)) %>%
  select(-c(tmp,X)) %>%
  select(Gene, everything())
head(matrix_renamed[1:5,1:5])

# Load the positive and negative markers
Endodermis_markers <- read.csv("Endo_Integrated_GeneList_filtered.csv")
Endo_mark <- Endodermis_markers %>%
  select(Gene)
head(Endo_mark)

Cortex_markers <- read.csv("Cortex_Integrated_GeneList_filtered.csv")
Cort_mark <- Cortex_markers %>%
  select(Gene)
head(Cort_mark)

######### Adding indicator variables

Endodermis_markers_ind <- Endo_mark %>%
  mutate(Prediction = 1) %>%
  select(Gene, Prediction) 

head(Endodermis_markers_ind)



Cortex_markers_ind <- Cort_mark %>%
  mutate(Prediction = 0) %>%
  select(Gene, Prediction) 

head(Cortex_markers_ind)

# Join them together 

Combined_markers <- bind_rows(Endodermis_markers_ind, Cortex_markers_ind)
head(Combined_markers)

## Combine with the matrix file
Combined_markers_mat <- Combined_markers %>%
  inner_join(matrix_renamed, by = "Gene")

write_csv(Combined_markers_mat , "Arabidopsis_genes_matrix_combined.csv")
