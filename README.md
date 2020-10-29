# DSPLMF
 A novel method for cancer Drug Sensitivity Prediction using Logistic Matrix Factorization
 
 
The implemenation of DSPLMF is presented in "main.python" file. Which runs the DSPLMF algorithm on cell lines and drugs and returns the probalities that cell lines are to sensitive to drugs. The file "functions.python" includes the code of fuctions that are used for evaluation and load data and code of logistic matrix factorization algorithm that is regularized by similarity matrices.



Instructions for using the DecisionTreeClassifier Folder:

By ‘DecisionTreeClassifier.py’ code, we estimate the similarity between cell lines based on IC50 values via three features gene expression profile, copy number alteration and single nucleotide mutation data. For this purpose,  you should convert the similarity matrix elements based on IC50 values to 0 and 1.  To do this, the values of each row of the matrix are sorted in descending order, and then the t-largest values are set to 1, and the remaining values are set to 0 (t=40 for GDSC dataset and t=20 for CCLE dataset). Then you should construct five .csv files with four columns (‘rna’ , ‘cnv’ , ‘mut’ ,’Liu’).  For example, ‘rna’ column contains the elements of the similarity matrix based on gene expression profile that are arranged in a column. ‘liu’ column contains the values of zero and one that you obtained above and are arranged in a column. Then you should construct ten files ‘train.csv’ and ‘test.csv’ by 5-fold cross-validation. Finally, you can run the ‘DecisionTreeClassifier.py’ code, and you will achieve the matrix (Matrix of k-nearest neighbor.txt) as the estimated matrix from the similarity between IC50 values. 
