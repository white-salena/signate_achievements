import pandas as pd

pred_A = pd.read_csv('competition/opt_2/dataset/submission_A.tsv', sep="\t")
pred_B = pd.read_csv('competition/opt_2/dataset/submission_B.tsv', sep="\t")
pred_C = pd.read_csv('competition/opt_2/dataset/submission_C.tsv', sep="\t")
pred_D = pd.read_csv('competition/opt_2/dataset/submission_D.tsv', sep="\t")

pred = pd.concat([pred_A, pred_B, pred_C, pred_D], ignore_index=True)

pred.to_csv('competition/opt_2/dataset/submissions.tsv', sep='\t', index=False, header=False)