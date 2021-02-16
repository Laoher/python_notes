# from pathlib import Path
#
# training_path =Path("/Users/Tyler/Documents/output_pdf_to_image/20210201 - Probation Assessment Report_Wang Xing (Tyler).doc")
#
# print(training_path.parent)
#
# print(training_path.parent / ("whafv"+".txt"))
#
# open(training_path.parent / ("whafv"+".txt"))

x = [0,3,45,None]

y = [i for i in x if i is not None and i-3==0]

print(y)
