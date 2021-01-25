# 语法 str.join(sequence)

str = "-";
seq = ("a", "b", "c"); # 字符串序列
print(str.join( seq ))  # a-b-c

list=['1','2','3','4','5']
print(''.join(list))  #12345

seq = {'hello':'nihao','good':2,'boy':3,'doiido':4}
print('-'.join(seq))  # hello-good-boy-doiido #字典只对键进行连接

print('kill %s' % ' '.join(['111','22']))  # kill 111 22
