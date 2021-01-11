f = open('test.json', 'r', encoding='utf-8')
f_new = open('test_new.json', 'w', encoding='utf-8')
lines = f.readlines()
for line in lines:
    if line[-1] is '\n':
        line = line[:26].replace('\'', '\"')+line[26:-3].replace('\"', '\'')+'\"'+line[-2:]
    else:
        line = line[:26].replace('\'', '\"') + line[26:-2].replace('\"', '\'') + '\"' + line[-1:]
    print(line)
    f_new.write(line)