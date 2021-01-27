import urllib.request
import re
keyname='短裙'
keyname=urllib.request.quote(keyname)
# print(keyname)
# headers=('User-Agent',"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36")
# opener=urllib.request.build_opener()
# opener.addheaders=[headers]
# urllib.request.install_opener(opener)
for i in range(0,3):
    url='https://s.taobao.com/search?q='+keyname+'&s='+str(i*44)
    data=urllib.request.urlopen(url).read().decode('utf-8','ignore')
    pat='pic_url":"//(.*?)"'
    imagelist=re.compile(pat).findall(data)
    print("imagelist")
    for j in range(0,len(imagelist)):
        thisimage=imagelist[j]
        print("hah")
        thisurl='http://'+thisimage
        file='D:/labeling & extracting & downloading tools/crawler/pic/'+str(i)+str(j)+'.jpg'
        urllib.request.urlretrieve(thisurl,file)