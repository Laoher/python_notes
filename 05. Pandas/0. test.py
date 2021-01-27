import heapq as hq

ls = [3,9]

data = [4,6,2,8,9,10,3,1,7]

for i in data:
    hq.heappush(ls,i)
    print(ls)

hq.heapreplace(ls,1)
print(hq.nlargest(3,ls))
print(hq.nsmallest(3,ls))