def find_closest_points(ps, k):
    distance = []
    for i in range(len(ps)):
        element = ps[i]
        distance.append((element,(element[0]**2+element[1]**2)**0.5))
    distance = sorted(distance, key =lambda item:item[1])

    answer =[]
    if len(ps)>=k:
        for i in range(k):
            answer.append(distance[i][0])

    return answer

#ps = [(6,8), (0.5, -0.9),(5,3), (1, 1), (-1, 1.5), (2, 3), (4, -1)]
ps =[]
K = 3

a = find_closest_points(ps,K)
print(a)