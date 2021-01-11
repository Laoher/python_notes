def LCS(str1, str2):
    N = len(str1)
    M = len(str2)

    L =[]
    for i in range(N+1):
        L.append([None]*(M+1))
    print(L)

    for i in range(N+1):
        for j in range(M+1):
            if i*j==0:
                L[i][j]=0
            elif str1[i-1] == str2[j-1]:
                L[i][j] = L[i-1][j-1] +1
            else:
                L[i][j] = max(L[i-1][j],L[i][j-1])
    return L[N][M]

str1 = "ABCD"
str2 = "EDCA"
print(LCS(str1, str2))
