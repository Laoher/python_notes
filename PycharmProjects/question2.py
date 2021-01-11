def print_matrix(ls,target):
    l=0
    r =len(ls)-1
    while l<=r:
        m = int(l+(r-l)/2)

        if ls[m]==target:
            return m
        elif ls[m]>ls[r]:
          if target >=ls[l] and target<ls[m]:
              r = m-1
          else:
              l=m+1
        else:
            if target>ls[m] and target<=ls[r]:
                l = m+1
            else:
                r = m-1
    return -1


a = print_matrix([3, 4, 5, 6, 0, 1, 2],2)
print(a)