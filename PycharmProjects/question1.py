def print_matrix(matrix):
    answer=[]
    visited = []
    i,j=0
    direction="right"

    while j != len(matrix[0])-1 and (i,j) not in visited:
        answer.append(matrix[i][j])

    while len(visited)<len(ls)*len(ls[0]):

        if (i,j) not in visited:
            visited.append((i,j))
            print(ls[i][j])


        if (i,j) in visited:
            continue
