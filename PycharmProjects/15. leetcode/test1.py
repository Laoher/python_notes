def match_pattern(pattern, str):
    if not str:
        if not pattern:
            return True
        else:
            return False
    match = {}


    str = str.split(" ")
    if len(pattern) != len(str):
        return False

    for i in range(len(pattern)):
        if pattern[i] not in match.keys():
            if str[i] in match.values():
                return False
            match[pattern[i]] = str[i]
        else:
            if match[pattern[i]] != str[i]:
                return False
    return True

print(match_pattern("abba", "dog cat cat dog"))
print(match_pattern("abba", "dog cat cat fox"))
print(match_pattern("","a"))
print(match_pattern("","ads"))
print(match_pattern("fsf","ab ab ca"))








