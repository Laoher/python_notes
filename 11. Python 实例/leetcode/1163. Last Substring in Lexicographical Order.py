class Solution:
    def lastSubstring(self, s: str) -> str:
        max_substring = ""
        max_alphabet = ""
        for index in range(len(s)):
            if s[index] >= max_alphabet:
                max_alphabet = s[index]
                if s[index:] > max_substring: max_substring = s[index:]

        return max_substring







a =Solution()
print(a.lastSubstring("lteaeeeteeecode"))