import re

# \ => use to drop the special meaning of character following it
# [] => use to represent Character Class
# ^ => matches the begining
# $ => matches the end
# . => matches any character except new line
# | => means OR
# ? => means zero or one occurrence
# * => means any number of occurrences
# + => means one or more occurrences
# {} => indicates number of occurences of preceeding regex to match
# () => enclose the group of regex to match

str = "harsh....chafle@gmail.com"

# findall() -> returns list if contains match otherwise returns empty list
# search() -> matches any characters if not return None

print(re.search("\..",str))       # drops meaning of . and considering it as char
print(re.search("[c]",str))       # searching 'c' in str
print(re.findall("[h]",str))      # returns the list containing all occurence of char 'h'
print(re.search("^f",str))        # finds that str starts with given char/string or not
print(re.search("com$",str))      # finds that str ends with given char/string or not
print(re.findall("com$",str))     # returns the list if ends with com

str2 = "charlie chachachaplin cooa chocolate"
print(re.findall("c.a",str2))     # matches any character except new line
print(re.findall("c..a",str2))
print(re.findall("cha|ate",str2)) # returns list if contains cha or ate
print(re.search("cha|ate",str2))  # matches str2 if contains cha or ate
print(re.findall("ch?a",str2))
print(re.findall("ch*a",str2))
print(re.findall("ch+o",str2))
print(re.findall("h{1,3}",str))
print(re.findall("ch(a|o)",str2)) 