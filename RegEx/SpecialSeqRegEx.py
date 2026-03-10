import re

a = "harry9 po@rter7"
print(re.findall("\Ah",a)) # matches the beginning
print(re.search("\br",a))
print(re.search("\By",a))
print(re.findall("\d",a))  # matches decimal numbers
print(re.findall("\D",a))  # opposite of the above
print(re.findall("\s",a))  # matches white spaces
print(re.findall("\S",a))  # matches non-white spaces
print(re.findall("\w",a))  # matches alpha numeric
print(re.findall("\W",a))  # matches non-alpha numeric
print(re.findall("7\Z",a)) # matches string end