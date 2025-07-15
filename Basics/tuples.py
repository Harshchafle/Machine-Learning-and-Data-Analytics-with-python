#   Tuples
tup1 = (1,2,3)
print(type(tup1))
print(tup1)
#tup1[0] = 9  ## tuples cant be changed
#if we want to change the tuple then convert it to list then reconcert it into tuple

# Typcasting of tuple into a list
list1 = list(tup1)
list1[0] = 9
tup1 = tuple(list1)
print(type(tup1))
print(tup1)