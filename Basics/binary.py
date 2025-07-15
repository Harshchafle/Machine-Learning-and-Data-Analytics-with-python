a = int(input("Enter the number : "))
print(a)
print(bin(a))
b = int(input("enter another number : "))
print(b)
print(bin(b))

print("Binary AND :",a & b,bin(a & b))
print("Binary OR :",a | b,bin(a | b))
print("Binary XOR :",a ^ b,bin(a ^ b))

# left shift and rigth shift
x = 5
print("5 : ",bin(x))
print("Right shift: ",bin(x >> 1))

x = 5
print("5 : ",bin(x))
print("left shift: ",bin(x << 1))
