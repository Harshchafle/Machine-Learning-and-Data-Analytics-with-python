print("Enter the marks")
number = int(input())
print("Number :",number)

if(number < 0):
    print("Negative number")
    exit()
elif(number>=0 & number<=80):
    print("C")
    exit()
elif(number>=80 & number<=90):
    print("B")
    exit()
elif(number>=90 & number<=100):
    print("A")
    exit()
else:
    print("Invalid input")
    exit()