#(1) Write a python program to find the roots of quadratic equation entered by the user.

def findRoots():
    a = int(input("Enter the value of Coefficient of x^2 : "))
    b = int(input("Enter the value of Coefficient of x : "))
    c = int(input("Enter the value of Constant : "))
    print("Quadratic equation : ",a,"x^2 + ",b ,"x + ",c)
    # descriminant
    d = pow(b,2) - 4*a*c

    # Checking the conditions for roots
    if(d == 0):
        x = (-b/(2*a))
        print("Roots are Real and Equal")
        print("x = ",x)
    elif(d < 0):
        d = d*(-1)
        x1 = complex(-b, pow(d,0.5))/(2*a)
        x2 = complex(-b, -pow(d,0.5))/(2*a)
        print("Complex Roots and unequal")
        print("x1 = ",x1)
        print("x2 = ",x2)
    elif(d > 0):
        d = sqrt(d)
        x1 = ((-b+d)/(2*a))
        x2 = ((-b-d)/(2*a))
        print("Roots are Real and unequal")
        print("x1 = ",x1)
        print("x2 = ",x2)

################################################################

#(2) Write a program to calculate Manhattan distance between two points A(x1,y1) , B(x2,y2) , AB = |x1 - x2| + |y1 - y2|

def calculateManhattanDistance(x1, y1, x2, y2):
    return abs(x1-x2)+abs(y1-y2)

#################################################################

#(3) To generate Report card of student Accept from user name of student marks in 3 subjects. generate percentage and add incentive of 0.5% on the percentage and compute final percentage . print the score card in following format

def generateReportCard():
    name = input("Enter the name of Student : ")
    marks = []
    marks.append(int(input("Enter marks in Maths : ")))
    marks.append(int(input("Enter marks in English : ")))
    marks.append(int(input("Enter marks in Science : ")))
    ttl = 0
    for i in range(0,3):
        ttl += marks[i]
        if(marks[i]<0 or marks[i] > 100):
            print(":Invalid Marks Entered !\nCant Generate Report Cards ")
            return
    
    percentage = ttl / 3
    incentive = percentage * 0.005
    perWithIncentive = percentage + percentage * 0.005
    if(perWithIncentive > 100):
        perWithIncentive = 100

    print("* * * * * * * * * * * * * * * * * * * * * *")
    print("Name : ",name)
    print("* * * * * * * * * * * * * * * * * * * * * *")
    print("Mathematics : ",marks[0])
    print("English : ",marks[1])
    print("Science : ",marks[2])
    print("* * * * * * * * * * * * * * * * * * * * * *")
    print("Total : ",ttl,"/300")
    print("Percentage : ", percentage)
    print("* * * * * * * * * * * * * * * * * * * * * *")
    print("Incentives : 0.5% : ", incentive)
    print("Final Percentage : ", perWithIncentive)
    print("* * * * * * * * * * * * * * * * * * * * * *")

#########################################################################
#(4) Program to find sum of a series

def findSum(x,n):
    if(n < 0):
        print("Series Not Matched for a given inputs")
    if(n == 0):
        print("Sum : Infinity")
    elif(n ==1):
        print("Sum : ",x)
    elif(n > 0):
        sum = 1
        for i in range(2, n+1):
            sum += pow(x,i)/i
        print("Sum : ",sum)

print(findSum(3,2))
#generateReportCard()
#print(calculateManhattanDistance(0,0,2,2))
#findRoots()
