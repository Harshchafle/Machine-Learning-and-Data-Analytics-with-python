def get_sum(a, b):
    return a+b

def get_product(a, b):
    return a*b

def print_name():
    print("__name__in p00_sample:",__name__)
    
print("file Scanning import statements scans file")
print(get_product(50,10))

if __name__ == "__main__" :
    print(get_sum(10,20))
    print_name()
    print("Inside the sample file!") 