class ParentClass:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def method1(self):
        print("Method 1 from ParentClass")

    def method2(self):
        print("Method 2 from ParentClass")


class ChildClass(ParentClass):
    def __init__(self, param3, param4):
        super().__init__(param3, param4)


# Creating an instance of ChildClass
child_obj = ChildClass("value3", "value4")

# Accessing parameters inherited from ParentClass
print(child_obj.param1)  # Output: value3
print(child_obj.param2)  # Output: value4

# Trying to access methods from ParentClass
# This will raise an AttributeError because methods are not inherited
child_obj.method1()  # Uncommenting this line will raise AttributeError
child_obj.method2()  # Uncommenting this line will raise AttributeError
