#Implementing a class
class animal:
    length: 30
    def running_distance(self,b):
        return b + 10

#Generating an object
dog = animal()
print(dog.length)
print(dog.running_distance(200))
