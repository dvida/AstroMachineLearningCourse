import numpy as np



### Introduction to object oriented programming ###



class UpTheGame:
    """ A class which holds the game parameters. The game consits of a 5x5 grid and the player moves around
    the grid by pressing the arrow keys. The player collects baloons that are randomly placed on the grid.
    The game ends when the player fills the air meter. Each baloon holds a random amount of air. """
    def __init__(self):
            
        # Game parameters
        self.grid_size = 5
        self.air_meter = 0
        self.air_meter_max = 100

        # Player position
        self.player_pos = self.randomPos()

        # Baloon positions
        self.baloons = []

        # Maximum number of baloons
        self.baloon_max = 3

        # Game status
        self.game_over = False


    def randomPos(self):
        """ Generate a random position inside the grid. """

        return [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]


    def movePlayer(self, x_direction, y_direction):
        """ Move the player by one grid cell. """

        # Move the player
        self.player_pos[0] += x_direction
        self.player_pos[1] += y_direction

        # Wrap position to be inside the grid
        self.player_pos[0] %= self.grid_size
        self.player_pos[1] %= self.grid_size

        # Check if the player is on a baloon
        if self.player_pos in self.baloons:

            # Remove the baloon from the grid
            self.baloons.remove(self.player_pos)

            # Add air to the air meter
            self.air_meter += random.randint(5, 20)

            # Check if the air meter is full
            if self.air_meter >= self.air_meter_max:
                self.game_over = True

    def addBaloon(self):
        """ Add a baloon to the grid. """

        # Generate a random position
        pos = self.randomPos()

        # Check if the position is already occupied
        if (pos not in self.baloons) and (pos != self.player_pos):
            self.baloons.append(pos)
        
        else:
            self.addBaloon()

    def printGrid(self):
        """ Print the current grid. """

        # Print the grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):

                # Check if the player is on this cell
                if [x, y] == self.player_pos:
                    print('P', end=' ')

                # Check if a baloon is on this cell
                elif [x, y] in self.baloons:
                    print('B', end=' ')

                # Empty cell
                else:
                    print('-', end=' ')

            print('')

        # Print the air meter
        print('Air meter: ', self.air_meter, '/', self.air_meter_max)

    def play(self):
        """ Play the game. """

        # Add a baloons to the grid
        for i in range(self.baloon_max):
            self.addBaloon()

        # Print the grid
        self.printGrid()

        # Loop until the game is over
        while not self.game_over:

            # Get the player's input (don't wait for enter)
            move = input('Move the player (w, a, s, d): ')

            # Move the player
            if move == 'w':
                self.movePlayer(0, -1)
            elif move == 'a':
                self.movePlayer(-1, 0)
            elif move == 's':
                self.movePlayer(0, 1)
            elif move == 'd':
                self.movePlayer(1, 0)

            # Add a baloon to the grid if there are less than the maximum number of baloons
            if len(self.baloons) < self.baloon_max:
                self.addBaloon()

            # Print the grid
            self.printGrid()

        # Print the game over message
        print('You have filled the air meter!')


game = UpTheGame()
game.play()





# Operator overloading and printable representation of an object

class Sphere:

    def __init__(self, volume):

        self.volume = volume
        self.radius = (self.volume/(4/3*np.pi))**(1/3)
        

    def __add__(self, other):

        merged = Sphere(self.volume + other.volume)

        return merged


    def __repr__(self):
        return 'Sphere of volume ' + str(self.volume) + ' m^3 and radius ' + str(self.radius) + ' m'


# More on operator overloading: https://docs.python.org/3/library/operator.html


s1 = Sphere(10) # 10 m3 volume
s2 = Sphere(2) # 2 m3 volume

print(s1)
print(s2)

# Add to spheres together
s3 = s1 + s2

# Check the new radius
print(s3)


# Class inheritance

class AstroObj:
    def __init__(self, name, ra, dec):

        self.name = name
        self.ra = ra
        self.dec = dec

    def angDist(self, other):
        """ Calculate the angular distance between two astronomical objects. """

        ra1 = np.radians(self.ra)
        dec1 = np.radians(self.dec)

        ra2 = np.radians(other.ra)
        dec2 = np.radians(other.dec)

        ang = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)

        return np.degrees(np.arccos(ang))


class Star(AstroObj):

    def __init__(self, name, ra, dec, spec_type):
        
        # Extend AstroObj
        AstroObj.__init__(self, name, ra, dec)
        
        self.spec_type = spec_type



class Galaxy(AstroObj):

    def __init__(self, name, ra, dec, z):

        # Extend AstroObj
        AstroObj.__init__(self, name, ra, dec)

        self.z = z



s1 = Star('Sirius', 101.2875, -16.7161, 'DA2')
g1 = Galaxy('NGC660', 25.7583, +13.645, 0.003)

print(s1.angDist(g1))


#########################################


# Everything in Python is an object!
# E.g. we can do something like this:

a = [1, 2, 3]
    
# This will give us the length of list 'a', because it is stored as its attribute
print(a.__len__)


#########################################


# Open and show the separate pandas file ...


#########################################




### List comprehension ###
# Transforming one list to another


# List of even numbers from 1 to 100
evens = [x for x in range(1, 101) if x%2 == 0]

# For easier understanding of the line above, let's convert it to words:
# "Take a number in a range from 1 to 100, only if it is divisible by 2

print(evens)


### C/P to show the equivalent code
evens = []
for x in range(1, 101):
    if x%2 == 0:
        evens.append(x)

print(evens)

###


# Let's unravel a 2D list
a = [[1, 2], [3, 4]]

a = [x for row in a for x in row]

print(a)



# The classic "Mathematicians order pizzas" joke:
# An infinite number of mathematicians enter a pizzeria. The first mathematician orders 1 pizza. The second
# one orders 1/2 of a pizza, the third one orders 1/4, the fourth one orders 1/8, etc.
# The server quickly looses this temper and just brings them 2 pizzas. Was he right?

pizzas = [1.0/(2**x) for x in range(50)]

# We see that the number quickly converges to 0, so we can use only 100 numbers
print(pizzas)

# The sum of all pizzas
print('Infinite pizzas:', sum(pizzas))



#########################################

# Question:
# Describe what will the 'form' list contain

lst = [4.1756, 2.3412, 8.5754, 7.124531]

form = ["x[{:d}] = {:5.2f}".format(i, x) for i, x in enumerate(lst)]

print(form)


#########################################


### Dictionaries ###
# A collection of (key: value) pairs

num2word = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}

# Print 'zero'
print(num2word[0])

# Go through all keys in the dictionary and return values
for key in num2word:
    print(num2word[key])


#########################################


#########################################

### Generators ###
# Functions which return next value in a sequence upon each call

def squares():
    """ Generator which returns the square value of every integer, starting with 1. """

    i = 1

    while True:

        # Return a squared number
        yield i**2

        i += 1

# Init the generator
sq = squares()

print(next(sq))
print(next(sq))
print(next(sq))
print(next(sq))



# Generator which exchanges between True and False upon every call
def truthGenerator():

    while True:
        yield True
        yield False

# Init the generator
truth = truthGenerator()

print(next(truth))
print(next(truth))
print(next(truth))
print(next(truth))



# Let's replace the lottery hostess with a robot...

import random

def lottery():

    # Returns 6 numbers between 1 and 40
    for i in range(6):
        yield random.randint(1, 40)

    # Returns a 7th number between 1 and 15
    yield random.randint(1,15)


for rand_num in lottery():
    print("And the next number is...",  rand_num)


# Now we can run our own illegal gambling den!



#########################################


### Sets ###
# Lists of unique elements

a = [1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 7]

# Convert a to a set
b = set(a)


# WARNING!
# We cannot index sets!
# This return an ERROR:
# print(b[0])

# If you want it back as a list, you could do:
# b = list(set(a))


c = set([2, 3, 10])

# Get the difference of two sets
print(b.difference(c))

# Get the intersection of two sets
print(b.intersection(c))

# More: check if one set is a subset of another, check if they are disjoint (their intersection is null)