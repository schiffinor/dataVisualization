"""
dictionary_practice.py
Roman Schiffino
CS 251: Data Analysis and Visualization
Lab 1a
"""

'''Task L1a
------------------------------------------------------------------------------------------------------------------------
Below this multi-line comment, create a Python dictionary that has:
- keys (strings) taken from the `fruit` list
- values (ints) taken from the `hundreds` list.
You should make use of the lists provided below.

Print out on two separate lines the keys and the values of your dictionary.

Uncomment the set of print statements below before Task L1b then execute this file to check your work.
'''
# KEEP ME
fruit = ['apple', 'banana', 'cantaloupe']
hundreds = [100, 200, 300]

# WRITE YOUR CODE HERE
newDict = dict(zip(fruit, hundreds))
print(newDict.keys())
print(newDict.values())

# UNCOMMENT BELOW TO CHECK WORK
print(75 * '-')
print('Task L1a')
print("You should see:\ndict_keys(['apple', 'banana', 'cantaloupe'])\ndict_values([100, 200, 300])")
print(75 * '-')

'''Task L1b
------------------------------------------------------------------------------------------------------------------------
Write code below that determines whether the list associated with the key 'cs251' in the below dictionary called
`number_dict` contains the number 56. If the list has it, print 'In there!' otherwise print 'Not in there!'.

Do not modify `number_dict` dictionary below.

Uncomment the set of print statements to check your work.

HINT: Checking whether the list contains 56 can be done concisely/efficiently with only one line of code.
'''

# KEEP ME
number_dict = {'hi': 1, 'there': [10, 20], 'cs251': [62, 96, 35, 70, 97, 89, 35, 34, 67, 62, 44, 4, 42, 33, 32, 78, 64]}

# YOUR CODE HERE
print("In there!" if 56 in number_dict["cs251"] else "Not in there!")

print(75 * '-')
print('Task L1b')
print('Your output above should be:\nNot in there!')
print(75 * '-')

'''Task L1c
------------------------------------------------------------------------------------------------------------------------
Write code below that determines the index (position) of the 4 in list associated with the key 'cs251' in the same
`number_dict` dictionary from above. Print out the index that you find.

Uncomment the set of print statements to check your work.

HINT: There is a helpful Python list method...
'''
# YOUR CODE HERE

print("The index of the 4 is {}".format(number_dict["cs251"].index(4)))

print(75 * '-')
print('Task L1c')
print('Your output above should be:\nThe index of the 4 is 11')
print(75 * '-')

'''Task L1d
------------------------------------------------------------------------------------------------------------------------
Create a Python dictionary with:
- keys (strings): 'shoe_store', 'music_store'
- make the values associated with the above 2 keys empty lists.

Loop through the `delivery` list below. If the current item is 'shoes', append it to the list associated with 'shoe_store',
otherwise append the item to the list associated with 'music_store'. After adding the current item in the `delivery` list
to your dictionary, print the lists associated with 'shoe_store' and 'music_store' from the dictionary on separate lines.
So there should be 10 print outs = 2 (number of stores) x 5 (number of delivery items)

Uncomment the set of print statements below to see the expected results.
NOTE: Matching the exact formatting of the test code print outs is not important here.
'''
# KEEP ME
delivery = ['shoes', 'music', 'music', 'shoes', 'music']

# YOUR CODE HERE
shoe_store = []
music_store = []
for i, item in enumerate(delivery):
    print("After adding item {}:".format(i))
    if item == 'shoes':
        shoe_store.append(item)
    elif item == 'music':
        music_store.append(item)
    else:
        raise ValueError("Invalid delivery item: {}".format(item))
    print(" store_store: {}\n music_store: {}".format(shoe_store, music_store))

print(75*'-')
print('Task L1d')
print("You should see:\nAfter adding item 0:\n store_store: ['shoes']\n music_store: []\nAfter adding item 1:",
      "\n store_store: ['shoes']\n music_store: ['music']\nAfter adding item 2:\n store_store: ['shoes']\n music_store:",
      " ['music', 'music']\nAfter adding item 3:\n store_store: ['shoes', 'shoes']\n music_store: ['music', 'music']\nAfter",
      "adding item 4:\n store_store: ['shoes', 'shoes']\n music_store: ['music', 'music', 'music']")
print(75*'-')
