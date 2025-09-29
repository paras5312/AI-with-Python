grocery = []
while True:
    user_input=input("Write a word to add or remove ")
    print("Choose 1 to Add, 2 to Delete, or 3 to Exit")
    user_choice = int(input("Select your option "))
    
    if user_choice > 3 or user_choice < 0:
        print("Invalid! please select number from above options")
        
    elif user_choice == 1:
        grocery.append(user_input)
        print(user_input, "is added to your shopping list.")
        
    elif user_choice == 2:
        if user_input in grocery:
            grocery.remove(user_input)
            print(user_input, "is deleted from your shopping list.")
        else:
            print("Item is not in the list.")
    else:
        total = len(grocery)
        print("there are", total, "number of item in the list.")
        break
    