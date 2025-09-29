items = [10,14,22,33,44,13,22,55,66,77]

while True:
    option  = int(input("Please enetr the order number from (1-10) or 0 to exit."))
    
    if option  == 0:
        total = sum(items)
        print("Total: ",total)
        payment = int(input("what is the payment."))
        if payment > total:
            print("change", payment - total)
        else:
            print("insufficent amount")
        break
    
    
    elif option > 10:
        print("Shopping list has only 10 item, please choose from 1-10.")
        
    else:
        print("Product: ", option, "Price: ", items[option -1 ])
        
        
        
        