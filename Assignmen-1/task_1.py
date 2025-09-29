def getText():
    while True:
        user_input = input("Type anything (enter 'exit' to stop): ").lower()
        if user_input == 'exit':
            break
        else:
            checkLength(user_input)
def checkLength(text):
    if len(text) < 10:
        text = "Input is too small"
    print(text)

getText()
