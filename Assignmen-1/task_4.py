def break_sentence(text, sep):
    piece = ""
    for letter in text:
        if letter == sep:
            if piece != "":
                print(piece)
                piece = ""
        else:
            piece = piece + letter
    if piece != "":
        print(piece)


def combine_words(items, sep):
    output = ""
    index = 0
    while index < len(items):
        output += items[index]
        if index != len(items) - 1:
            output += sep
        index += 1
    print(output)



sentence = input("Please enter sentence:")


words = []
word = ""
for letter in sentence:
    if letter == " ":
        if word != "":
            words.append(word)
            word = ""
    else:
        word += letter
if word != "":
    words.append(word)

combine_words(words, ",")


break_sentence(sentence, " ")