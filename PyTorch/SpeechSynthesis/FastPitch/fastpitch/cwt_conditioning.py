import torch

non_words = ['#','%','&','*','+','-','/','[',']','(',')','!','\\','\'','\"','(',')', ' ']

def upsample_word_label(word_info, cwt_labels):
    cwt_labels  = cwt_labels.tolist() 
    total_symbols = [x[1] for x in word_info]
    total_symbols = sum(total_symbols)
    cwt_labels_upsampled = []
    cwt_index = 0

    for word in word_info:
        if word[0] in non_words:
            x = [4] * word[1]
            cwt_labels_upsampled += [4]
        else:
            if not cwt_index > len(cwt_labels) - 1: #for debugging
                x = [cwt_labels[cwt_index]] * word[1]
                cwt_labels_upsampled += x
                cwt_index += 1
            else:
                print(word_info)
                print(len(word_info), len(cwt_labels))
#    print(cwt_labels_upsampled)
#    print(total_symbols)
#    print(len(cwt_labels_upsampled))
    if len(cwt_labels_upsampled) != total_symbols:
        print("BOOOOOOOOOOOOOOOOO")
    else:
        print("YESSSSSSSSSSSSSSS")

    return cwt_labels_upsampled
