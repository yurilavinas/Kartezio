import latexify

@latexify.function
# active learning, uncertanty metrics
def  count_different_pixels_weighted(array1, array2):
    different_pixels = 0

    for i in range(len(array1)):
        for j in range(len(array1[i])): 
            if array1[i][j] != array2[i][j]:
                if array1[i][j] == 0 or array2[i][j] == 0:
                    different_pixels += 2
                else:
                    different_pixels += 1

    return different_pixels/len(array1)/len(array1[0])

@latexify.function
def  count_different_pixels(array1, array2):
    different_pixels = 0

    for i in range(len(array1)):
        for j in range(len(array1[i])): 
            if array1[i][j] != array2[i][j]:
                different_pixels += 1

    return different_pixels/len(array1)/len(array1[0])
# active learning - end

print(count_different_pixels_weighted)
print(count_different_pixels)