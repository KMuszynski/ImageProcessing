def imgErosion(img, struct):
    rows = len(img)
    cols = len(img[0])
    
    struct_len = len(struct[0])
    
    eroded_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            fits = True
            for k in range(struct_len):
                if i < rows and j + k < cols:
                    if img[i][j + k] != struct[0][k]:
                        fits = False
                        break
                else:
                    fits = False
                    break
            
            if fits:
                eroded_img[i][j] = 1
    
    return eroded_img

def imgDilation(img, struct):
    rows = len(img)
    cols = len(img[0])
    
    struct_len = len(struct[0]) 
    
    dilated_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if img[i][j] == 1:
                for k in range(struct_len):
                    if i < rows and j + k < cols:
                        if dilated_img[i][j + k] == 0:
                            dilated_img[i][j + k] = 1
    
    return dilated_img

img = [
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,1,1,0,0,0,0],
    [0,1,1,1,1,0,0,0],
    [0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0]
]

struct = [[1, 1]]

eroded_img= imgErosion(img, struct)
dilated_img = imgDilation(img, struct)

print("\nErosion Result:")
for row in eroded_img:
    print(row)

print("\nDilation Result:")
for row in dilated_img:
    print(row)