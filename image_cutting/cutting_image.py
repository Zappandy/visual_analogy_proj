from PIL import Image
img = Image.open('./myimage.png')
wid, hgt = img.size
img.show()
print(wid, hgt)
nb_row = 4
nbr = 1
Wim = wid//nb_row
Him = hgt//nb_row

for k in range(1, nb_row+1):
    for i in range(0, nb_row-k+1):
        for j in range(0, nb_row-k+1):

            title = 'image'+str(nbr)+'.jpg'
            print(title)

            #

            # box = (0+(Wim)*ki, 0+(Him)*kj,Wim*(ki+1) ,Him*(kj+1) )
            box = (Wim*i, Him*j, Wim*(i+k), Him*(j+k))

            img2 = img.crop(box)
            img2 = img2.convert('RGB')
            img2.save(title)
            nbr += 1
        # img2.show()
