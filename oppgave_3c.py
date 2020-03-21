import numpy as np

def posisjon(n , L):
    r = np.empty([4*n**3, 3])
    pos = 0
    d = L/n
    print(d)
    i = np.arange(n)
    j = np.arange(n)
    k = np.arange(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                r[pos:pos + 4] = np.array([[i,j,k],[i,0.5+j,0.5+k],[i +0.5,j,0.5+k],[i+0.5,0.5+j,k]]) * d
                pos += 4
    return r

if __name__ == "__main__":
    box = posisjon(3,20)
    print(box)

    infile = open("posisjon_3c.txt","w")
    infile.write(f"{len(box)}\n")
    infile.write(f"type x y z\n")
    for h in range(len(box)):
        infile.write(f"Ar {box[h,0]} {box[h,1]} {box[h,2]}\n")
    infile.close()
