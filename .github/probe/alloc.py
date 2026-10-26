n = 1 << 30
b = bytearray(n)
for i in range(0, n, 4096):
    b[i] = 1
print("touched", n)
