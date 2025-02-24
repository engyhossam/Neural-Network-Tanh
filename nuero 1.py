def tanh(x):
    e = 2.71828  
    return (2 / (1 + (e ** (-2 * x)))) - 1  

i1, i2 = 0.05, 0.10

w1, w2, w3, w4 = -0.3, 0.2, -0.4, 0.1
w5, w6, w7, w8 = 0.3, -0.2, 0.4, -0.1

b1, b2 = 0.5, 0.7

# Hidden
h1 = tanh(i1 * w1 + i2 * w3 + b1)
h2 = tanh(i1 * w2 + i2 * w4 + b1)

# Output
o1 = tanh(h1 * w5 + h2 * w7 + b2)
o2 = tanh(h1 * w6 + h2 * w8 + b2)

print("Output 1:", o1)
print("Output 2:", o2)

