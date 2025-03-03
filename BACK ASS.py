import math

# تفعيل السيجمويد
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# مشتقة السيجمويد
def sigmoid_derivative(x):
    return x * (1 - x)

# المدخلات
i1, i2 = 0.05, 0.10

# الأوزان الأولية
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55

# البايس
b1, b2 = 0.35, 0.60

# الأهداف
t1, t2 = 0.01, 0.99

# ======= Forward Propagation =======
# Hidden Layer
h1_input = i1 * w1 + i2 * w3 + b1
h2_input = i1 * w2 + i2 * w4 + b1

h1 = sigmoid(h1_input)
h2 = sigmoid(h2_input)

# Output Layer
o1_input = h1 * w5 + h2 * w7 + b2
o2_input = h1 * w6 + h2 * w8 + b2

o1 = sigmoid(o1_input)
o2 = sigmoid(o2_input)

# طباعة تفاصيل الـ Forward
print(f"Forward Propagation:")
print(f"h1_input = {h1_input:.4f}, h1 = {h1:.4f}")
print(f"h2_input = {h2_input:.4f}, h2 = {h2:.4f}")
print(f"o1_input = {o1_input:.4f}, o1 = {o1:.4f}")
print(f"o2_input = {o2_input:.4f}, o2 = {o2:.4f}")

# ======= حساب الخطأ (MSE) =======
error1 = 0.5 * (t1 - o1) ** 2
error2 = 0.5 * (t2 - o2) ** 2
total_error = error1 + error2

print(f"Error o1 = {error1:.6f}, Error o2 = {error2:.6f}")
print(f"Total Error = {total_error:.6f}")

# ======= Backpropagation =======
# حساب Gradients عند o1 و o2
dE_o1 = o1 - t1
dE_o2 = o2 - t2

# حساب مشتقات الدوال (Chain Rule)
do1_net = sigmoid_derivative(o1)
do2_net = sigmoid_derivative(o2)

# Gradients عند w5, w6, w7, w8
dw5 = dE_o1 * do1_net * h1
dw6 = dE_o2 * do2_net * h1
dw7 = dE_o1 * do1_net * h2
dw8 = dE_o2 * do2_net * h2

# ======= تحديث الأوزان في الطبقة الأخيرة =======
w5 = w5 - 0.5 * dw5
w6 = w6 - 0.5 * dw6
w7 = w7 - 0.5 * dw7
w8 = w8 - 0.5 * dw8

# ======= Backpropagation للطبقة المخفية =======
dh1 = (dE_o1 * do1_net * w5) + (dE_o2 * do2_net * w6)
dh2 = (dE_o1 * do1_net * w7) + (dE_o2 * do2_net * w8)

dh1_net = sigmoid_derivative(h1)
dh2_net = sigmoid_derivative(h2)

dw1 = dh1 * dh1_net * i1
dw2 = dh2 * dh2_net * i1
dw3 = dh1 * dh1_net * i2
dw4 = dh2 * dh2_net * i2

# ======= تحديث الأوزان في الطبقة الأولى =======
w1 = w1 - 0.5 * dw1
w2 = w2 - 0.5 * dw2
w3 = w3 - 0.5 * dw3
w4 = w4 - 0.5 * dw4

# ======= عرض النتائج =======
print(f"\nNew weights after backpropagation:")
print(f"w1 = {w1:.4f}")
print(f"w2 = {w2:.4f}")
print(f"w3 = {w3:.4f}")
print(f"w4 = {w4:.4f}")
print(f"w5 = {w5:.4f}")
print(f"w6 = {w6:.4f}")
print(f"w7 = {w7:.4f}")
print(f"w8 = {w8:.4f}")
