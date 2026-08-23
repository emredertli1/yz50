# Inputs and Weights
inputs = [1.2, 0.5, -0.3]      # x1, x2, x3
weights = [0.2, 0.8, -0.5]     # w1, w2, w3
bias = 2.0                     # b

# Euler sayısı sabiti 
E = 2.718

# Forward Pass: Toplam (dot product + bias)
z = 0
for x, w in zip(inputs, weights):
    z += x * w
z += bias

# sigmoid activation function
output = 1 / (1 + (E ** (-z)))

print("Z Değeri (Ham Toplam):", z)
print("Sigmoid Çıktısı:", output)