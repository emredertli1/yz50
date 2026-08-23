#Inputs and Weights
inputs = [1.2, 0.5, -0.3]      # x1, x2, x3
weights = [0.2, 0.8, -0.5]     # w1, w2, w3
bias = 2.0                     # b

# Forward Pass: Toplam (dot product + bias)
z = 0
for x, w in zip(inputs, weights):
    z += x * w
z += bias

# RELU activation function
output = max(0, z)

print("Tek Nöron Çıktısı:", output)