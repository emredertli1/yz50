# 1. Giriş vektörü (4 özellikli bir örnek)
inputs = [1.0, 2.0, 3.0, 2.5]

# 2. Katman parametreleri (3 nöron, her birinin 4 ağırlığı ve 1 bias değeri var)
weights = [
    [0.2, 0.8, -0.5, 1.0],   # Nöron 1 ağırlıkları
    [0.5, -0.91, 0.26, -0.5], # Nöron 2 ağırlıkları
    [-0.26, -0.27, 0.17, 0.87] # Nöron 3 ağırlıkları
]

biases = [2.0, 3.0, 0.5]  # Nöron 1, 2 ve 3 için bias değerleri

# 3. Aktivasyon Fonksiyonu (ReLU)
def relu(x):
    return max(0.0, x)

# 4. Forward Pass (Her nöron için: dot product + bias -> aktivasyon)
layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    # Dot product: sum(w_i * x_i)
    neuron_output = 0.0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    
    # Bias ekleme: z = dot_product + bias
    neuron_output += neuron_bias
    
    # Aktivasyon: a = relu(z)
    activated_output = relu(neuron_output)
    
    layer_outputs.append(activated_output)

print("Katman Çıktısı (3 nöron):", layer_outputs)