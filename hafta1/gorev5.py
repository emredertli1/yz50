
def f(x):
    return 3 * x**2 - 4 * x + 5

#Başlangıç parametresi ve hiperparametreler
x = 3.0           # Başlangıç noktamız (f(3.0) = 20.0)
h = 0.0001        # Sayısal türev için küçük adım 
learning_rate = 0.01  # Parametre güncelleme adım büyüklüğü
epochs = 20       # Döngü sayısı

# 3. Gradient Descent Döngüsü
for epoch in range(epochs):
    # İleri hesaplama (Forward Pass) -> Mevcut Loss
    loss = f(x)
    
    # Sayısal Türev  -> Gradyan hesabı
    # dL/dx = (f(x + h) - f(x)) / h
    grad = (f(x + h) - f(x)) / h
    
    # Parametre Güncellemesi (Gradient Descent Step)
    # Gradyanın tersi yönünde adım atıyoruz
    x -= learning_rate * grad
    
    print(f"Epoch {epoch+1:2d} | Loss: {loss:.6f} | x: {x:.6f} | grad: {grad:.6f}")