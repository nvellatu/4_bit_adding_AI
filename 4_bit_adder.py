import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

# Generate some example data
num_samples = 1000

# Generate input array of tuples
inputs = []
targets = []
for _ in range(num_samples):
    a = np.random.randint(0, 16)  # 4-bit number
    b = np.random.randint(0, 16)  # 4-bit number
    inputs.append([int(bit) for bit in np.binary_repr(a, width=4)] + [int(bit) for bit in np.binary_repr(b, width=4)])
    targets.append([int(bit) for bit in np.binary_repr(a + b, width=5)])
inputs = np.array(inputs)
targets = np.array(targets)

# Define the model
input_shape = (8,)  # Shape of each input tuple
input_layer = Input(shape=input_shape)
hidden_layer = Dense(16, activation='relu')(input_layer)
hidden_layer2 = Dense(16, activation='relu')(hidden_layer)
output_layer = Dense(5, activation='sigmoid')(hidden_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(inputs, targets, epochs=100, batch_size=32)

# Test the model
test_input = np.array([[1, 0, 1, 0, 1, 1, 0, 1]])  
predicted_output = model.predict(test_input)
print("Predicted output:", np.round(predicted_output))

test_input = np.array([[1, 0, 0, 0, 0, 1, 0, 1]])  
predicted_output = model.predict(test_input)
print("Predicted output:", np.round(predicted_output))

test_input = np.array([[1, 0, 0, 1, 1, 1, 0, 0]])  
predicted_output = model.predict(test_input)
print("Predicted output:", np.round(predicted_output))

test_input = np.array([[1, 0, 1, 1, 1, 1, 0, 1]])  
predicted_output = model.predict(test_input)

print("Predicted output:", np.round(predicted_output))
