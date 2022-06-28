import tensorflow as tf
#from tensorflow import keras

interpreter = tf.lite.Interpreter("YoloV5-m-c-fp16.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()#[0]["index"]
output_index = interpreter.get_output_details()#[0]["index"]

print(input_index)

# Gather results for the randomly sampled test images
predictions = []

test_labels, test_imgs = [], []
for img, label in tqdm(test_batches.take(10)):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)