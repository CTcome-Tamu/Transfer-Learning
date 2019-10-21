import tensorflow.contrib.slim as slim

model_variables = slim.get_variables()
restore_variables = [var for var in model_variables]
for var in restore_variables:
    print(var.name)