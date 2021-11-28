# machine_learning_learning

Running through a bunch of tutorials as found on 

https://www.tensorflow.org/tutorials/keras/classification


# Machine Learning Types

This is mainly taken from the time series tutorial as that is one of the more difficult tutorials in the list and goes
through a LOT of different types.

Terms involved: 
 - window: width of the input data.
 - label_width: width of the ouput data.
 - offset: how offset the label data is from the input

## Single Step Models

These models take a single input and then make a prediction. For example, it may use today's data to predict tomorroww's
values. IE: Input = t0, output = t1. To create an input for this, you would create window=1, label_width=1, and offset=1

The problem with this for time series data is that is has no historical data, only the previous reading.

## Single Step Linear

This is the simplest type of model. The cool part is that you can see how the input values affect the output. The linear
model is just a dense network with no activation set. Example: 

```python
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
```

## Multi-layer, Single Step Dense Model

Instead of a single linear layer, this will have multiple layers with activation functions that are not linear. The
reason for non-linear activation function is outside the scope of this write up, but it allow for more complicated
predictions than linear activation would allow

```python
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
```

## Multi-Step, Multi-Step Dense Model

Look at that fancy, fancy name. Multi-step models means that instead of just a single time step, multiple steps are
given to the model. This is inputs: t1, t2...tn are used to predict: tn+1. The input data should look like: window=$step_size, 
label_width=1, and offset=1. 

In order to do this, we need to flatten the input data. The inputs data is columns x step_size, and this will flatten
that data into a single list.

```python
    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
```

## Convolution Neural Network

This is basically the previous model but in a better way. This model is the same as the above.

```python
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
```

## Recurrent Neural Network

Instead of manually feeding the previous data in, you can have the previous data automatically retained. 

BOOM! Done. Easy.
```python
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])
```

## Multi-Output Models

Instead of just outputting one value - such as temperature - the models can output many outputs. To do this, you just
need to have multiple labels. I.... I don't really know how to do this, because I've only done a single output
prediction.

```
    single_step_window = WindowGenerator(
        # `WindowGenerator` returns all features as labels if you 
        # don't set the `label_columns` argument.
        input_width=1, label_width=1, shift=1)

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1)

    for example_inputs, example_labels in wide_window.train.take(1):
      print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
      print(f'Labels shape (batch, time, features): {example_labels.shape}')
```


## Multi-step Predictions

Instead of making just one prediction, we could make a whole bunch of predictions. Kind of like running weather a model
and forecasting into the future. I also don't really know how to do this. But we'll leave this here to remind it's
possible later
