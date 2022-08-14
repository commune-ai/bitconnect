
# Configuring Pipelines



## [Config Template](./config/templates/)

When we are dealing with complicated ML pipelines, we want to make sure there is a central configuration that 
is modular, simple and flexible. 

## Config Schema

### Dataset

- specifies the dataset configuration which includes
- data manager configuration when pulling in coins
- in this case we are not using a module key (TODO)

### Trainer

- Configures the training environment of the pipeline
- specify the training module via the trainer module key
- each trainer will include with it a set of model roles 
which is useful if you have multiple models coordinating with each other

### Model

- Configures the model/models used for the pipeline
- the model class is specified by the module
- you can also train multiple models in sequence that
map a dictionary of tensors to a dictionay of tensors
- the keys within this fig indicates the roles of each model
- you can also specify modules within models by having extra layers of depth in the config

**Example**

- the following model includes a gaussian process and a transformer within the oracle role.
- the transformer also contains another model called nbeats which contain linear and fourier subcomponents
- **this allows for infinite expressibility of models**

```json
model:
  oracle:

    gp: 
      optimizer:
        lr: 0.01
    transformer:

      nbeats:

        linear:


        fourier:


      optimizer:
        lr: 0.001
        weight_decay: 1.e-4
        amsgrad: True

    loss_weight:
      mse: 1
      gp: 1

```



## [Hyperparameters](../hyper_parameters.py)

- specify the hyperparameter distributions using ray tune objects