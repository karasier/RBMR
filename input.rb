require_relative 'RBM'

# usage: var = RBMR::RBM.new([columns],number_of_data)
rbm = RBMR::RBM.new([3,4],1)

# randomize biases & weights
#rbm.randomize

# load biases & weights from a file
rbm.load_parameters("parameters.json")

# usage: var.input([inputs])
rbm.input([1,0,0])

# usage: var.run(number_of_steps) step→Gibbs Sampling step
rbm.run(1)

# update biases & weights
rbm.update_parameters

rbm.outputs

# save biases & weights to a file
rbm.save_parameters("parameters.json")
