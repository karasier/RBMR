require_relative 'RBM'

# usage: var = RBMR::RBM.new([columns],number_of_data)
rbm = RBMR::RBM.new([6,4],1)

# randomize biases & weights
#rbm.randomize

# load biases & weights from a file
rbm.load_parameters("parameters.json")

data = [[1,1,1,0,0,0]]

1000.times do |i|
  data.size.times do |j|
    # usage: var.input([inputs])
    rbm.input(data[j])

    # usage: var.run(number_of_steps) step→Gibbs Sampling step
    rbm.run(1)

  end
  # get cross entropy
  cost = rbm.compute_mean_cross_entropy
  puts "epoch : #{i} , cost : #{cost}"
  # update biases & weights
  rbm.update_parameters
end

puts "finished"
# check
rbm.input([0,0,1,1,1,1])
rbm.run(1)
rbm.outputs
rbm.input([1,1,0,0,0,1])
rbm.run(1)
rbm.outputs

# save biases & weights to a file
rbm.save_parameters("parameters.json")
