require_relative 'RBMR'

# usage: var = RBMR::RBM.new([columns],number_of_data)
rbm = RBMR::RBM.new([6,4])

# randomize biases & weights
#rbm.randomize

# load biases & weights from .json file
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
  cost = rbm.compute_mean_cross_entropy(data.size)
  puts "epoch : #{i} , cost : #{cost}"
end

puts "\nUnderstood!\n\n"

# check
rbm.reconstruct([1,1,1,0,0,0])

# reconstruct from feature
# rbm.input_hidden_layer(4.times.map{rand(0..1).to_i})

# save biases & weights to .json file
rbm.save_parameters("parameters.json")
