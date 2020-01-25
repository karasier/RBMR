require_relative 'RBMR'

# usage: rbm = RBMR::RBM.new([number of visible units,number of hidden units],visible unit's type(default: Bernoulli))
# Available -> :Bernoulli, :Gaussian
rbm = RBMR::RBM.new([6,5],:Bernoulli)

# randomize biases & weights
rbm.randomize

# load biases & weights from .json file
#rbm.load_parameters("parameters.json")

data = [[1,1,1,0,0,0]]

100.times do |i|
  data.size.times do |j|
    # usage: rbm.input([inputs])
    rbm.input(data[j])

    # usage: rbm.run(number_of_steps) steps → Contrastive Divergence steps
    rbm.run(1)
    rbm.compute_cross_entropy
  end
  # get error rate
  error_rate = rbm.get_error_rate(i+1)

  # get cross entropy
  cost = rbm.compute_mean_cross_entropy(data.size)
  puts "epoch : #{i} , cost : #{cost}, error_rate : #{error_rate}"
end

puts "\nUnderstood!\n\n"

rbm.reconstruct(data[0])
rbm.outputs

# reconstruct from feature
#rbm.input_hidden_layer(5.times.map{rand(0..1).to_i})

# save biases & weights to .json file
#rbm.save_parameters("parameters.json")