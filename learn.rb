require_relative './lib/rbmr'

# usage: rbm = RBMR::RBM.new([number of visible units,number of hidden units],visible unit's type(default: Bernoulli))
# Available -> :Bernoulli, :Gaussian
rbm = RBMR::RBM.new([6,5], :Bernoulli)

# randomize biases & weights
rbm.randomize

# load biases & weights from .json file
#rbm.load_network("network.json")

data = [[1,1,1,0,0,0]]

100.times do |i|
  data.size.times do |j|

    rbm.input(data[j])

    rbm.run(1)
    rbm.compute_cross_entropy
  end

  # Get cross entropy.
  cost = rbm.compute_mean_cross_entropy(data.size)
  puts "epoch : #{i} , cost : #{cost}"
end

puts "\nUnderstood!\n\n"
print "input: #{data[0]}\n"
print "visible: #{rbm.reconstruct(data[0])}\n"
print "P(v|h): #{rbm.get_visible_probability}\n"

# save biases & weights to .json file
#rbm.save_network("network.json")
