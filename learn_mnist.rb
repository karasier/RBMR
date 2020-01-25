require_relative 'RBMR'
require_relative 'mnist_loader'


puts "Loading images"

# MNISTの読み込み
mnist = MNISTLoader.new("assets/t10k-images-idx3-ubyte.gz", "assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images

puts "Initializing network"

# usage: rbm = RBMR::RBM.new([number of visible units,number of hidden units],visible units' type(default: Bernoulli))
rbm = RBMR::RBM.new([784,51],:Bernoulli)

# randomize biases & weights
rbm.randomize

# load biases & weights from .json file
#rbm.load_parameters("parameters.json")

imgs = images.map { |image| mnist.byte_to_float(image).flatten }
imgs = imgs.map { |image| mnist.binarize(image).flatten }
#imgs = images.map{ |image| image.flatten }

puts "Runnning..."

c = 0
1.times do
  imgs.each_with_index do |inputs,index|

    if c > 50
      c = 0
      break
    end

    # usage: rbm.input([inputs])
    rbm.input(inputs)

    # usage: rbm.run(number_of_steps) steps → Contrastive Divergence steps
    rbm.run(1)
    mnist.ascii_print(inputs)
    c += 1
  end
end

puts "\nUnderstood!\n\n"
gets

c = 0
imgs.each_with_index do |inputs,index|

  if c > 50
    break
  end
  mnist.ascii_print(inputs)
  outputs = rbm.reconstruct(inputs)
  mnist.ascii_print(outputs)

  c += 1
end

# reconstruct from feature
#rbm.input_hidden_layer(5.times.map{rand(0..1).to_i})

# save biases & weights to .json file
#rbm.save_parameters("parameters.json")
