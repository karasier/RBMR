require 'nmatrix'
require 'json'
require 'random_bell'

# Simple and fast library for building neural networks.
# @since 1.0.0
# @author Ryota Sakai
module RBMR
  # Describes a restricted boltzmann machine.
  class RBM
    # Creates a RBM from columns giving each the size
    # of a column of units.
    # @param [Array] columns the array showing the shape of a restricted boltzmann machine
    # @param [Symbol] type the visible units' type you want to use
    # ':Bernoulli' or ':Gaussian'
    # default -> :Bernoulli
    # If you set :Bernoulli, you can use Bernoulli-Bernoulli RBM.
    # If you set :Gaussian, you can use Gaussian-Bernoulli RBM.
    # @example initialization of the restricted boltzmann machine
    #   rbm = RBMR::RBM.new([5,4])
    #   rbm = RBMR::RBM.new([5,4],:Gaussian)
    def initialize(columns,type = :Bernoulli)
      # training_rate
      @training_rate = 0.1

      # Set the type of visible units.
      @type = type

      # Ensure columns is a proper array.
      @columns = columns.flatten

      # Creates the geometry of the bias matrices.
      # The hidden layer and the visible layer are stored in that order.
      # @biases[0] -> hidden layer's biases
      # @biases[1] -> visible layer's biases
      @biases_geometry = @columns.reverse.map { |col| [1,col] }

      # Create the geometry of the weight matrix
      @weights_geometry = @columns[0..1]

      # Create units matrices
      # The visible layer and the hidden layer are stored in that order.
      # @units[0] -> visible layer's units
      # @units[1] -> hidden layer's units
      @units = @columns.map{ |col| NMatrix.new([1,col],0.0) }

      # Create the matrices of conditional probability that the unit will be 1.
      # P(hidden|visible)→P(visible|hidden)の順で格納
      # @probability[0] -> P(hidden|visible)
      # @probability[1] -> P(visible|hidden)
      @probability = @columns.reverse.map{ |col| NMatrix.new([1,col],0.0) }

      # the array of biases derivatives
      @derivative_biases = @columns.map{ |col| NMatrix.new([1,col], 0.0) }

      # matrix of weights derivative
      @derivative_weights = @weights_geometry.map do |geo|
        NMatrix.new(geo,0.0)
      end

      # need for compute cross entropy
      @cross_entropy = NMatrix.new([1,@units[0].size],0.0)

      # Make a Gaussian distribution that has μ = 0, σ = 0.01.
      @bell = RandomBell.new(mu: 0, sigma: 0.01, range: -Float::INFINITY..Float::INFINITY)

      # computation method of sampling visible units
      @computation_methods = { Bernoulli: method(:compute_from_bernoulli), Gaussian: method(:compute_from_gaussian) }
      @computation_method = @computation_methods[@type]

      # Set Sigmoid function.
      @sigmoid = Sigmoid[:antiderivative]
    end


    # Set up the RBM to random values.
    def randomize
      # Create fast matrices for the biases.
      @biases = @biases_geometry.map { |geo| NMatrix.new(geo,0.0)}
      puts "@biases: #{@biases}"

      # Create random fast matrices for the weights.
      # The weights are initialized to follow the Gaussian distribution.
      weights_array = []
      @weights_geometry[0].times do |i|
        @weights_geometry[0].times do |j|
          weights_array.push(@bell.rand)
        end
      end

      # Store weights into @weights.
      @weights = []
      @weights.push(NMatrix.new(@weights_geometry,weights_array))
      puts "@weights: #{@weights}"
    end

    # Get the inputs of the restricted boltzmann machine.
    # @param [Array] values inputs of the restricted boltzmann machine.
    def input(values)
      @units[0] = N[values.flatten,:dtype => :float64]
      @inputs = @units[0].dup

      # If Gaussian-Bernoulli RBM is used, it standardize input data.
      @type == :Gaussian ? standardize : nil
    end

    # Input to the hidden layer.
    # The main use is for post-learning confirmation.
    # @param [Array] values inputs of the hidden layer
    def input_hidden_layer(values)
      @units[1] = N[values.flatten,:dtype => :float64]
      sample_visible_units
    end

    # Standardize input data.
    def standardize
      @mean = @units[0].mean(1)[0]
      @standard_deviation = @units[0].std(1)[0]
      @units[0] = (@units[0] - @mean) / @standard_deviation
      @inputs = @units[0].dup
    end

    # Unstandardize visible units and inputs.
    def unstandardize
      @units[0] = @units[0] * @standard_deviation + @mean
      @inputs = @inputs * @standard_deviation + @mean
    end

    # Compute P(hidden|visible) and sample hidden units.
    def sample_hidden_units
      # Compute conditional probability of hidden layer.
      pre_sigmoid = NMatrix::BLAS.gemm(@units[0],@weights[0],@biases[0])
      @probability[0] = @sigmoid.call(pre_sigmoid)

      # Sample hidden units from conditional probability.
      @probability[0].each_with_index do |prob,i|
        @units[1][i] = prob >= rand ? 1.0 : 0.0
      end
    end

    # Compute P(visible|hidden) and sample visible units.
    def sample_visible_units
      @computation_method.call
    end

    # Compute visible units from Bernoulli units.
    def compute_from_bernoulli
      # Compute conditional probability of visible layer.
      product_of_units_and_weights = NMatrix::BLAS.gemm(@weights[0],@units[1].transpose)
      pre_sigmoid = product_of_units_and_weights.transpose + @biases[1]
      @probability[1] = @sigmoid.call(pre_sigmoid)

      # Sample visible units from conditional probability.
      @probability[1].each_with_index do |prob,i|
        @units[0][i] = prob >= rand ? 1.0 : 0.0
      end
    end

    # Compute visible units from Gaussian units.
    def compute_from_gaussian
      # Compute product of hidden units and weights.
      product_of_units_and_weights = NMatrix::BLAS.gemm(@weights[0],@units[1].transpose)

      # Compute mean of Gaussian distribution.
      mean_of_gaussian_distribution = product_of_units_and_weights.transpose + @biases[1]

      # Sample visible units from Gaussian distribution.
      mean_of_gaussian_distribution.each_with_index do |mean,i|
        @units[0][i] = RandomBell.new(mu: mean, sigma: 1, range: -Float::INFINITY..Float::INFINITY).rand
      end

      # Compute Gaussian distribution.
      difference_of_units_and_mean = @units[0] - mean_of_gaussian_distribution
      @probability[1] = (-(difference_of_units_and_mean ** 2) / 2.0).exp / Math.sqrt(2.0 * Math::PI)
    end

    # Initialize derivatives.
    # The main use is for mini-batch learning.
    def initialize_derivatives
      # Initialize derivative of biases.
      @derivative_biases = @columns.map{ |col| NMatrix.new([1,col], 0.0) }

      # Initialize derivative of weights.
      @derivative_weights = @weights_geometry.map do |geo|
        NMatrix.new(geo,0.0)
      end
    end

    # Compute derivatives biases and weights.
    def compute_derivatives
      unit = @type == :Bernoulli ? @units[0] : @probability[1]

      @derivative_weights = NMatrix::BLAS.gemm(@inputs, @probability_hidden, nil, 1.0, 0.0, :transpose) - NMatrix::BLAS.gemm(unit, @probability[0], nil, 1.0, 0.0, :transpose)

      @derivative_biases[0] = (@inputs - unit)
      @derivative_biases[1] = (@probability_hidden - @probability[0])
    end

    # Update biases.
    def update_biases
      @biases[0] += @derivative_biases[1] * @training_rate
      @biases[1] += @derivative_biases[0] * @training_rate
    end

    # Update weights.
    def update_weights
      @weights[0] += @derivative_weights * @training_rate
    end

    # Update biases and weights.
    def update_parameters
      update_biases
      update_weights
    end

    # Sample visible units and hidden units.
    def sample(number_of_steps)
      sample_hidden_units

      # Store the conditional probability of hidden layer of first step.
      # Need for computing derivatives.
      @probability_hidden = @probability[0].dup

      sample_visible_units

      # Sample the times of number_of_steps.
      number_of_steps.times do |i|
        sample_hidden_units
        sample_visible_units
      end
    end

    # Compute cross entropy.
    def compute_cross_entropy
      log_probability = @probability[1].log
      log_probability_dash = (-@probability[1] + 1).log
      inputs_dash = (-@inputs + 1)
      @cross_entropy += ((@inputs * log_probability) + (inputs_dash * log_probability_dash))
    end

    # Compute mean cross entropy.
    # @param [Int] number_of_data the number of training data.
    def compute_mean_cross_entropy(number_of_data)
      mean_cross_entropy = -@cross_entropy.to_a.sum / number_of_data.to_f
      @cross_entropy = NMatrix.new([1,@units[0].size],0.0)
      return mean_cross_entropy
    end

    # Learn RBM.
    # @param [Int] number_of_steps the number of Contrastive Divergence steps
    def run(number_of_steps)
      sample(number_of_steps)
      compute_derivatives
      update_parameters
    end

    # Reconstruct input data.
    # The main use is for post-learning confirmation.
    # @param [Array] values the data you want to reconstruct
    def reconstruct(values)
      input(values)
      sample_hidden_units
      sample_visible_units
      return @units[0]
    end

    # Get outputs of neural network.
    # @return [Array] output of restricted boltzmann machine(= visible units)
    def get_outputs
      @type == :Gaussian ? unstandardize : nil
      return @units[0]
    end

    # Get conditional probability of visible layer.
    # @return [Array] conditional probability of visible layer.
    def get_visible_probability
      return @probability[1]
    end

    # Set training rate.
    # @param [Float] rate training rate
    def set_training_rate(rate = 0.1)
      @training_rate = rate
    end

    # Save learned network to JSON file.
    # @param [String] path file path
    def save_network(path)
      # Make hash of parameters.
      hash = { "type" => @type, "columns" => @columns, "biases" => @biases, "weights" => @weights }

      # Save file.
      File.open(path,"w+") do |f|
        f.puts(JSON.pretty_generate(hash))
      end
    end

    # Load learned network from JSON file.
    # @param [String] path file path
    def load_network(path)
      # Open file.
      File.open(path,"r+") do |f|
        # Load hash from JSON file.
        hash = JSON.load(f)

        # Set columns from hash.
        @columns = hash["columns"]

        # Set visible units' type.
        @type = hash["type"].to_sym

        # Initialize the restricted boltzmann machine.
        initialize(@columns,@type)

        # Load biases.
        biases_matrix = hash["biases"].to_a
        @biases = []
        @columns.size.times do |i|
          @biases.push(N[biases_matrix[i].split(',').map!{ |item| item.delete("/[\-]/").gsub(" ","").to_f}])
        end
        puts "#{@biases}"

        # Load weights.
        weights_matrix = hash["weights"].to_a
        @weights = []
        weights_array = weights_matrix[0].split(',').map!{ |item| item.delete("/[\-]/").gsub(" ","").to_f}.to_a
        @weights.push(NMatrix.new(@weights_geometry,weights_array))
        puts "#{@weights}"
      end
    end
  end

  # Apply sigmoid function to z.
  # @param [NMatrix] z a vector of NMatrix containing the multiply accumulation of inputs, weights and biases
  # @return [NMatrix] a vector of NMatrix that each elements are applied to sigmoid function
  def self.sigmoid(z)
    return ((-z).exp + 1) ** (-1)
  end

  # Differentiate sigmoid function.
  # @param [NMatrix] a a vector of NMatrix containing neuron statuses
  # @return [NMatrix] a vector of NMatrix that each elements are differentiated
  def self.differentiate_sigmoid(a)
    return (-a + 1.0) * a
  end

  Sigmoid = { antiderivative: method(:sigmoid), derivative: method(:differentiate_sigmoid) }
end
