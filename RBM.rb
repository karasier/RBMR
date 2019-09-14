require 'nmatrix'
require 'zlib'
require 'benchmark'
##
#      Simple and fast library for building restricted boltzmann machine.
########################################################################

module RBMR

    Sigmoid = proc { |x| 1 / (1 + Math::E**(-x)) }
    ##
    class RBM
        # Creates a RBM from columns giving each the size
        # of a column of neurons (input and output comprised).
        # If a block is given as argument, it will be used as
        # default transfer fuction (default: sigmoid)
        def initialize(*columns,&transfer)
            # 学習率
            @training_rate = 0.1
            # Ensure columns is a proper array.
            # 各層のニューロンの数を配列にする
            @columns = columns.flatten
            # The columns containing processing neurons (i.e., excluding the
            # inputs).
            @neuron_columns = @columns[1..-1]
            # Set the default transfer function
            @transfer = block_given? ? Sigmoid : transfer

            # Creates the geometry of the bias matrices
            # 可視層と隠れ層のバイアスを格納
            # 隠れ層→可視層の順で格納
            @biases_geometry = @columns.reverse.map { |col| [col,1] }
            # Create the geometry of the weight matrices
            # 可視層と隠れ層のユニット間の重みを格納
            @weights_geometry = @neuron_columns.zip(@columns[0..-1])

            # 可視層と隠れ層のユニットの値を格納
            # 可視層→隠れ層の順で格納
            @units = @columns.map{ |col| NMatrix.new([1,col],0.0).transpose }

            # 条件付き確率を格納
            # P(hidden|visible)→P(visible|hidden)の順で格納
            @probability = @columns.reverse.map{ |col| NMatrix.new([1,col],0.0).transpose }

            @probability[1].size.times do |i|
              @probability[1][i] = 1/@columns[0].to_f
            end

            @random_geometry = @biases_geometry.clone

            @visible_probability_0 = @probability[1].dup
        end


        # Set up the NN to random values.
        def randomize
            # Create random fast matrices for the biases.
            # NMatrixの配列を作成 バイアス
            @biases = @biases_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}
            @biases.size.times do |i|
              @biases[i] -= 0.5
            end
            puts "@biases: #{@biases}"
            # Create random fast matrices for the weights.
            # NMatrixの配列を作成 重み
            @weights = @weights_geometry.map do |geo|
                NMatrix.random(geo,:dtype => :float64)
            end
            @weights.size.times do |i|
              @weights[i] -= 0.5
            end
            puts "@weights: #{@weights}"

            @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}
        end

        # RBMへの入力を取得
        # 引数: *vaules→入力
        def input(*values)
          @inputs = N[values.flatten,:dtype => :float64].transpose
          @units[0] = @inputs.clone
        end

        # P(h|v)と隠れ層のユニットの値を計算
        def compute_visible
          @pre_sigmoid = NMatrix::BLAS.gemm(@units[0],@weights[0],@biases[0])
          @pre_sigmoid.each_with_index do |data,i|
            @probability[0][i] = Sigmoid.call(data)
          end

          @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}

          @probability[0].size.times do |i|
            if @probability[0][i] > @random_value[0][i] then
              @units[1][i] = 1
            else
              @units[1][i] = 0
            end
          end
        end

        # P(v|h)と可視層のユニットの値を計算
        def compute_hidden
          @pre_sigmoid = NMatrix::BLAS.gemm(@weights[0],@units[1],@biases[1])
          @pre_sigmoid.each_with_index do |data,i|
            @probability[1][i] = Sigmoid.call(data)
          end

          @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}

          @probability[1].size.times do |i|
            if @probability[1][i] > @random_value[1][i] then
              @units[0][i] = 1
            else
              @units[0][i] = 0
            end
          end
        end

        # 計算するメソッド
        def propagate
          compute_visible
          @hidden_probability_0 = @probability[0].dup
          10.times do |i|
            compute_hidden
            compute_visible
          end
          puts "@inputs: #{@inputs}"
          puts "P(h|v)_0: #{@hidden_probability_0}"
          puts "P(v|h)_0: #{@visible_probability_0}"
          puts "P(h|v)_1: #{@probability[0]}"
          puts "P(v|h)_1: #{@probability[1]}"
          puts "visible: #{@units[0]}"
        end

        def run(time)
          time.times do |trial|
            propagate
          end
        end
    end
end
