require 'nmatrix'
require 'zlib'
require 'benchmark'
##
#      Simple and fast library for building neural networks.
########################################################################

module RBMR

    #Sigmoid = proc { |x| 1 / (1 + Math::E**(-x)) }
    ##
    # Describes a standard fully connected NN based on backpropagation.
    class RBM
        # Creates a NN from columns giving each the size
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

            # Set the default transfer function
            @transfer = block_given? ? Sigmoid : transfer

            # Creates the geometry of the bias matrices
            # 可視層と隠れ層のバイアスを格納
            @biases_geometry = @columns.map { |col| [col,1] }

            # Create the geometry of the weight matrices
            # 可視層と隠れ層のユニット間の重みを格納
            @weights_geometry = [@columns[0],@columns[1]]
            puts @weights_geometry


            # 可視層と隠れ層のユニットの値を格納
            @visible_units_0 = NMatrix.new([1,@columns[0]],0.0).transpose
            @hidden_units_0 = NMatrix.new([1,@columns[1]],0.0).transpose
            @visible_units_1 = @visible_units_0.clone
            @hidden_units_1 = @hidden_units_0.clone

            # シグモイド適用前の計算値を格納
            @pre_sigmoid_hidden_0 = NMatrix.new([1,@columns[0]],0.0).transpose
            @pre_sigmoid_hidden_1 = @pre_sigmoid_visible_0.clone
            @pre_sigmoid_visible = NMatrix.new([1,@columns[1]],0.0).transpose

            # 条件付き確率を格納
            @probability_hidden_0 = @pre_sigmoid_hidden_0.clone
            @probability_hidden_1 = @probability_hidden_0.clone
            @probability_visible = @pre_sigmoid_visible.clone

            @random_geometry = @biases_geometry.clone
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
            @visible_units_0[0] = N[values.flatten,:dtype => :float64].transpose
        end

        def probability_hidden_0_compute
          puts "@visible_units_0: #{@visible_units_0}"
          puts "@weights: #{@weights}"
          puts "@biases[1]: #{@biases[1]}"
          @pre_sigmoid_hidden_0 = NMatrix::BLAS.gemm(@visible_units_0,@weights,@biases[1])
          @pre_sigmoid_hidden_0.each_with_index do |data,i|
            @probability_hidden_0[i] = Sigmoid.call(data)
          end
        end

        def probability_visible_compute
          @pre_sigmoid_visible = NMatrix::BLAS.gemm(@weights,@hidden_units_0,@biases[0])
          @pre_sigmoid_visible.each_with_index do |data,i|
            @probability_visible[i] = Sigmoid.call(data)
          end
        end

        def probability_hidden_1_compute
          @pre_sigmoid_hidden_1 = NMatrix::BLAS.gemm(@visible_units_1,@weights,@biases[1])
          @pre_sigmoid_hidden_1.each_with_index do |data,i|
            @probability_hidden_1[i] = Sigmoid.call(data)
          end
        end

        def hidden_units_0_compute
          @probability_hidden_0.size.times do |i|
            if @probability_hidden_0[i] > @random_value[1,i] then
              @hidden_units_0[i] = 1
            else
              @hidden_units_0[i] = 0
            end
          end
        end

        def visible_units_1_compute
          @probability_visible.size.times do |i|
            if @probability_visible[i] > @random_value[0,i] then
              @visible_units_1[i] = 1
            else
              @visible_units_1[i] = 0
            end
          end
        end

        def hidden_units_1_compute
          @probability_hidden_1.size.times do |i|
            if @probability_hidden_1[i] > @random_value[1,i] then
              @hidden_units_1[i] = 1
            else
              @hidden_units_1[i] = 0
            end
          end
        end
        # z = inputs * weights + biases
        # z = 入力値×重み+バイアス
        # zは長さの違う配列を複数持つ配列
        # zは活性化関数適用前の状態
        # 活性化関数への入力値を計算するメソッド
        # 引数:row 現在の層のインデックス →　現在が何層目かを表す。

        # 入力層から順方向に計算するメソッド
        # 最初に入力されたニューラルネットワークの層の回数計算する
        def propagate
          probability_hidden_0_compute
          hidden_units_0_compute
          probability_visible_compute
          visible_units_1_compute
          probability_hidden_1_compute
          hidden_units_1_compute
        end

        # 重みとバイアスを出力
        def backpropagate_outputs(row)
          puts "weights=#{@weights[row]}"
          puts "biases=#{@biases[row]}"
        end

        # 入力値と状態を出力
        def propagate_outputs(row)
          puts "a=#{@a[row]}"
          puts "z=#{@z[row]}"
        end

        def outputs
          @a[@neuron_columns.size]
        end

        def run(time)
          time.times do |trial|
            propagate
          end
        end
    end
end
