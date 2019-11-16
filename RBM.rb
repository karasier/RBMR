require 'nmatrix'
require 'zlib'
require 'benchmark'
require 'json'

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
        def initialize(*columns,number_of_data,&transfer)
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

            # データ数を格納
            @number_of_data = number_of_data
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

            # 0ステップ目のユニットの値を格納
            @visible_units_0 = @units[0].dup
            @hidden_units_0 = @units[1].dup

            # 条件付き確率を格納
            # P(hidden|visible)→P(visible|hidden)の順で格納
            @probability = @columns.reverse.map{ |col| NMatrix.new([1,col],0.0).transpose }

            #P(v|h)_0を計算
            @probability[1].size.times do |i|
              @probability[1][i] = 1/@number_of_data.to_f
            end

            # 最初のステップのP(v|h)
            @visible_probability_0 = @probability[1].dup

            # バイアス更新用にユニットの期待値を格納
            @expected_units_0 = @columns.map{ |col| NMatrix.new([1,col],0.0).transpose }
            @expected_units_k = @columns.map{ |col| NMatrix.new([1,col],0.0).transpose }

            # 重み更新用にユニットの期待値のアダマール積を格納
            @expected_weights_derivative_0 = @weights_geometry.map do |geo|
                NMatrix.new(geo,0.0)
            end
            @expected_weights_derivative_k = @weights_geometry.map do |geo|
                NMatrix.new(geo,0.0)
            end

            # ユニット値のサンプリング用
            @random_geometry = @biases_geometry.clone

            @cross_entropy = @units[0].dup

            @bias_visible_derivate = @units[0].dup
            @bias_hidden_derivate = @units[1].dup
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
          @visible_units_0 = N[values.flatten,:dtype => :float64].transpose
          @units[0] = @visible_units_0.clone
        end

        # P(h|v)と隠れ層のユニットの値を計算
        def compute_visible
          # 条件付き確率を計算
          @pre_sigmoid = NMatrix::BLAS.gemm(@units[0],@weights[0],@biases[0])
          @pre_sigmoid.each_with_index do |data,i|
            @probability[0][i] = Sigmoid.call(data)
          end

          @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}

          # 隠れ層のユニットの値を計算
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
          # 条件付き確率を計算
          @pre_sigmoid = NMatrix::BLAS.gemm(@weights[0],@units[1],@biases[1])
          @pre_sigmoid.each_with_index do |data,i|
            @probability[1][i] = Sigmoid.call(data)
          end

          @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}

          # 可視層のユニットの値を計算
          @probability[1].size.times do |i|
            if @probability[1][i] > @random_value[1][i] then
              @units[0][i] = 1
            else
              @units[0][i] = 0
            end
          end
        end

        # ユニットの期待値(0ステップ目)を計算
        def compute_expected_units_0
          # 可視層の期待値
          @expected_units_0[0] += @visible_units_0 * @visible_probability_0
          # 隠れ層の期待値
          @expected_units_0[1] += @hidden_units_0 * @hidden_probability_0
        end

        # ユニットの期待値(kステップ目)を計算
        def compute_expected_units_k
          # 可視層の期待値
          @expected_units_k[0] += @units[0] * @probability[1]
          # 隠れ層の期待値
          @expected_units_k[1] += @units[1] * @probability[0]
        end

        # 重みの導関数の期待値(0ステップ目)を計算
        def compute_expected_weights_derivative_0
          @expected_weights_derivative_0[0] += NMatrix::BLAS.gemm(@hidden_probability_0,@visible_units_0.transpose)
        end

        # 重みの導関数の期待値(kステップ目)を計算
        def compute_expected_weights_derivative_k
          @expected_weights_derivative_k[0] += NMatrix::BLAS.gemm(@probability[0],@units[0].transpose)
        end

        # バイアスの導関数の計算
        def compute_expected_biases
          @bias_visible_derivate += (@visible_units_0 - @units[0])
          @bias_hidden_derivate += (@hidden_probability_0 - @probability[0])
        end

        # 期待値の計算
        def compute_expected_values
          compute_expected_biases
          compute_expected_units_0
          compute_expected_units_k
          compute_expected_weights_derivative_0
          compute_expected_weights_derivative_k
        end

        # バイアスの更新用
        def update_biases
          @biases[0] +=  @bias_hidden_derivate * @training_rate
          @biases[1] +=  @bias_visible_derivate * @training_rate
        end

        # 重みの更新用
        def update_weights
          @weights[0] += (@expected_weights_derivative_0[0] - @expected_weights_derivative_k[0]) * @training_rate
        end

        # 計算するメソッド
        def sampling(number_of_steps)
          compute_visible
          # 最初のステップの隠れ層のユニット値とP(h|v)を保存
          @hidden_units_0 = @units[1].dup
          @hidden_probability_0 = @probability[0].dup
          number_of_steps.times do |i|
            compute_hidden
            compute_visible
          end

          compute_cross_entropy
        end

        # 重みとバイアスの更新
        def update_parameters
          update_biases
          update_weights
        end

        # 交差エントロピーの計算
        def compute_cross_entropy
          @log_probability = @probability[1].dup
          @probability[1].each_with_index do |data,i|
            @log_probability[i] = Math.log(data)
          end

          @log_probability_dash = @probability[1].dup
          @probability[1].each_with_index do |data,i|
            @log_probability_dash[i] = Math.log(1 - data)
          end

          @visible_units_0_dash = @visible_units_0.dup
          @visible_units_0_dash.each_with_index do |data,i|
            @visible_units_0_dash[i] = 1 - data
          end

          @cross_entropy += ((@visible_units_0 * @log_probability) + (@visible_units_0_dash * @log_probability_dash))
        end

        # 平均交差エントロピーの計算
        def compute_mean_cross_entropy
          # 交差エントロピーの最小化はKLダイバージェンスの最小化と等しい
          # 真の確率分布のエントロピーは一定のため
          @mean_cross_entropy = -@cross_entropy.sum/@number_of_data.to_f
          return @mean_cross_entropy
        end

        # 実行用
        def run(number_of_steps)
          sampling(number_of_steps)
          compute_expected_values
        end

        # 結果の出力用
        def outputs
          puts "visible_0: #{@visible_units_0}"
          puts "hidden_0: #{@hidden_units_0}"
          puts "visible_k: #{@units[0]}"
          puts "hidden_k: #{@units[1]}"
          puts "P(h|v)_0: #{@hidden_probability_0}"
          puts "P(v|h)_0: #{@visible_probability_0}"
          puts "P(h|v)_k: #{@probability[0]}"
          puts "P(v|h)_k: #{@probability[1]}"
        end

        # パラメータをファイルに保存するメソッド
        def save_parameters(filename)
          hash = {"@number_of_data" => @number_of_data,"@columns" => @columns,"@biases" => @biases,"@weights" => @weights}
          File.open(filename,"w+") do |f|
            f.puts(JSON.pretty_generate(hash))
          end
        end

        # パラメータをファイルから読み込むメソッド
        def load_parameters(filename)
          File.open(filename,"r+") do |f|
            hash = JSON.load(f)
            @number_of_data = hash["@number_of_data"]
            @columns = hash["@columns"]
            initialize(@columns,@number_of_data)

            biases_matrix = hash["@biases"].to_a
            @biases = []
            @columns.size.times do |i|
              @biases.push(N[biases_matrix[i].split(',').map!{ |item| item.delete("/[\-]/").gsub(" ","").to_f}].transpose)
            end
            puts "#{@biases}"


            weights_matrix = hash["@weights"].to_a
            @weights = []
            weights_array = weights_matrix[0].split(',').map!{ |item| item.delete("/[\-]/").gsub(" ","").to_f}.to_a
            @weights.push(NMatrix.new(@weights_geometry[0],weights_array))
            puts "#{@weights}"
          end
        end
    end
end
