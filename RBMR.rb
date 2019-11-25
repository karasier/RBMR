require 'nmatrix'
require 'zlib'
require 'benchmark'
require 'json'
require 'random_bell'

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

            # 0ステップ目のユニットの値を格納
            @visible_units_0 = NMatrix.new([1,@units[0].size],0.0).transpose

            # ユニットが1をとる条件付き確率を格納
            # P(hidden|visible)→P(visible|hidden)の順で格納
            @probability = @columns.reverse.map{ |col| NMatrix.new([1,col],0.0).transpose }

            # ユニット値のサンプリング用の一様乱数 区間[0,1]
            @random_geometry = @biases_geometry.clone

            # バイアスの導関数
            @derivative_visible_bias = NMatrix.new([1,@units[0].size],0.0).transpose
            @derivative_hidden_bias = NMatrix.new([1,@units[1].size],0.0).transpose

            # 重みの導関数
            @derivative_weights = @weights_geometry.map do |geo|
                NMatrix.new(geo,0.0)
            end

            # 交差エントロピー計算用
            @cross_entropy = NMatrix.new([1,@units[0].size],0.0).transpose

            # μ = 0, σ = 0.01のガウス分布を作成
            @bell = RandomBell.new(mu: 0, sigma: 0.01, range: -0.03..0.03)

            @error = 0
        end


        # Set up the NN to random values.
        def randomize
            # Create random fast matrices for the biases.
            # NMatrixの配列を作成 バイアス
            @biases = @biases_geometry.map { |geo| NMatrix.new(geo,0.0)}
            puts "@biases: #{@biases}"

            # Create random fast matrices for the weights.
            # NMatrixの配列を作成 重み
            # ガウス分布に従うように重みをランダムに初期化
            weights_array = []
            @weights_geometry[0][0].times do |i|
              @weights_geometry[0][1].times do |j|
                weights_array.push(@bell.rand)
              end
            end

            @weights = []
            @weights.push(NMatrix.new(@weights_geometry[0],weights_array))

            puts "@weights: #{@weights}"

            @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}
        end

        # RBMへの入力を取得
        # 引数: *vaules→入力
        def input(*values)
          @visible_units_0 = N[values.flatten,:dtype => :float64].transpose
          @units[0] = @visible_units_0.dup
        end

        # 隠れ層への入力
        def input_hidden_layer(*values)
          @units[1] = N[values.flatten,:dtype => :float64].transpose
          compute_hidden
          puts "hidden_units: #{@units[1]}"
          puts "visible units: #{@units[0]}"
          puts "P(v|h): #{@probability[1]}"
        end

        # P(h|v)と隠れ層のユニットの値を計算
        def compute_visible
          # 隠れ層の条件付き確率を計算
          @pre_sigmoid = NMatrix::BLAS.gemm(@weights[0],@units[0],@biases[0])
          @pre_sigmoid.each_with_index do |data,i|
            @probability[0][i] = Sigmoid.call(data)
          end

          @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}

          # 隠れ層のユニットの値を計算
          @probability[0].size.times do |i|
            if @probability[0][i] >= @random_value[0][i] then
              @units[1][i] = 1
            else
              @units[1][i] = 0
            end
          end
        end

        # P(v|h)と可視層のユニットの値を計算
        def compute_hidden
          # 可視層の条件付き確率を計算
          @pre_sigmoid = NMatrix::BLAS.gemm(@weights[0],@units[1],nil,1.0,0.0,:transpose) + @biases[1]
          @pre_sigmoid.each_with_index do |data,i|
            @probability[1][i] = Sigmoid.call(data)
          end

          @random_value = @random_geometry.map { |geo| NMatrix.random(geo,:dtype => :float64)}

          # 可視層のユニットの値を計算
          @probability[1].size.times do |i|
            if @probability[1][i] >= @random_value[1][i] then
              @units[0][i] = 1
            else
              @units[0][i] = 0
            end
          end
        end


        # 導関数の初期化
        def initialize_derivatives
          # バイアスの導関数の初期化
          @derivative_visible_bias = NMatrix.new([1,@units[0].size],0.0).transpose
          @derivative_hidden_bias = NMatrix.new([1,@units[1].size],0.0).transpose

          # 重みの導関数の初期化
          @derivative_weights = @weights_geometry.map do |geo|
              NMatrix.new(geo,0.0)
          end
        end

        # 重みの導関数を計算
        def weights_derivative
          @derivative_weights = NMatrix::BLAS.gemm(@hidden_probability_0,@visible_units_0.transpose) - NMatrix::BLAS.gemm(@probability[0],@units[0].transpose)
        end

        # バイアスの導関数を計算
        def bias_derivative
          @derivative_visible_bias = (@visible_units_0 - @units[0])
          @derivative_hidden_bias = (@hidden_probability_0 - @probability[0])
        end

        # 導関数の計算
        def compute_derivatives
          bias_derivative
          weights_derivative
        end

        # バイアスの更新
        def update_biases
          @biases[0] += @derivative_hidden_bias * @training_rate
          @biases[1] += @derivative_visible_bias * @training_rate
        end

        # 重みの更新
        def update_weights
          @weights[0] += @derivative_weights * @training_rate
        end

        # 重みとバイアスの更新
        def update_parameters
          update_biases
          update_weights
        end

        # 計算するメソッド
        def sampling(number_of_steps)
          compute_visible
          # 最初のステップの隠れ層のユニット値とP(h|v)を保存
          @hidden_probability_0 = @probability[0].dup
          compute_hidden
          number_of_steps.times do |i|
            compute_visible
            compute_hidden
          end

          compute_cross_entropy
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
        def compute_mean_cross_entropy(number_of_data)
          # 交差エントロピーの最小化はKLダイバージェンスの最小化と等しい
          # 真の確率分布のエントロピーは一定のため
          @mean_cross_entropy = -@cross_entropy.sum/number_of_data.to_f
          @cross_entropy = NMatrix.new([1,@units[0].size],0.0).transpose
          return @mean_cross_entropy
        end

        def get_error_rate(times)
          error = false

          @units[0].size.times do |i|
            if @units[0][i] != @visible_units_0[i] then
              error = true
            end
          end

          if error then
            @error += 1
          end

          #puts "erorr_rate:#{@error.to_f / times.to_f}"
          return @error.to_f/times.to_f
        end
        # 実行用
        def run(number_of_steps)
          sampling(number_of_steps)
          compute_derivatives
          update_parameters
        end

        def reconstruct(*values)
          input(values)
          compute_visible
          compute_hidden
          compute_visible
          outputs
        end

        # 結果の出力用
        def outputs
          puts "input: #{@visible_units_0}"
          puts "visible units: #{@units[0]}"
          puts "P(v|h): #{@probability[1]}"
        end

        # パラメータをファイルに保存するメソッド
        def save_parameters(filename)
          hash = {"@columns" => @columns,"@biases" => @biases,"@weights" => @weights}
          File.open(filename,"w+") do |f|
            f.puts(JSON.pretty_generate(hash))
          end
        end

        # パラメータをファイルから読み込むメソッド
        def load_parameters(filename)
          File.open(filename,"r+") do |f|
            hash = JSON.load(f)
            initialize(@columns)

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
