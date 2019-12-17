require 'nmatrix'
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
        def initialize(*columns)
            # 学習率
            @training_rate = 0.1
            # Ensure columns is a proper array.
            # 各層のニューロンの数を配列にする
            @columns = columns.flatten

            # Creates the geometry of the bias matrices
            # 可視層と隠れ層のバイアスを格納
            # 隠れ層→可視層の順で格納
            @biases_geometry = @columns.reverse.map { |col| [1,col] }

            # Create the geometry of the weight matrices
            # 可視層と隠れ層のユニット間の重みを格納
            @weights_geometry = @columns[0..1]

            # Create units matrices
            # 可視層と隠れ層のユニットの値を格納
            # 可視層→隠れ層の順で格納
            @units = @columns.map{ |col| NMatrix.new([1,col],0.0) }

            # ユニットが1をとる条件付き確率を格納
            # P(hidden|visible)→P(visible|hidden)の順で格納
            @probability = @columns.reverse.map{ |col| NMatrix.new([1,col],0.0) }

            # バイアスの導関数
            @derivative_visible_bias = NMatrix.new([1,@units[0].size],0.0)
            @derivative_hidden_bias = NMatrix.new([1,@units[1].size],0.0)

            # 重みの導関数
            @derivative_weights = @weights_geometry.map do |geo|
                NMatrix.new(geo,0.0)
            end

            # 交差エントロピー計算用
            @cross_entropy = NMatrix.new([1,@units[0].size],0.0)

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
            @weights_geometry[0].times do |i|
              @weights_geometry[0].times do |j|
                weights_array.push(@bell.rand)
              end
            end

            @weights = []
            @weights.push(NMatrix.new(@weights_geometry,weights_array))
            puts "@weights: #{@weights}"
        end

        # RBMへの入力を取得
        # 引数: *vaules→入力
        def input(*values)
          @units[0] = N[values.flatten,:dtype => :float64]
          @inputs = @units[0].dup
        end

        # 隠れ層への入力
        def input_hidden_layer(*values)
          @units[1] = N[values.flatten,:dtype => :float64]
          compute_hidden
          puts "hidden_units: #{@units[1]}"
          puts "visible units: #{@units[0]}"
          puts "P(v|h): #{@probability[1]}"
        end

        # P(h|v)と隠れ層のユニットの値を計算
        def compute_visible
          # 隠れ層の条件付き確率を計算
          @pre_sigmoid = NMatrix::BLAS.gemm(@units[0],@weights[0],@biases[0])
          @probability[0] = NMatrix.ones_like(@pre_sigmoid)/((-@pre_sigmoid).exp + 1)
          #@pre_sigmoid.each_with_index do |data,i|
          #  @probability[0][i] = Sigmoid.call(data)
          #end

          # 隠れ層のユニットの値を計算
          @probability[0].each_with_index do |prob,i|
            @units[1][i] = prob >= rand ? 1.0 : 0.0
          end
        end

        # P(v|h)と可視層のユニットの値を計算
        def compute_hidden
          # 可視層の条件付き確率を計算
          @pre_sigmoid = NMatrix::BLAS.gemm(@units[1],@weights[0],nil,1.0,0.0,false,:transpose) + @biases[1]
          @probability[1] = NMatrix.ones_like(@pre_sigmoid)/((-@pre_sigmoid).exp + 1)
          #@pre_sigmoid.each_with_index do |data,i|
          #  @probability[1][i] = Sigmoid.call(data)
          #end

          # 可視層のユニットの値を計算
          @probability[1].each_with_index do |prob,i|
            @units[0][i] = prob >= rand ? 1.0 : 0.0
          end
        end


        # 導関数の初期化
        def initialize_derivatives
          # バイアスの導関数の初期化
          @derivative_visible_bias = NMatrix.new([1,@units[0].size],0.0)
          @derivative_hidden_bias = NMatrix.new([1,@units[1].size],0.0)

          # 重みの導関数の初期化
          @derivative_weights = @weights_geometry.map do |geo|
              NMatrix.new(geo,0.0)
          end
        end

        # 重みの導関数を計算
        def weights_derivative
          @derivative_weights = NMatrix::BLAS.gemm(@inputs,@hidden_probability,nil,1.0,0.0,:transpose) - NMatrix::BLAS.gemm(@units[0],@probability[0],nil,1.0,0.0,:transpose)
        end

        # バイアスの導関数を計算
        def bias_derivative
          @derivative_visible_bias = (@inputs - @units[0])
          @derivative_hidden_bias = (@hidden_probability - @probability[0])
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
          @hidden_probability = @probability[0].dup
          compute_hidden
          number_of_steps.times do |i|
            compute_visible
            compute_hidden
          end
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

          @inputs_dash = @inputs.dup
          @inputs_dash.each_with_index do |data,i|
            @inputs_dash[i] = 1 - data
          end

          @cross_entropy += ((@inputs * @log_probability) + (@inputs_dash * @log_probability_dash))
        end

        # 平均交差エントロピーの計算
        def compute_mean_cross_entropy(number_of_data)
          # 交差エントロピーの最小化はKLダイバージェンスの最小化と等しい
          # 真の確率分布のエントロピーは一定のため
          @mean_cross_entropy = -@cross_entropy.to_a.sum/number_of_data.to_f
          @cross_entropy = NMatrix.new([1,@units[0].size],0.0)
          return @mean_cross_entropy
        end

        def get_error_rate(times)
          error = false

          @units[0].size.times do |i|
            if @units[0][i] != @inputs[i] then
              error = true
            end
          end

          if error then
            @error += 1
          end

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
          outputs
        end

        # 結果の出力用
        def outputs
          puts "input: #{@inputs}"
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
              @biases.push(N[biases_matrix[i].split(',').map!{ |item| item.delete("/[\-]/").gsub(" ","").to_f}])
            end
            puts "#{@biases}"


            weights_matrix = hash["@weights"].to_a
            @weights = []
            weights_array = weights_matrix[0].split(',').map!{ |item| item.delete("/[\-]/").gsub(" ","").to_f}.to_a
            @weights.push(NMatrix.new(@weights_geometry,weights_array))
            puts "#{@weights}"
          end
        end
    end
end
