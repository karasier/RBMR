require_relative 'RBM'

rbm = RBMR::RBM.new([2,3])
rbm.input([1,1])
rbm.randomize
rbm.run(1)
