require_relative 'RBM'

rbm = RBMR::RBM.new([2,2])
rbm.input([1,0])
rbm.randomize
rbm.run(1)
