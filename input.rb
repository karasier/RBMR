require_relative 'RBM'

rbm = RBMR::RBM.new([4,3])
rbm.input([1,0,0,0])
rbm.randomize
rbm.run(1)
