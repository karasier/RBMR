require_relative 'RBM'

rbm = RBMR::RBM.new([3,4],1)
#rbm.randomize
rbm.load_parameters("parameters.json")
rbm.input([1,0,0])
rbm.run(1)
rbm.update_parameters
rbm.outputs
rbm.save_parameters("parameters.json")
