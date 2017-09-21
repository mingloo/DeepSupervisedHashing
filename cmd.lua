cmd = torch.CmdLine()
cmd:text('Options')
-- Dataset
cmd:option('-dataset', 'Fashion-MNIST', 'datasets: CIFAR10/MNIST/Fashion-MNIST')

-- Model parameters
cmd:option('-bits', 48, 'the hashing bits length')
cmd:option('-batchsize', 20, 'the batch size')
cmd:option('-epochs', 100, 'the total iterations')

-- Miscellaneous
cmd:option('-seed', 123, 'Torch manual random number generator seed')
cmd:option('-gpuid', 0, '0-indexed id of GPU to use. -1 = CPU')
cmd:option('-output', 'output/', 'output path')
cmd:option('-data', 'data/', 'data path')
cmd:option('-checkpoint_interval', 10, 'how many iterations we save the model as checkpoint')
cmd:option('-checkpoint', '', 'the timestamp of previous checkpoint')
cmd:option('-skip_training', false, 'skip training and do evaluation only')

return cmd
