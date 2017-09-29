require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'optim'
require 'misc.PairwiseHashingCriterion'

local models = require 'models'

local cmd = require 'cmd'
-- Parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.DoubleTensor')
lfs.mkdir(opt.output)
lfs.mkdir(opt.data)

if opt.gpuid >= 0 then
  require 'cunn'
  require 'cutorch'
  require 'cudnn'
  cutorch.setDevice(opt.gpuid + 1)
  cutorch.manualSeed(opt.seed)
end

local trainData = {}
local testData = {}
if opt.dataset == 'CIFAR10' then
  -- CIFAR 10
  cifar10 = require 'misc.cifar10'
  trainData, testData = cifar10.load()
elseif opt.dataset == 'Fashion-MNIST' then
  -- Fashion-MNIST
  fashion_mnist = require 'misc.fashion_mnist'
  trainData, testData = fashion_mnist.load()
elseif opt.dataset == 'MNIST' then
  -- MNIST
  mnist = require 'misc.mnist'
  trainData, testData = mnist.load()
else
  print('Please specify one of the following datasets: CIFAR10/MNIST/Fashion-MNIST')
  return
end
print('trainData:', trainData)
print('testData:', testData)

local model = {}
if opt.checkpoint == '' then
  -- training model from scratch
  local model1 = models.getModel(opt.dataset)
  local model2 = nn.Sequential()
  model2 = model1:clone('weight', 'bias', 'gradWeight', 'gradBias')

  model = nn.ParallelTable()
  model:add(model1)
  model:add(model2)
else
  -- training from previous checkpoint --
  local name = {}
  if opt.gpuid >= 0 then
    name = opt.dataset..'-model-GPU-'..opt.epochs..'-b'..opt.bits
  else
    name = opt.dataset..'-model-CPU-'..opt.epochs..'-b'..opt.bits
  end
  model = torch.load(paths.concat(opt.output, opt.checkpoint, name)..'.t7')
end

print(model)

-- Criterion
crit = nn.PairwiseHashingCriterion(2 * opt.bits, 0.01)

-- Transfer to GPU
if opt.gpuid >= 0 then
  model:cuda()
  crit:cuda()
end

-- Retrieval the gradient parameters
params, gradParams = model:getParameters()

-- Initialization
local method = 'xavier'
model = require('weight-init')(model, method)

-- Optimization Initial Configs
sgdState = sgdState or {
    learningRate = 1e-2,
    -- learningRateDecay = 0.001,
    weightDecay = 0.005,
    momentum = 0.9
}

function train(model, dataset, iter)
  collectgarbage();

  model:training()

  local shuffle = torch.randperm(dataset.size)

  for t = 1, dataset.size, opt.batchsize do

    -- create mini batch
    local data = {}
    local labels = {}

    for i = t, math.min(t+opt.batchsize-1, dataset.size) do
      table.insert(data, dataset.data[shuffle[i]]:totable())
      table.insert(labels, dataset.labels[shuffle[i]])
    end

    local count = #data
    local N = count * (count - 1) / 2

    local couple_x1 = {}
    local couple_x2 = {}

    local batchLabels = torch.Tensor(N):zero()

    for ii = 1, count do
      for jj = ii + 1, count do
        table.insert(couple_x1, data[ii])
        table.insert(couple_x2, data[jj])
        if labels[ii] == labels[jj] then
          batchLabels[#couple_x1] = 0
        else
          batchLabels[#couple_x1] = 1
        end
      end
    end

    local batchData = {}
    if opt.gpuid >= 0 then
      table.insert(batchData, torch.Tensor(couple_x1):cuda())
      table.insert(batchData, torch.Tensor(couple_x2):cuda())
      batchLabels = batchLabels:cuda()
    else
      table.insert(batchData, torch.Tensor(couple_x1))
      table.insert(batchData, torch.Tensor(couple_x2))
    end

    -- feval function
    local feval = function(params_new)
      collectgarbage()
      if params ~= params_new then
        params:copy(params_new)
      end
      gradParams:zero()

      -- perform mini-batch gradient descent
      local output=model:forward(batchData)

      local loss = crit:forward(output, batchLabels)

      local dl_dx=crit:backward(output, batchLabels)

      model:backward(batchData, dl_dx)

      return loss, gradParams
    end

    _, eval = optim.sgd(feval, params, sgdState)

    print('Train Epoch '..iter..':\t['..t..'/'..trainData.size..']'..'\tloss: ' ..eval[1])

  end
end

function evaluation(model, dataset)
  model:evaluate()

  local output = torch.Tensor(dataset.size, opt.bits):zero()

  if opt.gpuid >= 0 then
    output = output:cuda()
  end

  for i = 1, dataset.size do
    if opt.gpuid >= 0 then
      output[i] = model:forward(dataset.data[i]:cuda())
    else
      output[i] = model:forward(dataset.data[i])
    end
  end

  output = output:gt(0)

  return output
end


-- training --
local timestamp = {}
if opt.skip_training == false then
  timestamp = os.date('%Y-%m-%d-%H-%M-%S', ts)
  lfs.mkdir(paths.concat(opt.output, timestamp))
  for iter = 1, opt.epochs do

    train(model, trainData, iter)

    if iter % opt.checkpoint_interval == 0 then
      if opt.gpuid >= 0 then
        torch.save(paths.concat(opt.output, timestamp)..'/'..opt.dataset..'-model-GPU-'..iter..'-b'..opt.bits..'.t7', model)
      else
        torch.save(paths.concat(opt.output, timestamp)..'/'..opt.dataset..'-model-CPU-'..iter..'-b'..opt.bits..'.t7', model)
      end

      -- evaluation --
      local trainB = evaluation(model:get(1), trainData)
      local testB = evaluation(model:get(1), testData)
      matio = require 'matio'
      matio.save(paths.concat(opt.output, timestamp)..'/'..opt.dataset..'-model-CPU-'..iter..'-b'..opt.bits..'-data.mat', {B_train=trainB:double(), B_test=testB:double(), train_L=trainData.labels:double(), test_L=testData.labels:double()})
    end
  end
  if opt.gpuid >= 0 then
    torch.save(paths.concat(opt.output, timestamp)..'/'..opt.dataset..'-model-GPU-'..opt.epochs..'-b'..opt.bits..'.t7', model)
  else
    torch.save(paths.concat(opt.output, timestamp)..'/'..opt.dataset..'-model-CPU-'..opt.epochs..'-b'..opt.bits..'.t7', model)
  end
else
  timestamp = opt.checkpoint
end

-- evaluation --
local trainB = evaluation(model:get(1), trainData)
local testB = evaluation(model:get(1), testData)

matio = require 'matio'
matio.save(paths.concat(opt.output, timestamp)..'/'..opt.dataset..'-model-CPU-'..opt.epochs..'-b'..opt.bits..'-data.mat', {B_train=trainB:double(), B_test=testB:double(), train_L=trainData.labels:double(), test_L=testData.labels:double()})
print('Done')
