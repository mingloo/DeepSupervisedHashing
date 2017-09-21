require 'torch'
require 'nn'
local cmd = require 'cmd'

models = {}

function models.getModel(dataset)

  local model = {}

  if opt.dataset == 'CIFAR10' then
    model = nn.Sequential()
    model:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialCrossMapLRN(3, 5e-05, 0.75))

    model:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialCrossMapLRN(3, 5e-05, 0.75))

    model:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    model:add(nn.Reshape(64*4*4))
    model:add(nn.Linear(64*4*4, 512))
    model:add(nn.ReLU())
    model:add(nn.Linear(512, opt.bits))
  elseif opt.dataset == 'MNIST' or opt.dataset == 'Fashion-MNIST' then
    model = nn.Sequential()
    model:add(nn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2, 2))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialCrossMapLRN(3, 5e-05, 0.75))

    model:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialCrossMapLRN(3, 5e-05, 0.75))

    model:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
    model:add(nn.ReLU())
    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    model:add(nn.Reshape(64*3*3))
    model:add(nn.Linear(64*3*3, 512))
    model:add(nn.ReLU())
    model:add(nn.Linear(512, opt.bits))
  end

  return model
end

return models





