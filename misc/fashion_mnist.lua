local fashion_mnist = {}

function fashion_mnist.load()
  -- load dataset
  local dataset = require 'fashion-mnist'
  -- load trainData
  local trainData = dataset.traindataset()
  trainData.data = trainData.data:view(trainData.size, 1, 28, 28)
  trainData.labels = trainData.label

  local testData = dataset.testdataset()
  testData.data = testData.data:view(testData.size, 1, 28, 28)
  testData.labels = testData.label

  -- cast data type
  trainData.data = trainData.data:double()
  testData.data = testData.data:double()

  -- pre-process
  -- data preparation
  local mean = {} -- store the mean, to normalize the test set in the future
  local stdv  = {} -- store the standard-deviation for the future
  for i=1,1 do -- over each image channel
    mean[i] = trainData.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainData.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

  for i=1,1 do -- over each image channel
    testData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

  return trainData, testData
end
return fashion_mnist
