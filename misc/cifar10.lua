local cifar10 = {}

function cifar10.load()
  -- download dataset
  if not paths.dirp('./data/cifar-10-batches-t7') then
    local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
    local tar = paths.basename(www)
    os.execute('wget ' .. www .. ';' .. 'tar xvf ' .. tar .. '; rm -rf ' .. tar .. '; mv cifar-10-batches-t7 data/')
  end

  -- load dataset
  local trainData = {
    data = torch.Tensor(50000, 3072):double(),
    labels = torch.Tensor(50000):byte(),
    size = 50000,
  }
  for i = 0,4 do
    subset = torch.load('./data/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
    trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():double()
    trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
  end
  trainData.labels = trainData.labels + 1

  subset = torch.load('./data/cifar-10-batches-t7/test_batch.t7', 'ascii')
  local testData = {
    data = subset.data:t():double(),
    labels = subset.labels[1],
    size = 10000
  }
  testData.labels = testData.labels + 1

  -- reshape data
  trainData.data = trainData.data:reshape(50000,3,32,32)
  testData.data = testData.data:reshape(10000,3,32,32)

  -- pre-process
  -- data preparation
  local mean = {} -- store the mean, to normalize the test set in the future
  local stdv  = {} -- store the standard-deviation for the future
  for i=1,3 do -- over each image channel
    mean[i] = trainData.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainData.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

  for i=1,3 do -- over each image channel
    testData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

  return trainData, testData
end
return cifar10
