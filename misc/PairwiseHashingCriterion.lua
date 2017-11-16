-- claim
local PairwiseHashingCriterion, parent = torch.class('nn.PairwiseHashingCriterion', 'nn.Criterion')

-- init
function PairwiseHashingCriterion:__init(margin, alpha)
  parent.__init(self)

  self.margin = margin or 1
  self.alpha = alpha or 0.01

  self.gradInput = {torch.Tensor(), torch.Tensor()}

  self.Li=torch.Tensor()
end
-- --------------------------------------------------------

-- updateOutput
-- (b1,b2,y) e.g. (img1,img2,0/1)
function PairwiseHashingCriterion:updateOutput(input,target)
  assert(#input==2,"Invalid number of inputs' dimension (should be 2)")

  local b1=input[1]  -- the 1st sample
  local b2=input[2]  -- the 2ed sample
  local y=target   -- label: 0/similar, 1/dissimilar

  local N=b1:size(1) -- batchsize

  -- compute self.Li
  local part_1=0.5*torch.cmul(1-y,(b1-b2):norm(2,2):pow(2))
  local temp=torch.max(torch.cat(self.margin-(b1-b2):norm(2,2):pow(2),torch.Tensor(N):zero():type(torch.type(b1)),2),2)
  local part_2=0.5*torch.cmul(y, temp)
  local part_3=self.alpha*((torch.abs(b1)-1):norm(1,2)+(torch.abs(b2)-1):norm(1,2))

  self.Li=part_1+part_2+part_3
  self.output=self.Li:sum()/N

  return self.output
end
-- --------------------------------------------------------

-- indicator(symmetric b1-b2,b2-b1 are the same): the different types of tensor will lead to differnt repeatTensor modes(ByteTensor,DoubleTensor)
local function indicator(b1,b2)
  local ind=(self.margin-(b1-b2):norm(2,2):pow(2)):gt(0):repeatTensor(1,b1:size(2))
  return ind
end
-- --------------------------------------------------------

-- mathsign
--
local function mathsign(value)
  local part_positive_1=1*(value:ge(1):type(value:type()))+1*torch.cmul(value:ge(-1):type(value:type()),value:le(0):type(value:type()))
  local part_negative_1=(-1)*(value:lt(-1):type(value:type()))+(-1)*torch.cmul(value:gt(0):type(value:type()),value:lt(1):type(value:type()))

  local result=part_positive_1+part_negative_1
  return result
end
-- --------------------------------------------------------

-- updateGradiInput
function PairwiseHashingCriterion:updateGradInput(input,target)
  local b1=input[1]  -- the 1st variable
  local b2=input[2]  -- the 2nd variable
  local y=target   -- label: 0/similar, 1/dissimilar

  local N=b1:size(1) -- batchsize

  local indicator=(self.margin-(b1-b2):norm(2,2):pow(2)):gt(0):repeatTensor(1,b1:size(2))

  self.gradInput[1]=torch.cmul((b1-b2),(1-y):repeatTensor(b1:size(2),1):t())+torch.cmul((b2-b1),indicator:type(b1:type())):cmul(y:repeatTensor(b1:size(2),1):t())+self.alpha*(mathsign(b1):type(b1:type()))
  self.gradInput[1]=(1/N)*self.gradInput[1]

  self.gradInput[2]=torch.cmul((b2-b1),(1-y):repeatTensor(b1:size(2),1):t())+torch.cmul((b1-b2),indicator:type(b1:type())):cmul(y:repeatTensor(b1:size(2),1):t())+self.alpha*(mathsign(b2):type(b2:type()))
  self.gradInput[2]=(1/N)*self.gradInput[2]

  return self.gradInput
end
-- --------------------------------------------------------
