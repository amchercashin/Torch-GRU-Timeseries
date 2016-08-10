require 'torch'
require 'rnn'
display = require 'display'

----Make-some-data---

local dataLoader = require("./DataLoad")
--local timeseq = dataLoader.makeData(5000)
local timeseq = dataLoader.loadData()
timeseq, means, sds = dataLoader.normalize(timeseq) --Normalize data and get row means and sd's
---------------------

----Options-------
gpu=1

--nIters = 2000
batchSize = 128
rho = 50
hiddenSize = 80
nFeatures = timeseq:size()[1]
nOutput = nFeatures  --TODO make possible to set Nfeatures and nOutput to different values
lr = 0.0003
train_part = 0.98
validate_each_steps = 100 --get validation error each validate_each_steps steps
nFullCycles = 1 --number of grand cycles: passes through all data
-------------------

---NN-defenition---
rnn = nn.Sequential()
   :add(nn.Linear(nFeatures, hiddenSize))
   --:add(nn.NormStabilizer())
   :add(nn.GRU(hiddenSize, hiddenSize))
   --:add(nn.NormStabilizer())
   --:add(nn.GRU(hiddenSize, hiddenSize))
   --:add(nn.NormStabilizer())
   --:add(nn.ReLU())
   --:add(nn.Linear(hiddenSize, 10))
   --:add(nn.ReLU())
   :add(nn.Linear(hiddenSize, nOutput))
   --:add(nn.HardTanh())
rnn = nn.Sequencer(rnn)
--rnn:training()
print(rnn)

criterion = nn.MSECriterion()
criterion = nn.SequencerCriterion(criterion)
---------------------

---Transform-to-all-possible-batches
local dataPrepare = require("./DataPrepare")
local all_slices, max_slices = dataPrepare.makeSlices(timeseq, rho, nFeatures) --get data transformed to slices (slice x nFeatures x rho) and number of slices

max_train_slices = math.floor(max_slices * train_part)
train_input_indeces = torch.randperm(max_train_slices-1):long() --shuffled indeces 
train_target_indeces = train_input_indeces + 1                  --targets indeces for inputs

val_input_indeces =  torch.range(max_train_slices+1, max_slices):long() -- not shuffled indeces for validation
--val_target_indeces = val_input_indeces + 1
----------------------------------

---TRAINING-THE-NETWORK---
maxIteration = torch.floor(max_train_slices / batchSize)  --max n of iteration to iterate over all data, for could leave some data sequences in the end
errors = torch.Tensor(maxIteration * nFullCycles,3)

---Place-all-on-CUDA---
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
  
  rnn = rnn:cuda()
  criterion = criterion:cuda()
  --all_slices = all_slices:cuda()
  --train_input_indeces = train_input_indeces:cuda()
  --train_target_indeces = train_target_indeces:cuda()
  --val_input_indeces = val_input_indeces:cuda()
  --val_target_indeces = val_target_indeces:cuda()
  errors = errors:cuda()
 -- predictions = predictions:cuda()
end
--------------------
require 'optim'
local params, gradParams = rnn:getParameters()
rnn:training()
for fullCycle = 1, nFullCycles do
  for iteration = 1, maxIteration do --redo later, can leave some slices aside because of floor   
     seq_iteration = fullCycle * maxIteration - maxIteration + iteration
     local inputs = all_slices:index(1, train_input_indeces):narrow(1, 1+(iteration-1)*batchSize, batchSize) --consequently scan through shuffled slices
     local targets = all_slices:index(1, train_target_indeces):narrow(1, 1+(iteration-1)*batchSize, batchSize)
     
     inputs = inputs:transpose(1,3):transpose(2,3) --to seqlength x batchsize x nFeatures shape
     targets = targets:transpose(1,3):transpose(2,3)   
     
     inputs = inputs:cuda()
     targets = targets:cuda()
     
     local function feval(params)
       gradParams:zero()
       local outputs = rnn:forward(inputs)
       err = criterion:forward(outputs, targets)
       print(string.format("Iteration %d ; NLL err = %f ", seq_iteration, err))
       local gradOutputs = criterion:backward(outputs, targets)
       rnn:backward(inputs, gradOutputs)
       
       return loss,gradParams
     end
     
     optim.adam(feval, params, {learningRate = lr})
     
     if math.fmod(seq_iteration, validate_each_steps) == 0 or seq_iteration == 1 then
       local val_inputs = all_slices:index(1, val_input_indeces) --get all validation set
       --local val_targets = all_slices:index(1, val_target_indeces)
       
       val_inputs = val_inputs:transpose(1,3):transpose(2,3) --to seqlength x batchsize x nFeatures shape
       --val_targets = val_targets:transpose(1,3):transpose(2,3)     
       
       val_inputs = val_inputs:cuda()
       --val_targets = val_targets:cuda()
       
       local val_outputs = rnn:forward(val_inputs[{ {}, {1,-2}, {} }])
       val_err = criterion:forward(val_outputs, val_inputs[{ {}, {2,-1}, {} }])
       print(string.format("VALIDATION ERROR = %f ", val_err))          
     end   
     
     errors[{ {seq_iteration}, {1} }] = seq_iteration
     errors[{ {seq_iteration}, {2} }] = err
     errors[{ {seq_iteration}, {3} }] = val_err
  end
end
torch.save('./SavedModel/trained-model.t7', rnn)
torch.save('./SavedModel/data_means.t7', means)
torch.save('./SavedModel/data_sds.t7', sds)
------------------------
local config1 = {
  title = "Loss over time",
  --labels = {"Validation", "Prediction"},
  xlabel = "iretarion",
  logscale = true,
}
display.plot(errors, config1)

---Reconstrukt timeseries from 1st validation slice into the future
rnn:evaluate()
npoints = max_slices - max_train_slices - 1     --number of data points to generate
predicted_points = torch.LongTensor(npoints):cuda()
for iteration = 1, npoints do
  if iteration == 1 then 
    outputs = all_slices:index(1, val_input_indeces):transpose(1,3):transpose(2,3)[{ {},{1},{} }] --take 1st validation slice as initial input
    outputs = outputs:cuda()
  end
  
  outputs = rnn:forward(outputs)
  predicted_points[iteration] = outputs[{ {-1},{},{1} }] * sds[1] + means[1]  --take last value from 1st feature predictions, rescale
  
end

real_values = timeseq[1][{ {-npoints,-1} }] * sds[1] + means[1] --last npoints from initial sequence, 1st feature
local config2 = {
  title = "Validation vs Prediction",
  --labels = {"Validation", "Prediction"},
  xlabel = "iretarion",
  --logscale = true,
}
display.plot(torch.cat(torch.cat(torch.linspace(1, npoints, npoints):cuda(),
                                 real_values:cuda(), 2
                                 ), 
                      predicted_points, 2
                    )
            , config2)

