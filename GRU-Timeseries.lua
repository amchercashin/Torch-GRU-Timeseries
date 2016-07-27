require 'torch'
require 'rnn'
display = require 'display'


----Options-------
gpu=1

--nIters = 2000
batchSize = 8
rho = 100
hiddenSize = 200
nFeatures = 2
nOutput = nFeatures  --TODO make possible to set Nfeatures and nOutput to different values
lr = 0.0006
train_part = 0.9
validate_each_steps = 100 --get validation error each validate_each_steps steps
-------------------

---NN-defenition---
rnn = nn.Sequential()
   :add(nn.Linear(nFeatures, hiddenSize))
   --:add(nn.NormStabilizer())
   :add(nn.GRU(hiddenSize, hiddenSize))
   --:add(nn.NormStabilizer())
   --:add(nn.GRU(hiddenSize, hiddenSize))
   :add(nn.Linear(hiddenSize, nOutput))
   --:add(nn.HardTanh())
rnn = nn.Sequencer(rnn)
--rnn:training()
print(rnn)

criterion = nn.MSECriterion()
criterion = nn.SequencerCriterion(criterion)
---------------------

----Make-some-data---
timesteps = 4000
timeseq = torch.Tensor(nFeatures, timesteps) --timeseries, nFeatures x timesteps
--timeseq[1] = torch.cos(torch.linspace(0, 200, timesteps))
timeseq[1] = torch.add(
                      torch.cmul(
                                torch.cos(torch.linspace(0, timesteps/4, timesteps)), 
                                torch.linspace(0, timesteps/4, timesteps)), 
                      torch.linspace(0, timesteps/4, timesteps))
timeseq[2] = torch.linspace(0, timesteps/4, timesteps)

---Normalize-data---
means, sds = {}, {}
for feature=1, nFeatures do
  means[feature] = timeseq[feature]:mean()
  sds[feature] = timeseq[feature]:std()
  timeseq[feature] = (timeseq[feature] - means[feature]) / sds[feature]
end

---Transform-to-all-possible-batches
max_slices = timeseq:size(timeseq:dim()) - rho + 1 --we-should-transform-our-long-sequnce-of-length-timesteps
all_slices = torch.Tensor(max_slices, nFeatures, rho) --to-many-sclices-of-rho-lenght
for t=1, rho do  
  all_slices[{ {}, {}, t }] = timeseq:t():narrow(1, t, max_slices) --all-possible-slices-ordered: slice x nFeatures x rho
end

max_train_slices = math.floor(max_slices * train_part)
train_input_indeces = torch.randperm(max_train_slices-1):long() --shuffled indeces 
train_target_indeces = train_input_indeces + 1                  --targets indeces for inputs

val_input_indeces =  torch.range(max_train_slices+1, max_slices-1):long() -- not shuffled indeces for validation
val_target_indeces = val_input_indeces + 1
----------------------------------

---TRAINING-THE-NETWORK---
maxIteration = torch.floor(max_train_slices / batchSize)  --max n of iteration to iterate over all data, for could leave some data sequences in the end
errors = torch.Tensor(maxIteration,3)
--predictions = torch.Tensor(val_input_indeces:size(), 2)
--predictions[1] = torch.linspace(0, val_input_indeces:size(), val_input_indeces:size()) -- to accumulate predictions from validation set of the first time sequence
--predictions[2] = 

---Place-all-on-CUDA---
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
  
  rnn = rnn:cuda()
  criterion = criterion:cuda()
  all_slices = all_slices:cuda()
  train_input_indeces = train_input_indeces:cuda()
  train_target_indeces = train_target_indeces:cuda()
  val_input_indeces = val_input_indeces:cuda()
  val_target_indeces = val_target_indeces:cuda()
  errors = errors:cuda()
 -- predictions = predictions:cuda()
end
--------------------
rnn:training()
for iteration = 1, maxIteration do --redo later, can leave some slices aside because of floor   
   local inputs = all_slices:index(1, train_input_indeces):narrow(1, 1+(iteration-1)*batchSize, batchSize) --consequently scan through shuffled slices
   local targets = all_slices:index(1, train_target_indeces):narrow(1, 1+(iteration-1)*batchSize, batchSize)
   
   inputs = inputs:transpose(1,3):transpose(2,3) --to seqlength x batchsize x nFeatures shape
   targets = targets:transpose(1,3):transpose(2,3)   
   
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
   
   rnn:zeroGradParameters()
   
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward(inputs, gradOutputs)
   rnn:updateParameters(lr)
   
   if math.fmod(iteration, validate_each_steps) == 0 or iteration == 1 then
     local val_inputs = all_slices:index(1, val_input_indeces) --get all validation set
     local val_targets = all_slices:index(1, val_target_indeces)
     
     val_inputs = val_inputs:transpose(1,3):transpose(2,3) --to seqlength x batchsize x nFeatures shape
     val_targets = val_targets:transpose(1,3):transpose(2,3)     
     
     local val_outputs = rnn:forward(val_inputs)
     val_err = criterion:forward(val_outputs, val_targets)
     print(string.format("VALIDATION ERROR = %f ", val_err))          
   end   
   
   errors[{ {iteration}, {1} }] = iteration
   errors[{ {iteration}, {2} }] = err
   errors[{ {iteration}, {3} }] = val_err
end
------------------------

display.plot(errors)

---Reconstrukt timeseries from 1st validation slice into the future
rnn:evaluate()
npoints = max_slices - max_train_slices - 1     --number of data points to generate
predicted_points = torch.LongTensor(npoints):cuda()
for iteration = 1, npoints do
  if iteration == 1 then 
    outputs = all_slices:index(1, val_input_indeces):transpose(1,3):transpose(2,3)[{ {},{1},{} }] --take 1st validation slice as initial input
  end
  --rnn:zeroGradParameters()
  outputs = rnn:forward(outputs)
  predicted_points[iteration] = outputs[{ {-1},{},{1} }] * sds[1] + means[1]  --take last value from 1st feature predictions, rescale
  
end

real_values = timeseq[1][{ {-npoints,-1} }] * sds[1] + means[1] --last npoints from initial sequence, 1st feature
display.plot(torch.cat(torch.cat(torch.linspace(1, npoints, npoints):cuda(),
                                 real_values:cuda(), 2
                                 ), 
                      predicted_points, 2
                    )
            )

--predictions = rnn:forward( all_slices:index(1, val_input_indeces):transpose(1,3):transpose(2,3)[{ {},{max_slices - max_train_slices - 1},{} }] )--predict values for last slice from validation set
--predictions = predictions:resize(rho, 2)[{ {},{1} }] * sds[1] + means[1] --take only first feature, resize, rescale
--real_values = all_slices[max_slices][1] * sds[1] + means[1]
--display.plot(torch.cat(torch.cat(torch.linspace(1, rho, rho):cuda(),
--                                 real_values, 2
--                                 ), 
--                      predictions,2
--                      )
--           )
