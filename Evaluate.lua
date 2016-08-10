require 'torch'
require 'rnn'
require 'cutorch'
require 'cunn'
display = require 'display'

rnn = torch.load('./SavedModel/trained-model.t7')
means = torch.load('./SavedModel/data_means.t7')
sds = torch.load('./SavedModel/data_sds.t7')

local dataLoader = require("./DataLoad")
local timeseq = dataLoader.loadData()

local rho = 50
local eval_part = 0.05
local nFeatures = timeseq:size()[1]

--Normalize with means and sds from training
for feature=1, nFeatures do
    timeseq[feature] = (timeseq[feature] - means[feature]) / sds[feature]
end

---Transform-to-all-possible-batches
local dataPrepare = require("./DataPrepare")
local all_slices, max_slices = dataPrepare.makeSlices(timeseq, rho, nFeatures) --get data transformed to slices (slice x nFeatures x rho) and number of slices

val_input_indeces =  torch.range(torch.ceil(max_slices * (1-eval_part)+1), max_slices-1):long()


--cutorch.setDevice(1)
npoints = val_input_indeces:size()[1]     --number of data points to generate

predicted_points = torch.LongTensor(npoints):cuda()
--rnn = rnn:cuda()
--all_slices = all_slices:cuda()
--val_input_indeces = val_input_indeces:cuda()
rnn:evaluate()
--for iteration = 1, npoints do
local inputs = all_slices:index(1, val_input_indeces):transpose(1,3):transpose(2,3) --take all validation slices as initial input, make appropriate format
inputs = inputs:cuda()

local outputs = rnn:forward(inputs)
predicted_points = outputs[{ {-1},{},{1} }] * sds[1] + means[1]  --take last value from 1st feature predictions, rescale
predicted_points = predicted_points:squeeze()

real_values = timeseq[1][{ {-npoints,-1} }] * sds[1] + means[1] --last npoints from initial sequence, 1st feature
real_values = real_values:cuda()

local MSE = torch.add(real_values, -1, predicted_points):pow(2):mean()
local naive_benchmark = timeseq[1][{ {-(npoints+1),-2} }] * sds[1] + means[1]
naive_benchmark = naive_benchmark:cuda()
local nbMSE = torch.add(real_values, -1, naive_benchmark):pow(2):mean()
print('model MSE: '..MSE)
print('naive benchmark MSE: '..nbMSE)

local real_rise = torch.gt(torch.add(real_values[{ {2,-1} }], -1, real_values[{ {1,-2} }]),0)
local predicted_rise = torch.gt(torch.add(predicted_points[{ {2,-1} }], -1, real_values[{ {1,-2} }]),0)
local correct_predictions = torch.eq(real_rise, predicted_rise)
correct_predictions = torch.sum(correct_predictions) / correct_predictions:size()[1]
print('correct predictions of rize or fall: '..correct_predictions..'%')

local config2 = {
  title = "Validation vs Prediction",
  --labels = {"Validation", "Prediction"},
  xlabel = "iretarion",
  --logscale = true,
}

display.plot(torch.cat(torch.cat(torch.linspace(1, npoints, npoints):cuda(),
                                 real_values, 2
                                 ), 
                      predicted_points, 2
                    )
            , config2)
          
require 'Dataframe'
df = Dataframe()
rl = torch.totable(real_values)
pp = torch.totable(predicted_points)

df:load_table{data=Df_Dict{RealValues=rl, PredictedValues=pp}}
df:to_csv('./Eval_Output/real_vs_predicted.csv')
