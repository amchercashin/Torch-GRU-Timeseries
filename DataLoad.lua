--Load or make some data
--outputs data in nfeatures x timesteps format

local dataLoader = {};

function dataLoader.makeData(timesteps)
  local timesteps = timesteps or 5000
  local nFeatures = 2
  
  local timeseq = torch.Tensor(nFeatures, timesteps) --timeseries, nFeatures x timesteps
--timeseq[1] = torch.cos(torch.linspace(0, 200, timesteps))
  timeseq[1] = torch.add(
                        torch.cmul(
                                  torch.cos(torch.linspace(0, timesteps/4, timesteps)), 
                                  torch.linspace(0, timesteps/4, timesteps)), 
                        torch.linspace(0, timesteps/4, timesteps))
  timeseq[2] = torch.linspace(0, timesteps/4, timesteps)
  
  return timeseq
end

function dataLoader.loadData(filePath)
  require 'Dataframe'
  local filePath = filePath or '/home/amchercashin/DataScience/Torch/Quotes_prediction_with_RNN/Data/data.csv'
  local df = Dataframe()
  df:load_csv{path=filePath, header=true}

  local timeseq = df:to_tensor()
  timeseq = timeseq:t()
  return timeseq
end

return dataLoader;