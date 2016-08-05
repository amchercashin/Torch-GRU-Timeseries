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
  local filePath = filePath or './Data/data.csv'
  local df = Dataframe()
  df:load_csv{path=filePath, header=true}

  local timeseq = df:to_tensor()
  timeseq = timeseq:t()
  return timeseq
end

function dataLoader.normalize(timeseq) --Normalize data in rows
  nFeatures = timeseq:size()[1]
  local means, sds = {}, {}
  for feature=1, nFeatures do
    means[feature] = timeseq[feature]:mean()
    sds[feature] = timeseq[feature]:std()
    timeseq[feature] = (timeseq[feature] - means[feature]) / sds[feature]
  end
  
  return timeseq, means, sds
end

return dataLoader;