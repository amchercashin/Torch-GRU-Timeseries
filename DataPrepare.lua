local dataPrepare = {};

function dataPrepare.makeSlices(timeseq, rho, nFeatures)
---Transform-to-all-possible-batches
  local max_slices = timeseq:size(timeseq:dim()) - rho + 1 --we-should-transform-our-long-sequnce-of-length-timesteps
  local all_slices = torch.Tensor(max_slices, nFeatures, rho) --to-many-sclices-of-rho-length
  for t=1, rho do  
    all_slices[{ {}, {}, t }] = timeseq:t():narrow(1, t, max_slices) --all-possible-slices-ordered: slice x nFeatures x rho
  end

  --local max_train_slices = math.floor(max_slices * train_part)
  --local train_input_indeces = torch.randperm(max_train_slices-1):long() --shuffled indeces 
  --local train_target_indeces = train_input_indeces + 1                  --targets indeces for inputs

  --local val_input_indeces =  torch.range(max_train_slices+1, max_slices-1):long() -- not shuffled indeces for validation
  --local val_target_indeces = val_input_indeces + 1
  
  return all_slices, max_slices
end
----------------------------------


return dataPrepare;