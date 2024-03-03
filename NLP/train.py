import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda
from tqdm import tqdm
from datetime import datetime
import os 
from data_loader import Dataset, DataLoader
currentDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
sys.path.append(os.path.join(parentDir, "src"))
from trackerPACT import PACT

device = 'cuda' if cuda.is_available() else 'cpu'

def prepare_result_dir(model, result_dir):
  try:
    if len(result_dir) == 0:
      if not os.path.exists('experiments'):
        os.makedirs('experiments')
      else:
        result_dir = "experiments"
      now = datetime.now()
      directory = now.strftime("%Y-%m-%d_%H%M%S_experiment_"+str(model))
      path = os.path.join(result_dir, directory)
      os.mkdir(path)
    else:
      path = result_dir
    return path
  except FileNotFoundError as e:
    print(e)

#Training function
def train(model, data_loader, optimizer, device, scheduler, n_examples):
  print("Training the Model")
  model = model.train()
  losses = []
  correct_predictions = 0
  for data in tqdm(data_loader):
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    targets = data["target"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      labels = targets
    )
    _, preds = torch.max(outputs[1], dim=1)  # the second return value is logits
    loss = outputs[0] #the first return value is loss
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

#Evaluation function 
def eval(model, data_loader, device, n_examples):
  print("Validating the Model")
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for data in tqdm(data_loader):
      input_ids = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      targets = data["target"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels = targets
      )
      _, preds = torch.max(outputs[1], dim=1)
      loss = outputs[0]
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

#Prediction function
def get_predictions(model, data_loader):
  print("Testing the Best-Perfomred Model")
  model = model.eval()
  sequences = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for data in tqdm(data_loader):
      texts = data["sequence"]
      input_ids = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      targets = data["target"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
	      labels = targets
      )
      _, preds = torch.max(outputs[1], dim=1)
      sequences.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs[1])
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return sequences, predictions, prediction_probs, real_values



events_groups = [['MIGRATIONS'],['FAULTS'],['CACHE-MISSES'],['CYCLES']]
@PACT(measure_period= 1, perf_measure_period = 0.01, events_groups = events_groups, tracker_file_name = "./PACTData/PACT.csv")
def performTraining(args):
    for epoch in range(args.epoch):
        print(f'Epoch {epoch + 1}/{args.epoch}')
        print("\n")
        train_acc, train_loss = train(
                    args.model,
                    args.train_data_loader,
                    args.optimizer,
                    args.device,
                    args.scheduler,
                    args.train_set.__len__()
            )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        print("\n")
    return

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model', type=str, default='bert-base-uncased')
  parser.add_argument('-train', type=str, default='')
  parser.add_argument('-dev', type=str, default='')
  parser.add_argument('-max_sequence_len', type=int, default=64)
  parser.add_argument('-epoch', type=int, default=10)
  parser.add_argument('-train_batch_size', type=int, default=16)
  parser.add_argument('-valid_batch_size', type=int, default=16)
  parser.add_argument('-res', type=str, default='')
  parser.add_argument('-lr', type=float, default=2e-5)
  parser.add_argument('-n_warmup_steps', type=int, default=0)
  args = parser.parse_args()
  args.model = args.model.lower()

  res_path = prepare_result_dir(args.model, args.res)
  logger = open(res_path + "/log.txt", "w")
  logger.write("Model: " + args.model + "\n")
  
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  
  train_set = Dataset(args.train, tokenizer, args.max_sequence_len)
  classes, encoded_classes, train_set_shape = train_set.get_info()
  logger.write("Label Encoding: " + str(classes) + "-->" + str(np.sort(encoded_classes)) + "\n")
  encoded_classes = encoded_classes.astype(str)
  logger.write("shape of the train set: {} \n".format(train_set_shape))
  train_data_loader = DataLoader(train_set, args.train_batch_size, shuffle  = False, num_workers = 0)

  model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = len(encoded_classes)) 
  model = model.to(device)
  args.model = args.model.replace("/", "_")
  optimizer = AdamW(params =  model.parameters(), lr = args.lr)
  total_steps = len(train_data_loader) * args.epoch
  scheduler = get_linear_schedule_with_warmup(
              optimizer,
              num_warmup_steps = args.n_warmup_steps,
              num_training_steps = total_steps
          )

  history = defaultdict(list)
  best_accuracy = best_epoch = 0

  args.logger = logger
  args.model = model
  args.train_data_loader = train_data_loader
  args.optimizer = optimizer
  args.device = device
  args.scheduler = scheduler
  args.train_set = train_set

  performTraining(args)

  print("Training Process Finished!")

  
if __name__ == "__main__":
  main()
