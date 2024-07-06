from datasets import load_dataset, DatasetDict, Dataset
#load_dataset --> function to access/load datasets from online repos like Hugging Face or local files,
#DatasetDict ---> class representing different collection of datasets(e.g. train, validation, test)
#DatasetDict ---> class provided by datasets library used for mananging multiple datasets(yes there can be  different categories of datasets like train , validation , test ), all stored within a single object fo this class. keys represent category of dataset and value represent a individual Dataset objects containting the actual data points.


#Dataset ---> Class representing single dataset (datasets often contains text and labels) that we choosed to work with, this class offers functionalites to access and manipulate the data.


from transformers import(
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)
# yes we can access state of the art model with  just pipeline function(give name of model and task ) of transformer library, but if we want more control (like training the model, fine-tuning or do some advanced thing with the model)  then we use other functions and classes of transformer library
#Autotokenizer ---> this class handles TOKENIZATION ---->process of converting text into sequence of numerical tokens, since model can only understand  number 

from peft import PeftModel , PeftConfig, get_peft_model, LoraConfig
import evaluate
# evaluate library is for evaluation of model(evlautation technique of the model differs from task the model is trained for e.g., accuracy for classification, BLEU score for machine translation)...... How well the fine-tuned model performs on unseen data and identify areas for improvements.
import torch
import numpy as np

model_checkpoint = 'distilbert-base-uncased'

#define label-maps
id2label = { 0: "Negative", 1: "Positive"}
label2id = { "Negative" : 0, "Positive" : 1}
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label = id2label, label2id = label2id)

dataset = load_dataset("shawhin/imdb-truncated") #this dataset is hosted on huggingface_hub

#since machine only understand numbers, lets tokenize them(Preprocessing)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space= True)
# add_prefix_space=True (optional param) ---> adds space at the beginning of each sentence, some models might get help from it


#return_tensors = "np" ---> this instruct the tokenizer on how to return the processed text data into NumPy Arryays ---> 

#create tokenize function
def tokenize_function(examples):
    #extract text
    text = examples["text"]

 #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors = "np",
        truncation = True,
        max_length = 512
    )

    return tokenized_inputs

#add pad token if none exists
if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

#tokenize training and validation datasets
tokenize_dataset = dataset.map(tokenize_function, batched = True)

#create data collator
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
#  DataCollatorWithPadding is a class provided by deep learning libraries like hugging face Transformers.It's a pre-defined data collator specifically built for handling padding of sequences.

accuracy = evaluate.load("accuracy")
#here accuracy is common metric for classification tasks
#p variable contains two argument, one is "prediction" ----> "models predicted label for each data point in the batch" and another is "labels" ---> "true label for datapoint in the batch in the dataset we injected"

# define an evaluation function to pass into trainer later
def compute_metrics(p):
      predictions, labels = p
      predictions = np.argmax(predictions, axis =1)
      return {"accuracy": accuracy.compute(predictions = predictions, references=labels)}

#Untrained model perfomance
text_list =  ["It was good.", "Not a fan, don't recommed.", 
"Better than the first one.", "This is not worth watching even once.", 
"This one is a pass."]

print("Untraind Model Performance:")
print("-------------------")
 #lets calcualte logit for each classes for each datapoint in given dataset
for text in text_list:
      #tokenize text
      inputs = tokenizer.encode(text, return_tensors = "pt") #return data in pytorch Array
      #compute logits.
      logits = model(inputs).logits
      #here we are not tranforming logits into probability distributions hai, idk why
      predictions = torch.argmax(logits)
    #convert number(label) from model into label name
      print(text + "- " + id2label[predictions.tolist()]) #converts tensors into list and then we map the numeric prediction to human text("positive", "negative", "neutral") with id2label dictionary


#LETS FINE-TUNE WITH LORA
#setup configuration for LoRA, it is done by creating object of LoraConfig class
peft_config= LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules=['q_lin'])
#Creating new version of model by changing few parameters with PEFT which we can call PEFT model, scale of trainable parameters was reduced to 100x , lets see what's up with the perfomance
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
#trainable params: 628,994 || all params: 67,584,004 || trainable%: 0.9307, just 1% 

#lets define hyperparmeters 
lr = 1e-3 #0.001, size of optimization step
batch_size = 4
num_epochs = 10

#defining Training Arguments
training_args = TrainingArguments(
      output_dir = model_checkpoint + "-lora-text-classification",
      learning_rate = lr,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      num_train_epochs=num_epochs,
      weight_decay = 0.01, #to prevent overfitting
      evaluation_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end = True,
)

# QUESTION . We have earlier defined learning rate ( lora_alpha) and droopout(to prevent overfiiting) for LoRA in peft configuration then why we are again defining it inside TrainingArguments, is it redundacy or are they both different so that we have to include twice, if different , does that mean whatever happening in LoRA is separate?

#Now lets train. 🥳

#create trainer object
trainer = Trainer(
      model = model, #our peft model
      args = training_args, #hyperparameters ,defines how to do training
      train_dataset=tokenize_dataset["train"], #tokenized training data
      eval_dataset=tokenize_dataset["validation"], #tokenized validation data
      tokenizer=tokenizer,
      data_collator=data_collator, #adds padding on each batch 
      compute_metrics=compute_metrics,#measure acccuracy on validation dataset
)

#train model
trainer.train()

#Trained Model Performance
#move model to .mps for better GPU computation on Apple's Chip
model.to('mps')
print("Trained_Model_Perfomance")
print("---------------")
for text in text_list:
      inputs = tokenizer.encode(text, return_tensors="pt").to("mps")
      logits = model(inputs).logits
      predictions = torch.max(logits,1).indices
      print(text + "-" + id2label[predictions.tolist()[0]]) #we get the array of predictions for each text in a batch_size , convert it into Python list, we only retrieve the current text in a text list i.e. [0] and convert it into human readable label.













    
      



