
import numpy as np
from dataset import Dataset
from transformers import Trainer, AutoModelForCausalLM, TrainingArguments

# create a fake dataset
data = np.random.randint(0, 1000, (8192, 64)).tolist()
dataset = Dataset.from_dict({"input_ids": data, "labels": data})

# Training a model using Trainer API
trainer = Trainer(
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B"),
    args=TrainingArguments(
        run_name="fake-training",
        report_to="trackio",
        train_dataset=dataset,
    )
)
trainer.train()
trainer.evaluate()